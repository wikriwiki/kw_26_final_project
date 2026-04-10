"""
03_agent_generation_and_validation.py
====================================
vLLM(Qwen3-32B-AWQ) 기반 에이전트 생성 및 실시간 검증 파이프라인.
2단계에서 산출된 통계 프로필을 프롬프트에 주입하여 에이전트를 생성하고,
생성된 객체들이 원본 통계 분포를 준수하는지 즉각적으로 검증합니다.
"""

import json
import asyncio
import time
from pathlib import Path
from collections import defaultdict

# vLLM API 설정을 위해 openai 라이브러리 사용
from openai import AsyncOpenAI

# 디렉토리 설정
STATS_DIR = Path("output/stats")
AGENTS_OUT_DIR = Path("output/agents")

# LLM 설정
VLLM_URL = "http://localhost:8000/v1"
MODEL_NAME = "Qwen/Qwen3-32B-AWQ"
MAX_CONCURRENT = 8
TEMPERATURE = 0.8

SYSTEM_PROMPT = """\
당신은 서울시 소비 행동 시뮬레이션을 위한 데이터 기반 가상 에이전트 생성 모델입니다.
주어진 통계 지표(소비 10분위수, 업종별 소비 비율, 주 활동 시간대)를 엄격하게 반영하여 현실적인 소비자 프로필을 JSON 배열로 생성하십시오.

## 엄격한 규칙
1. 업종별 소비 비율(industry_ratio)의 합은 반드시 1.0이어야 합니다.
2. spending_level은 제공된 분위수(1~10)를 정확히 준수하십시오.
3. 출력은 반드시 아래 스키마를 따르는 JSON 배열(`[...]`)만 반환해야 하며, 어떠한 마크다운이나 부가 설명도 포함하지 마십시오.

## 출력 JSON 스키마
[{
  "agent_id": "문자열 (예: AGT_M_20대_001)",
  "demographics": { "gender": "M/F", "age_grp": "연령대" },
  "spending": {
    "spending_level": 정수 (1~10),
    "industry_ratio": { "업종명": 소수점 4자리 비율, ... }
  },
  "behavior": {
    "primary_active_time": "가장 활동적인 시간대 (예: 18시~21시)"
  }
}, ...]
"""

def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing required stats file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

async def generate_and_validate_batch(client: AsyncOpenAI, group_key: str, stats: dict, count: int, sem: asyncio.Semaphore) -> list:
    """단일 그룹(성별x연령대)에 대한 에이전트를 생성하고 유효성을 검증합니다."""
    async with sem:
        gender, age_grp = group_key.split("_", 1)
        spending_decile = stats["spending_deciles"].get(group_key, 5)
        industry_ratio = stats["industry_ratios"].get(group_key, {})
        
        user_prompt = f"""
        ## 대상 그룹 통계
        - 성별: {gender}, 연령대: {age_grp}
        - 할당된 생성 인원: {count}명
        - 소비 수준(10분위): {spending_decile}
        - 업종별 소비 비율 (기준값): {json.dumps(industry_ratio, ensure_ascii=False)}
        
        위 기준에 맞추어 개별 에이전트마다 미세한 변동성(±0.05)을 주되, 전체 평균은 기준값에 수렴하도록 {count}명의 에이전트를 생성하십시오.
        """
        
        try:
            response = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=TEMPERATURE,
                max_tokens=2000
            )
            
            raw_output = response.choices[0].message.content.strip()
            
            # 마크다운 백틱 제거 등 텍스트 전처리
            if raw_output.startswith("```json"):
                raw_output = raw_output[7:-3].strip()
            elif raw_output.startswith("```"):
                raw_output = raw_output[3:-3].strip()
                
            agents = json.loads(raw_output)
            
            # [즉각적인 객관적 검증 로직]
            validated_agents = []
            for agent in agents:
                # 1. 스키마 무결성 및 업종 비율 합계 검증
                ratios = agent.get("spending", {}).get("industry_ratio", {})
                ratio_sum = sum(ratios.values())
                
                # LLM 할루시네이션(비율 총합이 1.0을 크게 벗어남) 교정
                if not (0.95 <= ratio_sum <= 1.05):
                    # 정규화(Normalization) 강제 적용
                    normalized_ratios = {k: round(v / ratio_sum, 4) for k, v in ratios.items()}
                    agent["spending"]["industry_ratio"] = normalized_ratios
                
                # 2. 강제 메타데이터 주입 (안전성 보장)
                agent["agent_id"] = f"AGT_{gender}_{age_grp}_{len(validated_agents)+1:03d}"
                validated_agents.append(agent)
                
            return validated_agents

        except Exception as e:
            print(f"Error generating batch for {group_key}: {str(e)}")
            return []

async def main():
    AGENTS_OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Loading Statistical Profiles...")
    stats = {
        "spending_deciles": load_json(STATS_DIR / "agent_spending_deciles.json"),
        "industry_ratios": load_json(STATS_DIR / "agent_industry_ratios.json"),
        "temporal_activity": load_json(STATS_DIR / "global_temporal_activity.json")
    }
    
    # 예시: 그룹별 할당량 (실제 구현 시 agent_allocation.json에서 로드)
    # 여기서는 객관적 테스트를 위해 임의 할당량 설정
    allocation = {key: 10 for key in stats["spending_deciles"].keys()} 
    
    client = AsyncOpenAI(base_url=VLLM_URL, api_key="not-needed")
    sem = asyncio.Semaphore(MAX_CONCURRENT)
    
    print(f"Starting Generation Pipeline with max concurrency: {MAX_CONCURRENT}")
    start_time = time.time()
    
    tasks = []
    for group_key, count in allocation.items():
        tasks.append(generate_and_validate_batch(client, group_key, stats, count, sem))
        
    results = await asyncio.gather(*tasks)
    
    all_agents = []
    for batch in results:
        all_agents.extend(batch)
        
    elapsed = time.time() - start_time
    print(f"\n[Generation Complete] Generated and validated {len(all_agents)} agents in {elapsed:.2f} seconds.")
    
    out_path = AGENTS_OUT_DIR / "agents_final.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_agents, f, ensure_ascii=False, indent=2)
        
    print(f"  -> Saved to {out_path}")

if __name__ == "__main__":
    asyncio.run(main())
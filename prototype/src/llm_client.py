"""
LLM 클라이언트 — Qwen3-32B (Ollama 로컬)

주간 전략 결정에 사용.
에이전트 프로필 + 환경 상태 → 5가지 액션 중 택 1.
"""
import json
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen3:30b"

SYSTEM_PROMPT = """\
당신은 서울에 사는 소비자입니다. 아래 프로필과 환경 정보를 바탕으로 이번 주 소비 전략을 결정하세요.

반드시 아래 JSON 형식으로만 응답하세요. 다른 텍스트 없이 JSON만 출력하세요.

{
    "action": "유지" | "전환" | "온라인전환" | "신규탐색" | "추천",
    "params": {
        "base_dong": "현재 소비 행정동코드",
        "target_dong": "전환 시 목표 행정동코드 (전환이 아니면 null)",
        "preferred_industries": ["선호 업종1", "업종2", "업종3"],
        "avoid_industries": ["회피 업종"],
        "budget_adjustment": 1.0,
        "online_ratio": 0.0,
        "explore_industry": null,
        "recommend_industry": null,
        "recommend_dong": null
    },
    "reasoning": "판단 이유 (한 문장)"
}

액션 설명:
- 유지: 현재 행정동에서 계속 소비
- 전환: 다른 행정동으로 주 소비지 변경
- 온라인전환: 오프라인 소비 줄이고 온라인 비율 높이기 (online_ratio 0~1)
- 신규탐색: 새로운 업종이나 지역 시도 (explore_industry 지정)
- 추천: 특정 장소를 다른 소비자에게 추천 (recommend_industry, recommend_dong 지정)
"""


def build_user_prompt(agent: dict, environment: dict, last_week: dict) -> str:
    """에이전트 프로필 + 환경 + 지난주 요약 → 프롬프트"""

    profile_text = (
        f"세그먼트: {agent['segment']}\n"
        f"성별: {agent['gender']}, 연령대: {agent['age_group']}\n"
        f"소비 행정동: {str(agent['adm_cd'])[:8]}\n"
        f"월 예산: {agent['monthly_spending']:,}원\n"
        f"가격민감도: {agent['price_sensitivity']}, 충성도: {agent['loyalty']}\n"
        f"트렌드민감도: {agent['trend_sensitivity']}, 사회적영향: {agent['social_influence_weight']}\n"
        f"선호업종: {agent.get('top_industries', [])}\n"
    )

    last_week_text = ""
    if last_week:
        last_week_text = (
            f"\n[지난주 소비 요약]\n"
            f"방문 행정동: {last_week.get('visited_dongs', [])}\n"
            f"소비 업종: {last_week.get('industries', [])}\n"
            f"총 소비: {last_week.get('total_spent', 0):,}원\n"
            f"만족도: {last_week.get('satisfaction', 0.5):.1f}\n"
        )

    env_text = ""
    if environment:
        env_text = (
            f"\n[환경 정보]\n"
            f"상권 상태: {environment.get('district_type', 'unknown')}\n"
            f"유동인구 변화: {environment.get('floating_pop_change', '0%')}\n"
            f"정책: {environment.get('active_policy', '없음')}\n"
            f"뉴스/이벤트: {environment.get('events', [])}\n"
        )

    # GraphRAG 컨텍스트 (에이전트에 주입된 경우)
    graph_text = ""
    graph_ctx = agent.get("_graph_rag_context", "")
    if graph_ctx:
        graph_text = f"\n[GraphRAG 검색 결과]:\n{graph_ctx}\n"

    return (
        f"[내 프로필]\n{profile_text}{last_week_text}{env_text}{graph_text}\n"
        f"이번 주 소비 전략을 JSON으로 결정하세요."
    )


def call_llm(prompt: str, timeout: int = 60) -> dict:
    """Ollama API로 Qwen3-32B 호출 → JSON 파싱"""

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "system": SYSTEM_PROMPT,
        "stream": False,
        "options": {
            "temperature": 0.7,
            "top_p": 0.9,
            "num_predict": 512,
        },
    }

    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
        resp.raise_for_status()
        raw = resp.json().get("response", "")

        # JSON 추출 (```json ... ``` 블록이 있을 수 있음)
        text = raw.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        # /think 태그 제거 (Qwen3 thinking mode)
        if "<think>" in text:
            text = text.split("</think>")[-1].strip()

        result = json.loads(text)
        return result

    except requests.exceptions.ConnectionError:
        return _fallback_decision()
    except (json.JSONDecodeError, requests.exceptions.Timeout):
        return _fallback_decision()
    except Exception:
        return _fallback_decision()


def _fallback_decision() -> dict:
    """LLM 호출 실패 시 기본 결정 (유지)"""
    return {
        "action": "유지",
        "params": {
            "base_dong": None,
            "target_dong": None,
            "preferred_industries": [],
            "avoid_industries": [],
            "budget_adjustment": 1.0,
            "online_ratio": 0.0,
            "explore_industry": None,
            "recommend_industry": None,
            "recommend_dong": None,
        },
        "reasoning": "LLM 연결 실패 — 기본값(유지) 적용",
    }


def decide_weekly(agent: dict, environment: dict, last_week: dict) -> dict:
    """에이전트 1명의 주간 전략 결정"""
    prompt = build_user_prompt(agent, environment, last_week)
    result = call_llm(prompt)

    # params 기본값 보정
    params = result.get("params", {})
    params.setdefault("base_dong", str(agent["adm_cd"])[:8])
    params.setdefault("target_dong", None)
    params.setdefault("preferred_industries", agent.get("top_industries", []))
    params.setdefault("avoid_industries", [])
    params.setdefault("budget_adjustment", 1.0)
    params.setdefault("online_ratio", 0.0)
    params.setdefault("explore_industry", None)
    params.setdefault("recommend_industry", None)
    params.setdefault("recommend_dong", None)
    result["params"] = params

    return result


def decide_weekly_batch(
    agents: list[dict],
    environment: dict,
    weekly_summaries: dict,
    max_workers: int = 4,
) -> dict:
    """전체 에이전트 주간 결정을 병렬 처리

    Returns: {agent_id: decision_dict}
    """
    decisions = {}

    def _decide(agent):
        last_week = weekly_summaries.get(agent["agent_id"], {})
        return agent["agent_id"], decide_weekly(agent, environment, last_week)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_decide, a): a for a in agents}
        for future in as_completed(futures):
            agent_id, decision = future.result()
            decisions[agent_id] = decision

    return decisions


if __name__ == "__main__":
    # 연결 테스트
    test_agent = {
        "agent_id": "consumer_0001",
        "segment": "commuter",
        "gender": "남",
        "age_group": "30_39세",
        "adm_cd": "1114055000",
        "monthly_spending": 740000,
        "price_sensitivity": 0.4,
        "loyalty": 0.6,
        "trend_sensitivity": 0.5,
        "social_influence_weight": 0.5,
        "top_industries": ["한식", "카페", "일식"],
    }
    test_env = {
        "district_type": "HL",
        "floating_pop_change": "+5%",
        "active_policy": "없음",
        "events": ["명동 신규 라멘집 오픈"],
    }

    print("Testing Qwen3-32B via Ollama...")
    result = decide_weekly(test_agent, test_env, {})
    print(json.dumps(result, ensure_ascii=False, indent=2))

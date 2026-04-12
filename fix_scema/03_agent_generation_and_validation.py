"""
03_agent_generation_and_validation.py
====================================
vLLM(Qwen3-32B-AWQ) 기반 에이전트 생성 + 다차원 심층 검증 파이프라인.
Feature Parity 100%: Original(generate_agents.py + validate_vs_raw.py) 통합.

주요 개선:
  1. 풍부한 에이전트 스키마 (거주지, 직장, 소비패턴, 행동지표, 성격)
  2. 실시간 통계적 정합성 검증 (MAPE 30% 이내)
  3. 견고한 JSON 파싱 (<think> 태그, 코드펜스, 다중 객체)
  4. Resume 기능 (partial/ 디렉토리)
  5. 인구 비례 배분 (agent_allocation.json 기반)
"""

import json
import asyncio
import argparse
import time
import random
import re
import logging
from pathlib import Path
from typing import Any
from collections import defaultdict

# ─────────────────────────────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────────────────────────────
DEFAULT_VLLM_URL = "http://localhost:8000/v1"
MODEL_NAME = "Qwen/Qwen3-32B-AWQ"

STATS_DIR  = Path("output/stats")
OUTPUT_DIR = Path("output/agents")

MAX_RETRIES  = 3
TEMPERATURE  = 0.85
MAX_TOKENS   = 2500
MAPE_THRESHOLD = 0.30  # 30% 허용 오차

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(OUTPUT_DIR / "generation.log", mode="a", encoding="utf-8")
        if OUTPUT_DIR.exists() else logging.StreamHandler()
    ],
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# System Prompt — 다차원 스키마
# ─────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
당신은 서울시 소비 행동 시뮬레이션을 위한 가상 에이전트를 생성하는 전문가입니다.
주어진 통계 데이터를 기반으로 현실적이고 개별화된 소비자 프로필을 JSON으로 생성합니다.

## 규칙
1. 반드시 아래 JSON 스키마를 정확히 따르세요.
2. spending_level/mobility_level은 10분위 범주(1=하위10%, 10=상위10%)입니다.
   같은 분위 안에서 로그정규분포를 가정하고, 분위 하한에 가까운 값이 더 많습니다.
3. industry_ratio는 업종별 소비비율(합계=1.0)입니다.
   이 비율을 중심으로 에이전트마다 ±0.05 범위에서 변동시키되 합계는 1.0을 유지하세요.
4. 연령대에 맞는 구체적 나이를 정하세요 (예: 30대 → 30~39세).
5. 직업은 연령대, 성별, 거주지 상권 특성을 고려하여 현실적으로 결정하세요.
6. 직장 위치는 workplace_flow 확률분포를 참조하여 결정하세요.
7. 평일 소비와 주말 소비의 업종 비중을 구분하세요.
8. 20세미만은 학생/아르바이트만 가능합니다.
9. 소비액은 소득 수준과 일관되어야 합니다.
10. 통근시간은 거주지와 직장 위치 거리에 비례해야 합니다.
11. 텔레콤 행동 지표(배달일수, 이동거리 등)의 에이전트 평균이 제시된 그룹 평균과 유사해야 합니다.
12. 반드시 JSON만 출력하세요. 설명이나 마크다운 없이 순수 JSON 배열만 출력합니다.

## 출력 스키마
각 에이전트는 다음 JSON 형식을 따라야 합니다:
{
  "agent_id": "AGT_{행정동코드}_{성별}_{연령대}_{3자리순번}",
  "residence": {
    "dong_code": "행정동 8자리 코드",
    "dong": "행정동 이름",
    "gu": "자치구 이름"
  },
  "personal": {
    "age": 구체적나이(정수),
    "gender": "M 또는 F",
    "age_group": "연령대",
    "job": "직업 (구체적으로)",
    "income_level": "하/중하/중/중상/상 중 하나",
    "life_stage": "라이프스테이지 (예: 학생, 사회초년생, 신혼, 자녀양육, 은퇴 등)"
  },
  "workplace": {
    "dong_code": "직장 행정동코드 (무직/학생이면 null)",
    "dong": "직장 행정동명",
    "commute_min": 통근시간(분, 정수)
  },
  "spending": {
    "spending_level": 10분위수(정수),
    "weekday_top_categories": { "업종": 비율, ... },
    "weekend_top_categories": { "업종": 비율, ... },
    "weekend_weekday_ratio": 주말대비평일소비비율(소수)
  },
  "behavior": {
    "delivery_days": 월간배달사용일수,
    "shopping_days": 월간쇼핑사용일수,
    "weekday_move_km": 평일이동거리(km),
    "weekend_move_km": 휴일이동거리(km),
    "home_hours_weekday": 평일재택시간,
    "mobility_level": 이동활발도10분위(정수)
  },
  "personality": {
    "spending_tendency": "절약형/보통/소비형 중 하나",
    "lifestyle": "한 단어~짧은 문구"
  }
}

복수 에이전트 요청 시, JSON 배열 []로 감싸서 출력하세요.
/no_think"""


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="LLM 기반 에이전트 대량 생성")
    p.add_argument("--vllm-url", default=DEFAULT_VLLM_URL)
    p.add_argument("--stats-dir", type=Path, default=STATS_DIR)
    p.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    p.add_argument("--max-concurrent", type=int, default=8,
                    help="동시 LLM 요청 수 (default: 8)")
    p.add_argument("--resume", action="store_true",
                    help="이전 중단 지점부터 재개")
    p.add_argument("--limit", type=int, default=0,
                    help="처리할 그룹 수 제한 (0=전체)")
    p.add_argument("--dry-run", action="store_true",
                    help="실제 LLM 호출 없이 프롬프트만 출력")
    return p.parse_args()


def load_json(path: Path) -> Any:
    if not path.exists():
        logger.warning(f"JSON not found: {path}")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def extract_json_from_text(text: str) -> list[dict]:
    """LLM 응답에서 JSON 추출 (think 태그, 코드펜스, 다중 객체 대응)"""
    text = text.strip()

    # <think>...</think> 태그 제거
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # ```json ... ``` 블록 추출
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if m:
        text = m.group(1).strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
        elif isinstance(parsed, dict):
            if "agents" in parsed:
                return parsed["agents"]
            return [parsed]
    except json.JSONDecodeError:
        pass

    # 여러 JSON 객체가 연속된 경우 — 스택 기반 파싱
    objects = []
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    objects.append(json.loads(text[start:i + 1]))
                except json.JSONDecodeError:
                    pass
                start = None
    return objects


# ─────────────────────────────────────────────────────────────────────
# Prompt Builder
# ─────────────────────────────────────────────────────────────────────
def build_user_prompt(
    group_key: str,
    count: int,
    profile: dict,
    dong_ctx: dict | None,
    wf: list | None,
    global_dist: dict | None,
    agg_stats: dict | None,
    consump_detail: dict | None,
) -> str:
    loc = profile.get("location", {})
    demo = profile.get("demographics", {})
    tel = profile.get("telecom", {})
    cons = profile.get("consumption", {})
    mob = profile.get("mobility", {})

    dong_code = loc.get("adm_cd_8", "?")
    dong_name = loc.get("dong", "?")
    gu_name   = loc.get("gu", "?")
    gender    = demo.get("gender", "?")
    age_grp   = demo.get("age_grp", "?")
    population = demo.get("population", 0)

    spending_level = cons.get("spending_level", 5)
    mobility_level = mob.get("mobility_level", 5)
    mob_wd_we_ratio = mob.get("weekend_weekday_ratio", 1.0)

    parts = []
    parts.append(f"## 그룹 정보")
    parts.append(f"- 위치: {gu_name} {dong_name} ({dong_code})")
    parts.append(f"- 성별: {'남성' if gender == 'M' else '여성'}")
    parts.append(f"- 나이대: {age_grp}")
    parts.append(f"- 그룹 인구: {population:.0f}명")
    parts.append(f"- 생성할 에이전트 수: **{count}명**")
    parts.append("")

    # 소비/이동 분위수
    parts.append("## 소비/이동 수준 (10분위: 1=하위10%, 10=상위10%)")
    parts.append(f"- 소비수준: {spending_level}/10")
    parts.append(f"- 이동활발도: {mobility_level}/10")
    parts.append(f"- 주말/평일 이동비: {mob_wd_we_ratio:.2f}")

    # consumption_detail
    if consump_detail:
        detail_level = consump_detail.get("detail_spending_level", spending_level)
        wd_ratio = consump_detail.get("weekday_ratio", {})
        we_ratio = consump_detail.get("weekend_ratio", {})
        wd_we_spend = consump_detail.get("weekend_weekday_spending_ratio", 1.0)
        parts.append(f"- 세부소비수준: {detail_level}/10")
        if wd_we_spend:
            parts.append(f"- 주말/평일 소비비: {wd_we_spend:.2f}")
        if wd_ratio:
            parts.append(f"- 평일 업종 비중: {json.dumps(dict(list(wd_ratio.items())[:7]), ensure_ascii=False)}")
        if we_ratio:
            parts.append(f"- 주말 업종 비중: {json.dumps(dict(list(we_ratio.items())[:7]), ensure_ascii=False)}")
    parts.append("")

    # 업종별 소비비율 (global)
    gender_age_key = f"{gender}_{age_grp}"
    if global_dist:
        ind_ratio = global_dist.get("industry_spending_ratio", {}).get(gender_age_key, {})
        if ind_ratio:
            top8 = dict(list(ind_ratio.items())[:8])
            parts.append("## 업종별 소비비율 (같은 성별x연령대 서울 전체, 합계=1.0)")
            for industry, ratio in top8.items():
                parts.append(f"- {industry}: {ratio:.1%}")
            parts.append("")

    # 텔레콤 지표
    key_metrics = {
        "출근 소요시간(분)": tel.get("tel_commute_time"),
        "배달 사용일수/월": tel.get("tel_delivery_days"),
        "쇼핑 사용일수/월": tel.get("tel_shopping_days"),
        "평일 이동거리(km)": round(tel.get("tel_wd_move_dist", 0) / 1000, 1) if tel.get("tel_wd_move_dist") else None,
        "휴일 이동거리(km)": round(tel.get("tel_we_move_dist", 0) / 1000, 1) if tel.get("tel_we_move_dist") else None,
        "지하철 이용일수/월": tel.get("tel_subway_days"),
        "평일 재택시간(시간)": round(tel.get("tel_home_wd_time", 0) / 3600, 1) if tel.get("tel_home_wd_time") else None,
    }
    has_metrics = any(v is not None for v in key_metrics.values())
    if has_metrics:
        parts.append("## 이 그룹의 평균 행동 지표 (텔레콤 기반)")
        for label, val in key_metrics.items():
            if val is not None:
                parts.append(f"- {label}: {val}")
        parts.append("")

    # aggregate_stats (mean/std)
    if agg_stats:
        parts.append("## 같은 성별x연령대 동 간 분포 (에이전트 변동 참고)")
        for metric_name in ["tel_commute_time", "tel_delivery_days",
                            "tel_wd_move_dist", "tel_subway_days"]:
            stat = agg_stats.get(metric_name)
            if stat:
                parts.append(
                    f"- {metric_name}: mean={stat['mean']:.1f}, "
                    f"std={stat['std']:.1f}, "
                    f"범위=[{stat['min']:.1f}, {stat['max']:.1f}]"
                )
        parts.append("")

    # 동 상권 환경
    if dong_ctx:
        parts.append("## 동 상권 환경")
        ctx_labels = {
            "b069_sales": "매출지수", "b069_infra": "인프라지수",
            "b069_store": "가맹점지수", "b069_pop": "인구지수",
            "b079_seoul_inflow_ratio": "서울유입비율",
        }
        for k, label in ctx_labels.items():
            v = dong_ctx.get(k)
            if v is not None:
                parts.append(f"- {label}: {v}")
        parts.append("")

    # 직장 확률분포
    if wf:
        parts.append("## 거주지->직장 확률분포 (출근 목적지)")
        for entry in wf[:6]:
            parts.append(f"- {entry['dong']}: {entry['probability']:.2%}")
        parts.append("")

    parts.append(f"위 통계를 기반으로 **{count}명**의 현실적인 에이전트를 JSON 배열로 생성하세요.")
    return "\n".join(parts)


# ─────────────────────────────────────────────────────────────────────
# 통계적 정합성 검증 (Inline Validation)
# ─────────────────────────────────────────────────────────────────────
def validate_agent_batch(
    agents: list[dict],
    group_key: str,
    agg_stats: dict | None,
    profile: dict,
) -> list[dict]:
    """생성된 에이전트 배치에 대한 다차원 검증 및 교정"""

    adm8, gender, age = group_key.rsplit("_", 2)

    for i, agent in enumerate(agents):
        # 1. agent_id 고유성 보장 (행정동 포함)
        agent["agent_id"] = f"AGT_{adm8}_{gender}_{age}_{i + 1:03d}"

        # 2. 거주지 메타데이터 강제 주입
        if "residence" not in agent:
            agent["residence"] = {}
        agent["residence"]["dong_code"] = adm8

        loc = profile.get("location", {})
        if not agent["residence"].get("dong"):
            agent["residence"]["dong"] = loc.get("dong", "")
        if not agent["residence"].get("gu"):
            agent["residence"]["gu"] = loc.get("gu", "")

        # 3. 인적사항 강제 보정
        if "personal" not in agent:
            agent["personal"] = {}
        agent["personal"]["gender"] = gender
        agent["personal"]["age_group"] = age

        # 4. spending industry_ratio 정규화 (합계 → 1.0)
        spending = agent.get("spending", {})
        for cat_key in ["weekday_top_categories", "weekend_top_categories"]:
            ratios = spending.get(cat_key, {})
            if ratios:
                ratio_sum = sum(ratios.values())
                if ratio_sum > 0 and not (0.95 <= ratio_sum <= 1.05):
                    spending[cat_key] = {
                        k: round(v / ratio_sum, 4) for k, v in ratios.items()
                    }

        # 5. workplace 스키마 보장
        if "workplace" not in agent:
            agent["workplace"] = {"dong_code": None, "dong": "", "commute_min": 0}

        # 6. behavior 스키마 보장
        if "behavior" not in agent:
            agent["behavior"] = {}

    # 7. 배치 레벨 통계적 정합성 검증 (MAPE 로깅)
    if agg_stats and len(agents) >= 3:
        demo_key = f"{gender}_{age}"
        stats = agg_stats.get(demo_key, {})

        metric_extractors = {
            "tel_commute_time": lambda a: _safe_float(a.get("workplace", {}).get("commute_min")),
            "tel_delivery_days": lambda a: _safe_float(a.get("behavior", {}).get("delivery_days")),
            "tel_shopping_days": lambda a: _safe_float(a.get("behavior", {}).get("shopping_days")),
            "tel_wd_move_dist": lambda a: (
                _safe_float(a.get("behavior", {}).get("weekday_move_km")) or 0) * 1000,
        }

        for metric_name, extractor in metric_extractors.items():
            stat = stats.get(metric_name)
            if not stat or stat.get("mean", 0) == 0:
                continue

            agent_vals = [extractor(a) for a in agents]
            agent_vals = [v for v in agent_vals if v is not None and v > 0]
            if not agent_vals:
                continue

            agent_mean = sum(agent_vals) / len(agent_vals)
            raw_mean = stat["mean"]
            mape = abs(agent_mean - raw_mean) / abs(raw_mean)

            if mape > MAPE_THRESHOLD:
                logger.warning(
                    f"[MAPE] {group_key} | {metric_name}: "
                    f"raw={raw_mean:.1f} agent={agent_mean:.1f} "
                    f"MAPE={mape:.1%} > {MAPE_THRESHOLD:.0%} threshold"
                )

    return agents


def _safe_float(v):
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


# ─────────────────────────────────────────────────────────────────────
# LLM Caller
# ─────────────────────────────────────────────────────────────────────
async def call_vllm(client, system_prompt: str, user_prompt: str, model: str = MODEL_NAME) -> str:
    response = await asyncio.to_thread(
        client.chat.completions.create,
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
    return response.choices[0].message.content


async def generate_group(
    client,
    group_key: str,
    count: int,
    profiles: dict,
    dong_ctx: dict,
    workplace_flow: dict,
    global_dist: dict,
    agg_stats: dict,
    consump_detail: dict,
    sem: asyncio.Semaphore,
) -> tuple[str, list[dict]]:
    """단일 그룹에 대한 LLM 호출 → 검증 → 에이전트 반환"""
    async with sem:
        parts = group_key.rsplit("_", 2)
        if len(parts) != 3:
            return group_key, []
        adm8, gender, age = parts

        profile = profiles.get(group_key, {})
        d_ctx = dong_ctx.get(adm8)
        wf = workplace_flow.get(adm8)
        demo_key = f"{gender}_{age}"
        agg = agg_stats.get(demo_key)
        cd = consump_detail.get(group_key)

        user_prompt = build_user_prompt(
            group_key=group_key,
            count=count,
            profile=profile,
            dong_ctx=d_ctx,
            wf=wf,
            global_dist=global_dist,
            agg_stats=agg,
            consump_detail=cd,
        )

        for attempt in range(MAX_RETRIES):
            try:
                raw = await call_vllm(client, SYSTEM_PROMPT, user_prompt)
                agents = extract_json_from_text(raw)

                if not agents:
                    if attempt < MAX_RETRIES - 1:
                        continue
                    logger.warning(f"No agents parsed for {group_key} after {MAX_RETRIES} retries")
                    return group_key, []

                # 검증 + 교정
                agents = validate_agent_batch(
                    agents[:count], group_key, agg_stats, profile
                )

                return group_key, agents

            except Exception as e:
                wait = 2 ** attempt + random.random()
                logger.error(f"Error for {group_key} (attempt {attempt+1}): {e}")
                await asyncio.sleep(wait)

        return group_key, []


# ─────────────────────────────────────────────────────────────────────
# Resume 로직
# ─────────────────────────────────────────────────────────────────────
def load_resume_state(partial_dir: Path) -> tuple[list[dict], set]:
    """partial/ 디렉토리에서 기존 결과를 로드하여 재개 상태 구성"""
    done_keys = set()
    existing_agents = []

    if not partial_dir.exists():
        return existing_agents, done_keys

    for pf in sorted(partial_dir.glob("batch_*.json")):
        try:
            with open(pf, "r", encoding="utf-8") as f:
                batch_data = json.load(f)
            batch_agents = batch_data.get("agents", [])
            batch_keys = batch_data.get("completed_keys", [])

            # 유효성 검증: agents가 실제 list[dict]인지
            if isinstance(batch_agents, list) and all(
                isinstance(a, dict) for a in batch_agents
            ):
                existing_agents.extend(batch_agents)
            else:
                logger.warning(f"Invalid agents format in {pf.name} — skipping")
                continue

            if isinstance(batch_keys, list):
                done_keys.update(batch_keys)
            else:
                logger.warning(f"Invalid completed_keys in {pf.name}")

        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Error loading {pf.name}: {e} — skipping")
            continue

    logger.info(f"Resume: {len(existing_agents)} agents, {len(done_keys)} groups loaded")
    return existing_agents, done_keys


# ─────────────────────────────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────────────────────────────
async def run(args):
    # 출력 디렉토리 생성
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 로깅 파일 핸들러 추가 (디렉토리 생성 후)
    log_path = args.output_dir / "generation.log"
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)

    # --- Load stats ---
    logger.info("Loading stats ...")
    profiles     = load_json(args.stats_dir / "agent_profiles.json")
    allocation   = load_json(args.stats_dir / "agent_allocation.json")
    dong_ctx     = load_json(args.stats_dir / "dong_context.json")
    workplace_flow = load_json(args.stats_dir / "workplace_flow.json")
    global_dist  = load_json(args.stats_dir / "global_distributions.json")
    agg_stats    = load_json(args.stats_dir / "aggregate_stats.json")
    consump_detail = load_json(args.stats_dir / "consumption_detail.json")

    if not allocation:
        logger.error("agent_allocation.json is empty — cannot proceed.")
        return

    total_agents = sum(v for v in allocation.values() if isinstance(v, (int, float)))
    total_groups = len([v for v in allocation.values() if isinstance(v, (int, float)) and v > 0])
    logger.info(f"  Total: {total_agents:,} agents, {total_groups:,} groups")

    # --- Key list ---
    keys = [k for k, v in allocation.items() if isinstance(v, (int, float)) and v > 0]
    if args.limit > 0:
        keys = keys[:args.limit]
        logger.info(f"  Limited to {args.limit} groups")

    # --- Resume ---
    partial_dir = args.output_dir / "partial"
    done_keys = set()
    existing_agents = []

    if args.resume:
        existing_agents, done_keys = load_resume_state(partial_dir)

    remaining_keys = [k for k in keys if k not in done_keys]
    logger.info(f"  Remaining: {len(remaining_keys)} groups")

    # --- Dry Run ---
    if args.dry_run:
        print("\nDry-run: showing first prompt")
        if remaining_keys:
            k = remaining_keys[0]
            parts = k.rsplit("_", 2)
            adm8, gender, age = parts
            prompt = build_user_prompt(
                k, allocation[k], profiles.get(k, {}),
                dong_ctx.get(adm8), workplace_flow.get(adm8),
                global_dist, agg_stats.get(f"{gender}_{age}"),
                consump_detail.get(k),
            )
            print("=" * 60)
            print("[SYSTEM PROMPT]")
            print(SYSTEM_PROMPT[:500] + "...")
            print("=" * 60)
            print("[USER PROMPT]")
            print(prompt)
        return

    # --- LLM ---
    from openai import OpenAI
    client = OpenAI(base_url=args.vllm_url, api_key="not-needed")
    sem = asyncio.Semaphore(args.max_concurrent)

    logger.info(f"Starting generation (concurrent={args.max_concurrent})")
    start_time = time.time()

    all_agents = list(existing_agents)
    batch_num = len(done_keys)

    # Chunk 단위 처리
    chunk_size = args.max_concurrent * 2
    for chunk_start in range(0, len(remaining_keys), chunk_size):
        chunk_keys = remaining_keys[chunk_start:chunk_start + chunk_size]

        tasks = [
            generate_group(
                client, k, int(allocation[k]),
                profiles, dong_ctx, workplace_flow,
                global_dist, agg_stats, consump_detail, sem,
            )
            for k in chunk_keys
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        chunk_agents = []
        chunk_done = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Task exception: {result}")
                continue
            gk, agents = result
            if agents:
                chunk_agents.extend(agents)
            chunk_done.append(gk)

        all_agents.extend(chunk_agents)
        done_keys.update(chunk_done)

        # Partial save
        if chunk_agents:
            batch_num += 1
            partial_dir.mkdir(parents=True, exist_ok=True)
            save_json(
                {"agents": chunk_agents, "completed_keys": chunk_done},
                partial_dir / f"batch_{batch_num:04d}.json",
            )

        elapsed = time.time() - start_time
        done_count = chunk_start + len(chunk_keys)
        pct = len(all_agents) / max(total_agents, 1) * 100
        eta = (elapsed / max(done_count, 1)) * (len(remaining_keys) - done_count)
        logger.info(
            f"  [{done_count}/{len(remaining_keys)}] "
            f"{len(all_agents):,}/{total_agents:,} agents ({pct:.1f}%) | "
            f"{elapsed:.0f}s elapsed | ETA {eta:.0f}s"
        )

    # --- Final save ---
    final_path = args.output_dir / "agents_final.json"
    save_json(all_agents, final_path)

    elapsed = time.time() - start_time
    logger.info(f"Done: {len(all_agents):,} agents in {elapsed:.1f}s")
    logger.info(f"Output: {final_path}")

    if len(all_agents) != total_agents:
        logger.warning(
            f"Target {total_agents:,} != actual {len(all_agents):,} — "
            f"use --resume to retry"
        )

    # --- Validation Summary ---
    print(f"\n{'=' * 60}")
    print(f"  Generation Complete")
    print(f"{'=' * 60}")
    print(f"  Total agents: {len(all_agents):,}")
    print(f"  Target:       {total_agents:,}")
    print(f"  Coverage:     {len(all_agents)/max(total_agents,1)*100:.1f}%")
    print(f"  Unique dongs: {len(set(a.get('residence',{}).get('dong_code') for a in all_agents))}")
    print(f"  Unique jobs:  {len(set(a.get('personal',{}).get('job') for a in all_agents))}")
    m_cnt = sum(1 for a in all_agents if a.get('personal', {}).get('gender') == 'M')
    f_cnt = sum(1 for a in all_agents if a.get('personal', {}).get('gender') == 'F')
    print(f"  Gender (M/F): {m_cnt} / {f_cnt}")
    print(f"  Log:          {log_path}")


def main():
    args = parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
"""
generate_agents.py
======================
Phase 2: vLLM (Qwen3-32B-AWQ) 기반 소비자 에이전트 ~5,000명 대량 생성

이전 검증된 코드의 안정적 인프라 + 이번 stats 구조에 맞는 프롬프트

사전 조건:
  1. WSL에서 vLLM 서버 실행 중:
     conda activate vllm
     vllm serve Qwen/Qwen3-32B-AWQ --gpu-memory-utilization 0.90 --max-model-len 4096 --port 8000 --trust-remote-code
  2. pip install openai tqdm

사용법:
  python generate_agents.py                       # 기본 실행
  python generate_agents.py --resume              # 중단 지점부터 재개
  python generate_agents.py --max-concurrent 8    # 동시 요청 수 조절
  python generate_agents.py --limit 20            # 20그룹만 시범 생성
"""

import json
import asyncio
import argparse
import time
import random
import re
from pathlib import Path

# 이 파일은 scripts/bdc/ 안에 있음 — 프로젝트 루트는 두 단계 위
PROJECT_ROOT = Path(__file__).resolve().parents[2]
from typing import Any

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_VLLM_URL = "http://localhost:8000/v1"
MODEL_NAME = "Qwen/Qwen3-32B-AWQ"

STATS_DIR = PROJECT_ROOT / "output" / "stats"
OUTPUT_DIR = PROJECT_ROOT / "output" / "agents"

MAX_RETRIES = 3
TEMPERATURE = 0.85
# 1명당 ~700토큰 가정, 최소 1200 / 최대 3500
MAX_TOKENS_PER_AGENT = 700
MAX_TOKENS_MIN = 1200
MAX_TOKENS_MAX = 3500

# ---------------------------------------------------------------------------
# System Prompt — 이번 stats 구조(분위수, 비율)에 맞춤
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
당신은 서울시 소비 행동 시뮬레이션을 위한 가상 에이전트를 생성하는 전문가입니다.
주어진 통계 데이터를 기반으로 현실적이고 개별화된 소비자 프로필을 JSON으로 생성합니다.

## 규칙
1. 반드시 아래 JSON 스키마를 정확히 따르세요.
2. weekday_spending_level / weekend_spending_level / mobility_level은 10분위(1=하위10%, 10=상위10%)입니다.
   평일 소비분위와 주말 소비분위는 서로 다를 수 있으며, 각각의 분위에 해당하는 실제 금액 범위가 프롬프트에 주어집니다.
3. **분위 내 샘플링 규칙 (중요)**: 1분위와 10분위는 long tail 분포입니다.
   - 1분위: **하한은 극단 outlier(예: 월 1회 결제자)**이므로 무시하고, 범위의 **상단 60~95%** 영역에서 뽑으세요. 하한 근처 값을 쓰지 마세요.
   - 10분위: 상한은 극단 고소비자이므로 무시하고, 범위의 **하단 5~40%** 영역에서 뽑으세요.
   - 2~9분위: 범위 중앙 근처에서 로그정규로 약간 변동.
4. 프롬프트에 제시된 [평일 소비액 범위] / [주말 소비액 범위] 안에서 daily_spending_weekday, daily_spending_weekend를 원 단위로 생성하세요.
   **주말/평일 소비비 제약 (최우선, 위반 시 자체 수정)**:
   - `weekend_weekday_spending_ratio = daily_spending_weekend / daily_spending_weekday`
   - 프롬프트에 **"주말/평일 목표비"**가 주어지면, 이 값을 **기준점**으로 ±20% 안으로 맞추세요.
     예: 목표비 1.14 → 허용범위 [0.91, 1.37]. 목표비 0.74 → 허용범위 [0.59, 0.89].
   - 계산된 ratio가 허용범위를 벗어나면 daily_spending_weekday 또는 daily_spending_weekend를 **스스로 조정**하세요.
   - 목표비가 없을 때만 폴백으로 0.3~3.0 범위를 지키세요.
   - ⚠ "주말이 항상 평일보다 적다"는 고정관념을 버리고, **목표비를 그대로 따를 것**. 20~30대는 주말이 평일보다 높은 경우가 많습니다(목표비 > 1.0).
5. industry_ratio는 업종별 소비비율(합계=1.0)입니다.
   이 비율을 중심으로 에이전트마다 ±0.05 범위에서 변동시키되 합계는 1.0을 유지하세요.
6. 연령대에 맞는 구체적 나이를 정하세요 (예: 30대 → 30~39세).
7. 직업은 연령대, 성별, 거주지 상권 특성을 고려하여 현실적으로 결정하세요.
8. 직장 위치는 workplace_flow 확률분포를 참조해 결정하세요.
   - `workplace.dong_code`: 프롬프트의 "거주지->직장 확률분포" 목록에서 **코드를 그대로** 복사
   - `workplace.dong`: 같은 목록의 **괄호 안 동명을 그대로** 복사 (예: `1101053 (사직동)` → dong_code="1101053", dong="사직동")
   - 괄호가 "(동명 불명)"이면 dong=null
   - 임의의 동명/건물명/자치구명 생성 절대 금지
9. 평일 소비와 주말 소비의 업종 비중(weekday_ratio/weekend_ratio)을 구분하여 출력하세요.
10. **텔레콤 지표**: 프롬프트의 배달/지하철 사용일수는 이미 월 기준(0~31). delivery_days는 그 값을 ±3일 변동해 쓰세요.
    shopping_days는 프롬프트에 없습니다 — 직업/연령/소득으로 **독립 추정**하세요 (일반적으로 4~20일/월, 학생 10~15, 30대 워킹 4~10, 고소득 6~15).
    home_hours_weekday / weekend는 프롬프트의 "하루 평균 시간" 값을 ±1.5시간 변동해 쓰고, 반드시 0~24 범위를 지킬 것.
11. 20세미만은 학생/아르바이트만 가능합니다.
12. 소비액은 소득 수준과 일관되어야 합니다.
13. 통근시간은 거주지와 직장 위치 거리에 비례해야 합니다.
14. 반드시 JSON만 출력하세요. 설명이나 마크다운 없이 순수 JSON 배열만 출력합니다.

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
    "weekday_spending_level": 평일소비10분위(정수),
    "weekend_spending_level": 주말소비10분위(정수),
    "daily_spending_weekday": 1일평균평일소비액(원, 정수),
    "daily_spending_weekend": 1일평균주말소비액(원, 정수),
    "weekend_weekday_spending_ratio": 주말평일소비비(소수),
    "weekday_top_categories": { "업종": 비율, ... },
    "weekend_top_categories": { "업종": 비율, ... }
  },
  "behavior": {
    "delivery_days": 월간배달사용일수,
    "shopping_days": 월간쇼핑사용일수,
    "weekday_move_km": 평일이동거리(km),
    "weekend_move_km": 휴일이동거리(km),
    "home_hours_weekday": 평일재택시간,
    "home_hours_weekend": 휴일재택시간,
    "mobility_level": 이동활발도10분위(정수)
  },
  "personality": {
    "spending_tendency": "절약형/보통/소비형 중 하나",
    "lifestyle": "한 단어~짧은 문구"
  }
}

복수 에이전트 요청 시, JSON 배열 []로 감싸서 출력하세요.
/no_think"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
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
    p.add_argument("--target-total", type=int, default=0,
                    help="전체 그룹을 랜덤 샘플링해 대략 N명이 되도록 축소 (0=전체)")
    p.add_argument("--seed", type=int, default=42,
                    help="--target-total 샘플링 시드")
    p.add_argument("--dry-run", action="store_true",
                    help="실제 LLM 호출 없이 프롬프트만 출력")
    return p.parse_args()


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def postprocess_agents(
    agents: list[dict],
    profiles: dict,
    dong_name_lookup: dict[str, str],
    stat7_to_dong: dict[str, str],
) -> dict:
    """생성 후 residence.gu/dong 및 workplace.dong을 신뢰 원본으로 덮어씀. 보정 카운트 반환."""
    # adm8 -> gu 룩업 구성
    adm8_to_gu: dict[str, str] = {}
    for pk, pv in profiles.items():
        adm8 = pk.split("_")[0]
        gu = pv.get("location", {}).get("gu")
        if gu and adm8 not in adm8_to_gu:
            adm8_to_gu[adm8] = gu

    fixed_gu = 0
    fixed_res_dong = 0
    fixed_wp_dong = 0
    cleared_wp = 0
    for a in agents:
        res = a.get("residence") or {}
        adm8 = res.get("dong_code")
        if adm8:
            true_gu = adm8_to_gu.get(adm8)
            true_dong = dong_name_lookup.get(adm8)
            if true_gu and res.get("gu") != true_gu:
                res["gu"] = true_gu
                fixed_gu += 1
            if true_dong and res.get("dong") != true_dong:
                res["dong"] = true_dong
                fixed_res_dong += 1
            a["residence"] = res

        wp = a.get("workplace") or {}
        stat7 = wp.get("dong_code")
        if stat7:
            true_wp = stat7_to_dong.get(str(stat7))
            if true_wp and wp.get("dong") != true_wp:
                wp["dong"] = true_wp
                fixed_wp_dong += 1
            elif not true_wp and wp.get("dong"):
                wp["dong"] = None
                cleared_wp += 1
            a["workplace"] = wp

    return {
        "fixed_residence_gu": fixed_gu,
        "fixed_residence_dong": fixed_res_dong,
        "fixed_workplace_dong": fixed_wp_dong,
        "cleared_workplace_dong": cleared_wp,
    }


def load_stat7_to_dong(csv_path: Path) -> dict[str, str]:
    """통계청 7자리 코드 → 행정동명 매핑 로드 (workplace_flow가 stat7 코드를 사용)"""
    import csv
    mapping: dict[str, str] = {}
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        next(reader, None)  # header
        for row in reader:
            if len(row) >= 5:
                stat7 = row[2].strip()
                dong = row[4].strip()
                if stat7 and dong:
                    mapping[stat7] = dong
    return mapping


def save_json(data: Any, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def extract_json_from_text(text: str) -> list[dict]:
    """LLM 응답에서 JSON 배열 또는 객체를 추출 (think 태그, 코드펜스 처리)"""
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

    # 여러 JSON 객체가 연속된 경우
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


# ---------------------------------------------------------------------------
# Prompt Builder — 이번 stats 구조(분위수, 비율, 평일/주말)에 맞춤
# ---------------------------------------------------------------------------
def _decile_range(boundaries: dict | None, metric: str, decile: int) -> tuple[float, float] | None:
    """decile_boundaries.json에서 특정 metric(예: weekday_spending_level)의 분위 범위 반환"""
    if not boundaries or metric not in boundaries:
        return None
    for b in boundaries[metric].get("boundaries", []):
        if b.get("decile") == decile:
            return (b.get("min"), b.get("max"))
    return None


def _fmt_won(x: float) -> str:
    return f"{int(round(x)):,}원"


def build_user_prompt(
    group_key: str,
    count: int,
    profile: dict,
    dong_ctx: dict | None,
    wf: list | None,
    global_dist: dict | None,
    agg_stats: dict | None,
    consump_detail: dict | None,
    decile_boundaries: dict | None = None,
    dong_name_lookup: dict | None = None,
    stat7_to_dong: dict | None = None,
) -> str:
    # group_key 예: "11110515_F_10대"
    gk_parts = group_key.rsplit("_", 2)
    gk_adm8, gk_gender, gk_age = gk_parts if len(gk_parts) == 3 else ("?", "?", "?")
    is_teen = (gk_age == "10대")

    loc = profile.get("location", {})
    demo = profile.get("demographics", {})
    tel = profile.get("telecom", {})
    cons = profile.get("consumption", {})
    mob = profile.get("mobility", {})

    # profile이 비어있으면(예: 10대) group_key와 dong_name_lookup으로 복원
    dong_code = loc.get("adm_cd_8") or gk_adm8
    dong_name = loc.get("dong") or (dong_name_lookup or {}).get(gk_adm8, "?")
    gu_name = loc.get("gu", "?")
    gender = demo.get("gender") or gk_gender
    age_grp = demo.get("age_grp") or gk_age
    population = demo.get("population", 0)

    wd_spend_level = cons.get("weekday_spending_level")
    we_spend_level = cons.get("weekend_spending_level")
    mobility_level = mob.get("mobility_level")
    mob_wd_we_ratio = mob.get("weekend_weekday_ratio", 1.0)

    parts = []
    parts.append(f"## 그룹 정보")
    parts.append(f"- 위치: {gu_name} {dong_name} ({dong_code})")
    parts.append(f"- 성별: {'남성' if gender == 'M' else '여성'}")
    parts.append(f"- 나이대: {age_grp}")
    if population:
        parts.append(f"- 그룹 인구: {population:.0f}명")
    parts.append(f"- 생성할 에이전트 수: **{count}명**")
    if is_teen:
        parts.append("- ⚠ **10대 그룹**: 통신사 데이터 미수집. 학생(중/고/대학 1~2학년) 전형 패턴으로 생성.")
        parts.append("  직업은 '학생', 소득 '하'~'중하', 통근시간은 등하교(30분 이하), 소비는 용돈 수준(weekday/weekend 소비분위 1~3)으로 제한.")
    parts.append("")

    def _tail_hint(decile: int | None) -> str:
        if decile == 1:
            return " [1분위: 하한은 극단 outlier, 범위 상단 60~95%에서 뽑을 것]"
        if decile == 10:
            return " [10분위: 상한은 극단 outlier, 범위 하단 5~40%에서 뽑을 것]"
        return ""

    # 소비/이동 분위수 + 실제 수치 범위 (decile_boundaries 기반)
    parts.append("## 소비/이동 수준 (10분위: 1=하위10%, 10=상위10%)")
    if wd_spend_level is not None:
        rng = _decile_range(decile_boundaries, "weekday_spending_level", wd_spend_level)
        suffix = f" → 실제 범위 [{_fmt_won(rng[0])} ~ {_fmt_won(rng[1])}]/일·인당" if rng else ""
        parts.append(f"- 평일 소비분위: {wd_spend_level}/10{suffix}{_tail_hint(wd_spend_level)}")
    if we_spend_level is not None:
        rng = _decile_range(decile_boundaries, "weekend_spending_level", we_spend_level)
        suffix = f" → 실제 범위 [{_fmt_won(rng[0])} ~ {_fmt_won(rng[1])}]/일·인당" if rng else ""
        parts.append(f"- 주말 소비분위: {we_spend_level}/10{suffix}{_tail_hint(we_spend_level)}")
    if mobility_level is not None:
        rng = _decile_range(decile_boundaries, "mobility_level", mobility_level)
        suffix = f" → 12개월 누적 유동인구 [{rng[0]:,.0f} ~ {rng[1]:,.0f}]" if rng else ""
        parts.append(f"- 이동활발도: {mobility_level}/10{suffix}{_tail_hint(mobility_level)}")
    parts.append(f"- 주말/평일 이동비: {mob_wd_we_ratio:.2f}")
    parts.append("")

    # consumption_detail — 평일/주말 업종 비중, 주말/평일 소비비
    if consump_detail:
        wd_ratio = consump_detail.get("weekday_ratio", {})
        we_ratio = consump_detail.get("weekend_ratio", {})
        wd_we_spend = consump_detail.get("weekend_weekday_spending_ratio")
        if wd_we_spend is not None:
            lo = round(wd_we_spend * 0.80, 2)
            hi = round(wd_we_spend * 1.20, 2)
            parts.append("## ★ 주말/평일 소비비 목표 (최우선 제약)")
            parts.append(f"- **목표비 = {wd_we_spend:.2f}** (BDC 실측값)")
            parts.append(f"- **weekend_weekday_spending_ratio 허용범위: [{lo:.2f}, {hi:.2f}]** (목표비 ±20%)")
            if wd_we_spend >= 1.0:
                parts.append(f"- 이 그룹은 **주말 소비 ≥ 평일 소비** 그룹 — daily_spending_weekend ≥ daily_spending_weekday 가 되도록 설정할 것.")
            else:
                parts.append(f"- 이 그룹은 **주말 소비 < 평일 소비** 그룹 — daily_spending_weekend < daily_spending_weekday.")
            parts.append(f"- 계산된 ratio가 허용범위를 벗어나면 자체 수정 필수.")
            parts.append("")
        parts.append("## 그룹 업종 소비 비중 (BDC 기반)")
        if wd_ratio:
            top = dict(sorted(wd_ratio.items(), key=lambda x: -x[1])[:7])
            parts.append(f"- 평일 상위 업종: {json.dumps(top, ensure_ascii=False)}")
        if we_ratio:
            top = dict(sorted(we_ratio.items(), key=lambda x: -x[1])[:7])
            parts.append(f"- 주말 상위 업종: {json.dumps(top, ensure_ascii=False)}")
        parts.append("")

    # 업종별 소비비율 (global) — 10대는 "20세미만" 키로 매핑
    gender_age_key = f"{gender}_20세미만" if is_teen else f"{gender}_{age_grp}"
    if global_dist:
        ind_ratio = global_dist.get("industry_spending_ratio", {}).get(gender_age_key, {})
        if ind_ratio:
            top8 = dict(list(ind_ratio.items())[:8])
            parts.append("## 업종별 소비비율 (같은 성별x연령대 서울 전체, 합계=1.0)")
            for industry, ratio in top8.items():
                parts.append(f"- {industry}: {ratio:.1%}")
            parts.append("")

        ww = global_dist.get("weekday_weekend_spending", {})
        if ww:
            parts.append(f"## 전체 주말/평일 소비비")
            parts.append(f"- 평일비중: {ww.get('weekday_ratio', 0):.2f}, 주말비중: {ww.get('weekend_ratio', 0):.2f}")
            parts.append("")

    # 텔레콤 지표 (tel_shopping_days는 원본 단위 불명확하여 제외)
    home_wd_h = tel.get("tel_home_wd_time", 0) / 3600 if tel.get("tel_home_wd_time") else None
    home_we_h = tel.get("tel_home_we_time", 0) / 3600 if tel.get("tel_home_we_time") else None
    # 월간 누적 시간(22평일×24h=528h) → 하루 평균(/22)으로 환산
    home_wd_daily = round(home_wd_h / 22, 1) if home_wd_h else None
    home_we_daily = round(home_we_h / 8, 1) if home_we_h else None  # 월 평균 주말일수≈8

    key_metrics = {
        "출근 소요시간(분)": round(tel.get("tel_commute_time"), 1) if tel.get("tel_commute_time") else None,
        "배달 사용일수/월": round(tel.get("tel_delivery_days"), 1) if tel.get("tel_delivery_days") else None,
        "지하철 이용일수/월": round(tel.get("tel_subway_days"), 1) if tel.get("tel_subway_days") else None,
        "평일 이동거리(km/일)": round(tel.get("tel_wd_move_dist", 0) / 1000, 2) if tel.get("tel_wd_move_dist") else None,
        "휴일 이동거리(km/일)": round(tel.get("tel_we_move_dist", 0) / 1000, 2) if tel.get("tel_we_move_dist") else None,
        "평일 재택시간(시간/일)": home_wd_daily,
        "휴일 재택시간(시간/일)": home_we_daily,
    }
    has_metrics = any(v is not None for v in key_metrics.values())
    if has_metrics:
        parts.append("## 이 그룹의 평균 행동 지표 (텔레콤 기반, 1인당 일 평균)")
        parts.append("※ shopping_days는 원본 단위 불명확으로 제외 — 연령/직업/소득 기반으로 독립 추정할 것 (일반적으로 4~20일/월).")
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

    # 직장 확률분포 — dong은 통계청 7자리 코드, stat7_to_dong으로 실제 동명 표시
    if wf:
        parts.append("## 거주지->직장 확률분포 (출근 목적지)")
        parts.append("※ 아래 코드와 동명을 **그대로** workplace.dong_code / workplace.dong에 사용할 것. 다른 동명 지어내지 말 것.")
        for entry in wf[:6]:
            code = entry['dong']
            name = (stat7_to_dong or {}).get(code)
            label = f"{code} ({name})" if name else f"{code} (동명 불명)"
            parts.append(f"- {label}: {entry['probability']:.2%}")
        parts.append("")

    parts.append(f"위 통계를 기반으로 **{count}명**의 현실적인 에이전트를 JSON 배열로 생성하세요.")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# LLM Caller
# ---------------------------------------------------------------------------
async def call_vllm(client, system_prompt: str, user_prompt: str, max_tokens: int, model: str = MODEL_NAME) -> str:
    response = await asyncio.to_thread(
        client.chat.completions.create,
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=TEMPERATURE,
        max_tokens=max_tokens,
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
    decile_boundaries: dict,
    dong_name_lookup: dict,
    stat7_to_dong: dict,
    sem: asyncio.Semaphore,
) -> tuple[str, list[dict]]:
    """하나의 그룹에 대해 LLM 호출 -> 에이전트 리스트 반환"""
    async with sem:
        parts = group_key.rsplit("_", 2)
        if len(parts) != 3:
            return group_key, []
        adm8, gender, age = parts

        profile = profiles.get(group_key, {})
        d_ctx = dong_ctx.get(adm8)
        wf = workplace_flow.get(adm8)
        # 10대는 aggregate_stats에 10대 키 없음 → 20세미만으로 fallback (없으면 None)
        demo_key = f"{gender}_20세미만" if age == "10대" else f"{gender}_{age}"
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
            decile_boundaries=decile_boundaries,
            dong_name_lookup=dong_name_lookup,
            stat7_to_dong=stat7_to_dong,
        )

        max_toks = max(MAX_TOKENS_MIN, min(MAX_TOKENS_MAX, MAX_TOKENS_PER_AGENT * count + 300))
        last_err = None
        for attempt in range(MAX_RETRIES):
            try:
                raw = await call_vllm(client, SYSTEM_PROMPT, user_prompt, max_toks)
                agents = extract_json_from_text(raw)

                if not agents:
                    last_err = f"no JSON parsed (raw len={len(raw)}): {raw[:200]!r}"
                    if attempt < MAX_RETRIES - 1:
                        continue
                    print(f"  [FAIL] {group_key}: {last_err}")
                    return group_key, []

                # agent_id 보정
                for i, agent in enumerate(agents):
                    expected_id = f"AGT_{adm8}_{gender}_{age}_{i + 1:03d}"
                    agent["agent_id"] = expected_id
                    if "residence" not in agent:
                        agent["residence"] = {}
                    agent["residence"]["dong_code"] = adm8
                    if "personal" not in agent:
                        agent["personal"] = {}
                    agent["personal"]["gender"] = gender
                    agent["personal"]["age_group"] = age

                return group_key, agents[:count]

            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"
                wait = 2 ** attempt + random.random()
                await asyncio.sleep(wait)

        print(f"  [FAIL] {group_key}: {last_err}")
        return group_key, []


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------
async def run(args):
    # --- Load stats ---
    print("Loading stats...")
    profiles = load_json(args.stats_dir / "agent_profiles.json")
    allocation = load_json(args.stats_dir / "agent_allocation.json")
    dong_ctx = load_json(args.stats_dir / "dong_context.json")
    workplace_flow = load_json(args.stats_dir / "workplace_flow.json")
    global_dist = load_json(args.stats_dir / "global_distributions.json")
    agg_stats = load_json(args.stats_dir / "aggregate_stats.json")

    # consumption_detail (optional)
    cd_path = args.stats_dir / "consumption_detail.json"
    consump_detail = load_json(cd_path) if cd_path.exists() else {}

    # decile_boundaries (BDC 콘솔 출력 기반, 분위 -> 실제 수치 범위)
    db_path = args.stats_dir / "decile_boundaries.json"
    decile_boundaries = load_json(db_path) if db_path.exists() else {}

    # 통계청 7자리 -> 행정동명 (workplace_flow 코드 해석용)
    csv_candidates = [
        PROJECT_ROOT / "data" / "mapping" / "code_mapping_mopas_nso.csv",
        PROJECT_ROOT / "data" / "InboundData" / "code_mapping_mopas_nso.csv",
    ]
    stat7_to_dong: dict[str, str] = {}
    for cp in csv_candidates:
        if cp.exists():
            stat7_to_dong = load_stat7_to_dong(cp)
            print(f"  Loaded stat7 mapping: {len(stat7_to_dong)} codes from {cp.name}")
            break
    if not stat7_to_dong:
        print("  WARN: stat7→dong 매핑 CSV 못 찾음. workplace.dong 이름이 부정확할 수 있음")

    # 동코드 -> 동명 lookup (10대처럼 profile이 없는 그룹용)
    dong_name_lookup: dict[str, str] = {}
    for pk, pv in profiles.items():
        parts = pk.split("_")
        if not parts:
            continue
        adm8 = parts[0]
        dong = pv.get("location", {}).get("dong")
        if dong and adm8 not in dong_name_lookup:
            dong_name_lookup[adm8] = dong

    # 10대 runtime 주입: consumption_detail의 _10대 키마다 allocation 1명씩 추가 (JSON 수정 없음)
    teen_added = 0
    for tk in consump_detail.keys():
        if tk.endswith("_10대") and tk not in allocation:
            allocation[tk] = 1
            teen_added += 1
    if teen_added:
        print(f"  Injected {teen_added} teen groups (1 agent each, runtime-only)")

    total_agents = sum(allocation.values())
    total_groups = len([v for v in allocation.values() if v > 0])
    print(f"  Total: {total_agents:,} agents, {total_groups:,} groups")

    # --- Key list ---
    keys = [k for k, v in allocation.items() if v > 0]
    if args.target_total > 0:
        # 비례 축소 (stochastic rounding) — 원 allocation의 그룹별 비중 보존
        rng = random.Random(args.seed)
        orig_total = sum(allocation[k] for k in keys)
        scale = args.target_total / orig_total
        scaled_alloc: dict[str, int] = {}
        for k in keys:
            raw = allocation[k] * scale
            floor = int(raw)
            frac = raw - floor
            n = floor + (1 if rng.random() < frac else 0)
            if n > 0:
                scaled_alloc[k] = n
        allocation = scaled_alloc
        keys = list(scaled_alloc.keys())
        total_agents = sum(scaled_alloc.values())
        print(f"  Target-total={args.target_total} (scale={scale:.4f}): {len(keys)} groups, sum={total_agents:,} agents")
    if args.limit > 0:
        keys = keys[:args.limit]
        print(f"  Limited to {args.limit} groups")

    # --- Resume ---
    done_keys: set = set()
    existing_agents: list = []
    partial_dir = args.output_dir / "partial"

    if args.resume and partial_dir.exists():
        for pf in sorted(partial_dir.glob("batch_*.json")):
            try:
                batch_data = load_json(pf)
                existing_agents.extend(batch_data.get("agents", []))
                done_keys.update(batch_data.get("completed_keys", []))
            except Exception:
                pass
        print(f"  Resume: {len(existing_agents)} agents loaded, {len(done_keys)} groups done")

    remaining_keys = [k for k in keys if k not in done_keys]
    print(f"  Remaining: {len(remaining_keys)} groups")

    if args.dry_run:
        print("\nDry-run: showing first prompt")
        if remaining_keys:
            k = remaining_keys[0]
            parts = k.rsplit("_", 2)
            adm8, gender, age = parts
            demo_key = f"{gender}_20세미만" if age == "10대" else f"{gender}_{age}"
            prompt = build_user_prompt(
                k, allocation[k], profiles.get(k, {}),
                dong_ctx.get(adm8), workplace_flow.get(adm8),
                global_dist, agg_stats.get(demo_key),
                consump_detail.get(k),
                decile_boundaries=decile_boundaries,
                dong_name_lookup=dong_name_lookup,
                stat7_to_dong=stat7_to_dong,
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

    # 러프 추정: 그룹당 평균 ~20초, 병렬로 max_concurrent 처리
    n_chunks = (len(remaining_keys) + args.max_concurrent * 2 - 1) // (args.max_concurrent * 2)
    rough_sec = n_chunks * 25  # chunk당 대략 25초
    rough_min = rough_sec // 60
    print(f"\nStarting generation (concurrent={args.max_concurrent})")
    print(f"  러프 추정: {len(remaining_keys)} 그룹, {n_chunks} 청크, 약 {rough_min}분 ({rough_sec}초)")
    print(f"  실제 ETA는 첫 청크 완료 후 표시됩니다")
    start_time = time.time()

    all_agents = list(existing_agents)
    batch_num = len(done_keys)

    # chunk 단위 처리
    chunk_size = args.max_concurrent * 2
    for chunk_start in range(0, len(remaining_keys), chunk_size):
        chunk_keys = remaining_keys[chunk_start:chunk_start + chunk_size]

        tasks = [
            generate_group(
                client, k, allocation[k],
                profiles, dong_ctx, workplace_flow,
                global_dist, agg_stats, consump_detail,
                decile_boundaries, dong_name_lookup, stat7_to_dong, sem,
            )
            for k in chunk_keys
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        chunk_agents = []
        chunk_done = []
        for result in results:
            if isinstance(result, Exception):
                continue
            gk, agents = result
            if agents:
                chunk_agents.extend(agents)
            chunk_done.append(gk)

        all_agents.extend(chunk_agents)
        done_keys.update(chunk_done)

        # partial save
        if chunk_agents:
            batch_num += 1
            partial_dir.mkdir(parents=True, exist_ok=True)
            save_json(
                {"agents": chunk_agents, "completed_keys": chunk_done},
                partial_dir / f"batch_{batch_num:04d}.json",
            )

        elapsed = time.time() - start_time
        done_count = chunk_start + len(chunk_keys)
        pct = len(all_agents) / total_agents * 100
        eta = (elapsed / max(done_count, 1)) * (len(remaining_keys) - done_count)
        rate = len(all_agents) / max(elapsed, 0.1)
        finish_at = time.strftime("%H:%M:%S", time.localtime(time.time() + eta))

        def fmt_dur(s: float) -> str:
            m, s = divmod(int(s), 60); h, m = divmod(m, 60)
            return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"

        print(f"  [{done_count}/{len(remaining_keys)} groups] "
              f"{len(all_agents):,}/{total_agents:,} agents ({pct:.1f}%) | "
              f"{rate:.1f} ag/s | 경과 {fmt_dur(elapsed)} | ETA {fmt_dur(eta)} (완료 예상 {finish_at})")

    # --- Post-processing: residence.gu/dong, workplace.dong 신뢰 원본으로 덮어쓰기 ---
    pp_report = postprocess_agents(all_agents, profiles, dong_name_lookup, stat7_to_dong)
    print(f"\nPost-process: {pp_report}")

    # --- Final save ---
    args.output_dir.mkdir(parents=True, exist_ok=True)
    final_path = args.output_dir / "agents_final.json"
    save_json(all_agents, final_path)

    elapsed = time.time() - start_time
    print(f"\nDone: {len(all_agents):,} agents in {elapsed:.1f}s")
    print(f"Output: {final_path}")

    if len(all_agents) != total_agents:
        print(f"Warning: target {total_agents:,} != actual {len(all_agents):,}")
        print("  -> python generate_agents.py --resume  로 재시도 가능")


def main():
    args = parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()

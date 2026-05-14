"""
prompt_layers.py
================
SGLang RadixAttention prefix cache 효율 극대화를 위한 레이어드 프롬프트 빌더.

원본 `generate_agents.py`의 `build_user_prompt()`는 그룹 고유 정보(동/성별/연령)를
앞쪽에 두고 서울 전체 분포를 뒤에 두는 구조였다. 이러면 같은 코호트의 다음 호출이
와도 prefix가 어긋나서 캐시가 안 잡힌다.

여기서는 공유 범위가 넓은 것을 앞으로 배치하여, generate_group()이 dong-major 또는
cohort-major 순으로 호출될 때 자연스럽게 prefix를 재사용하도록 한다.

레이어 구조 (front = 공유 ↑):
  L1  system 프롬프트 (12-규칙, JSON-only)            -- 전체 공유
  L2  서울 전체 분포 (weekday/weekend overall)        -- 전체 공유
  L3  동 상권 환경 + 직장 확률분포                     -- 같은 동의 모든 코호트 공유
  L4  코호트(성별×연령) 통계 + 업종 소비비율           -- 같은 코호트의 모든 동 공유
  L5  그룹 고유 (profile, consumption_detail, count)  -- 호출별 고유

`build_layers()`는 (system_prompt, user_prompt) 쌍을 반환한다. user_prompt는
L2..L5 를 `\\n\\n` 으로 이어붙인 단일 문자열이라, SGLang의 token-level radix cache가
공통 prefix를 정확히 잡아낸다.
"""

from __future__ import annotations

import json
from typing import Any


# ---------------------------------------------------------------------------
# System (L1)  — 12-규칙, 출력 스키마
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
당신은 서울시 소비 행동 시뮬레이션을 위한 가상 에이전트를 생성하는 전문가입니다.
주어진 통계 데이터를 기반으로 현실적이고 개별화된 소비자 프로필을 JSON으로 생성합니다.

## 규칙
1. 반드시 아래 JSON 스키마를 정확히 따르세요.
2. spending_level/mobility_level은 10분위 범주(1=하위10%, 10=상위10%)입니다.
   같은 분위 안에서 로그정규분포를 가정하고, 분위 하한에 가까운 값이 더 많고 상한에 가까운 값은 드뭅니다.
3. industry_ratio는 업종별 소비비율(합계=1.0)입니다.
   이 비율을 중심으로 에이전트마다 ±0.05 범위에서 변동시키되 합계는 1.0을 유지하세요.
4. 연령대에 맞는 구체적 나이를 정하세요 (예: 30대 → 30~39세).
5. 직업은 연령대, 성별, 거주지 상권 특성을 고려하여 현실적으로 결정하세요.
6. 직장 위치는 workplace_flow 확률분포를 참조하여 결정하세요.
7. 평일 소비와 주말 소비의 업종 비중(weekday_ratio/weekend_ratio)을 구분하세요.
8. weekend_weekday_spending_ratio가 0.8이면 주말 소비가 평일의 80% 수준이라는 뜻입니다.
9. 20세미만은 학생/아르바이트만 가능합니다.
10. 소비액은 소득 수준과 일관되어야 합니다.
11. 통근시간은 거주지와 직장 위치 거리에 비례해야 합니다.
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

복수 에이전트 요청 시, JSON 배열 []로 감싸서 출력하세요."""


# ---------------------------------------------------------------------------
# Layer builders (L2 ~ L5)
# ---------------------------------------------------------------------------
def layer_global(global_dist: dict | None) -> str:
    """L2 — 서울 전체 분포. 호출 전체에서 동일."""
    if not global_dist:
        return ""
    ww = global_dist.get("weekday_weekend_spending", {}) or {}
    if not ww:
        return ""
    return (
        "## 서울 전체 주말/평일 소비비\n"
        f"- 평일비중: {ww.get('weekday_ratio', 0):.2f}, "
        f"주말비중: {ww.get('weekend_ratio', 0):.2f}"
    )


def layer_dong(dong_ctx: dict | None, wf: list | None) -> str:
    """L3 — 동 상권 환경 + 직장 확률분포. 같은 동의 모든 코호트 공유."""
    out: list[str] = []

    if dong_ctx:
        out.append("## 동 상권 환경")
        labels = {
            "b069_sales": "매출지수",
            "b069_infra": "인프라지수",
            "b069_store": "가맹점지수",
            "b069_pop": "인구지수",
            "b079_seoul_inflow_ratio": "서울유입비율",
        }
        for k, label in labels.items():
            v = dong_ctx.get(k)
            if v is not None:
                out.append(f"- {label}: {v}")

    if wf:
        if out:
            out.append("")
        out.append("## 거주지->직장 확률분포 (출근 목적지)")
        for entry in wf[:6]:
            out.append(f"- {entry['dong']}: {entry['probability']:.2%}")

    return "\n".join(out)


def layer_cohort(
    gender: str,
    age: str,
    global_dist: dict | None,
    agg_stats: dict | None,
) -> str:
    """L4 — 성별×연령 코호트 통계. 같은 코호트의 모든 동에서 공유."""
    out: list[str] = []

    # 코호트별 업종 소비비율
    if global_dist:
        ind_ratio = (
            global_dist.get("industry_spending_ratio", {}) or {}
        ).get(f"{gender}_{age}", {})
        if ind_ratio:
            top8 = dict(list(ind_ratio.items())[:8])
            out.append("## 코호트 업종 소비비율 (같은 성별×연령대 서울 전체, 합계=1.0)")
            for industry, ratio in top8.items():
                out.append(f"- {industry}: {ratio:.1%}")

    # aggregate_stats: 코호트의 동 간 분포 (mean/std)
    if agg_stats:
        if out:
            out.append("")
        out.append("## 코호트의 동 간 분포 (에이전트 변동 참고)")
        for metric_name in (
            "tel_commute_time",
            "tel_delivery_days",
            "tel_wd_move_dist",
            "tel_subway_days",
        ):
            stat = agg_stats.get(metric_name)
            if stat:
                out.append(
                    f"- {metric_name}: mean={stat['mean']:.1f}, "
                    f"std={stat['std']:.1f}, "
                    f"범위=[{stat['min']:.1f}, {stat['max']:.1f}]"
                )

    return "\n".join(out)


def layer_group(
    group_key: str,
    count: int,
    profile: dict,
    consump_detail: dict | None,
) -> str:
    """L5 — 그룹(행정동×성별×연령) 고유 정보. 호출별 가변."""
    loc = profile.get("location", {}) or {}
    demo = profile.get("demographics", {}) or {}
    tel = profile.get("telecom", {}) or {}
    cons = profile.get("consumption", {}) or {}
    mob = profile.get("mobility", {}) or {}

    dong_code = loc.get("adm_cd_8", "?")
    dong_name = loc.get("dong", "?")
    gu_name = loc.get("gu", "?")
    gender = demo.get("gender", "?")
    age_grp = demo.get("age_grp", "?")
    population = demo.get("population", 0)

    spending_level = cons.get("spending_level", 5)
    mobility_level = mob.get("mobility_level", 5)
    mob_wd_we_ratio = mob.get("weekend_weekday_ratio", 1.0)

    parts: list[str] = []

    parts.append("## 그룹 정보")
    parts.append(f"- 위치: {gu_name} {dong_name} ({dong_code})")
    parts.append(f"- 성별: {'남성' if gender == 'M' else '여성'}")
    parts.append(f"- 나이대: {age_grp}")
    parts.append(f"- 그룹 인구: {population:.0f}명")
    parts.append(f"- 생성할 에이전트 수: **{count}명**")
    parts.append("")
    parts.append("## 소비/이동 수준 (10분위)")
    parts.append(f"- 소비수준: {spending_level}/10")
    parts.append(f"- 이동활발도: {mobility_level}/10")
    parts.append(f"- 주말/평일 이동비: {mob_wd_we_ratio:.2f}")

    if consump_detail:
        detail_level = consump_detail.get("detail_spending_level", spending_level)
        wd_ratio = consump_detail.get("weekday_ratio", {}) or {}
        we_ratio = consump_detail.get("weekend_ratio", {}) or {}
        wd_we_spend = consump_detail.get("weekend_weekday_spending_ratio", 1.0)
        parts.append(f"- 세부소비수준: {detail_level}/10")
        if wd_we_spend:
            parts.append(f"- 주말/평일 소비비: {wd_we_spend:.2f}")
        if wd_ratio:
            parts.append(
                "- 평일 업종 비중: "
                + json.dumps(dict(list(wd_ratio.items())[:7]), ensure_ascii=False)
            )
        if we_ratio:
            parts.append(
                "- 주말 업종 비중: "
                + json.dumps(dict(list(we_ratio.items())[:7]), ensure_ascii=False)
            )

    # 그룹의 평균 행동 지표
    key_metrics = {
        "출근 소요시간(분)": tel.get("tel_commute_time"),
        "배달 사용일수/월": tel.get("tel_delivery_days"),
        "쇼핑 사용일수/월": tel.get("tel_shopping_days"),
        "평일 이동거리(km)": (
            round(tel.get("tel_wd_move_dist", 0) / 1000, 1)
            if tel.get("tel_wd_move_dist") else None
        ),
        "휴일 이동거리(km)": (
            round(tel.get("tel_we_move_dist", 0) / 1000, 1)
            if tel.get("tel_we_move_dist") else None
        ),
        "지하철 이용일수/월": tel.get("tel_subway_days"),
        "평일 재택시간(시간)": (
            round(tel.get("tel_home_wd_time", 0) / 3600, 1)
            if tel.get("tel_home_wd_time") else None
        ),
        "휴일 재택시간(시간)": (
            round(tel.get("tel_home_we_time", 0) / 3600, 1)
            if tel.get("tel_home_we_time") else None
        ),
    }
    if any(v is not None for v in key_metrics.values()):
        parts.append("")
        parts.append("## 이 그룹의 평균 행동 지표 (텔레콤 기반)")
        for label, val in key_metrics.items():
            if val is not None:
                parts.append(f"- {label}: {val}")

    parts.append("")
    parts.append(
        f"위 통계를 기반으로 **{count}명**의 현실적인 에이전트를 JSON 배열로 생성하세요."
    )

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def build_layers(
    *,
    group_key: str,
    count: int,
    gender: str,
    age: str,
    profile: dict,
    dong_ctx: dict | None,
    wf: list | None,
    global_dist: dict | None,
    agg_stats: dict | None,
    consump_detail: dict | None,
) -> tuple[str, str, list[tuple[str, str]]]:
    """Layered prompt를 만들어 (system, user, debug_layers) 로 반환.

    debug_layers는 [(layer_name, text), ...] 형태로 --dry-run 출력용.
    """
    l2 = layer_global(global_dist)
    l3 = layer_dong(dong_ctx, wf)
    l4 = layer_cohort(gender, age, global_dist, agg_stats)
    l5 = layer_group(group_key, count, profile, consump_detail)

    user_blocks: list[str] = []
    debug: list[tuple[str, str]] = [("L1_system", SYSTEM_PROMPT)]
    for name, text in (("L2_global", l2), ("L3_dong", l3),
                       ("L4_cohort", l4), ("L5_group", l5)):
        debug.append((name, text))
        if text:
            user_blocks.append(text)

    user_prompt = "\n\n".join(user_blocks)
    return SYSTEM_PROMPT, user_prompt, debug

"""
build_rank_coupling.py  (방식 A — 줄세우기 / SES rank-coupling)
================================================================
우리 BDC 통계로 정량 에이전트를 먼저 생성한 뒤, 같은 (자치구·성별·연령) 셀 안에서
**우리 소비분위 순위 ↔ NVIDIA SES_proxy 순위** 를 매칭해 정성 서사를 부착한다.

핵심:
  - 두 완성 데이터(우리 정량 / NVIDIA 정성)를 사후 매칭
  - 셀 내 백분위 정렬로 cross-correlation(SES↔소비) 복원 → "명품족↔알바생" 모순 완화
  - SES_proxy 힌트가 매칭의 *전부* (이게 방식 A의 강점이자 약점)

LLM 호출 없음. 결정적(seed 고정).

사용:
  python -m scripts.persona.build_rank_coupling --limit 10
  python -m scripts.persona.build_rank_coupling --out output/personas/rank_full.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    PROJECT_ROOT, PersonaRecord,
    age_to_group, build_quant_from_cell, load_bdc_stats, load_nvidia_seoul,
    nvidia_cell, parse_cell_key, ses_proxy, split_nvidia_fields, write_personas,
)


# ---------------------------------------------------------------------------
# NVIDIA 풀 인덱싱 + fallback chain
# ---------------------------------------------------------------------------
def index_nvidia_pool(nv: list[dict]) -> dict[tuple, list[dict]]:
    """NVIDIA 레코드를 (gu, sex, age_group) 셀로 그룹화 + SES 정렬."""
    by_cell: dict[tuple, list[dict]] = defaultdict(list)
    for rec in nv:
        cell = nvidia_cell(rec)
        if cell[1]:   # 성별 정규화 성공한 것만
            by_cell[cell].append(rec)
    # 각 셀 SES_proxy 오름차순 정렬
    for cell in by_cell:
        by_cell[cell].sort(key=ses_proxy)
    return dict(by_cell)


# age_group 인접 이웃 정의 — sex·age 무시되는 fallback 차단용
_AGE_NEIGHBORS = {
    "10대":      ["20대"],
    "20대":      ["10대", "30대"],
    "30대":      ["20대", "40대"],
    "40대":      ["30대", "50대"],
    "50대":      ["40대", "60대"],
    "60대":      ["50대", "70대이상"],
    "70대이상":  ["60대"],
}


def _unique_filter(pool: list[dict], used_uuids: set | None) -> list[dict]:
    """이미 사용된 NVIDIA UUID 제외한 풀 반환."""
    if not used_uuids:
        return pool
    return [r for r in pool if r.get("uuid") not in used_uuids]


def pick_nvidia_by_rank(
    pool_index: dict[tuple, list[dict]],
    nv_all_sorted: list[dict],
    cell: tuple, percentile: float,
    used_uuids: set | None = None,
) -> tuple[dict, str]:
    """(gu,sex,age) 셀에서 percentile 위치의 NVIDIA. 이미 사용된 UUID는 제외.

    매칭 우선순위 (sex·age 매칭 보장 + 중복 회피):
      1) (gu, sex, age) 풀 중 사용 안 된 것
      2) (sex, age) 사용 안 된 것
      3) (sex, age 인접) 사용 안 된 것
      4) (sex만) 사용 안 된 것
    NVIDIA 18.5만 > BDC 1.5만이라 1:1 unique 매칭이 정상적으로 가능.

    percentile ∈ [0,1] — 우리 에이전트의 셀 내 소비분위 순위.
    반환: (NVIDIA record, 사용된 매칭 레벨)
    """
    gu, sex, age = cell

    # 1차: (gu, sex, age)
    pool = _unique_filter(pool_index.get(cell) or [], used_uuids)
    level = "gu_sex_age"

    # 2차: (sex, age) — gu 무시
    if not pool:
        merged = [r for c, lst in pool_index.items()
                  if c[1] == sex and c[2] == age for r in lst]
        merged.sort(key=ses_proxy)
        pool = _unique_filter(merged, used_uuids)
        level = "sex_age"

    # 3차: (sex, age 인접)
    if not pool:
        adjacent = set(_AGE_NEIGHBORS.get(age, []))
        if adjacent:
            merged = [r for c, lst in pool_index.items()
                      if c[1] == sex and c[2] in adjacent for r in lst]
            merged.sort(key=ses_proxy)
            pool = _unique_filter(merged, used_uuids)
            level = "sex_adjacent_age"

    # 4차: (sex만) — 마지노선
    if not pool:
        merged = [r for c, lst in pool_index.items() if c[1] == sex for r in lst]
        merged.sort(key=ses_proxy)
        pool = _unique_filter(merged, used_uuids)
        level = "sex_only"

    if not pool:
        # sex도 모자람 — 데이터 이상이지만 폴백
        pool = _unique_filter(nv_all_sorted, used_uuids) or nv_all_sorted
        level = "any_emergency"

    idx = round(percentile * (len(pool) - 1)) if len(pool) > 1 else 0
    return pool[idx], level


# ---------------------------------------------------------------------------
# 메인 빌드
# ---------------------------------------------------------------------------
def build(limit: int = 0, seed: int = 42,
          llm_reconcile: bool = False, llm_mode: str | None = None,
          llm_stub: bool = False) -> list[dict]:
    stats = load_bdc_stats()
    nv = load_nvidia_seoul()
    profiles = stats["profiles"]
    allocation = stats["allocation"]
    deciles = stats["deciles"]

    pool_index = index_nvidia_pool(nv)
    nv_all_sorted = sorted([r for r in nv if nvidia_cell(r)[1]], key=ses_proxy)

    rng = random.Random(seed)

    # 1) 우리 통계로 정량 에이전트 전부 생성 (gu 별로 모음)
    #    한 셀(adm8,성별,연령)에 N명이면 분위에 ±변동을 줘 개체 차이 부여
    by_gu_cell: dict[tuple, list[dict]] = defaultdict(list)   # (gu,sex,age) → agents
    agents: list[dict] = []

    # rank-coupling 은 셀 전체가 있어야 순위가 의미 있으므로 항상 전체 생성.
    # limit 는 마지막에 다양성 있게 추출만 한다.
    cell_keys = list(allocation.keys())

    for key in cell_keys:
        n = allocation[key]
        adm8, sex, age = parse_cell_key(key)
        prof = profiles.get(key)
        if not prof:
            continue
        gu = prof.get("location", {}).get("gu") or ""
        dong = prof.get("location", {}).get("dong") or ""
        for seq in range(n):
            # 셀 대표 분위에 ±1 변동 (개체 다양성)
            base_wd = int(prof.get("consumption", {}).get("weekday_spending_level") or 5)
            base_we = int(prof.get("consumption", {}).get("weekend_spending_level") or 5)
            lv_wd = max(1, min(10, base_wd + rng.choice([-1, 0, 0, 1])))
            lv_we = max(1, min(10, base_we + rng.choice([-1, 0, 0, 1])))
            quant = build_quant_from_cell(prof, deciles, rng,
                                          spending_level_override=(lv_wd, lv_we))
            agent = {
                "agent_id": f"AGT_{adm8}_{sex}_{age}_{seq:03d}",
                "residence": {"dong_code": adm8, "dong": dong, "gu": gu},
                "_cell": (gu, sex, age),
                "_consume_rank_key": (lv_wd + lv_we) / 2 + rng.uniform(-0.3, 0.3),
                "_quant": quant,
                "personal_core": {"age_group": age, "gender": sex},
            }
            by_gu_cell[(gu, sex, age)].append(agent)
            agents.append(agent)

    # 2) 셀 내 rank-coupling: 우리 소비순위 ↔ NVIDIA SES순위
    #    중복 방지: 한 번 사용된 NVIDIA uuid는 다른 agent에 재할당 금지
    out: list[dict] = []
    used_uuids: set = set()
    match_level_counts: dict[str, int] = defaultdict(int)
    for cell, cell_agents in by_gu_cell.items():
        cell_agents.sort(key=lambda a: a["_consume_rank_key"])
        m = len(cell_agents)
        for i, agent in enumerate(cell_agents):
            pct = i / (m - 1) if m > 1 else 0.5
            nv_rec, level = pick_nvidia_by_rank(
                pool_index, nv_all_sorted, cell, pct, used_uuids=used_uuids,
            )
            uuid = nv_rec.get("uuid")
            if uuid:
                used_uuids.add(uuid)
            match_level_counts[level] += 1
            out.append(_assemble(agent, nv_rec, level, pct))

    # 매칭 품질 보고 — sex_age 매칭률이 핵심 지표
    total_matched = sum(match_level_counts.values())
    if total_matched > 0:
        print(f"[NVIDIA 매칭] 총 {total_matched:,}명 / NVIDIA pool {len(nv_all_sorted):,}", file=sys.stderr)
        for lvl in ["gu_sex_age", "sex_age", "sex_adjacent_age", "sex_only", "any_emergency"]:
            cnt = match_level_counts.get(lvl, 0)
            if cnt > 0:
                pct = cnt / total_matched * 100
                print(f"  · {lvl}: {cnt:,} ({pct:.1f}%)", file=sys.stderr)

    if limit:
        out = _diverse_sample(out, limit, seed)

    if llm_reconcile:
        max_workers = int(os.environ.get("PERSONA_RECONCILE_WORKERS", "10"))
        print(f"[LLM 5줄 요약] {len(out):,}명 / workers={max_workers}", file=sys.stderr, flush=True)

        def _summarize_one(p: dict) -> None:
            summary = summarize_persona_llm(p, llm_mode=llm_mode or "qwen8b")
            # personality_lifestyle_raw 를 5줄 요약으로 교체
            if "personality" not in p:
                p["personality"] = {}
            p["personality"]["lifestyle"] = summary

        if llm_stub:
            for p in out:
                _summarize_one(p)
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futs = [pool.submit(_summarize_one, p) for p in out]
                done = 0
                for f in as_completed(futs):
                    f.result()
                    done += 1
                    if done % 100 == 0 or done == len(out):
                        print(f"[LLM 5줄 요약] {done:,}/{len(out):,} ({done/len(out)*100:.1f}%)",
                              file=sys.stderr, flush=True)

    for p in out:
        p.pop("_nvidia_raw", None)
        p.pop("_cell", None)
        p.pop("_consume_rank_key", None)
        p.pop("_quant", None)

    return out


def _diverse_sample(out: list[dict], limit: int, seed: int) -> list[dict]:
    """데모용 — 자치구·연령·소비분위가 골고루 섞이도록 추출."""
    rng = random.Random(seed)
    # (gu, age_group) 버킷별로 하나씩 라운드로빈
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for p in out:
        buckets[(p["residence"]["gu"], p["personal"]["age_group"])].append(p)
    keys = list(buckets.keys())
    rng.shuffle(keys)
    picked: list[dict] = []
    bi = 0
    while len(picked) < limit and keys:
        k = keys[bi % len(keys)]
        if buckets[k]:
            # 버킷 내 랜덤 위치 — percentile(저~고 소비) 다양성 확보
            picked.append(buckets[k].pop(rng.randrange(len(buckets[k]))))
        else:
            keys.remove(k)
            continue
        bi += 1
    return picked[:limit]


def _assemble(agent: dict, nv_rec: dict, match_level: str, pct: float) -> dict:
    """우리 정량 + NVIDIA 정성 → 최종 PersonaRecord dict."""
    q = agent["_quant"]
    llm_fields, reserved = split_nvidia_fields(nv_rec)
    age = agent["personal_core"]["age_group"]
    sex = agent["personal_core"]["gender"]

    rec = PersonaRecord(
        agent_id=agent["agent_id"],
        residence=agent["residence"],
        personal={
            "age_group": age, "gender": sex,
            "job": nv_rec.get("occupation") or "",          # A안: NVIDIA occupation 채택
            "income_level": _income_from_level(q["spending"]["weekday_spending_level"]),
            "life_stage": _life_stage(nv_rec),
        },
        workplace={"dong_code": None, "dong": None, "commute_min": None},  # 방식 A는 직장 미생성(후속)
        spending=q["spending"],
        behavior=q["behavior"],
        personality={"spending_tendency": q["tendency"],
                     "lifestyle": (nv_rec.get("persona") or "")[:60]},
        nvidia_persona=llm_fields,
        nvidia_reserved=reserved,
        match_meta={"method": "rank-coupling", "match_level": match_level,
                    "consume_percentile": round(pct, 3),
                    "nvidia_ses": round(ses_proxy(nv_rec), 3),
                    "nvidia_uuid": nv_rec.get("uuid")},
    )
    d = rec.to_dict()
    d["_nvidia_raw"] = nv_rec  # LLM 요약용 원본 보존
    return d


def _income_from_level(lv: int) -> str:
    return {1: "하", 2: "하", 3: "중하", 4: "중하", 5: "중", 6: "중",
            7: "중상", 8: "중상", 9: "상", 10: "상"}.get(lv, "중")


def _life_stage(nv_rec: dict) -> str:
    age = int(nv_rec.get("age") or 0)
    ms = nv_rec.get("marital_status") or ""
    ft = nv_rec.get("family_type") or ""
    if age < 27:
        return "사회초년생" if "미혼" in ms else "학생"
    if "자녀" in ft:
        return "자녀양육"
    if age >= 65:
        return "은퇴"
    if "배우자" in ms:
        return "기혼"
    return "독립"


def _build_persona_summary_prompt(agent: dict) -> str:
    """NVIDIA 자연어 페르소나 + BDC 정량 데이터를 LLM 봉합용 프롬프트로.

    구조:
      [BDC 정량] — 우리 통계로 계산한 소비 수준·행태 (이게 fact, 기준)
      [NVIDIA 자연어] — culinary/sports/arts/travel/family/professional 페르소나 텍스트
        ↳ LLM이 BDC 정량에 모순되는 부분을 조정해 5줄로 통합
    """
    p = agent
    personal = p.get("personal", {})
    spending = p.get("spending", {})
    behavior = p.get("behavior", {})
    residence = p.get("residence", {})
    workplace = p.get("workplace", {})
    personality = p.get("personality", {})
    nvidia = p.get("_nvidia_raw", {})

    # === BDC 정량 (fact) ===
    bdc_lines = ["## [BDC 정량 데이터 — fact, 기준값]"]
    bdc_lines.append(
        f"- 인구학: {personal.get('age_group','')} {personal.get('gender','')}, "
        f"직업: {personal.get('job','') or '미상'}, 소득: {personal.get('income_level','')}, "
        f"생애주기: {personal.get('life_stage','')}"
    )
    wd = spending.get('daily_spending_weekday', 0) or 0
    we = spending.get('daily_spending_weekend', 0) or 0
    bdc_lines.append(
        f"- 일일 소비: 평일 **{wd:,}원**, 주말 **{we:,}원** (소비성향 {personality.get('spending_tendency','') or '보통'})"
    )
    bdc_lines.append(
        f"- 행태: 배달 {behavior.get('delivery_days',0)}회/월, "
        f"쇼핑 {behavior.get('shopping_days',0)}일/월, "
        f"평일 이동 {behavior.get('weekday_move_km',0):.1f}km, "
        f"재택 {behavior.get('home_hours_weekday',0):.1f}h, "
        f"이동성분위 {behavior.get('mobility_level',0)}"
    )
    bdc_lines.append(
        f"- 거주/직장: {residence.get('gu','')} {residence.get('dong','')}, "
        f"통근 {workplace.get('commute_min',0) or 0}분"
    )

    # === NVIDIA 자연어 페르소나 ===
    nv_lines = ["", "## [NVIDIA 자연어 페르소나 — BDC 소비 수준에 맞춰 조정 필요]"]
    one_liner = nvidia.get("persona") or ""
    if one_liner:
        nv_lines.append(f"- 한 줄 요약: {str(one_liner)[:200]}")

    # 식습관·취미·문화 — 소비 수준과 직접 연관
    for key, label in [
        ("culinary_persona", "식습관/외식"),
        ("hobbies_and_interests", "취미·관심사"),
        ("sports_persona", "운동/여가"),
        ("arts_persona", "예술/문화"),
        ("travel_persona", "여행"),
        ("family_persona", "가족생활"),
        ("professional_persona", "직업/일상"),
    ]:
        v = nvidia.get(key)
        if v:
            nv_lines.append(f"- {label}: {str(v)[:220]}")

    # 배경 정보
    bg_extras = []
    for k, label in [("cultural_background", "문화배경"),
                     ("skills_and_expertise", "전문성"),
                     ("career_goals_and_ambitions", "목표"),
                     ("marital_status", "혼인"),
                     ("family_type", "가족형태")]:
        v = nvidia.get(k)
        if v:
            bg_extras.append(f"{label}: {str(v)[:100]}")
    if bg_extras:
        nv_lines.append(f"- 배경: {' / '.join(bg_extras[:3])}")

    return "\n".join(bdc_lines + nv_lines)


_SUMMARY_SYSTEM_PROMPT = (
    "당신은 시뮬레이션 에이전트의 '살아있는 페르소나'를 봉합·요약하는 전문가입니다.\n"
    "이 요약은 시뮬에서 LLM이 매일 의사결정할 때 행동의 근거가 됩니다.\n"
    "그래서 정적인 신상정보(나이·직업·동네 나열)보다 **그 사람의 성격·취향·선택 패턴**이 보여야 합니다.\n"
    "\n"
    "[입력]\n"
    "- BDC 정량 데이터: 통계 기반 소비 금액·행태 (fact, 절대 바꾸지 마세요).\n"
    "- NVIDIA 자연어 페르소나: 식습관·취미·여행·가족·일·문화배경 등 다면적 텍스트.\n"
    "\n"
    "[당신의 작업]\n"
    "1) BDC 일일 소비 금액·소비성향을 기준으로 NVIDIA 페르소나에서 모순되는 부분을 자연스럽게 조정.\n"
    "   - 예: NVIDIA '고급 레스토랑' + BDC 1만원 → '평소 분식·한식, 특별한 날만 살짝 사치'.\n"
    "   - 예: NVIDIA '검소 식사' + BDC 8만원 → '집밥은 절약, 외식·배달은 자주 즐김'.\n"
    "2) 단순 사실 나열 대신 **성격·태도·선택 경향**이 드러나는 5줄을 작성.\n"
    "   - 좋은 표현: '꼼꼼하고 단골을 챙기는 편', '새로운 곳보다 익숙한 곳을 선호', "
    "'몸이 피곤한 날엔 외출을 줄이는 성향'.\n"
    "   - 피할 표현: '서울 종로구 청운효자동 거주, 사무직 계약직, 25세 여성' 같은 신상 나열.\n"
    "\n"
    "[5줄 구성 (이 흐름을 따르되 자연스럽게)]\n"
    "①성향 한 줄: 핵심 성격·태도 (예: 절약하지만 가족 챙기는, 호기심 많고 새로운 곳 시도하는).\n"
    "②소비 패턴: 일일 예산 수준과 어떤 식으로 쓰는지 (예: '하루 평균 1만원대, 외식보다 집밥·편의점').\n"
    "③일상 동선·활동: 평일·주말·재택·이동 등 행태에서 드러나는 리듬.\n"
    "④관심·취향: 어떤 카테고리·장소·경험을 좋아하는지 (NVIDIA 페르소나에서 조정한 후).\n"
    "⑤관계·맥락: 가족·동료·이웃과의 관계, 의사결정에 영향을 주는 사회적 배경.\n"
    "\n"
    "[출력 규칙]\n"
    "- 5줄, 줄바꿈만으로 구분. 번호·불릿 기호 금지.\n"
    "- 각 줄은 완결된 문장. 자연스러운 서술형.\n"
    "- 신상 항목 나열보다 '이 사람이 어떻게 살고 무엇을 좋아하는가'가 묻어나야 함.\n"
    "- 일일 소비 수준은 한 번 정도만 자연스럽게 녹임 (반복 금지).\n"
)


def summarize_persona_llm(agent: dict, llm_mode: str = "qwen8b") -> str:
    """BDC 정량을 기준으로 NVIDIA 자연어 페르소나를 조정 + 5줄 요약."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "sim"))
    from llm_client import call_chat

    prompt_data = _build_persona_summary_prompt(agent)
    user = (
        "다음 페르소나를 BDC 정량 기준으로 조정한 뒤 5줄로 요약해 주세요.\n\n"
        f"{prompt_data}\n\n/no_think"
    )

    try:
        resp = call_chat(None, _SUMMARY_SYSTEM_PROMPT, user, temperature=0.7, max_tokens=400)
        result = (resp.choices[0].message.content or "").strip()
        # think 토큰 흔적 제거
        if "</think>" in result:
            result = result.split("</think>", 1)[1].strip()
        return result
    except Exception:
        # 실패 시 기존 1줄 lifestyle 유지
        return agent.get("personality", {}).get("lifestyle", "")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="생성 수 제한 (0=전체 15000)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path,
                    default=PROJECT_ROOT / "output" / "personas" / "samples" / "A_rank_coupling.json")
    ap.add_argument("--llm-reconcile", action="store_true",
                    help="rank-coupling 후 모든 페르소나를 LLM이 전수 검증, 모순이면 서사 봉합 (방식 A+LLM)")
    ap.add_argument("--summarize", action="store_true",
                    help="NVIDIA 전체 필드 → vLLM 5줄 요약 (llm_reconcile 대체)")
    ap.add_argument("--llm-stub", action="store_true",
                    help="LLM 서버 없이 결정적 stub fixer 사용 (테스트/오프라인)")
    ap.add_argument("--llm-mode", default="qwen8b",
                    help="LLM 모드 (qwen32b/qwen14b/qwen8b/exaone). 기본값: qwen8b")
    ap.add_argument("--jsonl", action="store_true",
                    help="JSONL 라인 출력 (대용량 권장 — 메모리 절약)")
    args = ap.parse_args()

    personas = build(limit=args.limit, seed=args.seed,
                     llm_reconcile=args.summarize or args.llm_reconcile,
                     llm_mode=args.llm_mode,
                     llm_stub=args.llm_stub)
    out = args.out
    if args.jsonl and out.suffix == ".json":
        out = out.with_suffix(".jsonl")
    write_personas(personas, out, jsonl=args.jsonl)
    label = "rank-coupling+LLM" if args.llm_reconcile else "rank-coupling"
    print(f"[{label}] {len(personas)} personas → {out}")
    # 매칭 레벨 분포
    from collections import Counter
    levels = Counter(p["_match"]["match_level"] for p in personas)
    print(f"  match levels: {dict(levels)}")
    if args.llm_reconcile:
        n_aud = sum(1 for p in personas if p["_match"].get("llm_audited"))
        n_incons = sum(1 for p in personas if p["_match"].get("llm_consistent") is False)
        n_fixed = sum(1 for p in personas if p["_match"].get("llm_reconciled"))
        mode_lbl = "stub" if args.llm_stub else (args.llm_mode or "env/default")
        print(f"  llm({mode_lbl}): 전수검증 {n_aud}/{len(personas)}, "
              f"모순발견 {n_incons}, 봉합 {n_fixed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

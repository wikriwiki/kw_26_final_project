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


def pick_nvidia_by_rank(
    pool_index: dict[tuple, list[dict]],
    nv_all_sorted: list[dict],
    cell: tuple, percentile: float,
) -> tuple[dict, str]:
    """(gu,sex,age) 셀에서 percentile 위치의 NVIDIA. 셀 비면 fallback.

    percentile ∈ [0,1] — 우리 에이전트의 셀 내 소비분위 순위.
    반환: (NVIDIA record, 사용된 매칭 레벨)
    """
    gu, sex, age = cell
    # 1차: (gu, sex, age)
    pool = pool_index.get(cell)
    level = "gu_sex_age"
    # 2차: (sex, age) — gu 무시
    if not pool:
        merged = [r for c, lst in pool_index.items() if c[1] == sex and c[2] == age for r in lst]
        merged.sort(key=ses_proxy)
        pool = merged or None
        level = "sex_age"
    # 3차: (sex) only
    if not pool:
        merged = [r for c, lst in pool_index.items() if c[1] == sex for r in lst]
        merged.sort(key=ses_proxy)
        pool = merged or None
        level = "sex"
    # 4차: 전체
    if not pool:
        pool = nv_all_sorted
        level = "any"

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
    out: list[dict] = []
    for cell, cell_agents in by_gu_cell.items():
        cell_agents.sort(key=lambda a: a["_consume_rank_key"])
        m = len(cell_agents)
        for i, agent in enumerate(cell_agents):
            pct = i / (m - 1) if m > 1 else 0.5
            nv_rec, level = pick_nvidia_by_rank(pool_index, nv_all_sorted, cell, pct)
            out.append(_assemble(agent, nv_rec, level, pct))

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
    """NVIDIA 전체 필드 + 정량 데이터를 합쳐 LLM 요약 프롬프트 생성."""
    p = agent
    # 기본 인구학
    personal = p.get("personal", {})
    spending = p.get("spending", {})
    behavior = p.get("behavior", {})
    residence = p.get("residence", {})
    workplace = p.get("workplace", {})
    personality = p.get("personality", {})
    nvidia = p.get("_nvidia_raw", {})  # NVIDIA 원본 레코드

    lines = [
        f"[인구학] 나이: {personal.get('age','')}세 ({personal.get('age_group','')}), "
        f"성별: {personal.get('gender','')}, 직업: {personal.get('job','')}, "
        f"소득: {personal.get('income_level','')}, 생애주기: {personal.get('life_stage','')}",
        f"[소비] 평일 {spending.get('daily_spending_weekday',0):,}원/일, "
        f"주말 {spending.get('daily_spending_weekend',0):,}원/일, "
        f"소비성향: {personality.get('spending_tendency','')}, "
        f"평일 주요지출: {list((spending.get('weekday_top_categories') or {}).keys())[:3]}",
        f"[행태] 배달 {behavior.get('delivery_days',0)}회/월, "
        f"쇼핑 {behavior.get('shopping_days',0)}일/월, "
        f"평일 이동 {behavior.get('weekday_move_km',0):.1f}km, "
        f"평일 재택 {behavior.get('home_hours_weekday',0):.1f}h, "
        f"이동성 분위 {behavior.get('mobility_level',0)}",
        f"[거주/직장] {residence.get('gu','')} {residence.get('dong','')} 거주, "
        f"통근 {workplace.get('commute_min',0)}분",
    ]
    # NVIDIA 추가 필드 (있는 것만)
    nvidia_extras = []
    for k in ["hobbies", "cultural_activities", "career_background", "skills",
              "education", "marital_status", "family_situation", "summary"]:
        v = nvidia.get(k)
        if v:
            nvidia_extras.append(f"{k}: {str(v)[:80]}")
    if nvidia_extras:
        lines.append("[NVIDIA 추가] " + " / ".join(nvidia_extras[:4]))

    return "\n".join(lines)


def summarize_persona_llm(agent: dict, llm_mode: str = "qwen8b") -> str:
    """NVIDIA 전체 필드를 vLLM으로 5줄 요약. 반환: 5줄 문자열."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "sim"))
    from llm_client import call_chat

    prompt_data = _build_persona_summary_prompt(agent)
    system = (
        "당신은 에이전트 페르소나 요약 전문가입니다. "
        "주어진 데이터를 바탕으로 이 사람의 생활 방식과 소비 패턴을 "
        "자연스러운 한국어 5줄로 요약하세요. "
        "각 줄은 완결된 문장이어야 하며, 숫자 나열이 아닌 서술형으로 작성하세요. "
        "소비 수준(하루 예산), 주요 활동 카테고리, 생활 패턴, 성격/성향을 반드시 포함하세요. "
        "출력은 5줄 텍스트만, 번호나 불릿 기호 없이."
    )
    user = f"다음 에이전트 데이터를 5줄로 요약하세요:\n\n{prompt_data}\n\n/no_think"

    try:
        resp = call_chat(None, system, user, temperature=0.7, max_tokens=300)
        result = resp.choices[0].message.content.strip()
        return result
    except Exception as e:
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

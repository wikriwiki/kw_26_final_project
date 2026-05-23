"""
build_conditional.py  (방식 B — 조건부 입히기 / conditional-graft)
====================================================================
NVIDIA 서울 사람을 base 로 두고(이미 학력·직업·집·취미가 현실적으로 엮인 진짜 사람),
그 사람에게 "이 직업·이 동네라면 소비를 이만큼 하겠지"를 우리 통계로 조건부 부여.

4단계:
  1) 동네     — NVIDIA gu 안의 행정동을 (성별·연령) 인구비례로 추첨 → adm8
  2) 소비     — 그 (adm8·성별·연령) 셀의 대표 소비분위 ± SES 힌트(옵션)
  3) 업종     — 셀 업종비율(L1) + 취미 보정(옵션)
  4) 행태     — 셀 telecom 분포 샘플

방식 A와 결정적 차이: 짝짓기가 아니라 *한 사람한테서 파생* → 모순 원천 차단.
SES 힌트·취미 보정은 on/off 옵션 (꺼도 셀 marginal 은 맞음 = 기본 안전).

LLM 호출 0, 결정적(seed).

사용:
  python -m scripts.persona.build_conditional --limit 10
  python -m scripts.persona.build_conditional --no-ses-hint --no-hobby-adjust
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    PROJECT_ROOT, PersonaRecord,
    age_to_group, build_quant_from_cell, industry_to_l1_ratio,
    load_bdc_stats, load_nvidia_seoul,
    nvidia_cell, nvidia_gu, nvidia_sex, parse_cell_key, ses_proxy,
    split_nvidia_fields, top_categories,
)


# ---------------------------------------------------------------------------
# 우리 통계 인덱싱 — (gu, sex, age) → [(adm8 key, population, profile)]
# ---------------------------------------------------------------------------
def index_profiles_by_gu_cell(profiles: dict) -> dict[tuple, list[tuple]]:
    idx: dict[tuple, list[tuple]] = defaultdict(list)
    for key, prof in profiles.items():
        adm8, sex, age = parse_cell_key(key)
        gu = prof.get("location", {}).get("gu") or ""
        pop = float(prof.get("demographics", {}).get("population") or 1.0)
        idx[(gu, sex, age)].append((key, pop, prof))
    return dict(idx)


def pick_dong_by_population(
    profile_index: dict[tuple, list[tuple]],
    cell: tuple, rng: random.Random,
) -> tuple | None:
    """(gu,sex,age) 셀 안에서 행정동을 인구비례 추첨. 셀 비면 fallback."""
    gu, sex, age = cell
    pool = profile_index.get(cell)
    level = "gu_sex_age"
    if not pool:   # gu 무시 → (sex, age)
        pool = [t for c, lst in profile_index.items() if c[1] == sex and c[2] == age for t in lst]
        level = "sex_age"
    if not pool:
        return None
    weights = [p for (_, p, _) in pool]
    chosen = rng.choices(pool, weights=weights, k=1)[0]
    return (*chosen, level)   # (key, pop, prof, level)


# ---------------------------------------------------------------------------
# 2단계: SES 힌트로 소비분위 이동
# ---------------------------------------------------------------------------
def shift_level_by_ses(base_level: int, ses: float, enable: bool,
                       rng: random.Random) -> int:
    """셀 대표분위를 SES(0~1) 로 ±이동. enable=False 면 작은 노이즈만.

    SES 0.5 = 이동 없음, 1.0 = +2, 0.0 = -2 (최대 ±2분위).
    """
    if enable:
        shift = round((ses - 0.5) * 4)        # [-2, +2]
    else:
        shift = rng.choice([-1, 0, 0, 1])     # 셀 marginal 중심 작은 변동
    return max(1, min(10, base_level + shift))


# ---------------------------------------------------------------------------
# 3단계: 취미·가족 → 업종 L1 보정
# ---------------------------------------------------------------------------
_HOBBY_KEYWORDS = {
    "여가": ("등산", "산책", "골프", "스크린", "노래방", "공연", "영화", "게임", "낚시", "캠핑", "당구", "볼링"),
    "건강": ("헬스", "운동", "요가", "필라테스", "트레이닝", "수영", "병원", "건강"),
    "식사": ("맛집", "한식", "미식", "요리", "식도락", "탐방"),
    "카페": ("카페", "커피", "디저트", "베이커리"),
    "쇼핑": ("쇼핑", "패션", "옷", "수집", "피규어"),
    "교육": ("독서", "공부", "학습", "서예", "강의", "자격증"),
}


def adjust_l1_by_hobbies(l1_ratio: dict[str, float], nv_rec: dict,
                         enable: bool, boost: float = 0.05) -> dict[str, float]:
    """취미·가족 키워드 매칭 L1 에 boost 가산 후 재정규화. enable=False 면 원본."""
    if not enable or not l1_ratio:
        return l1_ratio
    text = " ".join([
        str(nv_rec.get("hobbies_and_interests") or ""),
        str(nv_rec.get("hobbies_and_interests_list") or ""),
    ])
    adj = dict(l1_ratio)
    for l1, kws in _HOBBY_KEYWORDS.items():
        if any(k in text for k in kws):
            adj[l1] = adj.get(l1, 0.0) + boost
    # 자녀 있으면 교육·마트 가산
    if "자녀" in (nv_rec.get("family_type") or ""):
        adj["교육"] = adj.get("교육", 0.0) + boost
        adj["마트"] = adj.get("마트", 0.0) + boost
    total = sum(adj.values())
    return {k: round(v / total, 4) for k, v in sorted(adj.items(), key=lambda x: -x[1])} if total else l1_ratio


# ---------------------------------------------------------------------------
# 메인 빌드
# ---------------------------------------------------------------------------
def build(limit: int = 0, seed: int = 42,
          ses_hint: bool = True, hobby_adjust: bool = True,
          reconcile: bool = False) -> list[dict]:
    stats = load_bdc_stats()
    nv = load_nvidia_seoul()
    profiles = stats["profiles"]
    deciles = stats["deciles"]
    profile_index = index_profiles_by_gu_cell(profiles)

    rng = random.Random(seed)
    out: list[dict] = []
    seq_by_cell: dict[tuple, int] = defaultdict(int)

    nv_iter = list(nv)
    if limit:
        rng.shuffle(nv_iter)

    for nv_rec in nv_iter:
        cell = nvidia_cell(nv_rec)
        if not cell[1]:
            continue
        picked = pick_dong_by_population(profile_index, cell, rng)
        if not picked:
            continue
        key, _pop, prof, dong_level = picked
        adm8, sex, age = parse_cell_key(key)

        # 2단계: 소비분위 (셀 대표 ± SES 힌트)
        ses = ses_proxy(nv_rec)
        base_wd = int(prof.get("consumption", {}).get("weekday_spending_level") or 5)
        base_we = int(prof.get("consumption", {}).get("weekend_spending_level") or 5)
        lv_wd = shift_level_by_ses(base_wd, ses, ses_hint, rng)
        lv_we = shift_level_by_ses(base_we, ses, ses_hint, rng)

        quant = build_quant_from_cell(prof, deciles, rng,
                                      spending_level_override=(lv_wd, lv_we))

        # 3단계: 취미 보정 (build_quant 가 만든 top_categories 를 재계산)
        l1 = industry_to_l1_ratio(prof.get("consumption", {}).get("industry_ratio", {}))
        l1_adj = adjust_l1_by_hobbies(l1, nv_rec, hobby_adjust)
        quant["spending"]["weekday_top_categories"] = top_categories(l1_adj)
        quant["spending"]["weekend_top_categories"] = top_categories(l1_adj)

        seq = seq_by_cell[(adm8, sex, age)]
        seq_by_cell[(adm8, sex, age)] += 1

        persona = _assemble(nv_rec, prof, adm8, sex, age, seq, quant,
                            dong_level, ses, ses_hint, hobby_adjust)

        # 방식 C(hybrid): 규칙기반 모순 검출 + 봉합
        if reconcile:
            from reconcile import reconcile_spending
            persona = reconcile_spending(persona, ses, prof, deciles, rng)
        out.append(persona)

    if limit:
        out = out[:limit]
    return out


def _assemble(nv_rec, prof, adm8, sex, age, seq, quant,
              dong_level, ses, ses_hint, hobby_adjust) -> dict:
    llm_fields, reserved = split_nvidia_fields(nv_rec)
    rec = PersonaRecord(
        agent_id=f"AGT_{adm8}_{sex}_{age}_{seq:03d}",
        residence={"dong_code": adm8,
                   "dong": prof.get("location", {}).get("dong") or "",
                   "gu": prof.get("location", {}).get("gu") or ""},
        personal={
            "age_group": age, "gender": sex,
            "job": nv_rec.get("occupation") or "",
            "income_level": _income_from_level(quant["spending"]["weekday_spending_level"]),
            "life_stage": _life_stage(nv_rec),
        },
        workplace={"dong_code": None, "dong": None, "commute_min": None},
        spending=quant["spending"],
        behavior=quant["behavior"],
        personality={"spending_tendency": quant["tendency"],
                     "lifestyle": (nv_rec.get("persona") or "")[:60]},
        nvidia_persona=llm_fields,
        nvidia_reserved=reserved,
        match_meta={"method": "conditional-graft", "dong_pick_level": dong_level,
                    "nvidia_ses": round(ses, 3), "ses_hint": ses_hint,
                    "hobby_adjust": hobby_adjust, "nvidia_uuid": nv_rec.get("uuid")},
    )
    return rec.to_dict()


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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-ses-hint", action="store_true", help="2단계 SES 힌트 끔 (가장 안전)")
    ap.add_argument("--no-hobby-adjust", action="store_true", help="3단계 취미 보정 끔")
    ap.add_argument("--reconcile", action="store_true",
                    help="방식 C(hybrid): 규칙기반 모순 검출 + 봉합 레이어 적용")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    personas = build(limit=args.limit, seed=args.seed,
                     ses_hint=not args.no_ses_hint,
                     hobby_adjust=not args.no_hobby_adjust,
                     reconcile=args.reconcile)
    out = args.out or (PROJECT_ROOT / "output" / "personas" /
                       ("hybrid_sample.json" if args.reconcile else "conditional_graft_sample.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(personas, ensure_ascii=False, indent=2), encoding="utf-8")
    label = "hybrid(C)" if args.reconcile else "conditional-graft(B)"
    print(f"[{label}] {len(personas)} personas → {out}")
    from collections import Counter
    levels = Counter(p["_match"]["dong_pick_level"] for p in personas)
    print(f"  dong pick levels: {dict(levels)}")
    if args.reconcile:
        n_recon = sum(1 for p in personas if p["_match"].get("reconciled"))
        n_warn = sum(1 for p in personas if p["_match"].get("warnings"))
        print(f"  reconciled: {n_recon}/{len(personas)}, 잔여 경고: {n_warn}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""P010 민생회복 소비쿠폰 1차 — 명부 시민의 지급 계층(일반 15만 / 차상위·한부모 30만 / 기초생활수급자 40만) 배정.

페르소나에는 수급 자격이 없다. 그래서 공개 통계로 '이 자치구·이 나이대 시민이 수급자일 확률'을 만들고,
(정책ID, aid) 고정 해시로 사람마다 한 번 뽑는다. 런·시뮬레이션이 달라도 같은 사람은 같은 계층이다.

입력(이 폴더, 모두 한국은행 자료가 아님):
  ssis_welfare_recipients_by_sigungu_202512.csv  한국사회보장정보원 '복지사업 시군구별 수급권자 현황'(2025-12, data.go.kr 15062448)
  seoul_gu_age5_202506.csv                       행정안전부 주민등록 인구 자치구×5세(2025-06, 정책 기준일 6/18 의 달)
상수(출처):
  N_BASIC_NATIONAL = 2,710,000  1차 지급 기초생활수급자 수(정부 발표, 한국경제 2025-07-05 "기초생활보장 수급자 271만 명")
  N_NEAR_NATIONAL  =   380,000  1차 지급 차상위·한부모(같은 기사 "차상위계층 38만 명")
  SHARE_65_BASIC   = 0.444      기초생활보장 수급자 중 65세 이상(전국 2025년 말, 서울신문 2026-07-31) — 서울 값이 없어 전국 값을 쓴다
가정(보고서에 적을 것):
  · 자치구 몫: 2025-12 시군구 수급권자 수의 서울 자치구 비중을 1차 지급 시점 전국 수로 비례 축소한다.
  · 20세 미만 수급자 = 기초교육급여 수급자(학생)로 본다. 성인 수급자 = 맞춤형급여 − 교육급여.
  · 차상위·한부모 = 차상위 본인부담경감 + 차상위 자활 + 차상위 장애인(한부모는 이 자료에 없어 전국 38만으로 비례 맞춤).
    나이 분포는 일반 인구와 같다고 둔다(자료 없음).
  · 60대 페르소나는 그 자치구의 60~64세·65~69세 인구 비로 확률을 섞는다.
"""
import argparse
import csv
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
N_BASIC_NATIONAL = 2_710_000
N_NEAR_NATIONAL = 380_000
SHARE_65_BASIC = 0.444
AMOUNTS = {"basic": 400_000, "near": 300_000, "general": 150_000}
NEAR_CATS = ("차상위본인부담경감대상자", "차상위자활", "차상위장애인")
POLICY_ID = "P010"


def read_ssis(path):
    nat = {}
    gu = {}
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            n = int(r["수급권자수"])
            nat[r["사업명"]] = nat.get(r["사업명"], 0) + n
            if r["시도"] == "서울특별시":
                gu.setdefault(r["시군구"], {})[r["사업명"]] = n
    return nat, gu


def read_pop(path):
    """자치구 → {age5: count}, 자치구 코드(앞 5자리) → 자치구 이름."""
    pop, code2gu = {}, {}
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            g = r["sigungu_nm"]
            code2gu[r["admdong_cd"][:5]] = g
            if r["agegrd_nm"] != "계":
                pop.setdefault(g, {})[r["agegrd_nm"]] = int(r["popl_cnt"])
    return pop, code2gu


def _age_lo(label):
    return 100 if label.startswith("100") else int(label.split(" ")[0])


def tier_probs(nat, gu_rec, pop):
    """자치구 → {'basic': {'20_64': p, '65+': p}, 'near': p}."""
    k_basic = N_BASIC_NATIONAL / nat["기초생활보장(맞춤형급여)"]
    k_near = N_NEAR_NATIONAL / sum(nat[c] for c in NEAR_CATS)
    out = {}
    for g, rec in gu_rec.items():
        ages = pop[g]
        p_adult_young = sum(v for a, v in ages.items() if 20 <= _age_lo(a) < 65)
        p_old = sum(v for a, v in ages.items() if _age_lo(a) >= 65)
        p_all = sum(ages.values())
        basic_all = rec["기초생활보장(맞춤형급여)"] * k_basic
        basic_minor = rec.get("기초교육급여", 0) * k_basic
        basic_old = basic_all * SHARE_65_BASIC
        basic_young = max(0.0, basic_all - basic_minor - basic_old)
        near = sum(rec.get(c, 0) for c in NEAR_CATS) * k_near
        out[g] = {
            "basic": {"20_64": basic_young / p_adult_young, "65+": basic_old / p_old},
            "near": near / p_all,
            "_check": {"basic_seoul_scaled": basic_all, "near_scaled": near, "pop": p_all},
        }
    return out


def person_p(aid, probs, pop, code2gu):
    parts = aid.split("_")
    dong, band = parts[1], parts[3]
    g = code2gu.get(dong[:5])
    if g is None:
        raise KeyError(f"자치구를 찾지 못함: {aid}")
    pr = probs[g]
    if band == "60대":
        a = pop[g].get("60 - 64세", 0)
        b = pop[g].get("65 - 69세", 0)
        pb = (a * pr["basic"]["20_64"] + b * pr["basic"]["65+"]) / max(1, a + b)
    elif band in ("70대이상", "70대", "80대이상"):
        pb = pr["basic"]["65+"]
    else:
        pb = pr["basic"]["20_64"]
    return g, band, pb, pr["near"]


def u_of(aid):
    return int(hashlib.sha256(f"{POLICY_ID}|tier|{aid}".encode("utf-8")).hexdigest()[:13], 16) / float(16 ** 13)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roster", action="append", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    nat, gu_rec = read_ssis(os.path.join(HERE, "ssis_welfare_recipients_by_sigungu_202512.csv"))
    pop, code2gu = read_pop(os.path.join(HERE, "seoul_gu_age5_202506.csv"))
    probs = tier_probs(nat, gu_rec, pop)
    aids = []
    for r in a.roster:
        for x in json.load(open(r, encoding="utf-8")):
            if x not in aids:
                aids.append(x)
    tiers, exp = {}, {"basic": 0.0, "near": 0.0}
    for aid in aids:
        g, band, pb, pn = person_p(aid, probs, pop, code2gu)
        u = u_of(aid)
        t = "basic" if u < pb else ("near" if u < pb + pn else "general")
        tiers[aid] = t
        exp["basic"] += pb
        exp["near"] += pn
    aid_grants = {aid: AMOUNTS[t] for aid, t in sorted(tiers.items()) if t != "general"}
    seoul_basic = sum(v["_check"]["basic_seoul_scaled"] for v in probs.values())
    seoul_near = sum(v["_check"]["near_scaled"] for v in probs.values())
    seoul_pop = sum(v["_check"]["pop"] for v in probs.values())
    summary = {
        "people": len(aids),
        "assigned": {t: sum(1 for x in tiers.values() if x == t) for t in AMOUNTS},
        "expected": {k: round(v, 1) for k, v in exp.items()},
        "seoul_scaled_counts": {"basic": round(seoul_basic), "near": round(seoul_near),
                                "population_all_ages_202506": seoul_pop,
                                "basic_share_all_ages": round(seoul_basic / seoul_pop, 4),
                                "near_share_all_ages": round(seoul_near / seoul_pop, 4)},
        "per_roster": {},
    }
    for r in a.roster:
        ids = json.load(open(r, encoding="utf-8"))
        summary["per_roster"][os.path.basename(r)] = {
            t: sum(1 for x in ids if tiers[x] == t) for t in AMOUNTS}
    out = {
        "schema": "p010_eligibility_tiers_v1",
        "policy_id": POLICY_ID,
        "amounts": AMOUNTS,
        "aid_grants": aid_grants,
        "summary": summary,
        "inputs": {
            os.path.basename(p): hashlib.sha256(open(os.path.join(HERE, p), "rb").read()).hexdigest()
            for p in ("ssis_welfare_recipients_by_sigungu_202512.csv", "seoul_gu_age5_202506.csv")
        },
        "rosters": {os.path.basename(r): hashlib.sha256(open(r, "rb").read()).hexdigest() for r in a.roster},
        "constants": {"N_BASIC_NATIONAL": N_BASIC_NATIONAL, "N_NEAR_NATIONAL": N_NEAR_NATIONAL,
                      "SHARE_65_BASIC": SHARE_65_BASIC},
        "gu_probs": {g: {"basic_20_64": round(v["basic"]["20_64"], 5), "basic_65plus": round(v["basic"]["65+"], 5),
                         "near": round(v["near"], 5)} for g, v in sorted(probs.items())},
    }
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    sys.exit(main())

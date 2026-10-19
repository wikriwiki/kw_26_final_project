"""CPU-only, post-result proxies from the preserved P016 sector ledger.

Product-level sales cannot be inferred from a supermarket POI. Absence of a
product ledger is reported as absence of measurement, never a zero purchase.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "experiments/pdf_benchmark_expansion_20260928/p016_catalog.json"
OUT = ROOT / "output/pdf_benchmark_expansion_20260928"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rel(path):
    return path.relative_to(ROOT).as_posix()


def main():
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    files = {arm: ROOT / f"output/recovery_20260928/multipolicy_v53/p016/{arm}/{arm}/sector.ledger.jsonl"
             for arm in ["on", "off"]}
    evidence = [{"path": rel(p), "sha256": sha(p)} for p in files.values()]
    data = {arm: [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            for arm, path in files.items()}
    assert all(len(v) == 200 for v in data.values())
    keyed = {arm: {(r['aid'], r['day']): r for r in rows} for arm, rows in data.items()}
    assert len(keyed['on']) == 200 and keyed['on'].keys() == keyed['off'].keys()
    selected = {arm: [r for r in rows if "2020-07-30" <= r['day'] <= "2020-08-01"]
                for arm, rows in data.items()}
    assert all(len(rows) == 120 for rows in selected.values())
    totals = {arm: {key: sum(float(r.get(key) or 0) for r in rows)
                    for key in ['online_spent', 'total_spent', 'offline_spent']}
              for arm, rows in selected.items()}
    category = {arm: sum(float((r.get('by_l1') or {}).get('마트', 0)) for r in rows)
                for arm, rows in selected.items()}
    note = ("3일·40명 시민의 동일 날짜 ON/OFF 대리값. 원문은 4개월 참여 대형마트/온라인 매출의 기술 비교로 "
            "기간·대상·비교기준이 다름. 현재 표본은 소비금액 분위로만 추출되었으며 "
            "성별·연령·행정동·소득 실측분포 매칭을 충족한 표본이 아님.")
    rows = []

    def add(id_, value, unit, formula, components, warning):
        rows.append({"id": id_, "simulation": value, "simulation_unit": unit, "formula": formula,
                     "raw_components": components, "sample_citizens": 40, "scope_note": note,
                     "quality_notes": [warning], "direct_gap_allowed": False})

    online_on, online_off = (totals[a]['online_spent'] for a in ['on', 'off'])
    add("P016_DESCRIPTIVE_ONLINE_TOTAL", 100 * (online_on / online_off - 1) if online_off > 0 else None,
        "%", "100 × (sum(ON online_spent) / sum(OFF online_spent) − 1)",
        {"on_won": online_on, "off_won": online_off},
        "엔진 온라인 비중 0.7465의 고정 배분을 포함한 전체 온라인 지출. 농축산물 온라인 상품 구매액으로 해석하지 않음.")
    add("P016_DESCRIPTIVE_MART_TOTAL", 100 * (category['on'] / category['off'] - 1) if category['off'] > 0 else None,
        "%", "100 × (sum(ON by_l1['마트']) / sum(OFF by_l1['마트']) − 1)",
        {"on_won": category['on'], "off_won": category['off']},
        "마트 L1은 슈퍼마켓 등을 포함하며 원문 참여 대형마트와 분류가 다름. 기존 C1과 구조적으로 같은 금액에서 파생된 대리값으로, 별도 독립 성과가 아님.")
    per_person = defaultdict(float)
    for r in selected['on']:
        per_person[r['aid']] += float((r.get('by_l1') or {}).get('마트', 0)) / 3
    groups = defaultdict(list)
    for aid, spend in per_person.items():
        bits = aid.split('_')
        if len(bits) < 5:
            raise ValueError(f"unexpected agent id: {aid}")
        sex, age = bits[2], bits[3]
        groups[sex].append(spend)
        age_number = re.match(r'^(\d+)', age)
        if not age_number:
            raise ValueError(f"unexpected age band: {age}")
        number = int(age_number.group(1))
        group = 'under40' if number < 40 else '40_59' if number < 60 else '60_plus'
        groups[group].append(spend)
    for id_, numerator, denominator in [("SEX_F", 'F', 'M'), ('AGE_40_59', '40_59', 'under40'), ('AGE_60_PLUS', '60_plus', 'under40')]:
        a, b = groups[numerator], groups[denominator]
        mean_a = sum(a) / len(a) if a else None
        mean_b = sum(b) / len(b) if b else None
        value = math.log(mean_a / mean_b) if mean_a and mean_b and mean_a > 0 and mean_b > 0 else None
        add("P016_CONDITIONAL_" + id_, value, "비보정 평균지출 로그비", "log(ON 집단별 일평균 마트지출 / ON 기준집단 일평균 마트지출)",
            {"numerator_group": numerator, "denominator_group": denominator, "numerator_citizens": len(a),
             "denominator_citizens": len(b), "numerator_mean_won": mean_a, "denominator_mean_won": mean_b},
            "원문은 농축산물 구매액 회귀의 조건부 통제변수 계수이며 정책 효과가 아님. 대리값은 상품을 식별하지 않는 비보정 집단 평균비. ID의 연령대만 사용했고 정확한 나이와 미성년 소득은 관측하지 않음.")
    numeric = ROOT / "output/multi_policy_v53_20260928/p016/numeric.json"
    payload = {"schema": "pdf_benchmark_simulation_v1", "policy": "P016",
               "catalog_path": rel(CATALOG), "catalog_sha256": sha(CATALOG),
               "source_evidence": evidence, "numeric_path": rel(numeric), "numeric_sha256": sha(numeric),
               "timing": "post-result exploratory", "rows": rows,
               "missing_product_measurement": "곡물/채소/과일/축산/계란 상품행이 없어 해당 품목 구매액을 측정하지 못함. 수량0으로 간주하지 않음."}
    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / "p016_simulation.json"
    body = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if dest.exists() and dest.read_text(encoding='utf-8') != body:
        raise ValueError('existing CPU evidence would change')
    dest.write_text(body, encoding='utf-8')
    assert sha(CATALOG) == payload['catalog_sha256']
    print(json.dumps({"path": rel(dest), "sha256": sha(dest), "rows": [{"id": r['id'], "simulation": r['simulation']} for r in rows]}, ensure_ascii=False))


if __name__ == '__main__':
    main()

"""Preview dated COVID facts for a simulation; not yet wired into its runtime.

This deliberately excludes grant outcomes, vaccine rows awaiting review, and
future observed case counts. It does not force a consumption response.
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/experiments/covid_support_2021"


def build_context(day: date, district: str, *, data_dir: Path = DATA) -> dict:
    schedule = json.loads((data_dir / "distancing_schedule.json").read_text(encoding="utf-8"))
    matches = [r for r in schedule["regimes"] if r["from"] <= day.isoformat() <= r["until"]]
    if len(matches) != 1:
        raise ValueError("Date has no unique sourced distancing regime")
    regime = matches[0]
    overrides = [r for r in schedule["overrides"] if r["from"] <= day.isoformat() <= r["until"]]
    cases = json.loads((data_dir / "seoul_cases_daily.json").read_text(encoding="utf-8"))
    if not cases["coverage_from"] <= day.isoformat() <= cases["coverage_until"]:
        raise ValueError("Date outside case-data coverage")
    valid = [r for r in cases["records"] if r["city_reference_date"] < day.isoformat()]
    if len(valid) < 7:
        raise ValueError("Insufficient prior case history")
    window = sorted(valid, key=lambda r: r["source_date"])[-7:]
    if any(r["quality_flags"] for r in window):
        raise ValueError("Seven-day case window includes a quarantined day")
    if district not in window[-1]["areas"] or not district.endswith("구"):
        raise ValueError("Expected a Seoul district, not other/out-of-province totals")
    from datetime import timedelta
    if any(date.fromisoformat(b["source_date"]) - date.fromisoformat(a["source_date"]) != timedelta(days=1) for a, b in zip(window, window[1:])):
        raise ValueError("Seven-day case window includes a quarantined or missing day")
    covid_facts = {
        "day": day.isoformat(), "district": district,
        "latest_case_source_date": window[-1]["source_date"],
        "latest_city_reference_date": window[-1]["city_reference_date"],
        "seoul_cases_7d_sum": sum(r["total_cases"] for r in window),
        "district_cases_7d_sum": sum(r["areas"][district]["cases"] for r in window),
        "case_lag_basis": "Conservative assumed availability: city_reference_date strictly before dawn date. Retrospective revised data.",
        "distancing": regime, "overrides": overrides,
        "vaccination_status": "unknown", "vaccination_reason": "Individual history absent and downloaded aggregate counts require review",
        "runtime_integration": "not_implemented", "constraints_enforced": False,
        "source_ids": sorted(set(regime["source_ids"] + ["seoul_district_cases", "seoul_city_cases_data"] + [r["source_id"] for r in overrides])),
    }
    covid_facts["prompt_preview"] = render_facts(covid_facts)
    return covid_facts


def render_facts(context: dict) -> str:
    r = context["distancing"]
    cutoff = r["dine_in_cutoff"]
    lines = [
        f"오늘의 외부 환경: {context['day']}, 서울 {context['district']}. 코로나19 유행 상황이다.",
        f"최근 사용 가능한 7일 집계: 서울 확진 {context['seoul_cases_7d_sum']:,}명, 거주 자치구 {context['district_cases_7d_sum']:,}명.",
        f"이 집계의 마지막 발생자료 날짜는 {context['latest_case_source_date']}이다. 개인의 감염 여부나 감염 확률을 뜻하지 않는다.",
        f"식당·카페는 {cutoff} 이후 매장 취식이 제한되며 포장·배달은 가능하다." if cutoff else "식당·카페의 방역상 영업시간 제한은 해제된 기간이다. 각 매장의 실제 영업시간은 별도로 따른다.",
        "사적모임 조건: " + json.dumps(r["private_meetings"], ensure_ascii=False),
    ]
    if r.get("closed_facilities"):
        lines.append("집합금지 업종: " + ", ".join(r["closed_facilities"]) + ". 일반주점 전체와 같지 않다.")
    for override in context["overrides"]:
        lines.append(f"가정 내 가족모임 예외: 총 {override['total_max']}명, 접종 미완료 최대 {override['unvaccinated_max']}명. 가정 밖 식당 모임에는 적용되지 않는다.")
    lines += [
        "본인과 동행인의 접종 상태는 제공되지 않았다. 접종 완료자로 단정하거나 접종 예외 자격을 만들어내지 않는다.",
        "현재 페르소나의 나이·직업·가족·평소 소비 특성은 그대로인 가상실험이다. 이 날짜에 맞춰 출생연도나 성격을 재계산하지 않는다.",
        "이 정보는 활동의 배경과 허용 조건이다. 감염 우려에 대한 반응과 구매 필요는 본인의 상황에 따라 판단한다.",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--day", type=date.fromisoformat, required=True)
    parser.add_argument("--district", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = build_context(args.day, args.district)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(result["prompt_preview"])
    print("\n검증 범위: 입력 미리보기만 생성. Stage1/Stage2/Night 연결 및 제약 강제는 아직 구현되지 않음.")


if __name__ == "__main__":
    main()

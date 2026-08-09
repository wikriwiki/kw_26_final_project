"""테스트용 합성 run 산출물.

**정답을 아는 데이터**를 만든다. 대조군에는 시장 추세만, 처치군에는 시장 추세 + 정책
효과만 넣고 잡음을 넣지 않는다. 그래야 이중차분이 넣은 값을 정확히 되찾는지
소수점까지 검사할 수 있다. 실제 run 처럼 보이게 만드는 것이 목적이 아니다.
"""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

TREAT = ("식사", "카페")
CONTROL = ("건강", "쇼핑")

PRE_DAILY = {"식사": 100_000.0, "카페": 50_000.0, "건강": 80_000.0, "쇼핑": 40_000.0}
#: 대조군에 걸리는 시장 전체 추세 (사후/사전)
MARKET_GROWTH = 1.10
#: 처치군에만 추가로 걸리는 정책 효과
POLICY_LIFT = 1.25

START = date(2025, 7, 14)
PRE_DAYS = 4
POST_DAYS = 4
POLICY_FROM = START + timedelta(days=PRE_DAYS)
#: 한 업종의 하루치를 몇 건으로 쪼갤지 — 금액이 정확히 나눠떨어지게 고른다
EVENTS_PER_CELL = 4
AGENTS = 10


def expected_did_absolute() -> float:
    """설계상 정답. `analytics.did_two_by_two` 가 이 값을 되찾아야 한다."""
    treat_pre = sum(PRE_DAILY[c] for c in TREAT)
    treat_post = treat_pre * MARKET_GROWTH * POLICY_LIFT
    return treat_post - treat_pre * MARKET_GROWTH


def build(root: Path, *, with_metrics: bool = True) -> Path:
    """합성 run 을 만들고 root 를 돌려준다."""
    root = Path(root)
    (root / "metrics").mkdir(parents=True, exist_ok=True)
    (root / "timing").mkdir(parents=True, exist_ok=True)
    (root / "checkpoints").mkdir(parents=True, exist_ok=True)

    days = [START + timedelta(days=i) for i in range(PRE_DAYS + POST_DAYS)]
    summary = []
    with (root / "events.jsonl").open("w", encoding="utf-8") as fp:
        for index, day in enumerate(days):
            post = day >= POLICY_FROM
            for category, base in PRE_DAILY.items():
                amount = base
                if post:
                    amount *= MARKET_GROWTH
                    if category in TREAT:
                        amount *= POLICY_LIFT
                per_event = amount / EVENTS_PER_CELL
                for slot in range(EVENTS_PER_CELL):
                    policy_paid = round(per_event * 0.4) if (post and category in TREAT) else 0
                    fp.write(
                        json.dumps(
                            {
                                "day": day.isoformat(),
                                "l1": category,
                                "sub": None,
                                "amt": per_event,
                                "ex": 0,
                                "elig": True,
                                "wba": slot == 0,
                                "sp": json.dumps({"P777": policy_paid}) if policy_paid else "{}",
                                "day_type": "weekend" if day.weekday() >= 5 else "weekday",
                                "aid": f"A{slot:03d}",
                                "gu": "강남구",
                                "dong": None,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            summary.append({"day": day.isoformat(), "elapsed_sec": 60, "agent_elapsed_sec": 50})

            if with_metrics:
                with (root / "metrics" / f"day_{day.isoformat()}.jsonl").open("w", encoding="utf-8") as mf:
                    for aid in range(AGENTS):
                        decile = aid % 10 + 1
                        grant = 100_000 if (post and decile <= 5) else 0
                        mf.write(
                            json.dumps(
                                {
                                    "aid": f"A{aid:03d}",
                                    "status": "ok",
                                    "spend_decile": decile,
                                    "cm_intended_grant_today": grant,
                                    "cm_grant_carry_out": max(grant - 10_000, 0),
                                    "cm_policy_allocated_total": 10_000 if grant else 0,
                                    "cm_personal_total": 20_000 * (1.2 if grant else 1.0),
                                    "cm_online_total": 5_000,
                                    "avg_sat": 0.7,
                                }
                            )
                            + "\n"
                        )
            (root / "timing" / f"day_{day.isoformat()}.json").write_text(
                json.dumps({"day": day.isoformat(), "elapsed_sec": 60}), encoding="utf-8"
            )
            (root / "checkpoints" / f"done_{day.isoformat()}.json").write_text(
                json.dumps([f"A{aid:03d}" for aid in range(AGENTS)]), encoding="utf-8"
            )
        _ = index

    # `_build_fixtures.scan_run` 은 args + completed_at 이 있어야 run 을 완료로 본다.
    (root / "summary.json").write_text(
        json.dumps(
            {
                "args": {
                    "start": START.isoformat(),
                    "days": len(days),
                    "limit": AGENTS,
                    "workers": 4,
                },
                "completed_at": "2025-07-22T00:00:00Z",
                "updated_at": "2025-07-22T00:00:00Z",
                "summary": summary,
            }
        ),
        encoding="utf-8",
    )
    (root / "poi_summary.json").write_text(
        json.dumps({"poi_total": 100, "poi_eligible": 60}), encoding="utf-8"
    )
    return root


def policy() -> dict:
    return {
        "id": "P777",
        "name": "테스트 정책",
        "type": "grant",
        "description": "합성 데이터용",
        "effective_from": POLICY_FROM.isoformat(),
        "effective_until": (START + timedelta(days=30)).isoformat(),
        "target_districts": ["서울특별시"],
        "benefit_categories": list(TREAT),
        "poi_restricted": True,
        "grant_key": "spend_decile",
        "decile_grants": {str(i): (100_000 if i <= 5 else 0) for i in range(1, 11)},
    }

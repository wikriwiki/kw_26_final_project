"""보고서 v2 — 섹션 사이의 숫자 일관성 검사.

왜 필요한가
-----------
그래프가 늘어날수록 같은 값을 **여러 경로로** 계산하게 된다.
업종별 표의 합과 총계, 이중차분 요약과 업종별 DID 의 합, 겹쳐보기 구간의
합계가 서로 어긋나면 보고서 전체의 신뢰가 무너진다. 여기서는 그 항등식들을
실제로 다시 계산해 **보고서 안에 검사 결과를 싣는다.** 통과 여부를 숨기지 않는다.

각 검사는 다음을 남긴다.

``id``        기계가 참조할 식별자
``label``     사람이 읽을 문장
``status``    ``pass`` / ``fail`` / ``skip``
``expected``  기준값 (한쪽 경로의 계산)
``actual``    비교값 (다른 경로의 계산)
``diff``      차이
``tolerance`` 허용 오차 — 반올림 누적분만 허용하고 그 이상은 실패다
"""
from __future__ import annotations

from typing import Any

# 금액은 소수 둘째 자리에서 반올림해 저장하므로 항목 수에 비례한 누적오차만 허용한다.
MONEY_TOLERANCE = 1.0
RATIO_TOLERANCE = 1e-6
#: 비율은 소수 6자리로 저장한다. 그 비율을 곱해 만든 금액은 크기에 비례해 어긋난다.
REL_TOLERANCE = 1e-6


def _check(
    check_id: str,
    label: str,
    expected: float | None,
    actual: float | None,
    *,
    tolerance: float = MONEY_TOLERANCE,
    note: str | None = None,
) -> dict[str, Any]:
    if expected is None or actual is None:
        return {
            "id": check_id,
            "label": label,
            "status": "skip",
            "expected": expected,
            "actual": actual,
            "diff": None,
            "tolerance": tolerance,
            "note": note or "비교에 필요한 값이 없어 검사하지 않았습니다.",
        }
    diff = float(actual) - float(expected)
    # 저장된 값은 이미 반올림돼 있다. 비율은 소수 6자리, 금액은 2자리다.
    # 그 비율을 2억짜리 금액에 곱하면 반올림분만으로 수백 원이 어긋난다 —
    # 실제로 out_EXP7500(사전 일평균 2.1억)에서 59원 차이로 항등식이 깨졌다.
    # 절대 허용치만 두면 금액이 커질수록 반드시 실패하므로 상대 허용치를 함께 둔다.
    allowed = tolerance + abs(float(expected)) * REL_TOLERANCE
    return {
        "id": check_id,
        "label": label,
        "status": "pass" if abs(diff) <= allowed else "fail",
        "expected": round(float(expected), 4),
        "actual": round(float(actual), 4),
        "diff": round(diff, 6),
        "tolerance": round(allowed, 6),
        "note": note,
    }


def _skip(check_id: str, label: str, note: str) -> dict[str, Any]:
    return {
        "id": check_id,
        "label": label,
        "status": "skip",
        "expected": None,
        "actual": None,
        "diff": None,
        "tolerance": None,
        "note": note,
    }


def run_checks(bundle: dict[str, Any]) -> dict[str, Any]:
    """번들 하나에 대해 모든 항등식을 검사한다."""
    checks: list[dict[str, Any]] = []
    totals = bundle.get("totals") or {}
    daily = bundle.get("daily") or []
    categories = bundle.get("categories") or []
    period = bundle.get("period") or {}
    did = bundle.get("did")
    did_rows = bundle.get("did_by_category") or []
    overlay = bundle.get("overlay") or {}
    mix = bundle.get("mix") or {}

    n_days = max(len(daily), 1)
    n_cats = max(len(categories), 1)

    # 1. 일자별 합계 == 총계
    checks.append(
        _check(
            "daily_sum_amt",
            "일자별 소비금액의 합이 전체 총계와 같다",
            totals.get("amt"),
            sum(row["amt"] for row in daily) if daily else None,
            tolerance=MONEY_TOLERANCE * n_days,
        )
    )
    checks.append(
        _check(
            "daily_sum_events",
            "일자별 이벤트 수의 합이 전체 이벤트 수와 같다",
            totals.get("events"),
            sum(row["events"] for row in daily) if daily else None,
            tolerance=0.5,
        )
    )
    checks.append(
        _check(
            "daily_sum_policy",
            "일자별 정책지급액의 합이 전체 정책지급액과 같다",
            totals.get("policy_paid"),
            sum(row["policy_paid"] for row in daily) if daily else None,
            tolerance=MONEY_TOLERANCE * n_days,
        )
    )

    # 2. 업종별 합계 == 총계
    checks.append(
        _check(
            "category_sum_amt",
            "업종별 소비금액의 합이 전체 총계와 같다",
            totals.get("amt"),
            sum(row["amt"] for row in categories) if categories else None,
            tolerance=MONEY_TOLERANCE * n_cats,
        )
    )
    checks.append(
        _check(
            "category_share_100",
            "업종별 구성비의 합이 100%다",
            100.0,
            sum(row["share"] for row in categories if row.get("share") is not None) if categories else None,
            tolerance=0.01,
        )
    )

    # 3. 정책지급 분해
    checks.append(
        _check(
            "mix_components",
            "정책지급액 + 자기부담액 = 총 소비금액",
            mix.get("amt"),
            (mix.get("policy_paid") or 0) + (mix.get("self_paid") or 0) if mix else None,
            tolerance=MONEY_TOLERANCE * n_days,
        )
    )
    by_policy = bundle.get("policy_paid_by_policy_id") or {}
    checks.append(
        _check(
            "policy_id_sum",
            "정책 ID별 지급액의 합이 전체 정책지급액과 같다",
            totals.get("policy_paid"),
            sum(by_policy.values()) if by_policy else None,
            tolerance=MONEY_TOLERANCE * max(len(by_policy), 1),
        )
    )
    over = [
        row["l1"]
        for row in categories
        if row.get("policy_paid") is not None
        and row.get("amt") is not None
        and row["policy_paid"] > row["amt"] + MONEY_TOLERANCE
    ]
    checks.append(
        {
            "id": "policy_le_amount",
            "label": "어떤 업종에서도 정책지급액이 소비금액을 넘지 않는다",
            "status": "pass" if not over else "fail",
            "expected": 0,
            "actual": len(over),
            "diff": len(over),
            "tolerance": 0,
            "note": None if not over else f"초과 업종: {', '.join(over)}",
        }
    )

    # 4. 기간 분할
    days = set(bundle.get("meta", {}).get("days") or [])
    pre, post = set(period.get("pre") or []), set(period.get("post") or [])
    checks.append(
        {
            "id": "period_partition",
            "label": "사전기간과 사후기간이 전체 일자를 겹침 없이 정확히 나눈다",
            "status": "pass" if (pre | post) == days and not (pre & post) else "fail",
            "expected": len(days),
            "actual": len(pre) + len(post),
            "diff": len(pre) + len(post) - len(days),
            "tolerance": 0,
            "note": f"사전 {len(pre)}일 · 사후 {len(post)}일 · 중복 {len(pre & post)}일",
        }
    )

    # 5. 이중차분 항등식
    if not did:
        checks.append(
            _skip(
                "did_identity",
                "이중차분 항등식",
                period.get("reason") or "이중차분을 계산할 수 없어 검사하지 않았습니다.",
            )
        )
        checks.append(_skip("did_category_sum", "업종별 DID 의 합 = 처치군 전체 DID", "이중차분 결과가 없습니다."))
        checks.append(_skip("did_bias", "단순 전후비교 − DID = 제거된 시장추세", "이중차분 결과가 없습니다."))
    else:
        counterfactual = did.get("counterfactual_post")
        control_growth = did.get("control_growth")
        treat_pre = did.get("treat_pre")
        treat_post = did.get("treat_post")
        checks.append(
            _check(
                "did_counterfactual",
                "반사실값 = 처치군 사전 × 대조군 성장률",
                counterfactual,
                (treat_pre * control_growth) if (treat_pre is not None and control_growth is not None) else None,
                tolerance=MONEY_TOLERANCE,
            )
        )
        checks.append(
            _check(
                "did_identity",
                "DID = 처치군 사후 − 반사실값",
                did.get("did_absolute"),
                (treat_post - counterfactual) if (treat_post is not None and counterfactual is not None) else None,
                tolerance=MONEY_TOLERANCE,
            )
        )
        checks.append(
            _check(
                "did_bias",
                "단순 전후비교 − DID = 제거된 시장추세",
                did.get("bias_removed"),
                (
                    (did.get("naive_before_after") - did.get("did_absolute"))
                    if (did.get("naive_before_after") is not None and did.get("did_absolute") is not None)
                    else None
                ),
                tolerance=MONEY_TOLERANCE,
            )
        )
        targeted_sum = sum(
            row["did_absolute"] for row in did_rows if row.get("targeted") and row.get("did_absolute") is not None
        )
        checks.append(
            _check(
                "did_category_sum",
                "정책 대상 업종별 DID 의 합 = 처치군 전체 DID",
                did.get("did_absolute"),
                targeted_sum if did_rows else None,
                tolerance=MONEY_TOLERANCE * max(len(did_rows), 1),
                note="같은 대조군 성장률로 반사실을 만들었으므로 두 값은 반드시 같아야 한다.",
            )
        )
        checks.append(
            _check(
                "did_relative",
                "상대 DID = 처치군 성장률 − 대조군 성장률",
                did.get("did_relative"),
                (
                    did.get("treat_growth") - did.get("control_growth")
                    if (did.get("treat_growth") is not None and did.get("control_growth") is not None)
                    else None
                ),
                tolerance=RATIO_TOLERANCE * 10,
            )
        )

    # 6. 업종 표와 DID 표가 같은 사전/사후 값을 쓴다
    if did_rows and categories:
        by_l1 = {row["l1"]: row for row in categories}
        mismatched = [
            row["l1"]
            for row in did_rows
            if row["l1"] in by_l1
            and abs((row.get("pre_daily") or 0) - (by_l1[row["l1"]].get("pre_daily_amt") or 0)) > MONEY_TOLERANCE
        ]
        checks.append(
            {
                "id": "category_did_alignment",
                "label": "업종 비교표와 DID 표가 같은 사전 일평균을 쓴다",
                "status": "pass" if not mismatched else "fail",
                "expected": 0,
                "actual": len(mismatched),
                "diff": len(mismatched),
                "tolerance": 0,
                "note": None if not mismatched else f"불일치 업종: {', '.join(mismatched)}",
            }
        )
    else:
        checks.append(_skip("category_did_alignment", "업종 비교표와 DID 표 정렬", "두 표 중 하나가 비어 있습니다."))

    # 7. 겹쳐보기 구간
    if overlay.get("available"):
        pre_window = overlay["overall"]["pre_days"]
        post_window = overlay["overall"]["post_days"]
        ok = set(pre_window) <= pre and set(post_window) <= post and len(pre_window) == len(post_window)
        checks.append(
            {
                "id": "overlay_window",
                "label": "겹쳐보기의 전/후 구간이 같은 길이이고 각각 올바른 기간에서 나왔다",
                "status": "pass" if ok else "fail",
                "expected": len(pre_window),
                "actual": len(post_window),
                "diff": len(post_window) - len(pre_window),
                "tolerance": 0,
                "note": f"{pre_window[0] if pre_window else '—'}~{pre_window[-1] if pre_window else '—'} vs "
                f"{post_window[0] if post_window else '—'}~{post_window[-1] if post_window else '—'}",
            }
        )
        overlay_pre_sum = sum(overlay["overall"]["pre"])
        ledger_pre_sum = sum(row["amt"] for row in daily if row["day"] in set(pre_window))
        checks.append(
            _check(
                "overlay_pre_total",
                "겹쳐보기 사전 곡선의 합이 해당 일자의 원장 합계와 같다",
                ledger_pre_sum,
                overlay_pre_sum,
                tolerance=MONEY_TOLERANCE * max(len(pre_window), 1),
            )
        )
        overlay_post_sum = sum(overlay["overall"]["post"])
        ledger_post_sum = sum(row["amt"] for row in daily if row["day"] in set(post_window))
        checks.append(
            _check(
                "overlay_post_total",
                "겹쳐보기 사후 곡선의 합이 해당 일자의 원장 합계와 같다",
                ledger_post_sum,
                overlay_post_sum,
                tolerance=MONEY_TOLERANCE * max(len(post_window), 1),
            )
        )
        category_sum = sum(
            sum(series["post"]) for series in overlay.get("by_category", {}).values()
        )
        checks.append(
            _check(
                "overlay_category_total",
                "겹쳐보기 업종별 사후 곡선의 합이 전체 사후 곡선의 합과 같다",
                overlay_post_sum,
                category_sum if overlay.get("by_category") else None,
                tolerance=MONEY_TOLERANCE * max(len(overlay.get("by_category", {})), 1) * 2,
            )
        )
    else:
        for check_id, label in (
            ("overlay_window", "겹쳐보기 구간 길이"),
            ("overlay_pre_total", "겹쳐보기 사전 합계"),
            ("overlay_post_total", "겹쳐보기 사후 합계"),
            ("overlay_category_total", "겹쳐보기 업종 합계"),
        ):
            checks.append(_skip(check_id, label, overlay.get("reason") or "겹쳐보기를 만들 수 없었습니다."))

    # 8. 이벤트 스터디 관측 수
    study = bundle.get("event_study") or {}
    if study.get("available"):
        checks.append(
            _check(
                "event_study_points",
                "이벤트 스터디의 관측 일수가 분석 일수와 같다",
                len(bundle.get("meta", {}).get("days") or []),
                len(study.get("points") or []),
                tolerance=0,
            )
        )
    else:
        checks.append(
            _skip("event_study_points", "이벤트 스터디 관측 일수", study.get("reason") or "이벤트 스터디를 만들 수 없었습니다.")
        )

    # 9. 분위 표
    deciles = bundle.get("deciles") or {}
    if deciles.get("available"):
        treated = [item for item in deciles["items"] if item.get("treated")]
        checks.append(
            {
                "id": "decile_treated_present",
                "label": "지급을 받은 분위가 한 개 이상 존재한다",
                "status": "pass" if treated else "fail",
                "expected": ">=1",
                "actual": len(treated),
                "diff": None,
                "tolerance": 0,
                "note": "지급 분위가 하나도 없으면 정책이 배선되지 않았다는 뜻이다.",
            }
        )
    else:
        checks.append(_skip("decile_treated_present", "지급 분위 존재", deciles.get("reason") or "분위 정보가 없습니다."))

    failed = [c for c in checks if c["status"] == "fail"]
    skipped = [c for c in checks if c["status"] == "skip"]
    passed = [c for c in checks if c["status"] == "pass"]
    return {
        "checks": checks,
        "counts": {"pass": len(passed), "fail": len(failed), "skip": len(skipped), "total": len(checks)},
        "consistent": not failed,
        "verdict": (
            "모든 항등식이 일치합니다."
            if not failed
            else f"{len(failed)}개 항등식이 어긋났습니다 — 보고서의 수치를 그대로 인용하면 안 됩니다."
        ),
        "failed_ids": [c["id"] for c in failed],
        "skipped_ids": [c["id"] for c in skipped],
    }

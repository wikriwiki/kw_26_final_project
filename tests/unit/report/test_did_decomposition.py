"""이중차분 분해 검사 — 축을 바꿔도 합은 같아야 한다.

보고서에 그림이 늘어난 만큼, 같은 값을 계산하는 경로도 늘었다. 세부업종으로
쪼개든 지역으로 쪼개든 날짜로 펴든, 모두 **하나의 대조군 성장률**을 쓰기 때문에
부분의 합은 전체와 정확히 같아야 한다. 근사가 아니라 항등식이다.

여기서는 그 항등식들을 합성 데이터로 검사한다. 합성 데이터는 정답을 알고 있으므로
"돌아간다"가 아니라 "넣은 값을 되찾는가"를 소수점까지 볼 수 있다.
"""
from __future__ import annotations

import tempfile
import unittest
from datetime import timedelta
from pathlib import Path

from scripts.report import analytics, consistency

from . import _demo_run


class DecompositionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp = tempfile.TemporaryDirectory(prefix="report-did-")
        cls.root = _demo_run.build(Path(cls.temp.name) / "out_DID")
        cls.bundle = analytics.build_bundle(
            run_id="DID", run_root=cls.root, policy=_demo_run.policy()
        )
        cls.did_absolute = cls.bundle["did"]["did_absolute"]

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp.cleanup()

    # ----------------------------------------------------------------- 시간축

    def test_daily_counterfactual_averages_back_to_the_two_by_two_estimate(self) -> None:
        """날짜별로 편 정의의 평균이 2×2 추정치와 같아야 한다.

        같지 않다면 시간 궤적 그림과 DID 표가 서로 다른 값을 주장하게 된다.
        """
        cf = self.bundle["did_counterfactual_daily"]
        self.assertTrue(cf["available"])
        self.assertAlmostEqual(cf["mean_gap_post"], self.did_absolute, delta=1.0)
        self.assertAlmostEqual(
            cf["cumulative_gap_total"] / cf["post_days"], self.did_absolute, delta=1.0
        )

    def test_counterfactual_tracks_the_actual_line_before_the_policy(self) -> None:
        """사전 구간에서는 두 선이 붙어 있어야 한다 — 합성 데이터에 사전추세를 넣지 않았다."""
        cf = self.bundle["did_counterfactual_daily"]
        pre = [point for point in cf["points"] if point["phase"] == "pre"]
        self.assertTrue(pre)
        for point in pre:
            self.assertAlmostEqual(point["treat"], point["counterfactual"], delta=1.0)

    def test_counterfactual_covers_every_analysed_day(self) -> None:
        cf = self.bundle["did_counterfactual_daily"]
        period = self.bundle["period"]
        self.assertEqual(len(cf["points"]), len(period["pre"]) + len(period["post"]))

    # ----------------------------------------------------------------- 업종축

    def test_subcategory_did_sums_to_the_treatment_group_did(self) -> None:
        sub = self.bundle["did_by_subcategory"]
        self.assertTrue(sub["available"])
        self.assertAlmostEqual(
            sum(item["did_absolute"] for item in sub["items"]), self.did_absolute, delta=1.0
        )

    def test_subcategory_did_sums_to_its_own_category_did(self) -> None:
        """한 업종 안의 세부업종 합이 그 업종의 DID 와 같아야 한다."""
        sub = self.bundle["did_by_subcategory"]
        by_category = {row["l1"]: row["did_absolute"] for row in self.bundle["did_by_category"]}
        for l1, total in sub["by_l1"].items():
            self.assertAlmostEqual(total, by_category[l1], delta=1.0, msg=f"{l1} 세부업종 합 불일치")

    def test_unlabelled_subcategories_are_kept_not_dropped(self) -> None:
        """세부업종이 비어 있어도 버리지 않는다 — 버리면 부분합이 전체와 어긋난다."""
        with tempfile.TemporaryDirectory(prefix="report-did-nosub-") as temp:
            root = Path(temp) / "out_NOSUB"
            _demo_run.build(root)
            path = root / "events.jsonl"
            text = path.read_text(encoding="utf-8")
            for name in _demo_run.SUBCATEGORIES:
                text = text.replace(f'"sub": "{name}"', '"sub": null')
            path.write_text(text, encoding="utf-8")
            bundle = analytics.build_bundle(run_id="NOSUB", run_root=root, policy=_demo_run.policy())
            sub = bundle["did_by_subcategory"]
            self.assertTrue(sub["available"])
            self.assertTrue(all(item["l2"] == analytics.UNCLASSIFIED for item in sub["items"]))
            self.assertAlmostEqual(
                sum(item["did_absolute"] for item in sub["items"]),
                bundle["did"]["did_absolute"],
                delta=1.0,
            )

    def test_pareto_shares_add_up_to_one_hundred_percent(self) -> None:
        pareto = self.bundle["did_pareto"]
        self.assertTrue(pareto["available"])
        self.assertAlmostEqual(pareto["items"][-1]["cumulative_pct"], 100.0, places=3)
        self.assertAlmostEqual(
            sum(item["share_pct"] for item in pareto["items"]), 100.0, places=3
        )

    # --------------------------------------------------------------- 지역·요일

    def test_region_did_sums_to_the_treatment_group_did(self) -> None:
        region = self.bundle["did_by_region"]
        self.assertTrue(region["available"])
        self.assertEqual(
            sorted(item["name"] for item in region["items"]), sorted(_demo_run.DISTRICTS)
        )
        self.assertAlmostEqual(region["total_did"], self.did_absolute, delta=1.0)

    def test_daytype_did_sums_to_the_treatment_group_did(self) -> None:
        daytype = self.bundle["did_by_daytype"]
        self.assertTrue(daytype["available"])
        self.assertAlmostEqual(daytype["total_did"], self.did_absolute, delta=1.0)

    # ------------------------------------------------------------------- 강건성

    def test_placebo_on_a_policy_free_window_returns_zero(self) -> None:
        """정책이 없던 구간에 가짜 시행일을 두면 추정치는 0 이어야 한다."""
        placebo = self.bundle["did_placebo"]
        self.assertTrue(placebo["available"])
        self.assertAlmostEqual(placebo["did_absolute"], 0.0, delta=1.0)
        self.assertIn(placebo["fake_policy_from"], self.bundle["period"]["pre"])

    def test_placebo_is_skipped_with_a_reason_when_the_pre_window_is_too_short(self) -> None:
        bundle = analytics.build_bundle(
            run_id="SHORT",
            run_root=self.root,
            policy=_demo_run.policy(),
            start=(_demo_run.POLICY_FROM - timedelta(days=2)).isoformat(),
            days=4,
        )
        placebo = bundle["did_placebo"]
        self.assertFalse(placebo["available"])
        self.assertTrue(placebo["reason"])

    def test_decile_did_uses_people_not_categories(self) -> None:
        decile = self.bundle["did_by_decile"]
        self.assertTrue(decile["available"])
        self.assertEqual(decile["granted_deciles"], [1, 2, 3, 4, 5])
        self.assertEqual(decile["other_deciles"], [6, 7, 8, 9, 10])
        # 지급 분위에만 소비를 더 넣었으므로 순효과는 양수여야 한다.
        self.assertGreater(decile["did_absolute"], 0)

    # ---------------------------------------------------------------- 겹쳐보기

    def test_overlay_cumulative_ends_at_the_window_total(self) -> None:
        overall = self.bundle["overlay"]["overall"]
        self.assertAlmostEqual(overall["post_cumulative"][-1], sum(overall["post"]), places=2)
        self.assertAlmostEqual(overall["pre_cumulative"][-1], sum(overall["pre"]), places=2)

    def test_overlay_index_is_anchored_on_the_pre_window_average(self) -> None:
        overall = self.bundle["overlay"]["overall"]
        base = sum(overall["pre"]) / len(overall["pre"])
        self.assertAlmostEqual(overall["index_base"], base, places=2)
        self.assertAlmostEqual(
            overall["post_index"][0], overall["post"][0] / base * 100, places=2
        )

    # ---------------------------------------------------------------- 일관성

    def test_every_new_identity_is_actually_checked(self) -> None:
        """새 그림마다 대응하는 항등식 검사가 실제로 돌아야 한다."""
        checks = {check["id"]: check for check in consistency.run_checks(self.bundle)["checks"]}
        required = {
            "cf_daily_mean_matches_did",
            "cf_cumulative_matches_did",
            "cf_points_cover_window",
            "subcategory_did_sum",
            "region_did_sum",
            "daytype_did_sum",
            "pareto_cumulative_ends_at_100",
            "pareto_total_matches_positive_did",
            "overlay_cumulative_last",
            "overlay_index_base",
            "placebo_uses_pre_period_only",
        }
        missing = required - set(checks)
        self.assertFalse(missing, f"검사가 없는 항등식: {sorted(missing)}")
        for check_id in sorted(required):
            self.assertEqual(
                checks[check_id]["status"], "pass", f"{check_id} 가 통과하지 못했습니다"
            )

    def test_a_broken_decomposition_is_caught_rather_than_averaged_away(self) -> None:
        """일부러 어긋나게 만들면 검사가 반드시 실패해야 한다.

        검사가 통과만 한다면 그 검사는 아무것도 지키지 않는 것과 같다.
        """
        broken = {**self.bundle}
        broken["did_by_region"] = {
            **self.bundle["did_by_region"],
            "total_did": self.did_absolute + 10_000,
        }
        result = consistency.run_checks(broken)
        failed = {check["id"] for check in result["checks"] if check["status"] == "fail"}
        self.assertIn("region_did_sum", failed)
        self.assertFalse(result["consistent"])


if __name__ == "__main__":
    unittest.main()


class IndependentRecomputationTests(unittest.TestCase):
    """보고서를 만든 코드를 쓰지 않고 원본에서 다시 세어 대조한다.

    같은 모듈로 두 번 계산하면 같은 버그를 두 번 얻는다. `analytics.py` 의 집계가
    통째로 틀려도 그 틀린 값들끼리는 완벽히 일치한다 — 자기 일관성은 정확성이
    아니다. 그래서 표준 라이브러리만으로 처음부터 다시 센 값과 맞춰 본다.
    """

    def test_the_independent_verifier_agrees_with_the_report(self) -> None:
        import json
        import subprocess
        import sys

        repo_root = Path(__file__).resolve().parents[3]
        script = repo_root / "scripts" / "report" / "verify_independently.py"
        self.assertTrue(script.is_file(), "독립 검증 스크립트가 없습니다")

        with tempfile.TemporaryDirectory(prefix="report-verify-") as temp:
            root = _demo_run.build(Path(temp) / "out_VERIFY")
            policy_path = Path(temp) / "policy.json"
            policy_path.write_text(
                json.dumps(_demo_run.policy(), ensure_ascii=False), encoding="utf-8"
            )
            out = Path(temp) / "REPORT.html"
            build = subprocess.run(
                [
                    sys.executable,
                    str(repo_root / "scripts" / "report" / "build_report_v2.py"),
                    "--run-id", "VERIFY",
                    "--run-root", str(root),
                    "--policy-json", str(policy_path),
                    "--out", str(out),
                ],
                capture_output=True,
                text=True,
                cwd=str(repo_root),
            )
            self.assertEqual(build.returncode, 0, build.stderr[-2000:])

            check = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--run-root", str(root),
                    "--data-json", str(out.with_suffix(".data.json")),
                    "--policy-json", str(policy_path),
                ],
                capture_output=True,
                text=True,
                cwd=str(repo_root),
            )
            self.assertEqual(
                check.returncode, 0, f"독립 재계산이 어긋났습니다\n{check.stdout}\n{check.stderr}"
            )
            self.assertIn("전부 일치", check.stdout)

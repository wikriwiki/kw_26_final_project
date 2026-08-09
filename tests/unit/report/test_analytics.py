"""보고서 v2 계산 엔진 검사.

핵심은 "돌아간다"가 아니라 **넣은 값을 정확히 되찾는가**다.
합성 데이터에 시장 추세와 정책 효과를 따로 넣고, 이중차분이 정책 효과만
분리해 내는지 소수점까지 확인한다.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.report import analytics

from . import _demo_run


class AnalyticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp = tempfile.TemporaryDirectory(prefix="report-v2-")
        cls.root = _demo_run.build(Path(cls.temp.name) / "out_TEST")
        cls.policy = _demo_run.policy()
        cls.bundle = analytics.build_bundle(
            run_id="TEST", run_root=cls.root, policy=cls.policy
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp.cleanup()

    def test_period_split_puts_effective_day_in_the_post_window(self) -> None:
        period = self.bundle["period"]
        self.assertEqual(len(period["pre"]), _demo_run.PRE_DAYS)
        self.assertEqual(len(period["post"]), _demo_run.POST_DAYS)
        self.assertIn(_demo_run.POLICY_FROM.isoformat(), period["post"])
        self.assertTrue(period["usable"])

    def test_target_categories_come_from_the_policy_file(self) -> None:
        targets = self.bundle["targets"]
        self.assertEqual(targets["source"], "policy.benefit_categories")
        self.assertEqual(sorted(targets["categories"]), sorted(_demo_run.TREAT))
        self.assertEqual(sorted(self.bundle["control_categories"]), sorted(_demo_run.CONTROL))

    def test_did_recovers_the_injected_policy_effect(self) -> None:
        did = self.bundle["did"]
        self.assertIsNotNone(did)
        # 대조군에는 시장 추세만 넣었으므로 성장률이 정확히 그 값이어야 한다.
        self.assertAlmostEqual(did["control_growth"], _demo_run.MARKET_GROWTH, places=6)
        self.assertAlmostEqual(
            did["treat_growth"], _demo_run.MARKET_GROWTH * _demo_run.POLICY_LIFT, places=6
        )
        self.assertAlmostEqual(did["did_absolute"], _demo_run.expected_did_absolute(), delta=1.0)

    def test_naive_before_after_overstates_the_effect(self) -> None:
        did = self.bundle["did"]
        # 단순 전후비교에는 시장 추세가 섞여 있다 — 그래서 DID 보다 커야 한다.
        self.assertGreater(did["naive_before_after"], did["did_absolute"])
        self.assertAlmostEqual(
            did["bias_removed"], did["naive_before_after"] - did["did_absolute"], places=4
        )

    def test_category_did_sums_to_the_treatment_group_did(self) -> None:
        rows = self.bundle["did_by_category"]
        targeted = sum(r["did_absolute"] for r in rows if r["targeted"])
        self.assertAlmostEqual(targeted, self.bundle["did"]["did_absolute"], delta=1.0)
        # 대조군 업종은 반사실과 정확히 같으므로 DID 가 0 이어야 한다.
        for row in rows:
            if not row["targeted"]:
                self.assertAlmostEqual(row["did_absolute"], 0.0, delta=1.0)

    def test_event_study_pre_period_is_flat(self) -> None:
        study = self.bundle["event_study"]
        self.assertTrue(study["available"])
        pre = [p["normalized_gap"] for p in study["points"] if p["rel_day"] < 0]
        self.assertTrue(pre)
        for value in pre:
            self.assertAlmostEqual(value, 0.0, places=6)

    def test_overlay_uses_equal_length_windows_from_each_period(self) -> None:
        overlay = self.bundle["overlay"]
        self.assertTrue(overlay["available"])
        overall = overlay["overall"]
        self.assertEqual(len(overall["pre"]), len(overall["post"]))
        self.assertEqual(overlay["window_days"], min(_demo_run.PRE_DAYS, _demo_run.POST_DAYS))
        self.assertTrue(set(overall["pre_days"]) <= set(self.bundle["period"]["pre"]))
        self.assertTrue(set(overall["post_days"]) <= set(self.bundle["period"]["post"]))

    def test_totals_match_the_sum_of_every_cross_tab(self) -> None:
        totals = self.bundle["totals"]
        self.assertAlmostEqual(sum(r["amt"] for r in self.bundle["daily"]), totals["amt"], places=2)
        self.assertAlmostEqual(
            sum(r["amt"] for r in self.bundle["categories"]), totals["amt"], places=2
        )

    def test_deciles_are_rolled_up_from_metrics(self) -> None:
        deciles = self.bundle["deciles"]
        self.assertTrue(deciles["available"])
        treated = [d for d in deciles["items"] if d["treated"]]
        self.assertEqual(sorted(d["decile"] for d in treated), [1, 2, 3, 4, 5])

    def test_missing_metrics_is_reported_not_guessed(self) -> None:
        with tempfile.TemporaryDirectory(prefix="report-v2-nometrics-") as temp:
            root = _demo_run.build(Path(temp) / "out_NM", with_metrics=False)
            bundle = analytics.build_bundle(run_id="NM", run_root=root, policy=_demo_run.policy())
            self.assertFalse(bundle["deciles"]["available"])
            self.assertIn("deciles", bundle["unknown"])
            # 없는 값을 0 으로 채우지 않는다
            self.assertIsNone(bundle["daily"][0]["avg_satisfaction"])

    def test_missing_policy_date_disables_did_instead_of_faking_it(self) -> None:
        policy = {**_demo_run.policy()}
        policy.pop("effective_from")
        bundle = analytics.build_bundle(run_id="TEST", run_root=self.root, policy=policy)
        self.assertIsNone(bundle["did"])
        self.assertIn("did", bundle["unknown"])
        self.assertFalse(bundle["period"]["usable"])
        self.assertTrue(bundle["period"]["reason"])

    def test_narrowing_the_window_folds_every_table_not_just_the_main_ones(self) -> None:
        """분석 창을 자르면 요일·지역·정책ID 표도 같이 접혀야 한다.

        접지 않으면 그 세 표만 전체 기간을 보게 되어 총계와 어긋난다.
        """
        narrow = analytics.build_bundle(
            run_id="TEST",
            run_root=self.root,
            policy=self.policy,
            start=_demo_run.START.isoformat(),
            days=_demo_run.PRE_DAYS + 1,
        )
        self.assertEqual(narrow["meta"]["day_count"], _demo_run.PRE_DAYS + 1)
        self.assertLess(narrow["totals"]["amt"], self.bundle["totals"]["amt"])
        self.assertAlmostEqual(
            sum(narrow["policy_paid_by_policy_id"].values()),
            narrow["totals"]["policy_paid"],
            places=2,
        )
        self.assertAlmostEqual(
            sum(cell["amt"] for cell in narrow["daytype"].values()),
            narrow["totals"]["amt"],
            places=2,
        )
        self.assertAlmostEqual(
            sum(cell["amt"] for cell in narrow["districts"].values()),
            narrow["totals"]["amt"],
            places=2,
        )

    def test_events_file_missing_raises_instead_of_returning_zeros(self) -> None:
        with tempfile.TemporaryDirectory(prefix="report-v2-empty-") as temp:
            with self.assertRaises(analytics.AnalyticsError):
                analytics.build_bundle(
                    run_id="X", run_root=Path(temp), policy=_demo_run.policy()
                )


if __name__ == "__main__":
    unittest.main()

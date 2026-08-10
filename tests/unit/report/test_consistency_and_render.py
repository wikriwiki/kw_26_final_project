"""일관성 검사와 렌더러 검사.

일관성 검사는 **실패를 실제로 잡아내야** 의미가 있다. 그래서 통과 경로만이 아니라
값을 일부러 어긋나게 만든 번들도 함께 넣어 fail 이 나오는지 확인한다.
"""
from __future__ import annotations

import re
import tempfile
import unittest
from pathlib import Path

from scripts.report import analytics, consistency, narrator, render_v2
from scripts.report.catalog import v2_applicability, v2_catalog_payload

from . import _demo_run


class ConsistencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp = tempfile.TemporaryDirectory(prefix="report-v2-cons-")
        cls.root = _demo_run.build(Path(cls.temp.name) / "out_TEST")
        cls.policy = _demo_run.policy()
        cls.bundle = analytics.build_bundle(run_id="TEST", run_root=cls.root, policy=cls.policy)
        cls.checks = consistency.run_checks(cls.bundle)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp.cleanup()

    def test_clean_bundle_passes_every_identity(self) -> None:
        failed = [c for c in self.checks["checks"] if c["status"] == "fail"]
        self.assertEqual(failed, [], f"실패한 항등식: {[c['id'] for c in failed]}")
        self.assertTrue(self.checks["consistent"])
        self.assertEqual(self.checks["counts"]["skip"], 0)

    def test_every_check_is_reported_not_silently_dropped(self) -> None:
        ids = {c["id"] for c in self.checks["checks"]}
        for required in (
            "daily_sum_amt",
            "category_sum_amt",
            "did_identity",
            "did_category_sum",
            "overlay_window",
            "overlay_post_total",
            "event_study_points",
        ):
            self.assertIn(required, ids)
        self.assertEqual(self.checks["counts"]["total"], len(self.checks["checks"]))

    def test_tampered_category_total_is_detected(self) -> None:
        broken = {**self.bundle, "categories": [dict(row) for row in self.bundle["categories"]]}
        broken["categories"][0]["amt"] += 1_000_000
        result = consistency.run_checks(broken)
        self.assertFalse(result["consistent"])
        self.assertIn("category_sum_amt", result["failed_ids"])

    def test_tampered_did_is_detected(self) -> None:
        broken = {**self.bundle, "did": {**self.bundle["did"], "did_absolute": 1.0}}
        result = consistency.run_checks(broken)
        self.assertFalse(result["consistent"])
        self.assertIn("did_identity", result["failed_ids"])
        self.assertIn("did_category_sum", result["failed_ids"])

    def test_policy_paid_over_amount_is_detected(self) -> None:
        broken = {**self.bundle, "categories": [dict(row) for row in self.bundle["categories"]]}
        broken["categories"][0]["policy_paid"] = broken["categories"][0]["amt"] * 2
        result = consistency.run_checks(broken)
        self.assertIn("policy_le_amount", result["failed_ids"])


class RenderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp = tempfile.TemporaryDirectory(prefix="report-v2-render-")
        cls.root = _demo_run.build(Path(cls.temp.name) / "out_TEST")
        cls.policy = _demo_run.policy()
        cls.bundle = analytics.build_bundle(run_id="TEST", run_root=cls.root, policy=cls.policy)
        cls.checks = consistency.run_checks(cls.bundle)
        # LLM 없이 결정론 경로만 검사한다 — 테스트가 네트워크에 의존하면 안 된다.
        cls.narration = narrator.narrate_report(cls.bundle, cls.checks, enabled=False)
        cls.html = render_v2.build_html(
            cls.bundle,
            cls.checks,
            cls.narration,
            policy=cls.policy,
            provenance="test",
            run_id="TEST",
            source_paths=["events.jsonl"],
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp.cleanup()

    def test_report_is_self_contained(self) -> None:
        self.assertNotIn("<link", self.html)
        self.assertNotIn("http://", self.html.replace("http://www.w3.org", ""))
        self.assertNotIn("<img", self.html)

    def test_forbidden_visual_styles_are_absent(self) -> None:
        # docs/GAUNTLET_WEB_CONSOLE.md §2.1 금지 목록
        for banned in ("gradient", "backdrop-filter", "glassmorphism"):
            self.assertNotIn(banned, self.html.lower())

    def test_every_section_and_chart_is_present(self) -> None:
        for anchor, _, title, _ in render_v2.SECTION_PLAN:
            self.assertIn(f'id="{anchor}"', self.html)
            self.assertIn(title, self.html)
        # 그림이 조용히 사라지지 않았는지 — 절 수보다 많은 SVG 가 있어야 한다
        self.assertGreater(self.html.count("<svg"), len(render_v2.SECTION_PLAN))

    def test_section_selection_renumbers_the_table_of_contents(self) -> None:
        partial = render_v2.build_html(
            self.bundle,
            self.checks,
            self.narration,
            policy=self.policy,
            sections=["s6", "s7"],
        )
        anchors = re.findall(r'<li><a href="#(s\d+)">(\d+)\.', partial)
        self.assertEqual([a for a, _ in anchors], ["s1", "s6", "s7", "s10", "s11"])
        self.assertEqual([n for _, n in anchors], ["1", "2", "3", "4", "5"])
        # 항상 포함되는 절은 빼달라고 해도 남는다
        self.assertIn('id="s10"', partial)
        self.assertNotIn('id="s3"', partial)

    def test_markdown_mirrors_the_html_numbers(self) -> None:
        markdown = render_v2.build_markdown(
            self.bundle, self.checks, self.narration, policy=self.policy
        )
        did = self.bundle["did"]["did_absolute"]
        self.assertIn(f"{did:+,.0f}", markdown)
        self.assertIn("이중차분", markdown)
        self.assertIn("일관성 검증", markdown)

    def test_narration_marks_its_own_source(self) -> None:
        self.assertFalse(self.narration["used_llm"])
        for entry in self.narration["sections"].values():
            self.assertEqual(entry["source"], "deterministic")
        self.assertIn("규칙 기반 서술", self.html)


class NoPrePeriodRenderTests(unittest.TestCase):
    """시행일이 run 첫날인 경우 — 실제 run(FINAL/P010, effective_from=첫날)이 이 모양이다.

    사전 기간이 비면 `pre_daily_amt` 가 전부 0 이 된다. 그 0 을 그대로 막대로 그리면
    **"정책 전에는 소비가 0이었다"** 로 읽힌다. 비교 자체가 성립하지 않는다는 사실을
    적고, 없는 값을 0 으로 그리지 않아야 한다.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.temp = tempfile.TemporaryDirectory(prefix="report-v2-nopre-")
        cls.root = _demo_run.build(Path(cls.temp.name) / "out_NOPRE")
        cls.policy = dict(_demo_run.policy(), effective_from=_demo_run.START.isoformat())
        cls.bundle = analytics.build_bundle(run_id="NOPRE", run_root=cls.root, policy=cls.policy)
        cls.checks = consistency.run_checks(cls.bundle)
        cls.narration = narrator.narrate_report(cls.bundle, cls.checks, enabled=False)
        cls.html = render_v2.build_html(
            cls.bundle, cls.checks, cls.narration, policy=cls.policy, run_id="NOPRE"
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp.cleanup()

    def test_the_run_really_has_no_pre_period(self) -> None:
        self.assertEqual(self.bundle["period"]["pre"], [])
        self.assertIsNone(self.bundle.get("did"))

    def test_identities_still_hold(self) -> None:
        failed = [c for c in self.checks["checks"] if c["status"] == "fail"]
        self.assertEqual(failed, [], f"실패한 항등식: {[c['id'] for c in failed]}")

    def test_category_section_does_not_plot_a_zero_before_series(self) -> None:
        section = _section_html(self.html, "s5")
        # 그림 안에 '시행 전' 계열이 없어야 한다 (막대·범례·툴팁 어디에도)
        charts_only = "".join(re.findall(r"<svg\b.*?</svg>", section, re.S))
        self.assertNotIn("시행 전", charts_only)
        self.assertIn("시행 후 일평균", charts_only)
        self.assertIn("시행 전 기간이 없습니다", section)

    def test_growth_columns_read_as_missing_not_zero(self) -> None:
        section = _section_html(self.html, "s5")
        body = section[section.index("<tbody>") : section.index("</tbody>")]
        # 표의 '시행 전 일평균'·'차이' 칸은 0 이 아니라 — 로 적힌다
        self.assertNotIn(">0원<", body)
        self.assertIn('<td class="n">—</td>', body)


def _section_html(html: str, anchor: str) -> str:
    """`id="sN"` 섹션 하나를 잘라낸다. 검사가 옆 절의 문자열에 걸리지 않게."""
    start = html.index(f'id="{anchor}"')
    end = html.find("<section", start)
    return html[start : end if end != -1 else len(html)]


class CatalogTests(unittest.TestCase):
    def test_did_sections_lock_without_an_effective_date(self) -> None:
        policy = {k: v for k, v in _demo_run.policy().items() if k != "effective_from"}
        decisions = v2_applicability(policy)
        for section in ("s4", "s6", "s7"):
            applicable, reason = decisions[section]
            self.assertFalse(applicable)
            self.assertTrue(reason)
        # 개요·검증·근거는 언제나 열려 있다
        for section in ("s1", "s10", "s11"):
            self.assertTrue(decisions[section][0])

    def test_missing_events_locks_every_data_section(self) -> None:
        payload = v2_catalog_payload(_demo_run.policy(), run_artifacts={"events": False, "metrics": False})
        by_id = {item["id"]: item for item in payload["items"]}
        for section in ("s3", "s4", "s5", "s6", "s7", "s8", "s9"):
            self.assertFalse(by_id[section]["applicable"], section)
            self.assertTrue(by_id[section]["disabled_reason"], section)
        self.assertEqual(payload["required"], ["s1", "s10", "s11"])


if __name__ == "__main__":
    unittest.main()

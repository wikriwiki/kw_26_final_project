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

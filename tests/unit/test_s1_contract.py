"""S1 데이터 계약 회귀 테스트.

이 테스트는 UI 목업이 아니라 ``web/fixtures``에 고정한 실제 산출물의
형태와 항등식을 검사한다. 픽스처를 재생성할 때마다 이 테스트를 먼저
통과시켜야 S2 이후 조각을 열 수 있다.
"""
from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "web" / "fixtures"


def load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


class S1ContractTests(unittest.TestCase):
    def test_fixture_inventory_covers_every_run_in_the_index(self) -> None:
        """run 목록과 픽스처 파일이 서로를 덮는지 본다.

        예전에는 run 이름 세 개와 파일 개수 36 을 그대로 박아 두었다. 그러면
        실제 run 을 하나 더 받아올 때마다 — 콘솔이 바로 그러라고 만들어졌는데 —
        이 검사가 깨진다. 이름을 세는 대신 **관계**를 검사한다.
        """
        index = load("runs.index.json")
        ids = {item["run_id"] for item in index["items"]}
        self.assertEqual(index["total"], len(index["items"]))
        self.assertTrue(ids >= {"BASE", "FINAL", "BASE7500"}, f"기준 run 이 빠졌습니다: {ids}")
        self.assertTrue({item["status"] for item in index["items"]} <= {"completed", "incomplete"})
        # 부분 실행 하나는 반드시 남겨 둔다 — "완료가 아닌 run" 경로가 화면에서 죽지 않게
        self.assertEqual(load("run.BASE7500.detail.json")["status"], "incomplete")
        for run_id in ids:
            for suffix in ("detail", "days", "failures", "events.summary"):
                self.assertTrue(
                    (FIXTURES / f"run.{run_id}.{suffix}.json").is_file(),
                    f"run.{run_id}.{suffix}.json 이 없습니다",
                )

    def test_every_resource_declares_top_level_unknown(self) -> None:
        resources = list(FIXTURES.glob("*.json"))
        self.assertGreaterEqual(len(resources), 36)
        for path in resources:
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertIn("unknown", payload, path.name)
            self.assertIsInstance(payload["unknown"], list, path.name)

        for name in ("runs.index.json", "policies.index.json"):
            for item in load(name)["items"]:
                self.assertIn("unknown", item, f"{name}: {item}")

    def test_incomplete_resources_keep_the_complete_key_set(self) -> None:
        pairs = (
            ("run.BASE.day.2025-07-21.bottlenecks.json", "run.BASE7500.day.2025-07-14.bottlenecks.json"),
            ("run.BASE.day.2025-07-21.slow.json", "run.BASE7500.day.2025-07-14.slow.json"),
            ("run.FINAL.day.2025-08-03.failed.json", "run.BASE7500.day.2025-07-14.failed.json"),
            ("run.BASE.events.summary.json", "run.BASE7500.events.summary.json"),
        )
        for complete_name, incomplete_name in pairs:
            complete = load(complete_name)
            incomplete = load(incomplete_name)
            self.assertEqual(set(complete), set(incomplete), incomplete_name)
            self.assertFalse(incomplete["available"], incomplete_name)
            self.assertTrue(incomplete["unknown"], incomplete_name)

    def test_spend_decile_partition_preserves_agent_identity(self) -> None:
        cases = (
            "run.BASE.day.2025-07-21.json",
            "run.FINAL.day.2025-08-17.json",
            "run.BASE7500.day.2025-07-14.json",
        )
        for name in cases:
            payload = load(name)
            self.assertEqual(
                sum(row["agents"] for row in payload["by_spend_decile"]),
                payload["agents_ok"],
                name,
            )
            null_agents = sum(
                row["agents"] for row in payload["by_spend_decile"] if row["spend_decile"] is None
            )
            self.assertEqual(null_agents, payload["spend_decile_unknown_agents"], name)
            self.assertEqual(
                "spend_decile" in payload["unknown"], null_agents > 0, name
            )

    def test_prompt_preview_is_the_preflight_block_not_a_prefix_sample(self) -> None:
        for path in sorted(FIXTURES.glob("policy.P*.validate.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["exit_code"], 0, path.name)
            self.assertTrue(payload["ok"], path.name)
            preview = payload["prompt_preview"]
            self.assertIn("배경:", preview, path.name)
            self.assertTrue(
                any(line.startswith("  배경:") for line in preview.splitlines()),
                path.name,
            )
            self.assertIn("db_wiring", payload["unknown"], path.name)

    def test_policy_effective_grant_key_preserves_missing_file_defaults(self) -> None:
        index = load("policies.index.json")
        by_id = {item["id"]: item for item in index["items"]}
        self.assertEqual(by_id["P010"]["grant_key_effective"], "spend_decile")
        self.assertEqual(by_id["P010"]["grant_key_source"], "file")
        for policy_id in ("P008", "P009", "P011"):
            self.assertIsNone(by_id[policy_id]["grant_key"])
            self.assertEqual(by_id[policy_id]["grant_key_effective"], "income")
            self.assertEqual(by_id[policy_id]["grant_key_source"], "default")

    def test_fixture_payloads_are_small_server_side_responses(self) -> None:
        sizes = {path.name: path.stat().st_size for path in FIXTURES.glob("*.json")}
        self.assertLessEqual(max(sizes.values()), 200 * 1024)
        rescue = load("run.BASE7500.day.2025-07-14.json")
        self.assertTrue(rescue["aggregated_server_side"])
        self.assertGreater(rescue["source_bytes"], 10 * 1024 * 1024)
        self.assertLess(rescue["source_bytes"] / (FIXTURES / "run.BASE7500.day.2025-07-14.json").stat().st_size, 4000)

    def test_reference_status_scan_matches_the_actual_run_files(self) -> None:
        # Actual-data validation is skipped only on machines that do not mount
        # the documented data root; the checked-in fixtures remain mandatory.
        data_root = Path(os.environ.get("SIM_DATA_ROOT", r"C:\Users\srdyh\gpu_exp_data\20260802"))
        if not data_root.exists():
            self.skipTest("documented SIM_DATA_ROOT is not mounted")

        sys.path.insert(0, str(ROOT))
        from web.fixtures import _build_fixtures as builder

        for run_id, (run_root, _log) in builder.RUNS.items():
            for metrics_path in sorted((run_root / "metrics").glob("day_*.jsonl")):
                day = metrics_path.name[4:-6]
                actual = builder.status_scan(metrics_path)
                expected = load(f"run.{run_id}.days.json")
                item = next(row for row in expected["items"] if row["day"] == day)
                self.assertEqual(actual["rows"], item["metrics_rows"], f"{run_id}/{day}")
                self.assertEqual(actual["agents_ok"], item["agents_ok"], f"{run_id}/{day}")
                self.assertEqual(actual["agents_error"], item["agents_error"], f"{run_id}/{day}")


if __name__ == "__main__":
    unittest.main()

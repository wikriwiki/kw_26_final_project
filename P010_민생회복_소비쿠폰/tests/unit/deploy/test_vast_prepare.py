"""Cost controls: quotes may be stale or have invalid/over-budget prices."""
from datetime import datetime, timedelta, timezone
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location("vast_prepare", ROOT / "deploy/vast/prepare.py")
prepare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare)


class RentalPlanTests(unittest.TestCase):
    def setUp(self):
        self.now = datetime(2026, 9, 22, tzinfo=timezone.utc)
        self.snapshot = {"captured_at_utc": self.now.isoformat(), "rental_type": "on-demand",
                         "disk_gb": 100, "query": prepare.DEFAULT_QUERY,
                         "offers": [{"id": 12, "dph_total": 0.5, "gpu_name": "RTX 5090", "num_gpus": 1}]}
        self.args = dict(hourly_limit=1, hours=2, total_budget=5, transfer_reserve=1, now=self.now)

    def test_plan_reserves_transfer_cost_and_never_opens_api_port(self):
        result = prepare.rental_plan(self.snapshot, 12, **self.args)
        self.assertEqual(result["estimated_total_usd"], 2)
        self.assertFalse(result["hard_billing_cap"])
        self.assertIn("--cancel-unavail", result["create_argv"])
        self.assertNotIn("--env", result["create_argv"])

    def test_hourly_limit_blocks(self):
        self.args["hourly_limit"] = 0.49
        with self.assertRaisesRegex(ValueError, "hourly rate"):
            prepare.rental_plan(self.snapshot, 12, **self.args)

    def test_total_budget_counts_transfer(self):
        self.args["total_budget"] = 1.99
        with self.assertRaisesRegex(ValueError, "plus transfer"):
            prepare.rental_plan(self.snapshot, 12, **self.args)

    def test_stale_offer_requires_refresh(self):
        self.args["now"] += timedelta(minutes=16)
        with self.assertRaisesRegex(ValueError, "stale"):
            prepare.rental_plan(self.snapshot, 12, **self.args)

    def test_unknown_offer_cannot_create_command(self):
        with self.assertRaisesRegex(ValueError, "absent"):
            prepare.rental_plan(self.snapshot, 13, **self.args)

    def test_bad_quote_cannot_bypass_budget(self):
        for value in (-1, 0, float("nan"), float("inf")):
            with self.subTest(value=value):
                self.snapshot["offers"][0]["dph_total"] = value
                with self.assertRaises(ValueError):
                    prepare.rental_plan(self.snapshot, 12, **self.args)

    def test_image_cannot_insert_shell_command_or_use_latest(self):
        for image in ("pytorch/pytorch:latest", "pytorch/pytorch", "x:v1;echo bad"):
            with self.subTest(image=image):
                with self.assertRaises(ValueError):
                    prepare.rental_plan(self.snapshot, 12, image=image, **self.args)


if __name__ == "__main__":
    unittest.main()

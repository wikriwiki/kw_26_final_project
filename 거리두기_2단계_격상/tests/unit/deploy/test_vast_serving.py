"""Serving must keep LG model provenance and avoid the wrong AWQ format."""
import os
from pathlib import Path
import shutil
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[3]
BASH = (str(Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Git/bin/bash.exe")
        if os.name == "nt" else shutil.which("bash"))


@unittest.skipUnless(BASH and Path(BASH).is_file(), "Bash unavailable")
class ServingModelTests(unittest.TestCase):
    def command(self, **overrides):
        env = dict(os.environ)
        for name in ("MODEL", "MODEL_REVISION", "QUANTIZATION"):
            env.pop(name, None)
        env.update(overrides)
        return subprocess.run([BASH, "deploy/vast/serve_sglang.sh", "--dry-run"],
                              cwd=ROOT, env=env, text=True, encoding="utf-8", capture_output=True)

    def test_default_is_pinned_lg_with_metadata_quantization(self):
        result = self.command()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("LGAI-EXAONE/EXAONE-4.5-33B-AWQ", result.stdout)
        self.assertIn("31e6a965d0661bbe4a8b895e22a77f8271772ba0", result.stdout)
        self.assertIn("--host 127.0.0.1", result.stdout)
        self.assertIn("-m sglang.launch_server", result.stdout)
        self.assertNotIn("--quantization", result.stdout)

    def test_other_provider_is_rejected_even_in_dry_run(self):
        result = self.command(MODEL="other-provider/model")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Only LG", result.stderr)

    def test_wrong_awq_quantization_format_is_rejected(self):
        result = self.command(QUANTIZATION="awq_marlin")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("compressed-tensors metadata", result.stderr)


@unittest.skipUnless(BASH and Path(BASH).is_file(), "Bash unavailable")
class ExperimentWindowTests(unittest.TestCase):
    def command(self, **overrides):
        env = dict(os.environ)
        for name in ("MODEL", "START_DATE", "DAYS", "WORKERS"):
            env.pop(name, None)
        env.update(BUNDLE_DIR="fixture-bundle", ARM="off", OUTPUT_DIR="fixture-output")
        env.update(overrides)
        return subprocess.run([BASH, "deploy/vast/run_experiment.sh", "--dry-run"],
                              cwd=ROOT, env=env, text=True, encoding="utf-8", capture_output=True)

    def test_default_covers_four_week_policy_window(self):
        result = self.command()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--start 2017-11-19", result.stdout)
        self.assertIn("--days 28", result.stdout)
        self.assertIn("--workers 4", result.stdout)

    def test_pilot_keeps_new_start_with_one_day_override(self):
        result = self.command(DAYS="1")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--start 2017-11-19", result.stdout)
        self.assertIn("--days 1", result.stdout)


if __name__ == "__main__":
    unittest.main()

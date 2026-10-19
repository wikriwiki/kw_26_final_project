"""Deployment must preserve the real price/mobility inputs, not LFS pointers."""
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location("vast_package", ROOT / "deploy/vast/package_source.py")
package = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(package)


class StatisticalReferencePackagingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.stats = self.root / "output/stats"
        self.stats.mkdir(parents=True)
        for name in package.REQUIRED_REFERENCES:
            (self.stats / name).write_text('{"value": 1}', encoding="utf-8")

    def test_all_available_statistics_are_hashed_and_optional_absence_is_recorded(self):
        (self.stats / "additional.json").write_bytes(b'{"more": 2}')
        result = package.collect_references(self.root)
        self.assertEqual(result["output/stats/additional.json"], hashlib.sha256(b'{"more": 2}').hexdigest())
        self.assertIsNone(result["output/stats/poi_menu_price.json"])

    def test_missing_required_reference_blocks_packaging(self):
        (self.stats / "unit_price.json").unlink()
        with self.assertRaisesRegex(ValueError, "missing required.*unit_price"):
            package.collect_references(self.root)

    def test_lfs_pointer_cannot_be_treated_as_recovered_data(self):
        (self.stats / "agent_profiles.json").write_text(
            "version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 123\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Git LFS pointer"):
            package.collect_references(self.root)

    def test_corrupt_json_cannot_be_packed_as_usable_input(self):
        (self.stats / "dong_centroids.json").write_bytes(b"{bad}")
        with self.assertRaisesRegex(ValueError, "invalid statistics JSON"):
            package.collect_references(self.root)


if __name__ == "__main__":
    unittest.main()

import unittest
from pathlib import Path

from scripts.validate_pan_cancer_manifest import validate


class ManifestTest(unittest.TestCase):
    def test_minimal_fixture_has_no_metadata_errors(self):
        fixture = Path(__file__).parent / "fixtures" / "minimal_manifest.jsonl"
        self.assertEqual(validate(fixture, check_files=False), [])


if __name__ == "__main__":
    unittest.main()

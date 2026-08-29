import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "prepare_paot2_manifest.py"
SPEC = importlib.util.spec_from_file_location("prepare_paot2_manifest", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class PAOT2ManifestTests(unittest.TestCase):
    def setUp(self):
        task_map = Path(__file__).parents[1] / "configs" / "paot2_task_map.json"
        self.rules = MODULE.load_rules(task_map)

    def test_simple_task_rules(self):
        expected = {
            "04_LiTS/img/lits_1.nii.gz": ("liver", [2]),
            "05_KiTS/img/kits_001.nii.gz": ("kidney", [2]),
            "10_Decathlon/Task06_Lung/imagesTr/lung_001.nii.gz": ("lung", [1]),
            "10_Decathlon/Task10_Colon/imgTs/100.nii.gz": ("colon", [1]),
        }
        for path, result in expected.items():
            tasks = MODULE.matching_tasks(path, self.rules)
            self.assertEqual(len(tasks), 1)
            self.assertEqual((tasks[0]["primary_organ"], tasks[0]["tumor_label_values"]), result)

    def test_multi_organ_filename_expands_to_two_episodes(self):
        tasks = MODULE.matching_tasks(
            "11_MSWAL/img/03_07_MSWAL_0510_0000.nii.gz", self.rules
        )
        self.assertEqual(
            {task["target_region"] for task in tasks},
            {"liver_tumor", "pancreatic_tumor"},
        )
        self.assertEqual(
            {tuple(task["tumor_label_values"]) for task in tasks}, {(3,), (5,)}
        )

    def test_pair_parser_accepts_tabs_and_spaces(self):
        with tempfile.TemporaryDirectory() as directory:
            pairs = Path(directory) / "pairs.txt"
            pairs.write_text("a.nii.gz\tb.nii.gz\nc.nii.gz   d.nii.gz\n", encoding="utf-8")
            self.assertEqual(
                list(MODULE.read_pairs(pairs)),
                [(1, "a.nii.gz", "b.nii.gz"), (2, "c.nii.gz", "d.nii.gz")],
            )


if __name__ == "__main__":
    unittest.main()

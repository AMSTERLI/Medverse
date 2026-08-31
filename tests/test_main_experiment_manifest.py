import importlib.util
import tempfile
import unittest
from collections import Counter
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "prepare_main_experiment_manifest.py"
SPEC = importlib.util.spec_from_file_location("prepare_main_experiment_manifest", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def make_row(source: str, task: str, patient: str, case: str) -> dict:
    return {
        "case_id": case,
        "patient_id": patient,
        "source_dataset": source,
        "target_region": task,
        "image": f"/{source}/imagesTr/{patient}_0000.nii.gz",
        "mask": f"/{source}/labelsTr/{patient}.nii.gz",
        "foreground_voxels": 10,
    }


class MainExperimentManifestTests(unittest.TestCase):
    def test_curated_binary_mask_mapping_and_case_id_are_preserved(self):
        row = make_row("LiTS", "liver_tumor", "lits:0", "lits:lits_0:liver_tumor")
        row["tumor_label_values"] = [1]
        normalized = MODULE.normalize_scope([row])[0]
        self.assertEqual(normalized["case_id"], row["case_id"])
        self.assertEqual(normalized["tumor_label_values"], [1])

    def test_grouped_split_is_deterministic_and_patient_safe(self):
        rows = []
        for source in ("a", "b"):
            for task in ("liver_tumor", "kidney_tumor"):
                for index in range(30):
                    patient = f"{source}_{task}_{index:02d}"
                    rows.append(make_row(source, task, patient, f"{patient}:{task}"))
        # One multi-task patient must still remain entirely in one partition.
        rows.append(make_row("a", "kidney_tumor", "a_liver_tumor_00", "shared:kidney"))

        first = MODULE.assign_grouped_stratified_split(rows, seed=17)
        second = MODULE.assign_grouped_stratified_split(rows, seed=17)
        self.assertEqual(first, second)
        for row in rows:
            row["split"] = first[MODULE.group_key(row)]
        MODULE.validate_no_patient_leakage(rows)
        counts = Counter(row["split"] for row in rows)
        self.assertGreater(counts["train"], counts["test"])
        self.assertGreater(counts["test"], counts["val"])

    def test_contexts_are_fixed_same_task_and_prefer_same_source(self):
        rows = [make_row("a", "liver_tumor", f"p{i}", f"case{i}") for i in range(8)]
        for row in rows:
            row["split"] = "train" if row["patient_id"] != "p7" else "test"
        MODULE.freeze_contexts(rows, k=3, seed=17)
        by_id = {row["case_id"]: row for row in rows}
        for row in rows:
            contexts = [by_id[case_id] for case_id in row["context_case_ids"]]
            self.assertEqual(len(contexts), 3)
            self.assertTrue(all(context["target_region"] == row["target_region"] for context in contexts))
            self.assertTrue(all(context["source_dataset"] == row["source_dataset"] for context in contexts))
            self.assertTrue(all(context["patient_id"] != row["patient_id"] for context in contexts))

    def test_roi_resolution_strips_modality_suffix(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "dataset"
            image_dir, label_dir, organ_dir = root / "imagesTr", root / "labelsTr", root / "organTr"
            image_dir.mkdir(parents=True)
            label_dir.mkdir()
            organ_dir.mkdir()
            image = image_dir / "case_0000.nii.gz"
            mask = label_dir / "case.nii.gz"
            roi = organ_dir / "case.nii.gz"
            image.touch()
            mask.touch()
            roi.touch()
            row = make_row("dataset", "liver_tumor", "p", "case")
            row["image"], row["mask"] = str(image), str(mask)
            self.assertEqual(MODULE.resolve_roi(row), roi)


if __name__ == "__main__":
    unittest.main()

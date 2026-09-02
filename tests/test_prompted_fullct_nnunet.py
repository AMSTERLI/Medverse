import importlib.util
from pathlib import Path

import nibabel as nib
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "export_prompted_fullct_nnunet_dataset.py"
SPEC = importlib.util.spec_from_file_location(SCRIPT.stem, SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)

EVALUATE_SCRIPT = ROOT / "scripts" / "evaluate_liver_kidney_original_space.py"
EVALUATE_SPEC = importlib.util.spec_from_file_location(EVALUATE_SCRIPT.stem, EVALUATE_SCRIPT)
EVALUATE = importlib.util.module_from_spec(EVALUATE_SPEC)
assert EVALUATE_SPEC.loader is not None
EVALUATE_SPEC.loader.exec_module(EVALUATE)


def save(path: Path, array: np.ndarray, affine: np.ndarray) -> str:
    nib.save(nib.Nifti1Image(array, affine), path)
    return str(path)


def base_row(tmp_path: Path, source: str, task: str) -> tuple[dict, np.ndarray]:
    shape = (12, 10, 8)
    affine = np.diag([0.8, 0.8, 2.5, 1.0])
    image = np.full(shape, -1000, dtype=np.float32)
    image[2:10, 2:8, 1:7] = 50
    tumor = np.zeros(shape, dtype=np.uint8)
    tumor[5:7, 5:7, 3:5] = 1
    row = {
        "case_id": f"{source}:case:{task}",
        "patient_id": f"{source}:case",
        "source_case_id": "case",
        "source_dataset": source,
        "target_region": task,
        "split": "train",
        "image": save(tmp_path / f"{source}_ct.nii.gz", image, affine),
        "mask": save(tmp_path / f"{source}_tumor.nii.gz", tumor, affine),
    }
    return row, affine


def test_lits_uses_native_organ_and_liver_prompt(tmp_path: Path):
    row, affine = base_row(tmp_path, "LiTS", "liver_tumor")
    native = np.zeros((12, 10, 8), dtype=np.uint8)
    native[2:10, 2:8, 1:7] = 1
    native[5:7, 5:7, 3:5] = 2
    row["raw_mask"] = save(tmp_path / "lits_native.nii.gz", native, affine)
    row["organ_mask"] = save(tmp_path / "unused_totalseg.nii.gz", np.zeros_like(native), affine)
    label_path, prompt_path = tmp_path / "label.nii.gz", tmp_path / "prompt.nii.gz"
    metadata = MODULE.build_label_and_prompt(row, label_path, prompt_path)
    label = np.asarray(nib.load(label_path).dataobj)
    prompt = np.asarray(nib.load(prompt_path).dataobj)
    assert set(np.unique(label)) == {0, 1, 2}
    assert np.all(label[5:7, 5:7, 3:5] == 2)
    assert np.all(prompt == 0)
    assert metadata["organ_supervision_source"] == "LiTS_native_semantic_label"


def test_kits_cyst_is_part_of_prompted_organ(tmp_path: Path):
    row, affine = base_row(tmp_path, "KiTS23", "kidney_tumor")
    native = np.zeros((12, 10, 8), dtype=np.uint8)
    native[2:8, 2:7, 1:6] = 1
    native[5:7, 5:7, 3:5] = 2
    native[8:10, 4:6, 3:5] = 3
    row["raw_mask"] = save(tmp_path / "kits_native.nii.gz", native, affine)
    label_path, prompt_path = tmp_path / "label.nii.gz", tmp_path / "prompt.nii.gz"
    MODULE.build_label_and_prompt(row, label_path, prompt_path)
    label = np.asarray(nib.load(label_path).dataobj)
    prompt = np.asarray(nib.load(prompt_path).dataobj)
    assert np.all(label[8:10, 4:6, 3:5] == 1)
    assert np.all(prompt == 1)


def test_mswal_uses_totalsegmentator_only_for_organ_supervision(tmp_path: Path):
    row, affine = base_row(tmp_path, "MSWAL", "kidney_tumor")
    organ = np.zeros((12, 10, 8), dtype=np.uint8)
    organ[3:10, 2:8, 1:7] = 1
    row["organ_mask"] = save(tmp_path / "mswal_totalseg.nii.gz", organ, affine)
    label_path, prompt_path = tmp_path / "label.nii.gz", tmp_path / "prompt.nii.gz"
    metadata = MODULE.build_label_and_prompt(row, label_path, prompt_path)
    label = np.asarray(nib.load(label_path).dataobj)
    assert np.all(label[organ.astype(bool) & (label != 2)] == 1)
    assert metadata["organ_supervision_source"] == "MSWAL_TotalSegmentator_pseudo_label"


def test_evaluation_scores_only_prompted_tumor_label(tmp_path: Path):
    shape = (8, 8, 8)
    affine = np.eye(4)
    truth = np.zeros(shape, dtype=np.uint8)
    truth[3:5, 3:5, 3:5] = 1
    prediction = np.zeros(shape, dtype=np.uint8)
    prediction[1:7, 1:7, 1:7] = 1
    prediction[3:5, 3:5, 3:5] = 2
    truth_path = save(tmp_path / "truth.nii.gz", truth, affine)
    prediction_path = Path(save(tmp_path / "prediction.nii.gz", prediction, affine))
    row = {
        "case_id": "case",
        "patient_id": "patient",
        "source_dataset": "LiTS",
        "target_region": "liver_tumor",
        "mask": truth_path,
    }
    result = EVALUATE.evaluate_case(row, prediction_path, 2.0, (2,))
    assert result["dice"] == 1.0

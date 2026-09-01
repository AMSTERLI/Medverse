import importlib.util
import json
from pathlib import Path

import nibabel as nib
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_scope_label_contract_and_context_leakage():
    module = load_script("prepare_main_experiment_manifest.py")
    rows = []
    for source, task in (("LiTS", "liver_tumor"), ("KiTS", "kidney_tumor"), ("MSWAL", "liver_tumor"), ("MSWAL", "kidney_tumor")):
        for index in range(20):
            rows.append({
                "case_id": f"old:{source}:{task}:{index}", "patient_id": f"{source}_{index}",
                "source_dataset": source, "target_region": task,
                "image": f"/{source}/{index}.nii.gz", "mask": f"/{source}/{index}_seg.nii.gz",
                "foreground_voxels": 10,
            })
    rows.append({
        "case_id": "excluded", "patient_id": "x", "source_dataset": "LiTS",
        "target_region": "kidney_tumor", "image": "/x.nii.gz", "mask": "/x_seg.nii.gz",
        "foreground_voxels": 10,
    })
    scoped = module.normalize_scope(rows)
    assert len(scoped) == 80
    assert {row["source_dataset"] for row in scoped} == {"LiTS", "KiTS23", "MSWAL"}
    assert next(row for row in scoped if row["source_dataset"] == "KiTS23")["tumor_label_values"] == [2]
    assignments = module.assign_grouped_stratified_split(scoped, 20260831)
    for row in scoped:
        row["split"] = assignments[module.group_key(row)]
    module.freeze_contexts(scoped, 3, 20260831)
    module.validate(scoped, require_audit=False)
    lookup = {row["case_id"]: row for row in scoped}
    for row in scoped:
        for context_id in row["context_case_ids"]:
            context = lookup[context_id]
            assert context["split"] == "train"
            assert context["target_region"] == row["target_region"]
            assert module.group_key(context) != module.group_key(row)


def test_original_spacing_crop_and_inverse_roundtrip(tmp_path: Path):
    materialize = load_script("materialize_totalseg_roi_dataset.py")
    restore = load_script("restore_roi_prediction.py")
    shape = (13, 11, 9)
    affine = np.array([[0, -2, 0, 20], [1.5, 0, 0, -10], [0, 0, 3, 5], [0, 0, 0, 1]], dtype=float)
    image = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    raw_label = np.zeros(shape, dtype=np.uint8)
    raw_label[5:8, 4:7, 3:6] = 2
    organ = np.zeros(shape, dtype=np.uint8)
    organ[3:10, 2:9, 1:8] = 1
    image_path, label_path, organ_path = (tmp_path / name for name in ("ct.nii.gz", "label.nii.gz", "organ.nii.gz"))
    nib.save(nib.Nifti1Image(image, affine), image_path)
    nib.save(nib.Nifti1Image(raw_label, affine), label_path)
    nib.save(nib.Nifti1Image(organ, affine), organ_path)
    row = {
        "case_id": "LiTS:p1:liver_tumor", "patient_id": "p1", "study_id": "p1",
        "source_dataset": "LiTS", "source_case_id": "p1", "target_region": "liver_tumor",
        "primary_organ": "liver", "image": str(image_path), "mask": str(label_path),
        "organ_mask": str(organ_path), "tumor_label_values": [2], "split": "train",
    }
    output = materialize.process_one((row, str(tmp_path / "derived"), 0.0))
    roi = nib.load(output["roi_tumor_mask"])
    restored_path = tmp_path / "restored.nii.gz"
    restore.restore(Path(output["roi_tumor_mask"]), Path(output["geometry_metadata"]), restored_path)
    restored = nib.load(restored_path)
    assert restored.shape == shape
    assert np.allclose(restored.affine, affine)
    assert np.array_equal(np.asanyarray(restored.dataobj), raw_label == 2)
    assert output["tumor_label_values"] == [1]
    assert output["roi_tumor_coverage"] == 1.0


def test_abnormally_small_organ_uses_reproducible_full_volume_fallback(tmp_path: Path):
    materialize = load_script("materialize_totalseg_roi_dataset.py")
    shape = (16, 14, 12)
    affine = np.eye(4)
    image = np.zeros(shape, dtype=np.float32)
    raw_label = np.zeros(shape, dtype=np.uint8)
    raw_label[12:15, 10:13, 8:11] = 2
    organ = np.zeros(shape, dtype=np.uint8)
    organ[1, 1, 1] = 1
    image_path, label_path, organ_path = (tmp_path / name for name in ("ct.nii.gz", "label.nii.gz", "organ.nii.gz"))
    nib.save(nib.Nifti1Image(image, affine), image_path)
    nib.save(nib.Nifti1Image(raw_label, affine), label_path)
    nib.save(nib.Nifti1Image(organ, affine), organ_path)
    row = {
        "case_id": "KiTS23:p1:kidney_tumor", "patient_id": "p1", "study_id": "p1",
        "source_dataset": "KiTS23", "source_case_id": "p1", "target_region": "kidney_tumor",
        "primary_organ": "kidney", "image": str(image_path), "mask": str(label_path),
        "organ_mask": str(organ_path), "tumor_label_values": [2], "split": "test",
    }
    output = materialize.process_one((row, str(tmp_path / "derived"), 150.0))
    assert output["full_volume_fallback"] is True
    assert output["roi_shape"] == list(shape)
    assert output["roi_tumor_coverage"] == 1.0
    assert output["qc_flags"] == ["organ_volume_abnormally_small_full_volume_fallback"]


def test_snapshot_nnunet_plan(tmp_path: Path, capsys):
    module = load_script("snapshot_nnunetv2_plan.py")
    plans = tmp_path / "plans.json"
    fingerprint = tmp_path / "fingerprint.json"
    plans.write_text(json.dumps({"configurations": {"3d_fullres": {"spacing": [1.2, 1.2, 2.5], "patch_size": [128, 128, 96], "batch_size": 2}}}))
    fingerprint.write_text("{}")
    import sys
    previous = sys.argv
    sys.argv = ["snapshot", "--plans", str(plans), "--fingerprint", str(fingerprint), "--output", str(tmp_path / "snapshot.json")]
    try:
        module.main()
    finally:
        sys.argv = previous
    snapshot = json.loads((tmp_path / "snapshot.json").read_text())
    assert snapshot["spacing_xyz_mm"] == [1.2, 1.2, 2.5]
    assert snapshot["nnunet_patch_size"] == [128, 128, 96]

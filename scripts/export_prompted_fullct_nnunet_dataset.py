"""Export full-CT, prompt-conditioned organ/tumor targets for nnU-Net v2.

The CT is never cropped around a tumor or an externally predicted organ.  Each
case receives one scalar prompt image (liver=0, kidney=1) and one hierarchical
label map: background=0, prompted organ=1, prompted tumor=2.  LiTS and KiTS23
use their native organ annotations; MSWAL uses the task-specific
TotalSegmentator mask already recorded in the frozen experiment manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np


PROMPT_CODE = {"liver_tumor": 0, "kidney_tumor": 1}
NATIVE_ORGAN_LABELS = {
    "LiTS": {1, 2},
    "KiTS23": {1, 2, 3},
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def safe_case_identifier(row: dict[str, Any]) -> str:
    readable = "_".join(str(row.get(key, "")) for key in ("source_dataset", "source_case_id", "target_region"))
    readable = "".join(char if char.isalnum() or char == "_" else "_" for char in readable)
    digest = hashlib.sha256(str(row["case_id"]).encode("utf-8")).hexdigest()[:10]
    return f"{readable[:110]}_{digest}"


def full_ct_path(row: dict[str, Any]) -> Path:
    return Path(row.get("source_image", row["image"]))


def full_tumor_path(row: dict[str, Any]) -> Path:
    return Path(row.get("tumor_mask", row["mask"]))


def native_label_path(row: dict[str, Any]) -> Path:
    value = row.get("raw_mask")
    if not value:
        raise ValueError(f"{row['case_id']}: native LiTS/KiTS23 raw_mask is missing")
    return Path(value)


def mswal_organ_path(row: dict[str, Any]) -> Path:
    value = row.get("source_organ_mask", row.get("organ_mask"))
    if not value:
        raise ValueError(f"{row['case_id']}: MSWAL TotalSegmentator organ mask is missing")
    return Path(value)


def require_shape(case_id: str, name: str, image_shape: tuple[int, ...], other_shape: tuple[int, ...]) -> None:
    if tuple(image_shape) != tuple(other_shape):
        raise ValueError(f"{case_id}: {name} shape {other_shape} != CT shape {image_shape}")


def save_atomic(image: nib.Nifti1Image, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name.replace(".nii.gz", ".tmp.nii.gz"))
    nib.save(image, temporary)
    os.replace(temporary, destination)


def link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    source = source.resolve()
    if destination.is_symlink() or destination.exists():
        if destination.resolve() != source:
            raise FileExistsError(f"{destination} already points to a different file")
        return
    os.symlink(source, destination)


def build_label_and_prompt(row: dict[str, Any], label_path: Path, prompt_path: Path) -> dict[str, Any]:
    source = str(row["source_dataset"])
    task = str(row["target_region"])
    if task not in PROMPT_CODE:
        raise ValueError(f"{row['case_id']}: unsupported prompt task {task}")
    image_nii = nib.load(full_ct_path(row))
    tumor_nii = nib.load(full_tumor_path(row))
    require_shape(row["case_id"], "tumor", image_nii.shape, tumor_nii.shape)
    if not np.allclose(image_nii.affine, tumor_nii.affine, atol=1e-3):
        raise ValueError(f"{row['case_id']}: tumor mask affine does not match CT")
    tumor = np.asanyarray(tumor_nii.dataobj) > 0.5

    affine_repaired = False
    if source in NATIVE_ORGAN_LABELS:
        organ_nii = nib.load(native_label_path(row))
        require_shape(row["case_id"], "native organ label", image_nii.shape, organ_nii.shape)
        native = np.rint(np.asanyarray(organ_nii.dataobj)).astype(np.int16, copy=False)
        organ = np.isin(native, tuple(NATIVE_ORGAN_LABELS[source]))
        affine_repaired = not np.allclose(image_nii.affine, organ_nii.affine, atol=1e-3)
        supervision_source = f"{source}_native_semantic_label"
    elif source == "MSWAL":
        organ_nii = nib.load(mswal_organ_path(row))
        require_shape(row["case_id"], "MSWAL TotalSegmentator organ", image_nii.shape, organ_nii.shape)
        if not np.allclose(image_nii.affine, organ_nii.affine, atol=1e-3):
            raise ValueError(f"{row['case_id']}: MSWAL organ mask affine does not match CT")
        organ = np.asanyarray(organ_nii.dataobj) > 0.5
        supervision_source = "MSWAL_TotalSegmentator_pseudo_label"
    else:
        raise ValueError(f"{row['case_id']}: unsupported source dataset {source}")

    label = np.zeros(image_nii.shape, dtype=np.uint8)
    label[organ] = 1
    label[tumor] = 2
    prompt_code = PROMPT_CODE[task]
    prompt = np.full(image_nii.shape, prompt_code, dtype=np.uint8)
    header = image_nii.header.copy()
    header.set_data_dtype(np.uint8)
    label_nii = nib.Nifti1Image(label, image_nii.affine, header=header)
    prompt_nii = nib.Nifti1Image(prompt, image_nii.affine, header=header)
    label_nii.set_data_dtype(np.uint8)
    prompt_nii.set_data_dtype(np.uint8)
    save_atomic(label_nii, label_path)
    save_atomic(prompt_nii, prompt_path)
    return {
        "organ_supervision_source": supervision_source,
        "native_organ_affine_repaired_by_index_alignment": affine_repaired,
        "prompt_code": prompt_code,
        "prompt_encoding": "scalar_task_channel:liver_tumor=0,kidney_tumor=1",
        "organ_voxels": int(organ.sum()),
        "tumor_voxels": int(tumor.sum()),
        "tumor_outside_organ_voxels": int(np.logical_and(tumor, ~organ).sum()),
    }


def process_one(payload: tuple[dict[str, Any], str]) -> dict[str, Any]:
    row, raw_dir_string = payload
    raw_dir = Path(raw_dir_string)
    identifier = safe_case_identifier(row)
    training = row["split"] != "test"
    images_dir = raw_dir / ("imagesTr" if training else "imagesTs")
    labels_dir = raw_dir / ("labelsTr" if training else "labelsTs")
    ct_destination = images_dir / f"{identifier}_0000.nii.gz"
    prompt_destination = images_dir / f"{identifier}_0001.nii.gz"
    label_destination = labels_dir / f"{identifier}.nii.gz"
    link(full_ct_path(row), ct_destination)
    metadata = build_label_and_prompt(row, label_destination, prompt_destination)
    output = dict(row)
    output.update(metadata)
    output.update({
        "nnunet_case_identifier": identifier,
        "full_ct_image": str(full_ct_path(row)),
        "prompt_image": str(prompt_destination),
        "hierarchical_label": str(label_destination),
        "nnunet_label_values": {"background": 0, "prompted_organ": 1, "prompted_tumor": 2},
        "spatial_policy": "full_CT_no_hard_organ_or_tumor_crop",
    })
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--nnunet-raw", type=Path, required=True)
    parser.add_argument("--nnunet-preprocessed", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--dataset-id", type=int, default=502)
    parser.add_argument("--dataset-name", default="PromptedFullCT")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-cases", type=int)
    args = parser.parse_args()

    rows = load_jsonl(args.manifest)
    if args.max_cases is not None:
        rows = rows[: args.max_cases]
    dataset = f"Dataset{args.dataset_id:03d}_{args.dataset_name}"
    raw_dir = args.nnunet_raw / dataset
    payloads = [(row, str(raw_dir)) for row in rows]
    if args.workers == 1:
        outputs = [process_one(payload) for payload in payloads]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            outputs = list(executor.map(process_one, payloads))

    split = {
        "train": [row["nnunet_case_identifier"] for row in outputs if row["split"] == "train"],
        "val": [row["nnunet_case_identifier"] for row in outputs if row["split"] == "val"],
    }
    dataset_json = {
        "channel_names": {"0": "CT", "1": "noNorm"},
        "channel_descriptions": {"0": "full_CT", "1": "task_prompt:liver=0,kidney=1"},
        "labels": {"background": 0, "prompted_organ": 1, "prompted_tumor": 2},
        "numTraining": len(split["train"]) + len(split["val"]),
        "file_ending": ".nii.gz",
        "overwrite_image_reader_writer": "NibabelIOWithReorient",
    }
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "dataset.json").write_text(json.dumps(dataset_json, indent=2), encoding="utf-8")
    preprocessed_dir = args.nnunet_preprocessed / dataset
    preprocessed_dir.mkdir(parents=True, exist_ok=True)
    (preprocessed_dir / "splits_final.json").write_text(json.dumps([split], indent=2), encoding="utf-8")
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("w", encoding="utf-8", newline="\n") as stream:
        for row in outputs:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "dataset": dataset,
        "cases": len(outputs),
        "split": dict(Counter(row["split"] for row in outputs)),
        "task": dict(Counter(row["target_region"] for row in outputs)),
        "organ_supervision": dict(Counter(row["organ_supervision_source"] for row in outputs)),
        "input_channels": ["full_CT", "scalar_text_task_prompt"],
        "labels": dataset_json["labels"],
        "hard_roi_crop": False,
        "raw_dir": str(raw_dir),
        "preprocessed_dir": str(preprocessed_dir),
    }
    args.output_manifest.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

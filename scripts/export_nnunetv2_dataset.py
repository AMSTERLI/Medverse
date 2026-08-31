"""Export the shared ROI manifest as one fixed-split nnU-Net v2 dataset."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def case_identifier(row: dict) -> str:
    name = Path(row.get("roi_tumor_mask", row.get("tumor_mask", row["mask"]))).name
    return name[:-7] if name.endswith(".nii.gz") else Path(name).stem


def link(source: str | Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_path = Path(source).resolve()
    if destination.is_symlink() or destination.exists():
        if destination.resolve() != source_path:
            raise FileExistsError(f"{destination} already points to a different file")
        return
    os.symlink(source_path, destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--nnunet-raw", type=Path, required=True)
    parser.add_argument("--nnunet-preprocessed", type=Path, required=True)
    parser.add_argument("--dataset-id", type=int, default=501)
    parser.add_argument("--dataset-name", default="PanCancerTotalSegROI")
    args = parser.parse_args()

    rows = load_jsonl(args.manifest)
    dataset = f"Dataset{args.dataset_id:03d}_{args.dataset_name}"
    raw_dir = args.nnunet_raw / dataset
    preprocessed_dir = args.nnunet_preprocessed / dataset
    split = {"train": [], "val": []}
    seen = set()
    for row in rows:
        identifier = case_identifier(row)
        if identifier in seen:
            raise ValueError(f"duplicate nnU-Net case identifier: {identifier}")
        seen.add(identifier)
        image = row.get("roi_image", row["image"])
        organ = row.get("roi_organ_mask", row.get("organ_mask"))
        tumor = row.get("roi_tumor_mask", row.get("tumor_mask", row.get("mask")))
        if organ is None or tumor is None:
            raise ValueError(f"{row['case_id']}: ROI organ/tumor mask is missing")
        if row["split"] == "test":
            link(image, raw_dir / "imagesTs" / f"{identifier}_0000.nii.gz")
            link(organ, raw_dir / "imagesTs" / f"{identifier}_0001.nii.gz")
            link(tumor, raw_dir / "labelsTs" / f"{identifier}.nii.gz")
        else:
            link(image, raw_dir / "imagesTr" / f"{identifier}_0000.nii.gz")
            link(organ, raw_dir / "imagesTr" / f"{identifier}_0001.nii.gz")
            link(tumor, raw_dir / "labelsTr" / f"{identifier}.nii.gz")
            split[row["split"]].append(identifier)

    dataset_json = {
        "channel_names": {"0": "CT", "1": "TotalSegmentator_organ_mask"},
        "labels": {"background": 0, "tumor": 1},
        "numTraining": len(split["train"]) + len(split["val"]),
        "file_ending": ".nii.gz",
        "overwrite_image_reader_writer": "NibabelIOWithReorient",
    }
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "dataset.json").write_text(json.dumps(dataset_json, indent=2), encoding="utf-8")
    preprocessed_dir.mkdir(parents=True, exist_ok=True)
    (preprocessed_dir / "splits_final.json").write_text(
        json.dumps([split], indent=2), encoding="utf-8"
    )
    summary = {
        "dataset": dataset,
        "train": len(split["train"]),
        "val": len(split["val"]),
        "test": sum(row["split"] == "test" for row in rows),
        "input_channels": 2,
        "raw_dir": str(raw_dir),
        "preprocessed_dir": str(preprocessed_dir),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

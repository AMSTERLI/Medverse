"""Add a dedicated nnU-Net fold that deliberately trains and validates on the tiny gate set."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path


def load_case_identifier():
    source = Path(__file__).with_name("export_nnunetv2_dataset.py")
    spec = importlib.util.spec_from_file_location("export_nnunetv2_dataset", source)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.case_identifier


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
    parser.add_argument("--splits-file", type=Path, required=True)
    parser.add_argument("--prediction-input", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=1)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or any(row["split"] != "train" for row in rows):
        raise ValueError("the overfit manifest must contain only training rows")
    case_identifier = load_case_identifier()
    identifiers = [case_identifier(row) for row in rows]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("duplicate nnU-Net identifiers in overfit manifest")
    splits = json.loads(args.splits_file.read_text(encoding="utf-8"))
    while len(splits) <= args.fold:
        splits.append({"train": [], "val": []})
    # Intentional train/validation identity: this is an overfit gate, never a result split.
    splits[args.fold] = {"train": identifiers, "val": identifiers}
    args.splits_file.write_text(json.dumps(splits, indent=2), encoding="utf-8")
    for row, identifier in zip(rows, identifiers):
        link(row.get("roi_image", row["image"]), args.prediction_input / f"{identifier}_0000.nii.gz")
        link(row.get("roi_organ_mask", row["organ_mask"]), args.prediction_input / f"{identifier}_0001.nii.gz")
    print(json.dumps({"fold": args.fold, "cases": len(rows), "prediction_input": str(args.prediction_input)}, indent=2))


if __name__ == "__main__":
    main()

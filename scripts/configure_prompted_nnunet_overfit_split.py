"""Use the same four prompted full-CT cases for train and validation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def link(source: str | Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    source = Path(source).resolve()
    if destination.is_symlink() or destination.exists():
        if destination.resolve() != source:
            raise FileExistsError(f"{destination} already points to a different file")
        return
    os.symlink(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--splits-file", type=Path, required=True)
    parser.add_argument("--prediction-input", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    identifiers = [row["nnunet_case_identifier"] for row in rows]
    if not identifiers or len(identifiers) != len(set(identifiers)):
        raise ValueError("overfit manifest must contain distinct exported cases")
    splits = json.loads(args.splits_file.read_text(encoding="utf-8"))
    while len(splits) <= args.fold:
        splits.append({"train": [], "val": []})
    splits[args.fold] = {"train": identifiers, "val": identifiers}
    args.splits_file.write_text(json.dumps(splits, indent=2), encoding="utf-8")
    for row in rows:
        identifier = row["nnunet_case_identifier"]
        link(row.get("nnunet_ct_image", row["full_ct_image"]), args.prediction_input / f"{identifier}_0000.nii.gz")
        link(row["prompt_image"], args.prediction_input / f"{identifier}_0001.nii.gz")
    print(json.dumps({"fold": args.fold, "cases": len(rows), "train_equals_val": True}, indent=2))


if __name__ == "__main__":
    main()

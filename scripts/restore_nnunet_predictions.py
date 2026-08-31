"""Restore nnU-Net ROI predictions and write a common prediction manifest."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


def load_function(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return getattr(module, name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--roi-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()
    case_identifier = load_function(Path(__file__).with_name("export_nnunetv2_dataset.py"), "case_identifier")
    restore = load_function(Path(__file__).with_name("restore_roi_prediction.py"), "restore")
    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    records = []
    for row in rows:
        if row["split"] != args.split:
            continue
        roi_prediction = args.roi_predictions / f"{case_identifier(row)}.nii.gz"
        if not roi_prediction.is_file():
            raise FileNotFoundError(f"{row['case_id']}: missing {roi_prediction}")
        token = hashlib.sha256(row["case_id"].encode()).hexdigest()[:12]
        output = args.output_dir / "original" / f"{token}.nii.gz"
        restore(roi_prediction, Path(row["geometry_metadata"]), output)
        records.append({"case_id": row["case_id"], "prediction": str(output), "roi_prediction": str(roi_prediction)})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_manifest = args.output_dir / "predictions.jsonl"
    with prediction_manifest.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps({"cases": len(records), "predictions": str(prediction_manifest)}, indent=2))


if __name__ == "__main__":
    main()

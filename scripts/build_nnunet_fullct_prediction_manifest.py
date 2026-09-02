"""Map full-CT nnU-Net prediction filenames back to experiment case IDs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split")
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    records = []
    for row in rows:
        if args.split and row["split"] != args.split:
            continue
        prediction = args.prediction_dir / f"{row['nnunet_case_identifier']}.nii.gz"
        if not prediction.is_file():
            raise FileNotFoundError(f"{row['case_id']}: missing {prediction}")
        records.append({"case_id": row["case_id"], "prediction": str(prediction)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps({"cases": len(records), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()

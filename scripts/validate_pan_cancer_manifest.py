"""Validate the minimum metadata and leakage constraints of a JSONL manifest."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


REQUIRED = {
    "case_id",
    "patient_id",
    "study_id",
    "image",
    "mask",
    "split",
    "cancer_type",
    "primary_organ",
    "target_region",
    "modality",
    "phase_or_sequence",
    "center",
    "annotation_protocol",
}
SPLITS = {"train", "val", "test"}


def validate(path: Path, check_files: bool = True) -> list[str]:
    errors: list[str] = []
    patient_splits: dict[str, set[str]] = defaultdict(set)
    case_ids: set[str] = set()

    with path.open("r", encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, 1):
            if not raw_line.strip():
                continue
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                errors.append(f"line {line_number}: invalid JSON: {exc}")
                continue
            missing = sorted(REQUIRED - row.keys())
            if missing:
                errors.append(f"line {line_number}: missing fields {missing}")
                continue
            if row["split"] not in SPLITS:
                errors.append(f"line {line_number}: invalid split {row['split']!r}")
            if row["case_id"] in case_ids:
                errors.append(f"line {line_number}: duplicate case_id {row['case_id']!r}")
            case_ids.add(row["case_id"])
            patient_splits[row["patient_id"]].add(row["split"])

            if check_files:
                for field in ("image", "mask"):
                    candidate = Path(row[field])
                    if not candidate.is_absolute():
                        candidate = path.parent / candidate
                    if not candidate.is_file():
                        errors.append(f"line {line_number}: {field} not found: {candidate}")

    for patient_id, splits in patient_splits.items():
        if len(splits) > 1:
            errors.append(f"patient leakage: {patient_id!r} occurs in {sorted(splits)}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--skip-file-check", action="store_true")
    args = parser.parse_args()
    errors = validate(args.manifest, check_files=not args.skip_file_check)
    if errors:
        raise SystemExit("Manifest validation failed:\n- " + "\n- ".join(errors))
    print(f"Manifest is valid: {args.manifest}")


if __name__ == "__main__":
    main()

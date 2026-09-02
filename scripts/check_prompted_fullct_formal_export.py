"""Fail fast if the formal prompted full-CT export violates its frozen split."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


ALLOWED_TASKS = {"liver_tumor", "kidney_tumor"}
PROMPT_CODES = {"liver_tumor": 0, "kidney_tumor": 1}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--splits-file", type=Path, required=True)
    parser.add_argument("--dataset-json", type=Path, required=True)
    parser.add_argument("--expected-train", type=int, required=True)
    parser.add_argument("--expected-val", type=int, required=True)
    parser.add_argument("--expected-test", type=int, required=True)
    args = parser.parse_args()

    rows = load_jsonl(args.manifest)
    expected = {
        "train": args.expected_train,
        "val": args.expected_val,
        "test": args.expected_test,
    }
    counts = Counter(str(row["split"]) for row in rows)
    if dict(counts) != expected:
        raise ValueError(f"split counts changed: actual={dict(counts)}, expected={expected}")

    identifiers = [str(row["nnunet_case_identifier"]) for row in rows]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("duplicate nnU-Net case identifiers")

    tasks = {str(row["target_region"]) for row in rows}
    if tasks != ALLOWED_TASKS:
        raise ValueError(f"unexpected task set: {tasks}")

    patient_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        split = str(row["split"])
        patient_splits[str(row["patient_id"])].add(split)
        task = str(row["target_region"])
        if int(row["prompt_code"]) != PROMPT_CODES[task]:
            raise ValueError(f"{row['case_id']}: task/prompt mismatch")
        required = ("nnunet_ct_image", "prompt_image", "hierarchical_label")
        missing = [key for key in required if not Path(row[key]).exists()]
        if missing:
            raise FileNotFoundError(f"{row['case_id']}: missing exports {missing}")
        if int(row["organ_voxels"]) <= 0 or int(row["tumor_voxels"]) <= 0:
            raise ValueError(f"{row['case_id']}: empty organ/tumor supervision")
    leaking = {patient: splits for patient, splits in patient_splits.items() if len(splits) > 1}
    if leaking:
        raise ValueError(f"patient split leakage detected for {len(leaking)} patients")

    split_definition = json.loads(args.splits_file.read_text(encoding="utf-8"))
    if len(split_definition) != 1:
        raise ValueError("formal experiment must contain exactly one frozen fold")
    fold = split_definition[0]
    if len(fold["train"]) != expected["train"] or len(fold["val"]) != expected["val"]:
        raise ValueError("splits_final.json does not match manifest split counts")
    if set(fold["train"]) & set(fold["val"]):
        raise ValueError("train and validation identifiers overlap")

    dataset = json.loads(args.dataset_json.read_text(encoding="utf-8"))
    if int(dataset["numTraining"]) != expected["train"] + expected["val"]:
        raise ValueError("dataset.json numTraining is inconsistent")
    if dataset["labels"] != {"background": 0, "prompted_organ": 1, "prompted_tumor": 2}:
        raise ValueError(f"unexpected labels: {dataset['labels']}")

    result = {
        "cases": len(rows),
        "split": dict(counts),
        "task": dict(Counter(str(row["target_region"]) for row in rows)),
        "source": dict(Counter(str(row["source_dataset"]) for row in rows)),
        "patient_split_leakage": 0,
        "status": "passed",
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

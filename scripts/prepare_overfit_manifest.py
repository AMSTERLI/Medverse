"""Select four non-empty training patients per task for the pre-training gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def score(text: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}|{text}".encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases-per-task", type=int, default=4, choices=(2, 3, 4))
    parser.add_argument("--context-k", type=int, default=3, choices=(1, 3))
    parser.add_argument("--seed", type=int, default=20260831)
    args = parser.parse_args()
    if args.cases_per_task <= args.context_k:
        raise ValueError("cases-per-task must exceed context-k so target and contexts are distinct")
    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_task = defaultdict(list)
    for row in rows:
        if row["split"] == "train" and int(row.get("foreground_voxels", 0)) > 0:
            by_task[row["target_region"]].append(row)
    selected = []
    for task in ("liver_tumor", "kidney_tumor"):
        candidates = sorted(by_task[task], key=lambda row: score(row["case_id"], args.seed))
        task_rows = [dict(row) for row in candidates[: args.cases_per_task]]
        if len(task_rows) != args.cases_per_task:
            raise ValueError(f"{task}: need {args.cases_per_task} non-empty train cases")
        for row in task_rows:
            contexts = [candidate for candidate in task_rows if candidate["patient_id"] != row["patient_id"]]
            row["context_case_ids"] = [candidate["case_id"] for candidate in contexts[: args.context_k]]
            row["context_strategy"] = "overfit_fixed_same_task"
            row["split"] = "train"
        selected.extend(task_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for row in selected:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"rows": len(selected), "cases_per_task": args.cases_per_task, "context_k": args.context_k}))


if __name__ == "__main__":
    main()

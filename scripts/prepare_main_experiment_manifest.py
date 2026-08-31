"""Build the leakage-safe fixed split used by the two-model main experiment.

The script deliberately ignores any legacy train/validation labels in the input
manifest. Patients are grouped by source dataset and patient id, assigned once
to a deterministic 70/10/20 split, and reused by both nnU-Net and Medverse.
It also resolves precomputed TotalSegmentator organ masks and freezes K same-task
context examples from the training partition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


SPLITS = ("train", "val", "test")
DEFAULT_RATIOS = {"train": 0.70, "val": 0.10, "test": 0.20}
ORGAN_DIR_BY_IMAGE_DIR = {
    "img": "organ",
    "imgTs": "organTs",
    "imagesTr": "organTr",
    "imagesTs": "organTs",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, raw in enumerate(stream, 1):
            if not raw.strip():
                continue
            row = json.loads(raw)
            required = {"case_id", "patient_id", "source_dataset", "target_region", "image", "mask"}
            missing = required - row.keys()
            if missing:
                raise ValueError(f"line {line_number}: missing {sorted(missing)}")
            rows.append(row)
    if not rows:
        raise ValueError("input manifest is empty")
    return rows


def _nii_stem(path: Path) -> str:
    return path.name[:-7] if path.name.endswith(".nii.gz") else path.stem


def candidate_roi_paths(row: dict[str, Any]) -> Iterable[Path]:
    image = Path(row["image"])
    mask = Path(row["mask"])
    organ_dir_name = ORGAN_DIR_BY_IMAGE_DIR.get(image.parent.name)
    if organ_dir_name is None:
        return
    organ_dir = image.parent.parent / organ_dir_name
    stems = [_nii_stem(image), _nii_stem(mask)]
    stems.extend(stem[:-5] for stem in list(stems) if stem.endswith("_0000"))
    for stem in dict.fromkeys(stems):
        yield organ_dir / f"{stem}.nii.gz"


def resolve_roi(row: dict[str, Any]) -> Path | None:
    existing = row.get("organ_mask")
    if existing and Path(existing).is_file():
        return Path(existing)
    return next((path for path in candidate_roi_paths(row) if path.is_file()), None)


def group_key(row: dict[str, Any]) -> str:
    # Dataset prefix prevents coincidental numeric patient ids from different
    # public datasets being treated as the same person.
    return f"{row['source_dataset']}::{row['patient_id']}"


def stratum(row: dict[str, Any]) -> str:
    return f"{row['source_dataset']}::{row['target_region']}"


def stable_fraction(text: str) -> float:
    value = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")
    return value / float(2**64 - 1)


def assign_grouped_stratified_split(
    rows: list[dict[str, Any]], seed: int, ratios: dict[str, float] | None = None
) -> dict[str, str]:
    ratios = ratios or DEFAULT_RATIOS
    if set(ratios) != set(SPLITS) or abs(sum(ratios.values()) - 1.0) > 1e-8:
        raise ValueError("split ratios must define train/val/test and sum to one")

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[group_key(row)].append(row)
    total_by_stratum = Counter(stratum(row) for row in rows)
    target = {
        split: {name: ratios[split] * count for name, count in total_by_stratum.items()}
        for split in SPLITS
    }
    assigned_counts = {split: Counter() for split in SPLITS}
    assigned_rows = Counter()
    target_rows = {split: ratios[split] * len(rows) for split in SPLITS}

    def priority(item: tuple[str, list[dict[str, Any]]]) -> tuple[float, float]:
        key, members = item
        rarity = sum(1.0 / total_by_stratum[stratum(row)] for row in members)
        return (-rarity, stable_fraction(f"{seed}|order|{key}"))

    assignments: dict[str, str] = {}
    for key, members in sorted(groups.items(), key=priority):
        contribution = Counter(stratum(row) for row in members)

        def cost(candidate_split: str) -> tuple[float, float]:
            score = 0.0
            for split in SPLITS:
                for name, desired in target[split].items():
                    value = assigned_counts[split][name]
                    if split == candidate_split:
                        value += contribution[name]
                    score += ((value - desired) / max(desired, 1.0)) ** 2
                row_value = assigned_rows[split] + (len(members) if split == candidate_split else 0)
                score += 0.1 * ((row_value - target_rows[split]) / max(target_rows[split], 1.0)) ** 2
            return score, stable_fraction(f"{seed}|tie|{key}|{candidate_split}")

        selected = min(SPLITS, key=cost)
        assignments[key] = selected
        assigned_counts[selected].update(contribution)
        assigned_rows[selected] += len(members)
    return assignments


def freeze_contexts(rows: list[dict[str, Any]], k: int, seed: int) -> None:
    pools: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["split"] == "train" and row.get("foreground_voxels", 1) > 0:
            pools[row["target_region"]].append(row)

    for row in rows:
        candidates = [
            candidate for candidate in pools[row["target_region"]]
            if group_key(candidate) != group_key(row)
        ]
        same_source = [c for c in candidates if c["source_dataset"] == row["source_dataset"]]
        other_source = [c for c in candidates if c["source_dataset"] != row["source_dataset"]]

        def ranked(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
            return sorted(
                items,
                key=lambda c: stable_fraction(f"{seed}|context|{row['case_id']}|{c['case_id']}"),
            )

        selected = (ranked(same_source) + ranked(other_source))[:k]
        if len(selected) < k:
            raise ValueError(f"{row['case_id']} has only {len(selected)} eligible contexts; need {k}")
        row["context_case_ids"] = [candidate["case_id"] for candidate in selected]


def validate_no_patient_leakage(rows: list[dict[str, Any]]) -> None:
    membership: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        membership[group_key(row)].add(row["split"])
    leaking = {key: value for key, value in membership.items() if len(value) != 1}
    if leaking:
        raise ValueError(f"patient leakage across splits: {list(leaking.items())[:5]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--context-k", type=int, default=3)
    parser.add_argument("--allow-missing-roi", action="store_true")
    args = parser.parse_args()

    rows = load_jsonl(args.input)
    assignments = assign_grouped_stratified_split(rows, args.seed)
    missing_roi = []
    for row in rows:
        row["legacy_split"] = row.get("split")
        row["split"] = assignments[group_key(row)]
        row["split_seed"] = args.seed
        roi = resolve_roi(row)
        if roi is None:
            missing_roi.append(row["case_id"])
        else:
            row["organ_mask"] = str(roi)
    if missing_roi and not args.allow_missing_roi:
        raise FileNotFoundError(
            f"missing TotalSegmentator organ ROI for {len(missing_roi)} rows; examples={missing_roi[:10]}"
        )

    freeze_contexts(rows, args.context_k, args.seed)
    validate_no_patient_leakage(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    split_task_source = Counter(
        (row["split"], row["target_region"], row["source_dataset"]) for row in rows
    )
    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "seed": args.seed,
        "context_k": args.context_k,
        "rows": len(rows),
        "patients": len({group_key(row) for row in rows}),
        "missing_roi": len(missing_roi),
        "split_rows": dict(Counter(row["split"] for row in rows)),
        "split_patients": dict(Counter(assignments.values())),
        "strata": {"|".join(key): value for key, value in sorted(split_task_source.items())},
    }
    summary_path = args.summary or args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

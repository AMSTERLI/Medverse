"""Build the frozen LiTS/KiTS23/MSWAL liver-kidney experiment manifest.

Only annotations that are complete for the requested task are admitted. The
split is grouped by dataset and patient and stratified by dataset/task. Fixed
same-task contexts are selected from non-empty training cases only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SPLITS = ("train", "val", "test")
RATIOS = {"train": 0.70, "val": 0.10, "test": 0.20}
SOURCE_ALIASES = {"LiTS": "LiTS", "KiTS": "KiTS23", "KiTS23": "KiTS23", "MSWAL": "MSWAL"}
TASK_CONTRACT = {
    ("LiTS", "liver_tumor"): {"primary_organ": "liver", "tumor_labels": [2], "allowed_labels": {0, 1, 2}},
    ("KiTS23", "kidney_tumor"): {"primary_organ": "kidney", "tumor_labels": [2], "allowed_labels": {0, 1, 2, 3}},
    ("MSWAL", "liver_tumor"): {"primary_organ": "liver", "tumor_labels": [3], "allowed_labels": None},
    ("MSWAL", "kidney_tumor"): {"primary_organ": "kidney", "tumor_labels": [4], "allowed_labels": None},
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, raw in enumerate(stream, 1):
            if not raw.strip():
                continue
            row = json.loads(raw)
            required = {"patient_id", "source_dataset", "target_region", "image", "mask"}
            missing = required - row.keys()
            if missing:
                raise ValueError(f"line {line_number}: missing {sorted(missing)}")
            rows.append(row)
    if not rows:
        raise ValueError("input manifest is empty")
    return rows


def stable_fraction(text: str) -> float:
    value = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")
    return value / float(2**64 - 1)


def safe_token(text: str) -> str:
    readable = "".join(c if c.isalnum() or c in "_-" else "_" for c in text)
    return f"{readable[:100]}_{hashlib.sha256(text.encode()).hexdigest()[:10]}"


def organ_case_key(row: dict[str, Any]) -> str:
    return safe_token(f"{row['source_dataset']}::{row['source_case_id']}::{Path(row['image']).name}")


def normalize_scope(input_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for original in input_rows:
        source = SOURCE_ALIASES.get(str(original["source_dataset"]))
        task = str(original["target_region"])
        if source is None or (source, task) not in TASK_CONTRACT:
            continue
        contract = TASK_CONTRACT[(source, task)]
        row = dict(original)
        row["legacy_source_dataset"] = row["source_dataset"]
        row["legacy_case_id"] = row.get("case_id")
        row["source_dataset"] = source
        row["source_case_id"] = str(row.get("source_case_id", row["patient_id"]))
        row["study_id"] = str(row.get("study_id", row["patient_id"]))
        row["case_id"] = f"{source}:{row['patient_id']}:{task}"
        row["primary_organ"] = contract["primary_organ"]
        row["tumor_label_values"] = list(contract["tumor_labels"])
        row["modality"] = "CT"
        row.setdefault("phase_or_sequence", "unknown_ct_phase")
        row.setdefault("annotation_protocol", f"{source}_official_segmentation")
        if row["case_id"] in seen:
            raise ValueError(f"duplicate scoped case_id: {row['case_id']}")
        seen.add(row["case_id"])
        rows.append(row)
    if not rows:
        raise ValueError("no LiTS/KiTS23/MSWAL liver_tumor or kidney_tumor rows were found")
    return rows


def inspect_label(row: dict[str, Any]) -> None:
    import nibabel as nib
    import numpy as np

    values, counts = np.unique(np.rint(np.asanyarray(nib.load(row["mask"]).dataobj)), return_counts=True)
    if not np.isfinite(values).all() or not np.allclose(values, np.rint(values)) or np.any(values < 0):
        raise ValueError(f"{row['case_id']}: labels must be finite non-negative integers")
    integer_values = {int(v) for v in values.tolist()}
    allowed = TASK_CONTRACT[(row["source_dataset"], row["target_region"])]["allowed_labels"]
    if allowed is not None and not integer_values <= allowed:
        raise ValueError(f"{row['case_id']}: unexpected label values {sorted(integer_values - allowed)}")
    count_by_value = {int(v): int(n) for v, n in zip(values.tolist(), counts.tolist())}
    row["source_label_values"] = sorted(integer_values)
    row["foreground_voxels"] = sum(count_by_value.get(v, 0) for v in row["tumor_label_values"])
    row["label_mapping_audited"] = True


def group_key(row: dict[str, Any]) -> str:
    return f"{row['source_dataset']}::{row['patient_id']}"


def stratum(row: dict[str, Any]) -> str:
    return f"{row['source_dataset']}::{row['target_region']}"


def assign_grouped_stratified_split(rows: list[dict[str, Any]], seed: int) -> dict[str, str]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[group_key(row)].append(row)
    total = Counter(stratum(row) for row in rows)
    desired = {split: {key: RATIOS[split] * n for key, n in total.items()} for split in SPLITS}
    counts = {split: Counter() for split in SPLITS}
    assignments: dict[str, str] = {}

    def group_order(item: tuple[str, list[dict[str, Any]]]) -> tuple[float, float]:
        key, members = item
        rarity = sum(1.0 / total[stratum(row)] for row in members)
        return -rarity, stable_fraction(f"{seed}|order|{key}")

    for key, members in sorted(groups.items(), key=group_order):
        contribution = Counter(stratum(row) for row in members)

        def cost(split: str) -> tuple[float, float]:
            error = 0.0
            for candidate in SPLITS:
                for name, target in desired[candidate].items():
                    value = counts[candidate][name] + (contribution[name] if candidate == split else 0)
                    error += ((value - target) / max(target, 1.0)) ** 2
            return error, stable_fraction(f"{seed}|tie|{key}|{split}")

        selected = min(SPLITS, key=cost)
        assignments[key] = selected
        counts[selected].update(contribution)
    return assignments


def resolve_organ_mask(row: dict[str, Any], totalseg_root: Path | None, allow_legacy: bool) -> Path | None:
    organ_name = "liver" if row["primary_organ"] == "liver" else "kidney_union"
    if totalseg_root is not None:
        candidate = totalseg_root / "organ_masks" / organ_case_key(row) / f"{organ_name}.nii.gz"
        if candidate.is_file():
            return candidate
    if not allow_legacy:
        return None
    candidate = Path(str(row.get("organ_mask", "")))
    if not candidate.is_file():
        return None
    # The legacy MSWAL pipeline reused one generic mask for different organs.
    if row["source_dataset"] == "MSWAL" and candidate.parent.name == "organ":
        return None
    return candidate


def freeze_contexts(rows: list[dict[str, Any]], k: int, seed: int) -> None:
    pools: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["split"] == "train" and int(row.get("foreground_voxels", 0)) > 0:
            pools[row["target_region"]].append(row)
    for row in rows:
        candidates = [c for c in pools[row["target_region"]] if group_key(c) != group_key(row)]
        candidates.sort(key=lambda c: stable_fraction(f"{seed}|context|{row['case_id']}|{c['case_id']}"))
        if len(candidates) < k:
            raise ValueError(f"{row['case_id']} has only {len(candidates)} eligible train contexts; need {k}")
        row["context_strategy"] = "random_same_task"
        row["context_case_ids"] = [candidate["case_id"] for candidate in candidates[:k]]


def validate(rows: list[dict[str, Any]], require_audit: bool) -> None:
    membership: dict[str, set[str]] = defaultdict(set)
    lookup = {row["case_id"]: row for row in rows}
    for row in rows:
        membership[group_key(row)].add(row["split"])
        if require_audit and not row.get("label_mapping_audited"):
            raise ValueError(f"{row['case_id']}: missing label audit")
        for context_id in row["context_case_ids"]:
            context = lookup[context_id]
            if context["split"] != "train" or context["target_region"] != row["target_region"]:
                raise ValueError(f"{row['case_id']}: invalid context {context_id}")
            if group_key(context) == group_key(row) or int(context.get("foreground_voxels", 0)) <= 0:
                raise ValueError(f"{row['case_id']}: leaking or empty context {context_id}")
    leaking = {key: value for key, value in membership.items() if len(value) != 1}
    if leaking:
        raise ValueError(f"patient leakage across splits: {list(leaking.items())[:5]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--totalseg-root", type=Path)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--context-k", type=int, default=3)
    parser.add_argument("--inspect-labels", action="store_true")
    parser.add_argument("--allow-legacy-organ-masks", action="store_true")
    parser.add_argument("--allow-missing-roi", action="store_true")
    args = parser.parse_args()

    rows = normalize_scope(load_jsonl(args.input))
    if args.inspect_labels:
        for row in rows:
            inspect_label(row)
    elif any("foreground_voxels" not in row for row in rows):
        raise ValueError("foreground_voxels missing; rerun with --inspect-labels")

    assignments = assign_grouped_stratified_split(rows, args.seed)
    missing_roi: list[str] = []
    for row in rows:
        row["legacy_split"] = row.get("split")
        row["split"] = assignments[group_key(row)]
        row["split_seed"] = args.seed
        organ = resolve_organ_mask(row, args.totalseg_root, args.allow_legacy_organ_masks)
        if organ is None:
            missing_roi.append(row["case_id"])
        else:
            row["organ_mask"] = str(organ)
    if missing_roi and not args.allow_missing_roi:
        raise FileNotFoundError(
            f"missing task-specific TotalSegmentator masks for {len(missing_roi)} rows; examples={missing_roi[:10]}"
        )

    freeze_contexts(rows, args.context_k, args.seed)
    validate(rows, require_audit=args.inspect_labels)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    strata = Counter((row["split"], row["source_dataset"], row["target_region"]) for row in rows)
    summary = {
        "input": str(args.input), "output": str(args.output), "seed": args.seed,
        "context_k": args.context_k, "context_strategy": "random_same_task",
        "rows": len(rows), "patients": len({group_key(row) for row in rows}),
        "missing_roi": len(missing_roi), "label_mapping_audited": args.inspect_labels,
        "split_rows": dict(Counter(row["split"] for row in rows)),
        "split_patients": dict(Counter(assignments.values())),
        "strata": {"|".join(key): value for key, value in sorted(strata.items())},
    }
    summary_path = args.summary or args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

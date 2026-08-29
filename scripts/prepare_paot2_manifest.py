"""Convert PAOT2 image-mask pair lists into task-aware ICL JSONL.

The input lists may use either tabs or spaces.  One physical scan may expand
into multiple binary task episodes when its filename contains multiple organ
tokens (for example ``03_07_MSWAL``).
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Iterable


def load_rules(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    return payload["rules"]


def filename_tokens(image_path: str) -> set[str]:
    name = PurePosixPath(image_path).name
    while name.endswith((".gz", ".nii")):
        name = Path(name).stem
    return set(name.split("_"))


def matching_tasks(image_path: str, rules: Iterable[dict]) -> list[dict]:
    tokens = filename_tokens(image_path)
    matches = []
    for rule in rules:
        if not image_path.startswith(rule["path_prefix"]):
            continue
        token = rule.get("filename_token")
        if token is not None and token not in tokens:
            continue
        matches.append(rule)
    return matches


def read_pairs(path: Path) -> Iterable[tuple[int, str, str]]:
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_number, raw in enumerate(stream, 1):
            stripped = raw.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = re.split(r"\s+", stripped)
            if len(fields) != 2:
                raise ValueError(
                    f"{path}:{line_number}: expected image and mask, got {len(fields)} fields"
                )
            yield line_number, fields[0].replace("\\", "/"), fields[1].replace("\\", "/")


def foreground_voxels(mask_path: Path, values: list[int]) -> int:
    try:
        import nibabel as nib
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("--inspect-labels requires nibabel and numpy") from exc

    image = nib.load(str(mask_path))
    data = np.asanyarray(image.dataobj)
    # Some files use integer storage plus scl_slope; rounding after nibabel has
    # applied scaling recovers their semantic class indices.
    data = np.rint(data)
    return int(np.isin(data, values).sum())


def resolve_data_path(value: str, data_root: Path | None) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() or data_root is None else data_root / candidate


def case_stem(image_path: str) -> str:
    name = PurePosixPath(image_path).name
    if name.endswith(".nii.gz"):
        name = name[:-7]
    elif name.endswith(".nii"):
        name = name[:-4]
    return name.removesuffix("_0000")


def convert_list(
    pair_path: Path,
    split: str,
    rules: list[dict],
    data_root: Path | None,
    inspect_labels: bool,
    drop_empty: bool,
) -> tuple[list[dict], Counter]:
    rows: list[dict] = []
    stats: Counter = Counter()
    for line_number, image_path, mask_path in read_pairs(pair_path):
        image_fs = resolve_data_path(image_path, data_root)
        mask_fs = resolve_data_path(mask_path, data_root)
        if data_root is not None and (not image_fs.is_file() or not mask_fs.is_file()):
            missing = image_fs if not image_fs.is_file() else mask_fs
            raise FileNotFoundError(f"{pair_path}:{line_number}: missing {missing}")

        tasks = matching_tasks(image_path, rules)
        if not tasks:
            raise ValueError(f"{pair_path}:{line_number}: no task rule for {image_path}")

        patient_id = case_stem(image_path)
        for task in tasks:
            voxels = None
            if inspect_labels:
                voxels = foreground_voxels(mask_fs, task["tumor_label_values"])
                if voxels == 0 and drop_empty:
                    stats[f"dropped_empty/{task['target_region']}"] += 1
                    continue
            task_name = task["target_region"]
            row = {
                "case_id": f"{split}:{patient_id}:{task_name}",
                "patient_id": patient_id,
                "study_id": patient_id,
                "image": str(image_fs if data_root is not None else image_path),
                "mask": str(mask_fs if data_root is not None else mask_path),
                "split": split,
                "source_dataset": task["source_dataset"],
                "cancer_type": task["cancer_type"],
                "primary_organ": task["primary_organ"],
                "target_region": task_name,
                "tumor_label_values": task["tumor_label_values"],
                "modality": "CT",
                "phase_or_sequence": "unknown_ct_phase",
                "center": "unknown",
                "annotation_protocol": "PAOT2_existing_label",
            }
            if voxels is not None:
                row["foreground_voxels"] = voxels
            rows.append(row)
            stats[f"kept/{task_name}"] += 1
    return rows, stats


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-pairs", type=Path, required=True)
    parser.add_argument("--val-pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument(
        "--task-map",
        type=Path,
        default=project_root / "configs" / "paot2_task_map.json",
    )
    parser.add_argument(
        "--inspect-labels",
        action="store_true",
        help="Count task foreground voxels using nibabel.",
    )
    parser.add_argument(
        "--keep-empty",
        action="store_true",
        help="Keep zero-foreground task episodes (only meaningful with --inspect-labels).",
    )
    args = parser.parse_args()

    rules = load_rules(args.task_map)
    all_rows: list[dict] = []
    all_stats: Counter = Counter()
    for pair_path, split in ((args.train_pairs, "train"), (args.val_pairs, "val")):
        rows, stats = convert_list(
            pair_path=pair_path,
            split=split,
            rules=rules,
            data_root=args.data_root,
            inspect_labels=args.inspect_labels,
            drop_empty=args.inspect_labels and not args.keep_empty,
        )
        all_rows.extend(rows)
        all_stats.update(stats)

    case_ids = [row["case_id"] for row in all_rows]
    if len(case_ids) != len(set(case_ids)):
        duplicates = [key for key, count in Counter(case_ids).items() if count > 1]
        raise ValueError(f"duplicate task case IDs: {duplicates[:10]}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as stream:
        for row in all_rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps({"output": str(args.output), "rows": len(all_rows), **all_stats}, indent=2))


if __name__ == "__main__":
    main()

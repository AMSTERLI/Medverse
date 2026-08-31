"""Run frozen TotalSegmentator organ localization once per unique CT scan."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from prepare_main_experiment_manifest import load_jsonl, normalize_scope, organ_case_key


ROI_NAMES = ("liver", "kidney_left", "kidney_right")


def save_atomic(nii: nib.Nifti1Image, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name.replace(".nii.gz", ".tmp.nii.gz"))
    nib.save(nii, temporary)
    os.replace(temporary, destination)


def valid_mask(path: Path, reference: nib.Nifti1Image) -> tuple[bool, int]:
    if not path.is_file():
        return False, 0
    nii = nib.load(path)
    foreground = int(np.count_nonzero(np.asanyarray(nii.dataobj) > 0.5))
    valid = tuple(nii.shape) == tuple(reference.shape) and np.allclose(nii.affine, reference.affine, atol=1e-3)
    return bool(valid), foreground


def run_one(rows: list[dict[str, Any]], output_root: Path, cli: str, device: str, force: bool) -> dict[str, Any]:
    representative = rows[0]
    image_path = Path(representative["image"])
    case_key = organ_case_key(representative)
    case_dir = output_root / "organ_masks" / case_key
    metadata_path = output_root / "metadata" / f"{case_key}.totalseg.json"
    required = {"liver" if row["primary_organ"] == "liver" else "kidney_union" for row in rows}
    reference = nib.load(image_path)

    expected = [case_dir / f"{name}.nii.gz" for name in ROI_NAMES + ("kidney_union",)]
    if not force and metadata_path.is_file() and all(path.is_file() for path in expected):
        validation = {path.stem.split(".")[0]: valid_mask(path, reference) for path in expected}
        if all(ok for ok, _ in validation.values()) and all(validation[name][1] > 0 for name in required):
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            payload["status"] = "skipped_valid"
            return payload

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    case_dir.mkdir(parents=True, exist_ok=True)
    command = [cli, "-i", str(image_path), "-o", "OUTPUT", "--roi_subset", *ROI_NAMES]
    if device:
        command.extend(["--device", device])
    payload: dict[str, Any] = {
        "case_key": case_key,
        "image": str(image_path),
        "source_dataset": representative["source_dataset"],
        "source_case_id": representative["source_case_id"],
        "requested_tasks": sorted(row["target_region"] for row in rows),
        "required_masks": sorted(required),
        "command": command,
        "model_task": "total",
        "roi_subset": list(ROI_NAMES),
    }
    try:
        with tempfile.TemporaryDirectory(dir=output_root / "tmp") as temporary:
            temporary_path = Path(temporary)
            actual_command = [str(temporary_path) if value == "OUTPUT" else value for value in command]
            completed = subprocess.run(actual_command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            payload["returncode"] = completed.returncode
            payload["log_tail"] = completed.stdout.splitlines()[-50:]
            if completed.returncode != 0:
                raise RuntimeError(f"TotalSegmentator exited with {completed.returncode}")

            masks: dict[str, np.ndarray] = {}
            for name in ROI_NAMES:
                source = temporary_path / f"{name}.nii.gz"
                if not source.is_file():
                    raise FileNotFoundError(f"TotalSegmentator did not create {source.name}")
                source_nii = nib.load(source)
                if tuple(source_nii.shape) != tuple(reference.shape) or not np.allclose(source_nii.affine, reference.affine, atol=1e-3):
                    raise ValueError(f"{name}: output grid does not match the original CT")
                masks[name] = np.asanyarray(source_nii.dataobj) > 0.5
                output = nib.Nifti1Image(masks[name].astype(np.uint8), reference.affine, reference.header)
                output.set_data_dtype(np.uint8)
                save_atomic(output, case_dir / f"{name}.nii.gz")

            kidney = masks["kidney_left"] | masks["kidney_right"]
            kidney_nii = nib.Nifti1Image(kidney.astype(np.uint8), reference.affine, reference.header)
            kidney_nii.set_data_dtype(np.uint8)
            save_atomic(kidney_nii, case_dir / "kidney_union.nii.gz")
            voxels = {name: int(mask.sum()) for name, mask in masks.items()}
            voxels["kidney_union"] = int(kidney.sum())
            spacing = np.asarray(reference.header.get_zooms()[:3], dtype=float)
            voxel_volume = float(np.prod(spacing))
            payload.update({
                "status": "pass" if all(voxels[name] > 0 for name in required) else "failed_empty_required_mask",
                "shape": list(reference.shape), "spacing_xyz_mm": spacing.tolist(),
                "foreground_voxels": voxels,
                "organ_volume_mm3": {name: count * voxel_volume for name, count in voxels.items()},
            })
    except Exception as exc:
        payload.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
    metadata_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cli", default="TotalSegmentator")
    parser.add_argument("--device", default="gpu")
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--max-cases-per-stratum", type=int)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        parser.error("require shard-count >= 1 and 0 <= shard-index < shard-count")
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "tmp").mkdir(parents=True, exist_ok=True)

    rows = normalize_scope(load_jsonl(args.manifest))
    scans: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        scans[organ_case_key(row)].append(row)
    items = sorted(scans.items())
    if args.max_cases_per_stratum is not None:
        selected_keys: set[str] = set()
        counts: defaultdict[tuple[str, str], int] = defaultdict(int)
        for key, scan_rows in items:
            for row in scan_rows:
                stratum = (row["source_dataset"], row["target_region"])
                if counts[stratum] < args.max_cases_per_stratum:
                    selected_keys.add(key)
                    counts[stratum] += 1
        items = [item for item in items if item[0] in selected_keys]
    if args.max_cases is not None:
        items = items[: args.max_cases]
    items = [item for index, item in enumerate(items) if index % args.shard_count == args.shard_index]

    reports = []
    for index, (_, scan_rows) in enumerate(items, 1):
        report = run_one(scan_rows, args.output_root, args.cli, args.device, args.force)
        reports.append(report)
        print(json.dumps({"case": index, "total": len(items), "case_key": report["case_key"], "status": report["status"]}), flush=True)
    summary = {
        "scans": len(reports),
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "status": dict(__import__("collections").Counter(report["status"] for report in reports)),
        "failures": [report["case_key"] for report in reports if not report["status"].startswith(("pass", "skipped"))],
    }
    summary_name = (
        "totalseg_summary.json"
        if args.shard_count == 1
        else f"totalseg_summary.shard-{args.shard_index:02d}-of-{args.shard_count:02d}.json"
    )
    (args.output_root / summary_name).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if summary["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

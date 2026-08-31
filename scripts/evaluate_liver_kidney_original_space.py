"""Evaluate binary tumor predictions on the original CT grid and build reports."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from scipy import ndimage, stats


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def prediction_lookup(path: Path) -> dict[str, Path]:
    rows = load_jsonl(path)
    result = {}
    for row in rows:
        value = row.get("prediction") or row.get("prediction_original_space")
        if value is None:
            raise ValueError(f"prediction manifest row {row.get('case_id')} has no prediction path")
        result[row["case_id"]] = Path(value)
    return result


def surfaces(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return mask.copy()
    return mask ^ ndimage.binary_erosion(mask, structure=ndimage.generate_binary_structure(3, 1), border_value=0)


def voxel_metrics(prediction: np.ndarray, truth: np.ndarray, spacing: tuple[float, ...], tolerance_mm: float) -> dict[str, float]:
    denominator = int(prediction.sum() + truth.sum())
    dice = 1.0 if denominator == 0 else 2.0 * float(np.logical_and(prediction, truth).sum()) / denominator
    pred_surface, truth_surface = surfaces(prediction), surfaces(truth)
    if not pred_surface.any() and not truth_surface.any():
        return {"dice": dice, "nsd": 1.0, "hd95_mm": 0.0}
    if not pred_surface.any() or not truth_surface.any():
        return {"dice": dice, "nsd": 0.0, "hd95_mm": float("inf")}
    to_truth = ndimage.distance_transform_edt(~truth_surface, sampling=spacing)[pred_surface]
    to_pred = ndimage.distance_transform_edt(~pred_surface, sampling=spacing)[truth_surface]
    distances = np.concatenate((to_truth, to_pred))
    nsd = (np.count_nonzero(to_truth <= tolerance_mm) + np.count_nonzero(to_pred <= tolerance_mm)) / len(distances)
    return {"dice": dice, "nsd": float(nsd), "hd95_mm": float(np.percentile(distances, 95))}


def lesion_metrics(prediction: np.ndarray, truth: np.ndarray) -> dict[str, float | int]:
    structure = ndimage.generate_binary_structure(3, 3)
    pred_labels, pred_count = ndimage.label(prediction, structure)
    truth_labels, truth_count = ndimage.label(truth, structure)
    overlaps: list[tuple[int, int, int]] = []
    for pred_id in range(1, pred_count + 1):
        ids, counts = np.unique(truth_labels[pred_labels == pred_id], return_counts=True)
        overlaps.extend((int(count), pred_id, int(truth_id)) for truth_id, count in zip(ids, counts) if truth_id > 0)
    used_pred, used_truth = set(), set()
    for _, pred_id, truth_id in sorted(overlaps, reverse=True):
        if pred_id not in used_pred and truth_id not in used_truth:
            used_pred.add(pred_id); used_truth.add(truth_id)
    tp, fp, fn = len(used_truth), pred_count - len(used_pred), truth_count - len(used_truth)
    precision = tp / (tp + fp) if tp + fp else (1.0 if truth_count == 0 else 0.0)
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "lesion_tp": tp, "lesion_fp": fp, "lesion_fn": fn,
        "lesion_precision": precision, "lesion_recall": recall,
        "lesion_f1": f1, "false_positive_lesions_per_case": fp,
    }


def maximum_component_diameter(mask: np.ndarray, affine: np.ndarray) -> float:
    labels, count = ndimage.label(mask, ndimage.generate_binary_structure(3, 3))
    maximum = 0.0
    for component in range(1, count + 1):
        points = np.argwhere(labels == component)
        if len(points) < 2:
            continue
        physical = nib.affines.apply_affine(affine, points)
        extent = physical.max(axis=0) - physical.min(axis=0)
        maximum = max(maximum, float(np.linalg.norm(extent)))
    return maximum


def evaluate_case(row: dict[str, Any], prediction_path: Path, tolerance_mm: float) -> dict[str, Any]:
    truth_path = Path(row.get("tumor_mask", row.get("source_mask", row["mask"])))
    truth_nii, pred_nii = nib.load(truth_path), nib.load(prediction_path)
    if truth_nii.shape != pred_nii.shape or not np.allclose(truth_nii.affine, pred_nii.affine, atol=1e-3):
        raise ValueError(f"{row['case_id']}: prediction is not on original truth grid")
    truth = np.asarray(truth_nii.dataobj) > 0.5
    prediction = np.asarray(pred_nii.dataobj) > 0.5
    metrics = voxel_metrics(prediction, truth, tuple(truth_nii.header.get_zooms()[:3]), tolerance_mm)
    metrics.update(lesion_metrics(prediction, truth))
    diameter = float(row.get("maximum_3d_diameter_mm", maximum_component_diameter(truth, truth_nii.affine)))
    return {
        "case_id": row["case_id"], "patient_id": row["patient_id"],
        "source_dataset": row["source_dataset"], "target_region": row["target_region"],
        "tumor_size_group": "small_lt20mm" if diameter < 20 else "large_ge20mm",
        "maximum_3d_diameter_mm": diameter, **metrics,
    }


def finite_mean(values: list[float]) -> float:
    finite = [value for value in values if np.isfinite(value)]
    return float(np.mean(finite)) if finite else float("inf")


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = ("dice", "nsd", "hd95_mm", "lesion_precision", "lesion_recall", "lesion_f1", "false_positive_lesions_per_case")
    groups: dict[str, list[dict[str, Any]]] = {"all_patient_macro": rows}
    for field in ("target_region", "source_dataset", "tumor_size_group"):
        for value in sorted({str(row[field]) for row in rows}):
            groups[f"{field}:{value}"] = [row for row in rows if str(row[field]) == value]
    result = {}
    for name, members in groups.items():
        result[name] = {"cases": len(members)}
        result[name].update({metric: finite_mean([float(row[metric]) for row in members]) for metric in metrics})
    organ_groups = [result[f"target_region:{task}"] for task in sorted({row["target_region"] for row in rows})]
    result["organ_macro"] = {
        metric: finite_mean([float(group[metric]) for group in organ_groups]) for metric in metrics
    }
    return result


def paired_comparison(primary: list[dict[str, Any]], comparison: list[dict[str, Any]], seed: int = 20260831) -> dict[str, Any]:
    left, right = {r["case_id"]: r for r in primary}, {r["case_id"]: r for r in comparison}
    ids = sorted(left.keys() & right.keys())
    if ids != sorted(left) or ids != sorted(right):
        raise ValueError("paired comparison requires identical test case IDs")
    delta = np.asarray([left[case]["dice"] - right[case]["dice"] for case in ids], dtype=float)
    rng = np.random.default_rng(seed)
    bootstrap = np.asarray([rng.choice(delta, len(delta), replace=True).mean() for _ in range(10000)])
    try:
        wilcoxon = stats.wilcoxon(delta)
        statistic, p_value = float(wilcoxon.statistic), float(wilcoxon.pvalue)
    except ValueError:
        statistic, p_value = 0.0, 1.0
    return {
        "metric": "dice", "cases": len(ids), "mean_paired_difference": float(delta.mean()),
        "median_paired_difference": float(np.median(delta)),
        "paired_bootstrap_95_ci": np.percentile(bootstrap, [2.5, 97.5]).tolist(),
        "wilcoxon_statistic": statistic, "wilcoxon_p_value": p_value,
        "per_case_difference": dict(zip(ids, delta.tolist())),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = ["| 分组 | N | Dice | NSD | HD95 (mm) | Lesion recall | FP/case |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name, values in summary.items():
        lines.append(
            f"| {name} | {values.get('cases', '')} | {values['dice']:.4f} | {values['nsd']:.4f} | "
            f"{values['hd95_mm']:.3f} | {values['lesion_recall']:.4f} | {values['false_positive_lesions_per_case']:.3f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--comparison-predictions", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--nsd-tolerance-mm", type=float, default=2.0)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()
    manifest = [row for row in load_jsonl(args.manifest) if row["split"] == args.split]
    predictions = prediction_lookup(args.predictions)
    rows = [evaluate_case(row, predictions[row["case_id"]], args.nsd_tolerance_mm) for row in manifest]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "per_case_metrics.csv", rows)
    report: dict[str, Any] = {"summary": summarize(rows), "nsd_tolerance_mm": args.nsd_tolerance_mm}
    if args.comparison_predictions:
        other = prediction_lookup(args.comparison_predictions)
        other_rows = [evaluate_case(row, other[row["case_id"]], args.nsd_tolerance_mm) for row in manifest]
        report["paired_comparison_primary_minus_comparison"] = paired_comparison(rows, other_rows)
    (args.output_dir / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(args.output_dir / "paper_table.md", report["summary"])
    print(json.dumps({"cases": len(rows), "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()

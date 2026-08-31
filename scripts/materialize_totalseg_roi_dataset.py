"""Materialize one shared TotalSegmentator-ROI dataset for both model arms."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Sequence

import nibabel as nib
import numpy as np
from nibabel.orientations import apply_orientation, axcodes2ornt, io_orientation, ornt_transform
from nibabel.processing import resample_from_to, resample_to_output


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def safe_case_name(row: dict[str, Any]) -> str:
    readable = "_".join(
        str(value).replace("-", "_").replace(":", "_").replace("/", "_")
        for value in (row["source_dataset"], row["patient_id"], row["target_region"])
    )
    readable = "".join(char if char.isalnum() or char == "_" else "_" for char in readable)
    digest = hashlib.sha256(row["case_id"].encode("utf-8")).hexdigest()[:10]
    return f"{readable[:120]}_{digest}"


def _load_same_grid(row: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    image_nii = nib.load(row["image"])
    mask_nii = nib.load(row["mask"])
    organ_nii = nib.load(row["organ_mask"])
    if image_nii.shape != mask_nii.shape or image_nii.shape != organ_nii.shape:
        raise ValueError(
            f"{row['case_id']}: shape mismatch image={image_nii.shape}, "
            f"mask={mask_nii.shape}, organ={organ_nii.shape}"
        )
    image = np.squeeze(np.asanyarray(image_nii.dataobj)).astype(np.float32, copy=False)
    mask = np.squeeze(np.asanyarray(mask_nii.dataobj))
    organ = np.squeeze(np.asanyarray(organ_nii.dataobj))
    if image.ndim != 3:
        raise ValueError(f"{row['case_id']}: expected 3D data, got {image.shape}")

    # Apply the image orientation transform to all three same-index arrays. A
    # few legacy derived masks carry stale affines despite sharing the CT grid.
    orientation = ornt_transform(io_orientation(image_nii.affine), axcodes2ornt(("R", "A", "S")))
    image = apply_orientation(image, orientation)
    mask = apply_orientation(mask, orientation)
    organ = apply_orientation(organ, orientation)
    canonical_affine = image_nii.affine @ nib.orientations.inv_ornt_aff(orientation, image_nii.shape)
    return image, mask, organ, canonical_affine


def _bbox(mask: np.ndarray, margin_mm: float, spacing: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    foreground = np.argwhere(mask)
    if len(foreground) == 0:
        raise ValueError("TotalSegmentator organ ROI is empty")
    start = foreground.min(axis=0)
    end = foreground.max(axis=0) + 1
    margin = np.ceil(margin_mm / np.asarray(spacing)).astype(int)
    start = np.maximum(0, start - margin)
    end = np.minimum(np.asarray(mask.shape), end + margin)
    return start, end


def _save_atomic(image: nib.Nifti1Image, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name.replace(".nii.gz", ".tmp.nii.gz"))
    nib.save(image, temporary)
    os.replace(temporary, destination)


def process_one(payload: tuple[dict[str, Any], str, tuple[float, float, float], float, tuple[float, float]]) -> dict[str, Any]:
    row, output_root_raw, target_spacing, margin_mm, hu_window = payload
    output_root = Path(output_root_raw)
    name = safe_case_name(row)
    image_path = output_root / "images" / f"{name}_0000.nii.gz"
    mask_path = output_root / "labels" / f"{name}.nii.gz"
    organ_path = output_root / "organs" / f"{name}.nii.gz"

    if not (image_path.is_file() and mask_path.is_file() and organ_path.is_file()):
        image, raw_mask, raw_organ, affine = _load_same_grid(row)
        organ = raw_organ > 0.5
        tumor = np.isin(np.rint(raw_mask), row["tumor_label_values"])
        spacing = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
        start, end = _bbox(organ, margin_mm, spacing)
        slices = tuple(slice(int(a), int(b)) for a, b in zip(start, end))
        crop_affine = affine.copy()
        crop_affine[:3, 3] = affine[:3, :3] @ start + affine[:3, 3]
        lower, upper = hu_window
        image_crop = np.clip(image[slices], lower, upper).astype(np.float32, copy=False)
        tumor_crop = tumor[slices].astype(np.uint8)
        organ_crop = organ[slices].astype(np.uint8)

        image_nii = nib.Nifti1Image(image_crop, crop_affine)
        image_resampled = resample_to_output(
            image_nii, voxel_sizes=target_spacing, order=1, mode="constant", cval=float(lower)
        )
        target_grid = (image_resampled.shape, image_resampled.affine)
        tumor_resampled = resample_from_to(
            nib.Nifti1Image(tumor_crop, crop_affine), target_grid, order=0, mode="constant", cval=0
        )
        organ_resampled = resample_from_to(
            nib.Nifti1Image(organ_crop, crop_affine), target_grid, order=0, mode="constant", cval=0
        )
        image_resampled.set_data_dtype(np.float32)
        tumor_resampled.set_data_dtype(np.uint8)
        organ_resampled.set_data_dtype(np.uint8)
        _save_atomic(image_resampled, image_path)
        _save_atomic(tumor_resampled, mask_path)
        _save_atomic(organ_resampled, organ_path)

        tumor_total = int(tumor.sum())
        tumor_inside = int(tumor[slices].sum())
        coverage = 1.0 if tumor_total == 0 else tumor_inside / tumor_total
        output_shape = list(image_resampled.shape)
    else:
        output_nii = nib.load(image_path)
        output_shape = list(output_nii.shape)
        tumor_total = int(row.get("foreground_voxels", -1))
        coverage = float(row.get("roi_tumor_coverage", 1.0))

    updated = dict(row)
    updated.update(
        {
            "source_image": row["image"],
            "source_mask": row["mask"],
            "source_organ_mask": row["organ_mask"],
            "image": str(image_path),
            "mask": str(mask_path),
            "organ_mask": str(organ_path),
            "roi_spacing": list(target_spacing),
            "roi_margin_mm": margin_mm,
            "roi_shape": output_shape,
            "roi_tumor_coverage": coverage,
            "foreground_voxels_source": tumor_total,
        }
    )
    return updated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--spacing", default="1.5,1.5,2.0")
    parser.add_argument("--margin-mm", type=float, default=30.0)
    parser.add_argument("--hu-window", default="-1000,1000")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--min-tumor-coverage", type=float, default=0.95)
    args = parser.parse_args()

    spacing = tuple(float(value) for value in args.spacing.split(","))
    hu_window = tuple(float(value) for value in args.hu_window.split(","))
    if len(spacing) != 3 or len(hu_window) != 2:
        raise ValueError("spacing requires 3 values and hu-window requires 2")
    rows = load_jsonl(args.manifest)
    if args.max_cases is not None:
        rows = rows[: args.max_cases]
    payloads = [(row, str(args.output_root), spacing, args.margin_mm, hu_window) for row in rows]
    if args.workers == 1:
        outputs = [process_one(payload) for payload in payloads]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            outputs = list(executor.map(process_one, payloads))

    low_coverage = [row for row in outputs if row["roi_tumor_coverage"] < args.min_tumor_coverage]
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("w", encoding="utf-8") as stream:
        for row in outputs:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "cases": len(outputs),
        "spacing": spacing,
        "margin_mm": args.margin_mm,
        "hu_window": hu_window,
        "low_tumor_coverage_cases": len(low_coverage),
        "low_tumor_coverage_examples": [row["case_id"] for row in low_coverage[:20]],
        "shape_min": np.min([row["roi_shape"] for row in outputs], axis=0).tolist(),
        "shape_median": np.median([row["roi_shape"] for row in outputs], axis=0).tolist(),
        "shape_max": np.max([row["roi_shape"] for row in outputs], axis=0).tolist(),
    }
    summary_path = args.output_manifest.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if low_coverage:
        raise RuntimeError(
            f"{len(low_coverage)} ROIs cover less than {args.min_tumor_coverage:.1%} of tumor labels"
        )


if __name__ == "__main__":
    main()

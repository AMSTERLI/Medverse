"""Create binary labels and original-spacing TotalSegmentator organ ROIs.

This stage deliberately does not resample. nnU-Net v2 fingerprints these
cropped NIfTIs and chooses the shared target spacing used by both model arms.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Sequence

import nibabel as nib
import numpy as np
from nibabel.orientations import apply_orientation, axcodes2ornt, io_orientation, ornt_transform


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_case_name(row: dict[str, Any]) -> str:
    readable = "_".join(str(row[key]) for key in ("source_dataset", "patient_id", "target_region"))
    readable = "".join(char if char.isalnum() or char == "_" else "_" for char in readable)
    digest = hashlib.sha256(row["case_id"].encode("utf-8")).hexdigest()[:10]
    return f"{readable[:120]}_{digest}"


def _canonical_arrays(row: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, nib.Nifti1Image, np.ndarray]:
    image_nii = nib.load(row["image"])
    raw_mask_nii = nib.load(row["mask"])
    organ_nii = nib.load(row["organ_mask"])
    for label, nii in (("raw tumor label", raw_mask_nii), ("TotalSegmentator organ", organ_nii)):
        if tuple(nii.shape) != tuple(image_nii.shape) or not np.allclose(nii.affine, image_nii.affine, atol=1e-3):
            raise ValueError(
                f"{row['case_id']}: {label} grid mismatch; image={image_nii.shape}, mask={nii.shape}, "
                f"affine_match={np.allclose(nii.affine, image_nii.affine, atol=1e-3)}"
            )
    orientation = ornt_transform(io_orientation(image_nii.affine), axcodes2ornt(("R", "A", "S")))
    image = apply_orientation(np.asanyarray(image_nii.dataobj), orientation).astype(np.float32, copy=False)
    raw_mask = apply_orientation(np.asanyarray(raw_mask_nii.dataobj), orientation)
    organ = apply_orientation(np.asanyarray(organ_nii.dataobj), orientation) > 0.5
    canonical_affine = image_nii.affine @ nib.orientations.inv_ornt_aff(orientation, image_nii.shape)
    if image.ndim != 3:
        raise ValueError(f"{row['case_id']}: expected 3D CT, got {image.shape}")
    return image, raw_mask, organ, image_nii, orientation


def _bbox(mask: np.ndarray, margin_mm: float, spacing: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    foreground = np.argwhere(mask)
    if not len(foreground):
        raise ValueError("TotalSegmentator required organ mask is empty")
    margin = np.ceil(margin_mm / np.asarray(spacing)).astype(int)
    start = np.maximum(0, foreground.min(axis=0) - margin)
    end = np.minimum(np.asarray(mask.shape), foreground.max(axis=0) + 1 + margin)
    return start, end


def _save_atomic(nii: nib.Nifti1Image, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name.replace(".nii.gz", ".tmp.nii.gz"))
    nib.save(nii, temporary)
    os.replace(temporary, destination)


def maximum_3d_diameter_mm(mask: np.ndarray, affine: np.ndarray) -> tuple[float, str]:
    points = np.argwhere(mask)
    if len(points) < 2:
        return 0.0, "convex_hull_exact"
    physical = nib.affines.apply_affine(affine, points)
    try:
        from scipy.spatial import ConvexHull, distance

        vertices = physical[ConvexHull(physical).vertices] if len(physical) >= 4 else physical
        return float(distance.pdist(vertices).max(initial=0.0)), "convex_hull_exact"
    except Exception:
        extent = physical.max(axis=0) - physical.min(axis=0)
        return float(np.linalg.norm(extent)), "physical_bbox_diagonal_fallback"


def process_one(payload: tuple[Any, ...]) -> dict[str, Any]:
    """Materialize one case while accepting the pre-v1 compatibility tuple.

    Older callers passed target spacing, margin and HU window. Resampling and
    intensity normalization are now intentionally deferred to the shared
    nnU-Net fingerprint/preprocessing stage, so only the legacy margin value is
    retained from that tuple.
    """
    if len(payload) == 3:
        row, output_root_raw, margin_mm = payload
    elif len(payload) == 5:
        row, output_root_raw, _legacy_spacing, margin_mm, _legacy_hu_window = payload
    else:
        raise ValueError(f"expected a 3- or 5-item process payload, got {len(payload)}")
    output_root = Path(output_root_raw)
    name = safe_case_name(row)
    full_tumor_path = output_root / "tumor_masks" / f"{name}.nii.gz"
    roi_image_path = output_root / "rois" / "images" / f"{name}_0000.nii.gz"
    roi_tumor_path = output_root / "rois" / "tumor_masks" / f"{name}.nii.gz"
    roi_organ_path = output_root / "rois" / "organ_masks" / f"{name}.nii.gz"
    metadata_path = output_root / "metadata" / f"{name}.json"

    image, raw_mask, organ, original_nii, orientation = _canonical_arrays(row)
    source_tumor = np.isin(np.rint(np.asanyarray(nib.load(row["mask"]).dataobj)), row["tumor_label_values"])
    tumor = apply_orientation(source_tumor, orientation)
    affine = original_nii.affine @ nib.orientations.inv_ornt_aff(orientation, original_nii.shape)
    spacing = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
    voxel_volume = float(abs(np.linalg.det(affine[:3, :3])))
    organ_volume = float(organ.sum() * voxel_volume)
    primary_organ = row.get("primary_organ", str(row["target_region"]).removesuffix("_tumor"))
    abnormal_threshold = 200_000.0 if primary_organ == "liver" else 30_000.0
    full_volume_fallback = organ_volume < abnormal_threshold
    if full_volume_fallback:
        # This decision only uses the predicted organ mask, so the exact same
        # fallback is available at inference time without looking at tumor GT.
        start = np.zeros(3, dtype=int)
        end = np.asarray(organ.shape, dtype=int)
    else:
        start, end = _bbox(organ, margin_mm, spacing)
    slices = tuple(slice(int(a), int(b)) for a, b in zip(start, end))
    crop_affine = affine.copy()
    crop_affine[:3, 3] = affine[:3, :3] @ start + affine[:3, 3]

    tumor_total = int(tumor.sum())
    tumor_inside = int(tumor[slices].sum())
    coverage = 1.0 if tumor_total == 0 else tumor_inside / tumor_total
    qc_flags: list[str] = []
    if full_volume_fallback:
        qc_flags.append("organ_volume_abnormally_small_full_volume_fallback")
    if coverage < 0.999:
        qc_flags.append("tumor_partly_outside_fixed_organ_roi")
    if tumor_total == 0:
        qc_flags.append("empty_tumor_annotation")

    full_tumor_nii = nib.Nifti1Image(source_tumor.astype(np.uint8), original_nii.affine, original_nii.header)
    full_tumor_nii.set_data_dtype(np.uint8)
    roi_image_nii = nib.Nifti1Image(image[slices].astype(np.float32), crop_affine)
    roi_tumor_nii = nib.Nifti1Image(tumor[slices].astype(np.uint8), crop_affine)
    roi_organ_nii = nib.Nifti1Image(organ[slices].astype(np.uint8), crop_affine)
    roi_image_nii.set_data_dtype(np.float32)
    roi_tumor_nii.set_data_dtype(np.uint8)
    roi_organ_nii.set_data_dtype(np.uint8)
    _save_atomic(full_tumor_nii, full_tumor_path)
    _save_atomic(roi_image_nii, roi_image_path)
    _save_atomic(roi_tumor_nii, roi_tumor_path)
    _save_atomic(roi_organ_nii, roi_organ_path)

    diameter, diameter_method = maximum_3d_diameter_mm(source_tumor, original_nii.affine)
    metadata = {
        "case_id": row["case_id"], "original_shape": list(original_nii.shape),
        "original_affine": original_nii.affine.tolist(), "canonical_shape": list(image.shape),
        "canonical_affine": affine.tolist(), "original_to_ras_orientation": orientation.tolist(),
        "crop_bbox_original_voxels": {"start": start.tolist(), "end_exclusive": end.tolist(), "space": "canonical_RAS"},
        "crop_affine": crop_affine.tolist(), "crop_margin_mm": margin_mm,
        "full_volume_fallback": full_volume_fallback,
        "inverse_transform": "insert ROI into canonical_RAS bbox, then apply inverse orientation",
        "spacing_xyz_mm": spacing.tolist(), "roi_shape": list(image[slices].shape),
        "roi_tumor_coverage": coverage, "qc_status": "pass" if not qc_flags else "warning",
        "qc_flags": qc_flags,
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    updated = dict(row)
    updated.update({
        "source_image": row["image"], "source_mask": row["mask"],
        "source_organ_mask": row["organ_mask"], "tumor_mask": str(full_tumor_path),
        "image": str(roi_image_path), "mask": str(roi_tumor_path),
        "roi_image": str(roi_image_path), "roi_tumor_mask": str(roi_tumor_path),
        "roi_organ_mask": str(roi_organ_path), "geometry_metadata": str(metadata_path),
        "tumor_label_values_source": list(row["tumor_label_values"]), "tumor_label_values": [1],
        "foreground_voxels": tumor_total, "tumor_volume_mm3": tumor_total * voxel_volume,
        "maximum_3d_diameter_mm": diameter, "maximum_3d_diameter_method": diameter_method,
        "spacing_xyz_mm": spacing.tolist(), "crop_bbox_original_voxels": metadata["crop_bbox_original_voxels"],
        "crop_margin_mm": margin_mm, "full_volume_fallback": full_volume_fallback,
        "roi_shape": metadata["roi_shape"],
        "roi_tumor_coverage": coverage, "image_sha256": file_sha256(row["image"]),
        "mask_sha256": file_sha256(row["mask"]), "organ_mask_sha256": file_sha256(row["organ_mask"]),
        "qc_status": metadata["qc_status"], "qc_flags": qc_flags,
    })
    return updated


def generate_overlay(row: dict[str, Any], destination: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    image = np.asanyarray(nib.load(row["roi_image"]).dataobj)
    organ = np.asanyarray(nib.load(row["roi_organ_mask"]).dataobj) > 0
    tumor = np.asanyarray(nib.load(row["roi_tumor_mask"]).dataobj) > 0
    center = np.rint(np.argwhere(tumor).mean(axis=0)).astype(int) if tumor.any() else np.asarray(image.shape) // 2
    lower, upper = np.percentile(image[np.isfinite(image)], (1, 99))
    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    for axis, index, dim, title in zip(axes, center, range(3), ("sagittal", "coronal", "axial")):
        slices = [slice(None)] * 3
        slices[dim] = int(index)
        ct = np.asarray(image[tuple(slices)]).T
        org = np.asarray(organ[tuple(slices)]).T
        tum = np.asarray(tumor[tuple(slices)]).T
        axis.imshow(ct, cmap="gray", vmin=lower, vmax=upper, origin="lower")
        if org.any(): axis.contour(org, levels=[0.5], colors="lime", linewidths=0.7)
        if tum.any(): axis.contour(tum, levels=[0.5], colors="red", linewidths=0.9)
        axis.set_title(title); axis.axis("off")
    figure.suptitle(f"{row['case_id']} | green=TotalSegmentator, red=tumor GT")
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout(); figure.savefig(destination, dpi=140); plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--margin-mm", type=float, default=30.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--qc-samples-per-stratum", type=int, default=3)
    parser.add_argument("--fail-on-geometric-qc", action="store_true")
    args = parser.parse_args()
    rows = load_jsonl(args.manifest)
    if args.max_cases is not None:
        rows = rows[: args.max_cases]
    payloads = [(row, str(args.output_root), args.margin_mm) for row in rows]
    if args.workers == 1:
        outputs = [process_one(payload) for payload in payloads]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            outputs = list(executor.map(process_one, payloads))

    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("w", encoding="utf-8") as stream:
        for row in outputs:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    chosen: dict[tuple[str, str], int] = defaultdict(int)
    for row in outputs:
        key = (row["source_dataset"], row["target_region"])
        if chosen[key] < args.qc_samples_per_stratum:
            destination = args.output_root / "qc" / f"{safe_case_name(row)}.png"
            generate_overlay(row, destination)
            chosen[key] += 1
    summary = {
        "cases": len(outputs), "resampling_performed": False,
        "spacing_policy": "deferred_to_nnunetv2_planner", "margin_mm": args.margin_mm,
        "qc_status": dict(Counter(row["qc_status"] for row in outputs)),
        "qc_flags": dict(Counter(flag for row in outputs for flag in row["qc_flags"])),
        "qc_overlays": sum(chosen.values()),
        "shape_min": np.min([row["roi_shape"] for row in outputs], axis=0).tolist(),
        "shape_median": np.median([row["roi_shape"] for row in outputs], axis=0).tolist(),
        "shape_max": np.max([row["roi_shape"] for row in outputs], axis=0).tolist(),
    }
    args.output_manifest.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    geometric_flags = {"organ_volume_abnormally_small", "tumor_partly_outside_fixed_organ_roi"}
    geometric_failures = [
        row["case_id"] for row in outputs if geometric_flags.intersection(row["qc_flags"])
    ]
    if args.fail_on_geometric_qc and geometric_failures:
        print(json.dumps({"geometric_qc_failures": geometric_failures[:50]}, indent=2))
        raise SystemExit(7)


if __name__ == "__main__":
    main()

"""Task-aware CT patch dataset for Medverse in-context learning."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, get_worker_info


def load_manifest(path: Path | str) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, raw in enumerate(stream, 1):
            if not raw.strip():
                continue
            row = json.loads(raw)
            required = {"case_id", "patient_id", "image", "mask", "split", "primary_organ", "target_region", "tumor_label_values"}
            missing = required - row.keys()
            if missing:
                raise ValueError(f"manifest line {line_number}: missing {sorted(missing)}")
            rows.append(row)
    return rows


def _resolve(path: str, data_root: Path | None) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() or data_root is None else data_root / candidate


def _array_from_nifti(nii: nib.spatialimages.SpatialImage) -> np.ndarray:
    data = np.squeeze(np.asanyarray(nii.dataobj))
    if data.ndim != 3:
        raise ValueError(f"expected a 3D NIfTI, got {data.shape}")
    return data


def _load_aligned_volumes(
    image_path: Path, mask_path: Path, organ_path: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, tuple[float, float, float]]:
    """Load image, target and optional organ prompt on one verified grid."""
    image_nii, mask_nii = nib.load(str(image_path)), nib.load(str(mask_path))
    if tuple(image_nii.shape) != tuple(mask_nii.shape) or not np.allclose(image_nii.affine, mask_nii.affine, atol=1e-3):
        raise ValueError(f"grid mismatch: {image_path} vs {mask_path}")
    organ_nii = nib.load(str(organ_path)) if organ_path is not None else None
    if organ_nii is not None and (
        tuple(image_nii.shape) != tuple(organ_nii.shape)
        or not np.allclose(image_nii.affine, organ_nii.affine, atol=1e-3)
    ):
        raise ValueError(f"grid mismatch: {image_path} vs {organ_path}")
    image_nii, mask_nii = nib.as_closest_canonical(image_nii), nib.as_closest_canonical(mask_nii)
    if organ_nii is not None:
        organ_nii = nib.as_closest_canonical(organ_nii)
    spacing = tuple(float(value) for value in image_nii.header.get_zooms()[:3])
    organ = _array_from_nifti(organ_nii) if organ_nii is not None else None
    return _array_from_nifti(image_nii), _array_from_nifti(mask_nii), organ, spacing


def _interpolate(array: np.ndarray, shape: Sequence[int], mode: str) -> torch.Tensor:
    tensor = torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32))[None, None]
    kwargs: dict[str, Any] = {"size": tuple(int(v) for v in shape), "mode": mode}
    if mode != "nearest":
        kwargs["align_corners"] = False
    return F.interpolate(tensor, **kwargs)[0]


def _crop_or_pad(tensor: torch.Tensor, center: Sequence[int], patch_size: int) -> torch.Tensor:
    """Extract a centered cubic patch, padding outside the volume with zero."""
    spatial = tensor.shape[-3:]
    starts = [int(c) - patch_size // 2 for c in center]
    src_start = [max(0, value) for value in starts]
    src_end = [min(size, start + patch_size) for size, start in zip(spatial, starts)]
    patch = tensor[:, src_start[0]:src_end[0], src_start[1]:src_end[1], src_start[2]:src_end[2]]
    before = [max(0, -value) for value in starts]
    after = [patch_size - before[i] - patch.shape[i + 1] for i in range(3)]
    return F.pad(patch, (before[2], after[2], before[1], after[1], before[0], after[0]))


def _sample_center(image: torch.Tensor, mask: torch.Tensor, rng: random.Random, positive: bool) -> tuple[int, int, int]:
    candidate_mask = mask[0] > 0.5 if positive else ((image[0] > 0.25) & (mask[0] <= 0.5))
    candidates = torch.nonzero(candidate_mask, as_tuple=False)
    if len(candidates) == 0:
        candidates = torch.nonzero(torch.ones_like(mask[0], dtype=torch.bool), as_tuple=False)
    return tuple(int(value) for value in candidates[rng.randrange(len(candidates))].tolist())


def load_ct_volume(
    row: dict[str, Any], data_root: Path | None, target_spacing: tuple[float, float, float],
    hu_window: tuple[float, float], include_organ_channel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    image_value = row.get("roi_image", row["image"])
    mask_value = row.get("roi_tumor_mask", row.get("tumor_mask", row["mask"]))
    organ_value = row.get("roi_organ_mask", row.get("organ_mask"))
    image_path, mask_path = _resolve(image_value, data_root), _resolve(mask_value, data_root)
    organ_path = _resolve(organ_value, data_root) if include_organ_channel and organ_value else None
    if include_organ_channel and organ_path is None:
        raise ValueError(f"{row['case_id']}: organ-mask input channel requested but no mask is present")
    try:
        image, mask, organ, spacing = _load_aligned_volumes(image_path, mask_path, organ_path)
    except ValueError as exc:
        raise ValueError(f"{row['case_id']}: {exc}") from exc
    if not np.isfinite(image).all():
        raise ValueError(f"non-finite CT values in {image_path}")
    lower, upper = hu_window
    image = (np.clip(image.astype(np.float32, copy=False), lower, upper) - lower) / (upper - lower)
    semantic_mask = np.isin(np.rint(mask), row["tumor_label_values"]).astype(np.float32)
    new_shape = tuple(max(1, int(round(size * source / target))) for size, source, target in zip(image.shape, spacing, target_spacing))
    inputs = _interpolate(image, new_shape, "trilinear")
    if include_organ_channel:
        organ_tensor = _interpolate((organ > 0.5).astype(np.float32), new_shape, "nearest")
        inputs = torch.cat((inputs, organ_tensor), dim=0)
    return inputs, _interpolate(semantic_mask, new_shape, "nearest")


class PAOT2ICLDataset(Dataset):
    """Return tumor-aware patches for training and full targets for validation."""

    def __init__(
        self, manifest: Path | str | Sequence[dict[str, Any]], split: str, context_split: str = "train",
        num_context: int = 2, image_size: int = 128, target_spacing: tuple[float, float, float] = (2.0, 2.0, 3.0),
        hu_window: tuple[float, float] = (-1000.0, 1000.0), positive_probability: float = 0.7,
        full_volume_target: bool = False, data_root: Path | str | None = None, seed: int = 17,
        require_inspected_context: bool = True, drop_empty_targets: bool | None = None,
        max_cases_per_task: int | None = None, include_organ_channel: bool = False,
    ) -> None:
        super().__init__()
        rows = load_manifest(manifest) if isinstance(manifest, (str, Path)) else list(manifest)
        if require_inspected_context and any("foreground_voxels" not in row for row in rows):
            raise ValueError("manifest has not been inspected; regenerate it with prepare_paot2_manifest.py --inspect-labels")
        selected = [row for row in rows if row["split"] == split]
        should_drop_empty = drop_empty_targets if drop_empty_targets is not None else split == "train"
        if should_drop_empty:
            selected = [row for row in selected if row.get("foreground_voxels", 1) > 0]
        if max_cases_per_task is not None:
            counts: dict[str, int] = defaultdict(int)
            limited = []
            for row in selected:
                task = row["target_region"]
                if counts[task] < max_cases_per_task:
                    limited.append(row)
                    counts[task] += 1
            selected = limited
        self.rows = selected
        if not self.rows:
            raise ValueError(f"manifest contains no usable {split!r} rows")
        self.split, self.num_context, self.image_size = split, num_context, image_size
        self.target_spacing, self.hu_window = target_spacing, hu_window
        self.include_organ_channel = include_organ_channel
        self.positive_probability, self.full_volume_target = positive_probability, full_volume_target
        self.data_root = Path(data_root) if data_root is not None else None
        self.seed = seed
        self.rows_by_case_id = {row["case_id"]: row for row in rows}
        if len(self.rows_by_case_id) != len(rows):
            raise ValueError("manifest case_id values must be unique")

        pools: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            if row["split"] == context_split and row.get("foreground_voxels", 1) > 0:
                pools[row["target_region"]].append(row)
        self.context_pools = dict(pools)
        for row in self.rows:
            fixed_ids = row.get("context_case_ids")
            if fixed_ids is not None:
                if len(fixed_ids) != num_context:
                    raise ValueError(
                        f"{row['case_id']} freezes {len(fixed_ids)} contexts; need {num_context}"
                    )
                missing = [case_id for case_id in fixed_ids if case_id not in self.rows_by_case_id]
                if missing:
                    raise ValueError(f"{row['case_id']} references missing contexts: {missing}")
                fixed_rows = [self.rows_by_case_id[case_id] for case_id in fixed_ids]
                invalid = [
                    candidate["case_id"] for candidate in fixed_rows
                    if candidate["split"] != context_split
                    or candidate["target_region"] != row["target_region"]
                    or (
                        candidate.get("source_dataset") == row.get("source_dataset")
                        and candidate["patient_id"] == row["patient_id"]
                    )
                    or candidate.get("foreground_voxels", 1) <= 0
                ]
                if invalid:
                    raise ValueError(f"{row['case_id']} has invalid fixed contexts: {invalid}")
            else:
                available = sum(
                    not (
                        candidate.get("source_dataset") == row.get("source_dataset")
                        and candidate["patient_id"] == row["patient_id"]
                    )
                    for candidate in self.context_pools.get(row["target_region"], [])
                )
                if available < num_context:
                    raise ValueError(f"{row['case_id']} has {available} contexts; need {num_context}")

    def __len__(self) -> int:
        return len(self.rows)

    def _rng(self, index: int) -> random.Random:
        worker = get_worker_info()
        worker_seed = torch.initial_seed() if self.split == "train" else self.seed
        if worker is not None:
            worker_seed = worker.seed
        return random.Random(worker_seed + index)

    def _positive_patch(self, row: dict[str, Any], rng: random.Random) -> tuple[torch.Tensor, torch.Tensor]:
        image, mask = load_ct_volume(
            row, self.data_root, self.target_spacing, self.hu_window, self.include_organ_channel
        )
        center = _sample_center(image, mask, rng, positive=True)
        return _crop_or_pad(image, center, self.image_size), _crop_or_pad(mask, center, self.image_size)

    def __getitem__(self, index: int) -> dict[str, Any]:
        target_row, rng = self.rows[index], self._rng(index)
        fixed_ids = target_row.get("context_case_ids")
        if fixed_ids is not None:
            context_rows = [self.rows_by_case_id[case_id] for case_id in fixed_ids]
        else:
            candidates = [
                row for row in self.context_pools[target_row["target_region"]]
                if not (
                    row.get("source_dataset") == target_row.get("source_dataset")
                    and row["patient_id"] == target_row["patient_id"]
                )
            ]
            context_rows = rng.sample(candidates, self.num_context)
        target_image, target_mask = load_ct_volume(
            target_row, self.data_root, self.target_spacing, self.hu_window, self.include_organ_channel
        )
        if not self.full_volume_target:
            positive = bool(target_mask.sum() > 0) and rng.random() < self.positive_probability
            center = _sample_center(target_image, target_mask, rng, positive=positive)
            target_image, target_mask = _crop_or_pad(target_image, center, self.image_size), _crop_or_pad(target_mask, center, self.image_size)
        context_pairs = [self._positive_patch(row, rng) for row in context_rows]
        return {
            "target_in": target_image, "target_out": target_mask,
            "context_in": torch.stack([pair[0] for pair in context_pairs]),
            "context_out": torch.stack([pair[1] for pair in context_pairs]),
            "case_id": target_row["case_id"], "target_region": target_row["target_region"],
            "context_ids": [row["case_id"] for row in context_rows],
        }

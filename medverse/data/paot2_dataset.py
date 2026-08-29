"""Minimal task-aware CT dataset for Medverse in-context learning."""

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
            required = {
                "case_id",
                "patient_id",
                "image",
                "mask",
                "split",
                "primary_organ",
                "target_region",
                "tumor_label_values",
            }
            missing = required - row.keys()
            if missing:
                raise ValueError(f"manifest line {line_number}: missing {sorted(missing)}")
            rows.append(row)
    return rows


def _resolve(path: str, data_root: Path | None) -> Path:
    candidate = Path(path)
    if candidate.is_absolute() or data_root is None:
        return candidate
    return data_root / candidate


def _array_from_nifti(nii: nib.spatialimages.SpatialImage) -> np.ndarray:
    data = np.asanyarray(nii.dataobj)
    data = np.squeeze(data)
    if data.ndim != 3:
        raise ValueError(f"expected a 3D NIfTI, got {data.shape}")
    return data


def _load_aligned_pair(image_path: Path, mask_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load an image-mask pair without separating a legacy voxel alignment.

    Some PAOT2 labels have an identity affine even though their voxel arrays
    are aligned with the corresponding CT.  Canonicalizing those files
    independently would flip only the CT.  When shapes agree but affines do
    not, preserve the original shared voxel index grid used by the prior
    project.  When affines agree, canonicalize both files normally.
    """

    image_nii = nib.load(str(image_path))
    mask_nii = nib.load(str(mask_path))
    if tuple(image_nii.shape) != tuple(mask_nii.shape):
        raise ValueError(
            f"shape mismatch: {image_nii.shape} for {image_path} vs "
            f"{mask_nii.shape} for {mask_path}"
        )

    if np.allclose(image_nii.affine, mask_nii.affine, atol=1e-3):
        image_nii = nib.as_closest_canonical(image_nii)
        mask_nii = nib.as_closest_canonical(mask_nii)
    return _array_from_nifti(image_nii), _array_from_nifti(mask_nii)


def _resize(array: np.ndarray, size: int, mode: str) -> torch.Tensor:
    tensor = torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32))[None, None]
    kwargs = {"size": (size, size, size), "mode": mode}
    if mode != "nearest":
        kwargs["align_corners"] = False
    return F.interpolate(tensor, **kwargs)[0]


def load_ct_task(
    row: dict[str, Any],
    data_root: Path | None,
    image_size: int,
    hu_window: tuple[float, float],
) -> tuple[torch.Tensor, torch.Tensor]:
    image_path = _resolve(row["image"], data_root)
    mask_path = _resolve(row["mask"], data_root)
    try:
        image, mask = _load_aligned_pair(image_path, mask_path)
    except ValueError as exc:
        raise ValueError(f"{row['case_id']}: {exc}") from exc
    if not np.isfinite(image).all():
        raise ValueError(f"non-finite CT values in {image_path}")

    lower, upper = hu_window
    image = np.clip(image.astype(np.float32, copy=False), lower, upper)
    image = (image - lower) / (upper - lower)
    semantic_mask = np.isin(np.rint(mask), row["tumor_label_values"]).astype(np.float32)
    return _resize(image, image_size, "trilinear"), _resize(semantic_mask, image_size, "nearest")


class PAOT2ICLDataset(Dataset):
    """Return target and same-task context tensors required by Medverse.

    This deliberately uses a whole-volume resize for the first idea test.  It
    avoids validation-time ground-truth crops, but is not the final
    full-resolution preprocessing protocol.
    """

    def __init__(
        self,
        manifest: Path | str | Sequence[dict[str, Any]],
        split: str,
        context_split: str = "train",
        num_context: int = 2,
        image_size: int = 128,
        hu_window: tuple[float, float] = (-1000.0, 1000.0),
        data_root: Path | str | None = None,
        seed: int = 17,
        require_inspected_context: bool = True,
    ) -> None:
        super().__init__()
        rows = load_manifest(manifest) if isinstance(manifest, (str, Path)) else list(manifest)
        self.rows = [row for row in rows if row["split"] == split]
        if not self.rows:
            raise ValueError(f"manifest contains no {split!r} rows")
        self.split = split
        self.num_context = num_context
        self.image_size = image_size
        self.hu_window = hu_window
        self.data_root = Path(data_root) if data_root is not None else None
        self.seed = seed

        pools: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            if row["split"] != context_split:
                continue
            if require_inspected_context and "foreground_voxels" not in row:
                raise ValueError(
                    "context rows have not been inspected; regenerate the manifest "
                    "with prepare_paot2_manifest.py --inspect-labels"
                )
            if row.get("foreground_voxels", 1) > 0:
                pools[row["target_region"]].append(row)
        self.context_pools = dict(pools)

        for row in self.rows:
            available = sum(
                candidate["patient_id"] != row["patient_id"]
                for candidate in self.context_pools.get(row["target_region"], [])
            )
            if available < num_context:
                raise ValueError(
                    f"{row['case_id']} has only {available} distinct-patient contexts; "
                    f"need {num_context}"
                )

    def __len__(self) -> int:
        return len(self.rows)

    def _rng(self, index: int) -> random.Random:
        worker = get_worker_info()
        if self.split == "train":
            worker_seed = torch.initial_seed() if worker is not None else random.randrange(2**31)
            return random.Random(worker_seed + index)
        return random.Random(self.seed + index)

    def __getitem__(self, index: int) -> dict[str, Any]:
        target_row = self.rows[index]
        candidates = [
            row
            for row in self.context_pools[target_row["target_region"]]
            if row["patient_id"] != target_row["patient_id"]
        ]
        context_rows = self._rng(index).sample(candidates, self.num_context)

        target_image, target_mask = load_ct_task(
            target_row, self.data_root, self.image_size, self.hu_window
        )
        context_pairs = [
            load_ct_task(row, self.data_root, self.image_size, self.hu_window)
            for row in context_rows
        ]
        return {
            "target_in": target_image,
            "target_out": target_mask,
            "context_in": torch.stack([pair[0] for pair in context_pairs]),
            "context_out": torch.stack([pair[1] for pair in context_pairs]),
            "case_id": target_row["case_id"],
            "target_region": target_row["target_region"],
            "context_ids": [row["case_id"] for row in context_rows],
        }

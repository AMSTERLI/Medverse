"""Restore a binary ROI prediction to the original CT voxel grid."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.orientations import apply_orientation, axcodes2ornt, io_orientation, ornt_transform


def restore(prediction: Path, metadata: Path, output: Path) -> None:
    geometry = json.loads(metadata.read_text(encoding="utf-8"))
    prediction_nii = nib.load(prediction)
    prediction_data = np.asanyarray(prediction_nii.dataobj)
    start = np.asarray(geometry["crop_bbox_original_voxels"]["start"], dtype=int)
    end = np.asarray(geometry["crop_bbox_original_voxels"]["end_exclusive"], dtype=int)
    expected_shape = tuple((end - start).tolist())
    if tuple(prediction_data.shape) != expected_shape:
        raise ValueError(f"prediction shape {prediction_data.shape} != ROI shape {expected_shape}")
    canonical = np.zeros(tuple(geometry["canonical_shape"]), dtype=np.uint8)
    slices = tuple(slice(int(a), int(b)) for a, b in zip(start, end))
    canonical[slices] = prediction_data > 0.5
    original_affine = np.asarray(geometry["original_affine"], dtype=float)
    inverse = ornt_transform(axcodes2ornt(("R", "A", "S")), io_orientation(original_affine))
    original = apply_orientation(canonical, inverse).astype(np.uint8)
    if tuple(original.shape) != tuple(geometry["original_shape"]):
        raise ValueError(f"restored shape {original.shape} != original shape {geometry['original_shape']}")
    result = nib.Nifti1Image(original, original_affine)
    result.set_data_dtype(np.uint8)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name.replace(".nii.gz", ".tmp.nii.gz"))
    nib.save(result, temporary)
    os.replace(temporary, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    restore(args.prediction, args.metadata, args.output)
    print(json.dumps({"prediction": str(args.prediction), "output": str(args.output), "space": "original_CT"}))


if __name__ == "__main__":
    main()

import tempfile
import unittest
from pathlib import Path

try:
    import nibabel as nib
    import numpy as np
    import torch  # noqa: F401 - required by the dataset module

    from medverse.data.paot2_dataset import PAOT2ICLDataset, _crop_or_pad, _load_aligned_pair

    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False


@unittest.skipUnless(HAS_DEPS, "requires nibabel and torch")
class PAOT2AlignmentTests(unittest.TestCase):
    def test_mismatched_affines_preserve_shared_voxel_indices(self):
        image = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        mask = (image > 10).astype(np.int16)
        image_affine = np.diag([-2.0, 2.0, 3.0, 1.0])
        mask_affine = np.eye(4)
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "image.nii.gz"
            mask_path = Path(directory) / "mask.nii.gz"
            nib.save(nib.Nifti1Image(image, image_affine), image_path)
            nib.save(nib.Nifti1Image(mask, mask_affine), mask_path)
            loaded_image, loaded_mask, spacing = _load_aligned_pair(image_path, mask_path)
        np.testing.assert_array_equal(loaded_image, image)
        np.testing.assert_array_equal(loaded_mask, mask)
        self.assertEqual(spacing, (2.0, 2.0, 3.0))

    def test_matching_affines_canonicalize_both_arrays_together(self):
        image = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        mask = (image > 10).astype(np.int16)
        affine = np.diag([-2.0, 2.0, 3.0, 1.0])
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "image.nii.gz"
            mask_path = Path(directory) / "mask.nii.gz"
            nib.save(nib.Nifti1Image(image, affine), image_path)
            nib.save(nib.Nifti1Image(mask, affine), mask_path)
            loaded_image, loaded_mask, spacing = _load_aligned_pair(image_path, mask_path)
        np.testing.assert_array_equal(loaded_image, image[::-1])
        np.testing.assert_array_equal(loaded_mask, mask[::-1])
        self.assertEqual(spacing, (2.0, 2.0, 3.0))

    def test_crop_or_pad_keeps_requested_center_and_shape(self):
        volume = torch.zeros((1, 5, 6, 7))
        volume[0, 0, 0, 0] = 1
        patch = _crop_or_pad(volume, (0, 0, 0), 4)
        self.assertEqual(tuple(patch.shape), (1, 4, 4, 4))
        self.assertEqual(float(patch[0, 2, 2, 2]), 1.0)

    def test_dataset_honors_frozen_context_case_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for index in range(4):
                image = np.full((4, 4, 4), index, dtype=np.float32)
                mask = np.ones((4, 4, 4), dtype=np.int16)
                image_path, mask_path = root / f"image{index}.nii.gz", root / f"mask{index}.nii.gz"
                nib.save(nib.Nifti1Image(image, np.eye(4)), image_path)
                nib.save(nib.Nifti1Image(mask, np.eye(4)), mask_path)
                rows.append({
                    "case_id": f"case{index}",
                    "patient_id": f"patient{index}",
                    "image": str(image_path),
                    "mask": str(mask_path),
                    "split": "val" if index == 0 else "train",
                    "primary_organ": "liver",
                    "target_region": "liver_tumor",
                    "tumor_label_values": [1],
                    "foreground_voxels": 64,
                })
            rows[0]["context_case_ids"] = ["case3", "case1"]
            dataset = PAOT2ICLDataset(
                rows, split="val", num_context=2, image_size=4,
                target_spacing=(1.0, 1.0, 1.0), full_volume_target=False,
            )
            item = dataset[0]
            self.assertEqual(item["context_ids"], ["case3", "case1"])


if __name__ == "__main__":
    unittest.main()

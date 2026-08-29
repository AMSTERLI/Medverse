import tempfile
import unittest
from pathlib import Path

try:
    import nibabel as nib
    import numpy as np
    import torch  # noqa: F401 - required by the dataset module

    from medverse.data.paot2_dataset import _load_aligned_pair

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
            loaded_image, loaded_mask = _load_aligned_pair(image_path, mask_path)
        np.testing.assert_array_equal(loaded_image, image)
        np.testing.assert_array_equal(loaded_mask, mask)

    def test_matching_affines_canonicalize_both_arrays_together(self):
        image = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        mask = (image > 10).astype(np.int16)
        affine = np.diag([-2.0, 2.0, 3.0, 1.0])
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "image.nii.gz"
            mask_path = Path(directory) / "mask.nii.gz"
            nib.save(nib.Nifti1Image(image, affine), image_path)
            nib.save(nib.Nifti1Image(mask, affine), mask_path)
            loaded_image, loaded_mask = _load_aligned_pair(image_path, mask_path)
        np.testing.assert_array_equal(loaded_image, image[::-1])
        np.testing.assert_array_equal(loaded_mask, mask[::-1])


if __name__ == "__main__":
    unittest.main()

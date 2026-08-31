import importlib.util
import tempfile
import unittest
from pathlib import Path


try:
    import nibabel as nib
    import numpy as np

    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False


SCRIPT = Path(__file__).parents[1] / "scripts" / "materialize_totalseg_roi_dataset.py"
SPEC = importlib.util.spec_from_file_location("materialize_totalseg_roi_dataset", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
if HAS_DEPS:
    assert SPEC.loader is not None
    SPEC.loader.exec_module(MODULE)


@unittest.skipUnless(HAS_DEPS, "requires nibabel and numpy")
class ROIMaterializationTests(unittest.TestCase):
    def test_process_one_uses_only_organ_bbox_and_preserves_tumor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            affine = np.diag([2.0, 2.0, 2.0, 1.0])
            image = np.full((20, 20, 20), -1000, dtype=np.float32)
            image[5:15, 5:15, 5:15] = 50
            organ = np.zeros_like(image, dtype=np.uint8)
            organ[5:15, 5:15, 5:15] = 1
            label = np.zeros_like(image, dtype=np.uint8)
            label[8:11, 8:11, 8:11] = 2
            paths = {}
            for name, array in (("image", image), ("mask", label), ("organ", organ)):
                path = root / f"{name}.nii.gz"
                nib.save(nib.Nifti1Image(array, affine), path)
                paths[name] = str(path)
            row = {
                "case_id": "case",
                "patient_id": "patient",
                "source_dataset": "source",
                "target_region": "liver_tumor",
                "image": paths["image"],
                "mask": paths["mask"],
                "organ_mask": paths["organ"],
                "tumor_label_values": [2],
                "foreground_voxels": 27,
            }
            result = MODULE.process_one((row, str(root / "out"), (2.0, 2.0, 2.0), 2.0, (-1000.0, 1000.0)))
            self.assertEqual(result["roi_tumor_coverage"], 1.0)
            self.assertTrue(Path(result["image"]).is_file())
            self.assertEqual(set(np.unique(np.asanyarray(nib.load(result["mask"]).dataobj))), {0, 1})


if __name__ == "__main__":
    unittest.main()

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def rows_for_contexts():
    rows = []
    for task in ("liver_tumor", "kidney_tumor"):
        for index in range(7):
            rows.append({
                "case_id": f"{task}:{index}", "patient_id": f"p:{task}:{index}",
                "source_dataset": "source", "target_region": task,
                "split": "train" if index < 6 else "test", "foreground_voxels": 10,
            })
    return rows


def test_random_and_wrong_contexts_are_train_only_and_patient_safe():
    module = load_script("build_liver_kidney_contexts.py")
    for strategy in ("random_same_task", "wrong_task"):
        rows = module.build_contexts(rows_for_contexts(), strategy, k=3, seed=17)
        lookup = {row["case_id"]: row for row in rows}
        for row in rows:
            for context_id in row["context_case_ids"]:
                context = lookup[context_id]
                assert context["split"] == "train"
                assert module.patient_key(context) != module.patient_key(row)
                assert (context["target_region"] == row["target_region"]) == (strategy != "wrong_task")


def test_voxel_and_lesion_metrics_have_expected_extremes():
    module = load_script("evaluate_liver_kidney_original_space.py")
    truth = np.zeros((12, 12, 12), dtype=bool)
    truth[3:6, 3:6, 3:6] = True
    perfect = module.voxel_metrics(truth, truth, (1.0, 1.0, 1.0), 1.0)
    assert perfect == {"dice": 1.0, "nsd": 1.0, "hd95_mm": 0.0}
    lesions = module.lesion_metrics(truth, truth)
    assert lesions["lesion_tp"] == 1
    assert lesions["lesion_fp"] == 0
    assert lesions["lesion_recall"] == 1.0
    empty = module.voxel_metrics(np.zeros_like(truth), truth, (1.0, 1.0, 1.0), 1.0)
    assert empty["dice"] == 0.0
    assert empty["nsd"] == 0.0
    assert np.isinf(empty["hd95_mm"])

"""Resume nnU-Net Blosc2 preprocessing without deleting completed cases.

nnU-Net's standard preprocessor deliberately recreates the entire configuration
directory. On network filesystems a transient Blosc2 write failure late in a
large dataset would therefore discard hours of valid work on the next run. This
entry point validates completed case triplets, removes only incomplete outputs,
and retries the remaining cases independently.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import pickle
import time
from pathlib import Path
from typing import Any

import blosc2
from batchgenerators.utilities.file_and_folder_operations import load_json
from tqdm import tqdm

from nnunetv2.paths import nnUNet_preprocessed, nnUNet_raw
from nnunetv2.preprocessing.preprocessors.prompted_fullct_preprocessor import PromptedFullCTPreprocessor
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.utilities.utils import get_filenames_of_train_images_and_targets


def output_paths(output_base: Path) -> tuple[Path, Path, Path]:
    return (
        output_base.with_suffix(".b2nd"),
        output_base.with_name(output_base.name + "_seg.b2nd"),
        output_base.with_suffix(".pkl"),
    )


def remove_case_outputs(output_base: Path) -> None:
    for path in output_paths(output_base):
        if path.exists() or path.is_symlink():
            path.unlink()


def case_is_complete(output_base: Path) -> bool:
    data_path, seg_path, properties_path = output_paths(output_base)
    if any(not path.is_file() or path.stat().st_size == 0 for path in (data_path, seg_path, properties_path)):
        return False
    try:
        data = blosc2.open(urlpath=data_path, mode="r")
        seg = blosc2.open(urlpath=seg_path, mode="r")
        if tuple(data.shape[1:]) != tuple(seg.shape[1:]):
            return False
        with properties_path.open("rb") as stream:
            properties = pickle.load(stream)
        return isinstance(properties, dict)
    except Exception:
        return False


def preprocess_one(payload: tuple[str, list[str], str, dict[str, Any], str, dict[str, Any], int]) -> dict[str, Any]:
    output_base_string, images, label, plans, configuration, dataset_json, retries = payload
    output_base = Path(output_base_string)
    plans_manager = PlansManager(plans)
    configuration_manager = plans_manager.get_configuration(configuration)
    preprocessor = PromptedFullCTPreprocessor(verbose=False)
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        remove_case_outputs(output_base)
        try:
            preprocessor.run_case_save(
                str(output_base),
                images,
                label,
                plans_manager,
                configuration_manager,
                dataset_json,
            )
            if not case_is_complete(output_base):
                raise RuntimeError("preprocessor returned without a valid output triplet")
            return {"case": output_base.name, "attempt": attempt, "status": "completed"}
        except Exception as error:
            last_error = error
            remove_case_outputs(output_base)
            if attempt < retries:
                time.sleep(2 * attempt)
    assert last_error is not None
    raise RuntimeError(f"{output_base.name} failed after {retries} attempts: {last_error}") from last_error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-id", type=int, required=True)
    parser.add_argument("--configuration", default="3d_fullres")
    parser.add_argument("--plans-identifier", required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()
    if args.workers < 1 or args.retries < 1:
        raise ValueError("workers and retries must be positive")

    dataset_name = maybe_convert_to_dataset_name(args.dataset_id)
    raw_dataset = Path(nnUNet_raw) / dataset_name
    preprocessed_dataset = Path(nnUNet_preprocessed) / dataset_name
    plans_path = preprocessed_dataset / f"{args.plans_identifier}.json"
    dataset_json_path = preprocessed_dataset / "dataset.json"
    plans = load_json(str(plans_path))
    dataset_json = load_json(str(dataset_json_path))
    plans_manager = PlansManager(plans)
    configuration_manager = plans_manager.get_configuration(args.configuration)
    output_dir = preprocessed_dataset / configuration_manager.data_identifier
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = get_filenames_of_train_images_and_targets(str(raw_dataset), dataset_json)

    complete = []
    pending = []
    for identifier, item in dataset.items():
        output_base = output_dir / identifier
        if case_is_complete(output_base):
            complete.append(identifier)
        else:
            remove_case_outputs(output_base)
            pending.append(
                (
                    str(output_base),
                    list(item["images"]),
                    str(item["label"]),
                    plans,
                    args.configuration,
                    dataset_json,
                    args.retries,
                )
            )

    print(json.dumps({"total": len(dataset), "reused": len(complete), "pending": len(pending)}, indent=2))
    if pending:
        with multiprocessing.get_context("spawn").Pool(args.workers) as pool:
            for result in tqdm(
                pool.imap_unordered(preprocess_one, pending),
                total=len(pending),
                desc="Resuming preprocessing",
            ):
                print(json.dumps(result), flush=True)

    invalid = [identifier for identifier in dataset if not case_is_complete(output_dir / identifier)]
    if invalid:
        raise RuntimeError(f"preprocessing remains incomplete for {len(invalid)} cases: {invalid[:10]}")
    print(json.dumps({"total": len(dataset), "complete": len(dataset), "status": "passed"}, indent=2))


if __name__ == "__main__":
    main()

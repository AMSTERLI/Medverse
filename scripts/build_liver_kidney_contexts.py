"""Freeze random, retrieval, or wrong-task contexts without target-mask leakage."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom


TASKS = ("liver_tumor", "kidney_tumor")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def stable_score(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def patient_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["source_dataset"]), str(row["patient_id"])


def input_only_embedding(row: dict[str, Any], grid: int = 16) -> np.ndarray:
    """Frozen hand-crafted CT/organ embedding; never reads the tumor mask."""
    image_path = Path(row.get("roi_image", row["image"]))
    organ_path = Path(row.get("roi_organ_mask", row["organ_mask"]))
    image_nii, organ_nii = nib.load(image_path), nib.load(organ_path)
    if image_nii.shape != organ_nii.shape or not np.allclose(image_nii.affine, organ_nii.affine, atol=1e-3):
        raise ValueError(f"{row['case_id']}: CT/organ grid mismatch")
    image = np.asarray(image_nii.dataobj, dtype=np.float32)
    organ = np.asarray(organ_nii.dataobj) > 0.5
    clipped = np.clip(image, -1000.0, 1000.0) / 1000.0
    factors = tuple(grid / max(size, 1) for size in image.shape)
    low_image = zoom(clipped, factors, order=1, prefilter=False)
    low_organ = zoom(organ.astype(np.float32), factors, order=0, prefilter=False)
    values = clipped[organ] if organ.any() else clipped.ravel()
    histogram = np.histogram(values, bins=24, range=(-1.0, 1.0), density=True)[0].astype(np.float32)
    spacing = np.asarray(image_nii.header.get_zooms()[:3], dtype=np.float32)
    physical_shape = np.asarray(image.shape, dtype=np.float32) * spacing
    vector = np.concatenate((
        low_image.ravel(), low_organ.ravel(), histogram,
        physical_shape / 500.0,
        np.asarray([float(organ.mean()), float(values.mean()), float(values.std())], dtype=np.float32),
    )).astype(np.float32)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0 else vector


def build_contexts(
    rows: list[dict[str, Any]], strategy: str, k: int, seed: int,
    embedding_dir: Path | None = None,
) -> list[dict[str, Any]]:
    if k not in (1, 3, 5):
        raise ValueError("K must be one of 1, 3, or 5")
    lookup = {row["case_id"]: row for row in rows}
    if len(lookup) != len(rows):
        raise ValueError("duplicate case_id in manifest")
    bank = [
        row for row in rows
        if row["split"] == "train" and row["target_region"] in TASKS
        and int(row.get("foreground_voxels", 0)) > 0
    ]
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in bank:
        by_task[row["target_region"]].append(row)

    embeddings: dict[str, np.ndarray] = {}
    if strategy == "retrieval_same_task":
        if embedding_dir is not None:
            embedding_dir.mkdir(parents=True, exist_ok=True)
        for row in rows:
            cache = embedding_dir / f"{hashlib.sha256(row['case_id'].encode()).hexdigest()}.npy" if embedding_dir else None
            if cache is not None and cache.is_file():
                embedding = np.load(cache)
            else:
                embedding = input_only_embedding(row)
                if cache is not None:
                    np.save(cache, embedding)
            embeddings[row["case_id"]] = embedding

    output = []
    for original in rows:
        row = dict(original)
        task = row["target_region"]
        candidate_task = task
        if strategy == "wrong_task":
            candidate_task = TASKS[1] if task == TASKS[0] else TASKS[0]
        candidates = [candidate for candidate in by_task[candidate_task] if patient_key(candidate) != patient_key(row)]
        if len(candidates) < k:
            raise ValueError(f"{row['case_id']}: only {len(candidates)} eligible {candidate_task} contexts; need {k}")
        if strategy == "retrieval_same_task":
            query = embeddings[row["case_id"]]
            ranked = sorted(
                ((float(np.dot(query, embeddings[candidate["case_id"]])), candidate) for candidate in candidates),
                key=lambda item: (-item[0], item[1]["case_id"]),
            )
            selected = [candidate for _, candidate in ranked[:k]]
            row["context_candidates"] = [
                {"case_id": candidate["case_id"], "cosine_similarity": similarity}
                for similarity, candidate in ranked[: min(20, len(ranked))]
            ]
            row["retrieval_embedding"] = "frozen_input_only_ct_organ_v1"
        else:
            candidates.sort(key=lambda candidate: stable_score(
                f"{seed}|{strategy}|{row['case_id']}|{candidate['case_id']}"
            ))
            selected = candidates[:k]
        row["context_strategy"] = strategy
        row["context_case_ids"] = [candidate["case_id"] for candidate in selected]
        output.append(row)

    for row in output:
        for context_id in row["context_case_ids"]:
            context = lookup[context_id]
            if context["split"] != "train" or patient_key(context) == patient_key(row):
                raise AssertionError(f"context leakage: {row['case_id']} -> {context_id}")
            same = context["target_region"] == row["target_region"]
            if (strategy == "wrong_task" and same) or (strategy != "wrong_task" and not same):
                raise AssertionError(f"task rule violated: {row['case_id']} -> {context_id}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--strategy", choices=("random_same_task", "retrieval_same_task", "wrong_task"), required=True)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--embedding-dir", type=Path)
    args = parser.parse_args()
    rows = build_contexts(load_jsonl(args.manifest), args.strategy, args.k, args.seed, args.embedding_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {
        "strategy": args.strategy, "k": args.k, "seed": args.seed, "rows": len(rows),
        "bank_split": "train", "requires_nonempty_context": True,
        "target_mask_used_for_retrieval": False,
    }
    args.output.with_suffix(".summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

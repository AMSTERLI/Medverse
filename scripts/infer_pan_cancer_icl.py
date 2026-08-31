"""Run Medverse ICL sliding-window inference and restore original CT geometry."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

from medverse.data import PAOT2ICLDataset
from medverse.models.pan_cancer_medverse import PanCancerMedverse
from train_pan_cancer_icl import prediction_probability, sliding_window_logits


def load_restore():
    path = Path(__file__).with_name("restore_roi_prediction.py")
    spec = importlib.util.spec_from_file_location("restore_roi_prediction", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.restore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--target-spacing", required=True)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    spacing = tuple(float(value) for value in args.target_spacing.split(","))
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    saved = checkpoint.get("args", {})
    image_size = int(saved.get("image_size", 128))
    channels = tuple(int(value) for value in str(saved.get("channels", "32,64,128,256,512")).split(","))
    num_context = int(saved.get("num_context", 3))
    organ_channel = bool(saved.get("organ_channel", True))
    model = PanCancerMedverse(
        inner_channels=channels, conv_layers_per_stage=int(saved.get("conv_layers", 2)),
        img_size=image_size, in_channels=2 if organ_channel else 1, use_ccti=False,
    ).to(args.device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    dataset = PAOT2ICLDataset(
        manifest=args.manifest, split=args.split, context_split="train", num_context=num_context,
        image_size=image_size, target_spacing=spacing, full_volume_target=True,
        include_organ_channel=organ_channel, positive_probability=1.0,
    )
    restore = load_restore()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for index, row in enumerate(dataset.rows):
        sample = dataset[index]
        target = sample["target_in"].unsqueeze(0).to(args.device)
        context_in = sample["context_in"].unsqueeze(0).to(args.device)
        context_out = sample["context_out"].unsqueeze(0).to(args.device)
        with torch.inference_mode(), torch.cuda.amp.autocast(enabled=str(args.device).startswith("cuda")):
            output = sliding_window_logits(
                model, target, context_in, context_out, image_size, args.overlap, min(num_context, 2)
            )
            probability = prediction_probability(output, str(saved.get("prediction_mode", "logits")))
        roi_nii = nib.load(row.get("roi_image", row["image"]))
        resized = F.interpolate(
            probability.float(), size=tuple(int(v) for v in roi_nii.shape), mode="trilinear", align_corners=False
        )[0, 0].cpu().numpy()
        token = hashlib.sha256(row["case_id"].encode()).hexdigest()[:12]
        roi_prediction = args.output_dir / "roi" / f"{token}.nii.gz"
        original_prediction = args.output_dir / "original" / f"{token}.nii.gz"
        roi_prediction.parent.mkdir(parents=True, exist_ok=True)
        original_prediction.parent.mkdir(parents=True, exist_ok=True)
        nii = nib.Nifti1Image((resized >= 0.5).astype(np.uint8), roi_nii.affine, roi_nii.header)
        nii.set_data_dtype(np.uint8); nib.save(nii, roi_prediction)
        restore(roi_prediction, Path(row["geometry_metadata"]), original_prediction)
        records.append({
            "case_id": row["case_id"], "prediction": str(original_prediction),
            "roi_prediction": str(roi_prediction), "context_case_ids": row["context_case_ids"],
        })
        print(json.dumps({"case": index + 1, "total": len(dataset), "case_id": row["case_id"]}), flush=True)
    manifest_path = args.output_dir / "predictions.jsonl"
    with manifest_path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps({"cases": len(records), "predictions": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()

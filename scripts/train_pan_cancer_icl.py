"""Minimal train/evaluate entry point for PAOT2 Medverse ICL experiments."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

# Allow running the script directly from a source checkout without installing
# the package first.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from medverse.data import PAOT2ICLDataset
from medverse.models.pan_cancer_medverse import PanCancerMedverse


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    probability = torch.sigmoid(logits)
    dims = tuple(range(1, probability.ndim))
    intersection = (probability * target).sum(dims)
    denominator = probability.sum(dims) + target.sum(dims)
    return (1.0 - (2.0 * intersection + eps) / (denominator + eps)).mean()


def segmentation_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, target) + dice_loss(logits, target)


def hard_dice(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    prediction = torch.sigmoid(logits) >= 0.5
    truth = target >= 0.5
    dims = tuple(range(1, prediction.ndim))
    intersection = (prediction & truth).sum(dims).float()
    denominator = prediction.sum(dims).float() + truth.sum(dims).float()
    return (2.0 * intersection + eps) / (denominator + eps)


def move_batch(batch: dict, device: torch.device) -> tuple[torch.Tensor, ...]:
    return tuple(
        batch[key].to(device, non_blocking=True)
        for key in ("target_in", "target_out", "context_in", "context_out")
    )


def load_weights(model: torch.nn.Module, path: Path, strict: bool) -> None:
    payload = torch.load(path, map_location="cpu")
    state = payload.get("model", payload.get("state_dict", payload))
    normalized = {}
    for key, value in state.items():
        for prefix in ("model.", "net."):
            if key.startswith(prefix):
                key = key[len(prefix) :]
        normalized[key] = value
    if strict:
        model.load_state_dict(normalized, strict=True)
        return
    current = model.state_dict()
    compatible = {key: value for key, value in normalized.items() if key in current and current[key].shape == value.shape}
    result = model.load_state_dict(compatible, strict=False)
    print(
        json.dumps(
            {
                "loaded_tensors": len(compatible),
                "missing_tensors": len(result.missing_keys),
                "unexpected_tensors": len(result.unexpected_keys),
            }
        )
    )


@torch.no_grad()
def evaluate(model, loader, device, context_chunk: int, max_steps: int | None = None) -> dict:
    model.eval()
    by_task: dict[str, list[float]] = defaultdict(list)
    losses = []
    for step, batch in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break
        target_in, target_out, context_in, context_out = move_batch(batch, device)
        logits = model(target_in, context_in, context_out, l=context_chunk)
        losses.append(float(segmentation_loss(logits, target_out)))
        scores = hard_dice(logits, target_out).cpu().tolist()
        for task, score in zip(batch["target_region"], scores):
            by_task[task].append(score)
    task_dice = {task: float(np.mean(values)) for task, values in sorted(by_task.items())}
    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "macro_dice": float(np.mean(list(task_dice.values()))) if task_dice else float("nan"),
        "task_dice": task_dice,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/paot2_ccti"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--num-context", type=int, default=2)
    parser.add_argument("--channels", default="16,32,64,128,256")
    parser.add_argument("--conv-layers", type=int, default=2)
    parser.add_argument("--ccti-mode", choices=("none", "all", "random", "learned"), default="learned")
    parser.add_argument("--ccti-ratio", type=float, default=0.25)
    parser.add_argument("--disable-ccti", action="store_true")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--pretrained", type=Path)
    parser.add_argument("--checkpoint", type=Path, help="Strictly load an MVP checkpoint.")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--max-train-steps", type=int)
    parser.add_argument("--max-val-steps", type=int)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument(
        "--allow-uninspected-context",
        action="store_true",
        help="Fast smoke-test only: allow manifests without foreground_voxels.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()

    if args.image_size % 16 or args.image_size % 4:
        raise ValueError("--image-size must be divisible by both 16 and patch_num=4")
    seed_everything(args.seed)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    common = dict(
        manifest=args.manifest,
        context_split="train",
        num_context=args.num_context,
        image_size=args.image_size,
        data_root=args.data_root,
        seed=args.seed,
        require_inspected_context=not args.allow_uninspected_context,
    )
    train_dataset = PAOT2ICLDataset(split="train", **common)
    val_dataset = PAOT2ICLDataset(split="val", **common)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )

    channels = tuple(int(value) for value in args.channels.split(","))
    model = PanCancerMedverse(
        inner_channels=channels,
        conv_layers_per_stage=args.conv_layers,
        img_size=args.image_size,
        use_ccti=not args.disable_ccti,
        ccti_mode=args.ccti_mode,
        ccti_channel_ratio=args.ccti_ratio,
    ).to(device)
    if args.pretrained:
        load_weights(model, args.pretrained, strict=False)
    if args.checkpoint:
        load_weights(model, args.checkpoint, strict=True)

    print(
        json.dumps(
            {
                "train_episodes": len(train_dataset),
                "val_episodes": len(val_dataset),
                "device": str(device),
                "use_ccti": not args.disable_ccti,
                "ccti_mode": args.ccti_mode if not args.disable_ccti else "disabled",
                "channels": channels,
            }
        )
    )
    context_chunk = min(args.num_context, 2)
    if args.eval_only:
        print(json.dumps(evaluate(model, val_loader, device, context_chunk, args.max_val_steps), indent=2))
        return

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    amp_enabled = device.type == "cuda" and not args.no_amp
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    best_macro_dice = -1.0
    for epoch in range(args.epochs):
        model.train()
        running_loss = []
        for step, batch in enumerate(train_loader):
            if args.max_train_steps is not None and step >= args.max_train_steps:
                break
            target_in, target_out, context_in, context_out = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                logits = model(target_in, context_in, context_out, l=context_chunk)
                loss = segmentation_loss(logits, target_out)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss.append(float(loss.detach()))
            if args.log_every > 0 and (step + 1) % args.log_every == 0:
                print(
                    json.dumps(
                        {
                            "epoch": epoch + 1,
                            "train_step": step + 1,
                            "train_steps_total": len(train_loader),
                            "train_loss_running": float(np.mean(running_loss)),
                        }
                    ),
                    flush=True,
                )

        metrics = evaluate(model, val_loader, device, context_chunk, args.max_val_steps)
        report = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(running_loss)) if running_loss else float("nan"),
            **metrics,
        }
        print(json.dumps(report, ensure_ascii=False))
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1,
            "metrics": metrics,
            "args": vars(args),
        }
        torch.save(checkpoint, args.output_dir / "last.pt")
        if metrics["macro_dice"] > best_macro_dice:
            best_macro_dice = metrics["macro_dice"]
            torch.save(checkpoint, args.output_dir / "best.pt")


if __name__ == "__main__":
    main()

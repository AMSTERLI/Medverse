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
from torch.utils.data import DataLoader, WeightedRandomSampler

from medverse.data import PAOT2ICLDataset
from medverse.models.pan_cancer_medverse import PanCancerMedverse


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prediction_probability(output: torch.Tensor, prediction_mode: str) -> torch.Tensor:
    return torch.sigmoid(output) if prediction_mode == "logits" else output.clamp(0.0, 1.0)


def dice_loss(
    output: torch.Tensor, target: torch.Tensor, prediction_mode: str = "logits", eps: float = 1e-5
) -> torch.Tensor:
    probability = prediction_probability(output, prediction_mode)
    dims = tuple(range(1, probability.ndim))
    intersection = (probability * target).sum(dims)
    denominator = probability.sum(dims) + target.sum(dims)
    return (1.0 - (2.0 * intersection + eps) / (denominator + eps)).mean()


def smooth_l3_l1(output: torch.Tensor, target: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    error = torch.abs(output - target)
    return torch.where(
        error < beta,
        0.333 * error.pow(3) / beta**2,
        error + 0.333 * beta**3 - beta,
    ).mean()


def segmentation_loss(
    output: torch.Tensor, target: torch.Tensor, positive_weight: float = 8.0,
    loss_mode: str = "bce_dice", prediction_mode: str = "logits",
) -> torch.Tensor:
    if loss_mode == "bce_dice":
        pos_weight = torch.as_tensor(positive_weight, device=output.device, dtype=output.dtype)
        return F.binary_cross_entropy_with_logits(output, target, pos_weight=pos_weight) + dice_loss(
            output, target, prediction_mode
        )
    original = 50.0 * smooth_l3_l1(output, target)
    if loss_mode == "smoothl3":
        return original
    if loss_mode == "smoothl3_dice":
        return original + dice_loss(output, target, prediction_mode)
    raise ValueError(f"unknown loss mode: {loss_mode}")


def hard_dice(
    output: torch.Tensor, target: torch.Tensor, prediction_mode: str = "logits", eps: float = 1e-5
) -> torch.Tensor:
    prediction = prediction_probability(output, prediction_mode) >= 0.5
    truth = target >= 0.5
    dims = tuple(range(1, prediction.ndim))
    intersection = (prediction & truth).sum(dims).float()
    denominator = prediction.sum(dims).float() + truth.sum(dims).float()
    return (2.0 * intersection + eps) / (denominator + eps)


def _window_starts(size: int, patch_size: int, overlap: float) -> list[int]:
    if size <= patch_size:
        return [0]
    stride = max(1, int(round(patch_size * (1.0 - overlap))))
    starts = list(range(0, size - patch_size + 1, stride))
    if starts[-1] != size - patch_size:
        starts.append(size - patch_size)
    return starts


@torch.no_grad()
def sliding_window_logits(model, target, context_in, context_out, patch_size, overlap, context_chunk):
    """Infer an arbitrarily sized target by averaging overlapping patch logits."""
    original_shape = target.shape[-3:]
    pad = [max(0, patch_size - size) for size in original_shape]
    target = F.pad(target, (0, pad[2], 0, pad[1], 0, pad[0]))
    output = torch.zeros_like(target)
    count = torch.zeros_like(target)
    starts = [_window_starts(size, patch_size, overlap) for size in target.shape[-3:]]
    for d in starts[0]:
        for h in starts[1]:
            for w in starts[2]:
                patch = target[:, :, d:d + patch_size, h:h + patch_size, w:w + patch_size]
                logits = model(patch, context_in, context_out, l=context_chunk)
                output[:, :, d:d + patch_size, h:h + patch_size, w:w + patch_size] += logits
                count[:, :, d:d + patch_size, h:h + patch_size, w:w + patch_size] += 1
    output = output / count.clamp_min(1)
    return output[:, :, :original_shape[0], :original_shape[1], :original_shape[2]]


def move_batch(batch: dict, device: torch.device) -> tuple[torch.Tensor, ...]:
    return tuple(
        batch[key].to(device, non_blocking=True)
        for key in ("target_in", "target_out", "context_in", "context_out")
    )


def load_weights(model: torch.nn.Module, path: Path, strict: bool) -> dict[str, int]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state = payload.get("model", payload.get("state_dict", payload))
    normalized = {}
    for key, value in state.items():
        for prefix in ("model.", "net."):
            if key.startswith(prefix):
                key = key[len(prefix) :]
        normalized[key] = value
    if strict:
        model.load_state_dict(normalized, strict=True)
        return {"loaded_tensors": len(normalized), "missing_tensors": 0, "unexpected_tensors": 0}
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
    return {
        "loaded_tensors": len(compatible),
        "missing_tensors": len(result.missing_keys),
        "unexpected_tensors": len(result.unexpected_keys),
    }


def set_trainable_scope(model: torch.nn.Module, scope: str) -> dict[str, int]:
    """Freeze pretrained weights while keeping the experiment heads trainable."""
    for name, parameter in model.named_parameters():
        if scope == "heads":
            parameter.requires_grad = (
                name.startswith("target_decoder.ccti_blocks")
                or name.startswith("target_decoder.output_block")
            )
        elif scope == "decoder":
            parameter.requires_grad = name.startswith("target_decoder")
        elif scope == "all":
            parameter.requires_grad = True
        else:
            raise ValueError(f"unknown trainable scope: {scope}")
    return {
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "total_parameters": sum(p.numel() for p in model.parameters()),
    }


@torch.no_grad()
def evaluate(
    model, loader, device, context_chunk: int, patch_size: int, overlap: float,
    positive_weight: float, loss_mode: str, prediction_mode: str,
    max_steps: int | None = None, log_every: int = 20,
) -> dict:
    model.eval()
    by_task: dict[str, list[float]] = defaultdict(list)
    losses, empty_false_positive_rates = [], []
    positive_cases = empty_cases = 0
    for step, batch in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break
        target_in, target_out, context_in, context_out = move_batch(batch, device)
        logits = sliding_window_logits(
            model, target_in, context_in, context_out, patch_size, overlap, context_chunk
        )
        losses.append(float(segmentation_loss(
            logits, target_out, positive_weight, loss_mode, prediction_mode
        )))
        foreground = target_out.flatten(1).sum(1) > 0
        scores = hard_dice(logits, target_out, prediction_mode).cpu().tolist()
        predictions = (prediction_probability(logits, prediction_mode) >= 0.5).flatten(1)
        for task, score, is_positive, prediction in zip(batch["target_region"], scores, foreground.tolist(), predictions):
            if is_positive:
                by_task[task].append(score)
                positive_cases += 1
            else:
                empty_false_positive_rates.append(float(prediction.float().mean().cpu()))
                empty_cases += 1
        if log_every > 0 and (step + 1) % log_every == 0:
            print(json.dumps({"val_step": step + 1, "val_steps_total": len(loader)}), flush=True)
    task_dice = {task: float(np.mean(values)) for task, values in sorted(by_task.items())}
    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "macro_dice": float(np.mean(list(task_dice.values()))) if task_dice else float("nan"),
        "task_dice": task_dice,
        "positive_cases": positive_cases,
        "empty_cases": empty_cases,
        "empty_false_positive_rate": float(np.mean(empty_false_positive_rates)) if empty_false_positive_rates else float("nan"),
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
    parser.add_argument("--target-spacing", default="2.0,2.0,3.0")
    parser.add_argument("--positive-patch-probability", type=float, default=0.7)
    parser.add_argument("--positive-weight", type=float, default=8.0)
    parser.add_argument(
        "--loss-mode", choices=("bce_dice", "smoothl3", "smoothl3_dice"), default="bce_dice"
    )
    parser.add_argument("--prediction-mode", choices=("logits", "regression"), default="logits")
    parser.add_argument("--val-overlap", type=float, default=0.5)
    parser.add_argument("--val-cases-per-task", type=int, default=20)
    parser.add_argument(
        "--val-positive-patches",
        action="store_true",
        help=(
            "Diagnostic evaluation: crop validation targets around foreground exactly like "
            "positive training patches instead of evaluating full volumes."
        ),
    )
    parser.add_argument("--disable-task-balancing", action="store_true")
    parser.add_argument("--num-context", type=int, default=2)
    parser.add_argument("--channels", default="16,32,64,128,256")
    parser.add_argument("--conv-layers", type=int, default=2)
    parser.add_argument("--ccti-mode", choices=("none", "all", "random", "learned"), default="learned")
    parser.add_argument("--ccti-ratio", type=float, default=0.25)
    parser.add_argument("--disable-ccti", action="store_true")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--pretrained", type=Path)
    parser.add_argument(
        "--freeze-backbone-epochs", type=int, default=0,
        help="Train only CCTI/output heads for this many initial epochs.",
    )
    parser.add_argument(
        "--unfreeze-scope", choices=("decoder", "all"), default="decoder",
        help="Parameters to train after the frozen-head warm-up.",
    )
    parser.add_argument("--unfreeze-lr", type=float, default=3e-5)
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
    target_spacing = tuple(float(value) for value in args.target_spacing.split(","))
    if len(target_spacing) != 3 or any(value <= 0 for value in target_spacing):
        raise ValueError("--target-spacing must contain three positive comma-separated values")
    if not 0 <= args.positive_patch_probability <= 1 or not 0 <= args.val_overlap < 1:
        raise ValueError("patch probability must be in [0,1] and validation overlap in [0,1)")
    seed_everything(args.seed)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    common = dict(
        manifest=args.manifest,
        context_split="train",
        num_context=args.num_context,
        image_size=args.image_size,
        target_spacing=target_spacing,
        positive_probability=args.positive_patch_probability,
        data_root=args.data_root,
        seed=args.seed,
        require_inspected_context=not args.allow_uninspected_context,
    )
    train_dataset = PAOT2ICLDataset(split="train", full_volume_target=False, **common)
    val_dataset = PAOT2ICLDataset(
        split="val", full_volume_target=not args.val_positive_patches,
        drop_empty_targets=args.val_positive_patches,
        max_cases_per_task=args.val_cases_per_task, **common
    )
    sampler = None
    if not args.disable_task_balancing:
        task_counts: dict[str, int] = defaultdict(int)
        for row in train_dataset.rows:
            task_counts[row["target_region"]] += 1
        weights = [1.0 / task_counts[row["target_region"]] for row in train_dataset.rows]
        sampler = WeightedRandomSampler(
            weights, num_samples=len(train_dataset), replacement=True,
            generator=torch.Generator().manual_seed(args.seed),
        )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
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
    pretrained_report = None
    if args.pretrained:
        pretrained_report = load_weights(model, args.pretrained, strict=False)
        if pretrained_report["loaded_tensors"] == 0:
            raise RuntimeError("the pretrained checkpoint did not match any model tensors")
    if args.checkpoint:
        load_weights(model, args.checkpoint, strict=True)

    initial_scope = "heads" if args.freeze_backbone_epochs > 0 else args.unfreeze_scope
    parameter_report = set_trainable_scope(model, initial_scope)
    print(
        json.dumps(
            {
                "train_episodes": len(train_dataset),
                "val_episodes": len(val_dataset),
                "device": str(device),
                "use_ccti": not args.disable_ccti,
                "ccti_mode": args.ccti_mode if not args.disable_ccti else "disabled",
                "channels": channels,
                "patch_size": args.image_size,
                "target_spacing": target_spacing,
                "positive_patch_probability": args.positive_patch_probability,
                "positive_weight": args.positive_weight,
                "loss_mode": args.loss_mode,
                "prediction_mode": args.prediction_mode,
                "task_balancing": sampler is not None,
                "validation": (
                    "positive_patch" if args.val_positive_patches else "full_volume_sliding_window"
                ),
                "pretrained": str(args.pretrained) if args.pretrained else None,
                "pretrained_report": pretrained_report,
                "trainable_scope": initial_scope,
                **parameter_report,
            }
        )
    )
    context_chunk = min(args.num_context, 2)
    if args.eval_only:
        print(json.dumps(evaluate(
            model, val_loader, device, context_chunk, args.image_size, args.val_overlap,
            args.positive_weight, args.loss_mode, args.prediction_mode,
            args.max_val_steps, args.log_every,
        ), indent=2))
        return

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    amp_enabled = device.type == "cuda" and not args.no_amp
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    best_macro_dice = -1.0
    for epoch in range(args.epochs):
        if epoch == args.freeze_backbone_epochs and args.freeze_backbone_epochs > 0:
            parameter_report = set_trainable_scope(model, args.unfreeze_scope)
            for group in optimizer.param_groups:
                group["lr"] = args.unfreeze_lr
            print(json.dumps({
                "epoch": epoch + 1,
                "event": "unfreeze",
                "trainable_scope": args.unfreeze_scope,
                "learning_rate": args.unfreeze_lr,
                **parameter_report,
            }), flush=True)
        model.train()
        running_loss = []
        for step, batch in enumerate(train_loader):
            if args.max_train_steps is not None and step >= args.max_train_steps:
                break
            target_in, target_out, context_in, context_out = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                logits = model(target_in, context_in, context_out, l=context_chunk)
                loss = segmentation_loss(
                    logits, target_out, args.positive_weight, args.loss_mode, args.prediction_mode
                )
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

        metrics = evaluate(
            model, val_loader, device, context_chunk, args.image_size, args.val_overlap,
            args.positive_weight, args.loss_mode, args.prediction_mode,
            args.max_val_steps, args.log_every,
        )
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
            "args": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
        }
        torch.save(checkpoint, args.output_dir / "last.pt")
        if metrics["macro_dice"] > best_macro_dice:
            best_macro_dice = metrics["macro_dice"]
            torch.save(checkpoint, args.output_dir / "best.pt")


if __name__ == "__main__":
    main()

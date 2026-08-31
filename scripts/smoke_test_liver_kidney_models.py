"""GPU smoke test for the nnU-Net-style baseline and Medverse BAM arm."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from dynamic_network_architectures.architectures.unet import PlainConvUNet
from medverse.models.pan_cancer_medverse import PanCancerMedverse


def baseline(device: torch.device, size: int) -> dict:
    model = PlainConvUNet(
        input_channels=2, n_stages=4, features_per_stage=(8, 16, 32, 64),
        conv_op=torch.nn.Conv3d, kernel_sizes=3, strides=(1, 2, 2, 2),
        n_conv_per_stage=1, num_classes=1, n_conv_per_stage_decoder=1,
        conv_bias=True, norm_op=torch.nn.InstanceNorm3d,
        norm_op_kwargs={"eps": 1e-5, "affine": True},
        dropout_op=None, nonlin=torch.nn.LeakyReLU,
        nonlin_kwargs={"inplace": True}, deep_supervision=False,
    ).to(device)
    inputs = torch.randn(1, 2, size, size, size, device=device)
    truth = (torch.rand(1, 1, size, size, size, device=device) > 0.9).float()
    output = model(inputs)
    loss = F.binary_cross_entropy_with_logits(output, truth)
    loss.backward()
    with tempfile.NamedTemporaryFile(suffix=".pt") as stream:
        torch.save(model.state_dict(), stream.name)
        clone = PlainConvUNet(
            input_channels=2, n_stages=4, features_per_stage=(8, 16, 32, 64),
            conv_op=torch.nn.Conv3d, kernel_sizes=3, strides=(1, 2, 2, 2),
            n_conv_per_stage=1, num_classes=1, n_conv_per_stage_decoder=1,
            conv_bias=True, norm_op=torch.nn.InstanceNorm3d,
            norm_op_kwargs={"eps": 1e-5, "affine": True}, dropout_op=None,
            nonlin=torch.nn.LeakyReLU, nonlin_kwargs={"inplace": True}, deep_supervision=False,
        )
        clone.load_state_dict(torch.load(stream.name, map_location="cpu", weights_only=True), strict=True)
    return {"output_shape": list(output.shape), "loss": float(loss.detach()), "checkpoint_roundtrip": True}


def icl_bam(device: torch.device, size: int) -> dict:
    torch.manual_seed(19)
    model = PanCancerMedverse(
        inner_channels=(4, 8, 16, 32), conv_layers_per_stage=1,
        img_size=size, in_channels=2, use_ccti=False,
    ).to(device)
    assert len(model.target_decoder.ccti_blocks) == 0
    calls = []
    hooks = []
    for block in list(model.target_decoder.enc_blocks) + list(model.target_decoder.dec_blocks):
        hooks.append(block.cross_attention.register_forward_hook(
            lambda _module, _inputs, kwargs, output: calls.append({
                "target_shape": list(kwargs["target_feat"].shape),
                "context_shape": list(kwargs["context_feat"].shape),
                "output_shape": list(output.shape),
            }), with_kwargs=True,
        ))
    target = torch.randn(1, 2, size, size, size, device=device)
    truth = (torch.rand(1, 1, size, size, size, device=device) > 0.9).float()
    per_k = {}
    reference = None
    for k in (1, 3):
        context_in = torch.randn(1, k, 2, size, size, size, device=device)
        context_out = torch.zeros(1, k, 1, size, size, size, device=device)
        context_out[..., size // 4: size // 2, size // 4: size // 2, size // 4: size // 2] = 1
        output = model(target, context_in, context_out, l=min(k, 2))
        per_k[str(k)] = list(output.shape)
        if k == 1:
            reference = output.detach()
            loss = F.binary_cross_entropy_with_logits(output, truth)
            loss.backward()
    with torch.no_grad():
        empty_context = torch.zeros(1, 1, 1, size, size, size, device=device)
        changed = model(target, context_in[:, :1], empty_context, l=1)
        context_effect = float((reference - changed).abs().mean())
    for hook in hooks:
        hook.remove()
    if not calls or context_effect <= 1e-8:
        raise RuntimeError("BAM context-use proof failed: no cross-attention call or output effect")
    with tempfile.NamedTemporaryFile(suffix=".pt") as stream:
        torch.save(model.state_dict(), stream.name)
        clone = PanCancerMedverse(
            inner_channels=(4, 8, 16, 32), conv_layers_per_stage=1,
            img_size=size, in_channels=2, use_ccti=False,
        )
        clone.load_state_dict(torch.load(stream.name, map_location="cpu", weights_only=True), strict=True)
    return {
        "output_shapes_by_k": per_k, "loss": float(loss.detach()),
        "bam_cross_attention_calls": len(calls), "bam_context_effect_mean_abs": context_effect,
        "ccti_disabled": True, "empty_context_tested": True, "checkpoint_roundtrip": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA smoke requested but unavailable")
    report = {
        "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "no_icl_nnunet_style": baseline(device, args.image_size),
        "icl_medverse_original_bam": icl_bam(device, args.image_size),
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

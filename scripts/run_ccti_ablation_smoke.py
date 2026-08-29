"""Run all channel-selection controls on the same synthetic feature pair."""

import math

import torch

from medverse.models.nn.ccti import ChannelSelectiveContextTargetInteraction


def main() -> None:
    torch.manual_seed(17)
    target = torch.randn(1, 16, 4, 4, 4)
    context = torch.randn(1, 2, 16, 4, 4, 4)
    print("mode\tselected\ttrainable_params\tmean_abs_target_delta")
    for mode in ("none", "all", "random", "learned"):
        torch.manual_seed(17)
        block = ChannelSelectiveContextTargetInteraction(16, channel_ratio=0.25, mode=mode)
        target_out, _ = block(target, context, retrieval_similarity=torch.tensor([0.7]))
        selected = 0 if mode == "none" else block.last_routing["indices"].shape[1]
        params = sum(parameter.numel() for parameter in block.parameters() if parameter.requires_grad)
        delta = (target_out - target).abs().mean().item()
        print(f"{mode}\t{selected}\t{params}\t{delta:.8f}")

    assert math.ceil(16 * 0.25) == 4


if __name__ == "__main__":
    main()

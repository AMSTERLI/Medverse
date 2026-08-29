"""CPU-friendly shape/gradient smoke test for the standalone CCTI block."""

import argparse

import torch

from medverse.models.nn.ccti import ChannelSelectiveContextTargetInteraction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("none", "all", "random", "learned"), default="learned")
    parser.add_argument("--ratio", type=float, default=0.25)
    args = parser.parse_args()

    torch.manual_seed(17)
    target = torch.randn(2, 16, 4, 4, 4, requires_grad=True)
    context = torch.randn(2, 2, 16, 4, 4, 4, requires_grad=True)
    block = ChannelSelectiveContextTargetInteraction(
        channels=16,
        channel_ratio=args.ratio,
        mode=args.mode,
        bidirectional=True,
    )
    target_out, context_out = block(target, context, retrieval_similarity=torch.tensor([[0.8], [0.4]]))
    assert target_out.shape == target.shape
    assert context_out.shape == context.shape
    (target_out.mean() + context_out.mean()).backward()

    selected = block.last_routing.get("indices")
    selected_count = 0 if selected is None else selected.shape[1]
    print({"mode": args.mode, "selected_channels": selected_count, "shape": list(target_out.shape)})


if __name__ == "__main__":
    main()

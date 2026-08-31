"""Segmentation-only Medverse variant for pan-cancer 3D experiments."""

from __future__ import annotations

from typing import Sequence

from .Medverse import Medverse


class PanCancerMedverse(Medverse):
    """A binary 3D segmentation model whose task is defined by image-mask context.

    Cancer type, organ, modality and target-region metadata belong in the data
    sampler/retriever.  They are intentionally not accepted as network inputs.
    """

    def __init__(
        self,
        inner_channels: Sequence[int] = (16, 32, 64, 128, 256),
        conv_layers_per_stage: int = 2,
        patch_num: int = 4,
        hidden_size: int = 66,
        img_size: int = 128,
        in_channels: int = 1,
        use_ccti: bool = True,
        ccti_mode: str = "learned",
        ccti_channel_ratio: float = 0.25,
        ccti_bidirectional: bool = True,
        ccti_stage_indices: Sequence[int] = (0, 1, 2),
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=1,
            stages=len(inner_channels),
            dim=3,
            inner_channels=list(inner_channels),
            conv_layers_per_stage=conv_layers_per_stage,
            patch_num=patch_num,
            hidden_size=hidden_size,
            img_size=img_size,
            use_ccti=use_ccti,
            ccti_mode=ccti_mode,
            ccti_channel_ratio=ccti_channel_ratio,
            ccti_bidirectional=ccti_bidirectional,
            ccti_stage_indices=tuple(ccti_stage_indices),
        )

    def routing_diagnostics(self):
        """Return detached channel scores/indices for logging and ablations."""
        return {
            int(stage): block.last_routing
            for stage, block in self.target_decoder.ccti_blocks.items()
            if block.last_routing
        }

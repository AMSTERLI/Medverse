"""Channel-selective context-target interaction for 3D Medverse features.

This module adapts the channel router and bidirectional interaction idea from
BCSI to Medverse's semantic-context/target feature streams.  It deliberately
does not contain BCSI's labelled/unlabelled memory queues.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F


class CancerAwareChannelRouter(nn.Module):
    """Score channels from a target/context pair and select an experiment arm."""

    VALID_MODES = {"none", "all", "random", "learned"}

    def __init__(self, channels: int, channel_ratio: float = 0.25, mode: str = "learned"):
        super().__init__()
        if mode not in self.VALID_MODES:
            raise ValueError(f"mode must be one of {sorted(self.VALID_MODES)}, got {mode!r}")
        if not 0 < channel_ratio <= 1:
            raise ValueError("channel_ratio must be in (0, 1]")

        hidden = max(channels // 4, 8)
        self.channels = channels
        self.channel_ratio = channel_ratio
        self.mode = mode
        self.score = nn.Sequential(
            nn.Linear(2 * channels + 1, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels),
        )

    def forward(
        self,
        target: torch.Tensor,
        context: torch.Tensor,
        retrieval_similarity: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, channels = target.shape[:2]
        spatial_dims = tuple(range(2, target.ndim))
        context_spatial_dims = tuple(range(3, context.ndim))
        target_descriptor = target.mean(dim=spatial_dims)
        context_descriptor = context.mean(dim=(1,) + context_spatial_dims)

        if retrieval_similarity is None:
            similarity = target.new_zeros(batch, 1)
        else:
            similarity = retrieval_similarity.to(device=target.device, dtype=target.dtype)
            similarity = similarity.reshape(batch, -1).mean(dim=1, keepdim=True)

        logits = self.score(torch.cat((target_descriptor, context_descriptor, similarity), dim=1))
        gates = torch.sigmoid(logits)

        if self.mode == "none":
            indices = torch.empty(batch, 0, dtype=torch.long, device=target.device)
        elif self.mode == "all":
            indices = torch.arange(channels, device=target.device).expand(batch, -1)
        else:
            k = max(1, math.ceil(channels * self.channel_ratio))
            ranking = torch.rand_like(logits) if self.mode == "random" else logits
            indices = ranking.topk(k=k, dim=1, sorted=True).indices

        if not indices.numel():
            selected_gates = gates[:, :0]
        elif self.mode == "learned":
            # The continuous gate supplies gradients to the learned router even
            # though the Top-K membership itself is discrete.
            selected_gates = gates.gather(1, indices)
        else:
            # Keep all-channel and random-channel controls free of a learned
            # amplitude gate so they isolate the selection strategy.
            selected_gates = torch.ones_like(indices, dtype=target.dtype)
        return indices, selected_gates, logits


class ChannelSelectiveContextTargetInteraction(nn.Module):
    """Efficient bidirectional interaction over selected channel tokens.

    Spatial maps are pooled into small channel tokens before attention.  Only
    selected channels are gathered and scattered, while BAM remains responsible
    for fine spatial matching in the following Medverse decoder block.
    """

    def __init__(
        self,
        channels: int,
        channel_ratio: float = 0.25,
        mode: str = "learned",
        bidirectional: bool = True,
        spatial_dims: int = 3,
        pool_size: int = 2,
    ):
        super().__init__()
        if spatial_dims not in (2, 3):
            raise ValueError("spatial_dims must be 2 or 3")
        self.router = CancerAwareChannelRouter(channels, channel_ratio, mode)
        self.bidirectional = bidirectional
        self.spatial_dims = spatial_dims
        self.pool_size = pool_size
        token_dim = pool_size ** spatial_dims
        self.target_to_context = nn.MultiheadAttention(token_dim, num_heads=1, batch_first=True)
        self.context_to_target = nn.MultiheadAttention(token_dim, num_heads=1, batch_first=True)
        self.target_norm = nn.LayerNorm(token_dim)
        self.context_norm = nn.LayerNorm(token_dim)
        # Small non-zero scales preserve pretrained behaviour while allowing
        # gradients to reach the interaction weights on the first step.
        self.target_scale = nn.Parameter(torch.tensor(1e-3))
        self.context_scale = nn.Parameter(torch.tensor(1e-3))
        self.last_routing: Dict[str, torch.Tensor] = {}

    def _gather(self, feature: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        index = indices.view(indices.shape + (1,) * (feature.ndim - 2))
        return feature.gather(1, index.expand(-1, -1, *feature.shape[2:]))

    def _pool_tokens(self, feature: torch.Tensor) -> torch.Tensor:
        output_size = (self.pool_size,) * self.spatial_dims
        if self.spatial_dims == 3:
            pooled = F.adaptive_avg_pool3d(feature, output_size)
        else:
            pooled = F.adaptive_avg_pool2d(feature, output_size)
        return pooled.flatten(start_dim=2)

    def _expand_tokens(self, tokens: torch.Tensor, spatial_shape: Tuple[int, ...]) -> torch.Tensor:
        pooled_shape = (self.pool_size,) * self.spatial_dims
        update = tokens.reshape(tokens.shape[0], tokens.shape[1], *pooled_shape)
        mode = "trilinear" if self.spatial_dims == 3 else "bilinear"
        return F.interpolate(update, size=spatial_shape, mode=mode, align_corners=False)

    @staticmethod
    def _scatter_add(base: torch.Tensor, indices: torch.Tensor, update: torch.Tensor) -> torch.Tensor:
        index = indices.view(indices.shape + (1,) * (base.ndim - 2))
        return base.scatter_add(1, index.expand_as(update), update)

    def forward(
        self,
        target: torch.Tensor,
        context: torch.Tensor,
        retrieval_similarity: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if context is None or self.router.mode == "none":
            return target, context
        if target.ndim != self.spatial_dims + 2 or context.ndim != self.spatial_dims + 3:
            raise ValueError(
                f"expected target/context ranks {self.spatial_dims + 2}/{self.spatial_dims + 3}, "
                f"got {target.ndim}/{context.ndim}"
            )

        indices, selected_gates, logits = self.router(target, context, retrieval_similarity)
        selected_target = self._gather(target, indices)
        context_mean = context.mean(dim=1)
        selected_context = self._gather(context_mean, indices)
        target_tokens = self.target_norm(self._pool_tokens(selected_target))
        context_tokens = self.context_norm(self._pool_tokens(selected_context))

        if self.bidirectional:
            context_delta, _ = self.target_to_context(
                query=context_tokens, key=target_tokens, value=target_tokens, need_weights=False
            )
            enhanced_context_tokens = context_tokens + context_delta
        else:
            context_delta = torch.zeros_like(context_tokens)
            enhanced_context_tokens = context_tokens

        target_delta, _ = self.context_to_target(
            query=target_tokens,
            key=enhanced_context_tokens,
            value=enhanced_context_tokens,
            need_weights=False,
        )
        gate_shape = selected_gates.shape + (1,)
        target_delta = target_delta * selected_gates.view(gate_shape)
        context_delta = context_delta * selected_gates.view(gate_shape)

        target_update = self._expand_tokens(target_delta, tuple(target.shape[2:]))
        target = self._scatter_add(target, indices, self.target_scale * target_update)

        if self.bidirectional:
            batch, context_count, channels = context.shape[:3]
            context_update = self._expand_tokens(context_delta, tuple(context.shape[3:]))
            context_update = context_update.unsqueeze(1).expand(-1, context_count, -1, *context.shape[3:])
            expanded_indices = indices[:, None].expand(-1, context_count, -1).reshape(-1, indices.shape[1])
            context_flat = context.reshape(batch * context_count, channels, *context.shape[3:])
            update_flat = context_update.reshape(-1, context_update.shape[2], *context_update.shape[3:])
            context = self._scatter_add(context_flat, expanded_indices, self.context_scale * update_flat)
            context = context.reshape(batch, context_count, channels, *context.shape[2:])

        self.last_routing = {
            "indices": indices.detach(),
            "scores": torch.sigmoid(logits).detach(),
            "selected_gates": selected_gates.detach(),
        }
        return target, context

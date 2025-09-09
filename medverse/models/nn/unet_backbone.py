from typing import *
from .vmap import Vmap, vmap
import torch
import torch.nn as nn
import math
from collections.abc import Iterable
from pydantic import validate_arguments
from dataclasses import dataclass, field
from .layers import ConvBlock_target_encoder, ConvBlock_context_c2t,DefaultUnetOutputBlock,\
        ConvBlock_context_t2c, UnetDownsampleAndCreateShortcutBlock, UnetUpsampleAndConcatShortcutBlock,DefaultUnetStageBlock

from einops import rearrange

@dataclass(eq=False, repr=False)
class TargetEncoder(nn.Module):
    """
    Customizable U-Net class. Implement individual stages by passing an 
    implementation of the DefaultUnetStageBlock and DefaultUnetOutputBlock classes.
    This class takes care of channels, shortcuts, and up/downsampling.
    This class uses exclusively 1x1 convolutions without activation functions for channel mapping.

    Structure, channels, size

        in_channels, size H
    Embedding
        inner_channels[0], size H
    UnetStageBlock
        inner_channels[0], size H
    UnetDownsampleAndCreateShortcutBlock
        inner_channels[1], size H/2
    ...
        inner_channels[1], size H/2
    UnetUpsampleAndConcatShortcutBlock
        inner_channels[0], size H
    UnetStageBlock
        inner_channels[0], size H
    UnetOutputBlock
        out_channels, size H


    Args:
        nn (_type_): _description_
    """
    stages: int
    in_channels: int
    out_channels: int
    inner_channels: Union[int, List[int]]
    kwargs: Dict[str, Any] = field(default_factory=dict)
    dim: Literal[2, 3] = 3
    unet_block_cls_t2c: nn.Module = DefaultUnetStageBlock
    context_filled: bool = True


    def __post_init__(self):
        super().__init__()

        self.inner_channels = self.parse_channels(self.inner_channels)
        assert len(self.inner_channels) == self.stages
        self._build()

    @validate_arguments
    def parse_channels(self, inner_channels: Union[int, List[int]]) -> List[int]:
        if isinstance(inner_channels, int):
            # single int given. Expand to Iterable over stages.
            return [inner_channels] * self.stages
        elif isinstance(inner_channels, Iterable):
            if len(inner_channels) == 1:
                # single int given as Iterable. Expand to Iterable over stages.
                return inner_channels * self.stages
            else:
                return inner_channels

    def _build(self):
        conv_fn = getattr(nn, f"Conv{self.dim}d")
        self.target_embedding = conv_fn(in_channels=self.in_channels,
                                        out_channels=self.inner_channels[0],
                                        kernel_size=1,
                                        padding='same')

        self.enc_blocks = nn.ModuleList()
        self.downsample_blocks = nn.ModuleList()


        for i in range(self.stages):
            if i < self.stages - 1:
                self.enc_blocks.append(
                    self.unet_block_cls_t2c(channels=self.inner_channels[i],
                                        dim=self.dim,
                                        kwargs=self.kwargs)
                )
                
                self.downsample_blocks.append(
                    UnetDownsampleAndCreateShortcutBlock(
                        in_channels=self.inner_channels[i],
                        out_channels=self.inner_channels[i+1],
                        dim=self.dim,
                        context_filled=False,
                    )
                )

                
    def forward(self,
                target_in: torch.Tensor,
                ) -> torch.Tensor:
        """Performs the forward pass

        Args:
            target_in (torch.Tensor): Target input, shape BxCinxHxW or BxCinxHxWxD

        Returns:
            torch.Tensor: Target feature list , shape [BxCoutxHxW] or [BxCoutxHxWxD]
            
        """


        # apply input embeddings
        target = self.target_embedding(target_in)  # BCHWL

        # feed through encoder
        shortcuts = []
        for i in range(self.stages):
            if i < self.stages - 1:
                # run block
                target = self.enc_blocks[i](target)
                
                # downsample
                _, target, shortcut = self.downsample_blocks[i](None, target)
                shortcuts.append(shortcut)
                

        # apply output block
        return shortcuts, target



@dataclass(eq=False, repr=False)
class ContextUNet(nn.Module):
    """
    Structure, channels, size

        in_channels, size H
    Embedding
        inner_channels[0], size H
    UnetStageBlock
        inner_channels[0], size H
    UnetDownsampleAndCreateShortcutBlock
        inner_channels[1], size H/2
    ...
        inner_channels[1], size H/2
    UnetUpsampleAndConcatShortcutBlock
        inner_channels[0], size H
    UnetStageBlock
        inner_channels[0], size H
    UnetOutputBlock
        out_channels, size H


    Args:
        nn (_type_): _description_
    """
    stages: int
    in_channels: int
    out_channels: int
    inner_channels: Union[int, List[int]]
    kwargs: Dict[str, Any] = field(default_factory=dict)
    dim: Literal[2, 3] = 2
    unet_block_cls_t2c: nn.Module = DefaultUnetStageBlock
    unet_block_cls_c2t: nn.Module = DefaultUnetStageBlock
    context_filled: bool = True
    image_size: int = 128
    attention_dim: int = 66
    patch_num: int = 4

    def __post_init__(self):
        super().__init__()

        self.inner_channels = self.parse_channels(self.inner_channels)
        assert len(self.inner_channels) == self.stages
        self._build()

    @validate_arguments
    def parse_channels(self, inner_channels: Union[int, List[int]]) -> List[int]:
        if isinstance(inner_channels, int):
            # single int given. Expand to Iterable over stages.
            return [inner_channels] * self.stages
        elif isinstance(inner_channels, Iterable):
            if len(inner_channels) == 1:
                # single int given as Iterable. Expand to Iterable over stages.
                return inner_channels * self.stages
            else:
                return inner_channels

    def _build(self):
        conv_fn = getattr(nn, f"Conv{self.dim}d")
        self.context_embedding = Vmap(conv_fn(in_channels=self.in_channels+self.out_channels,
                                              out_channels=self.inner_channels[0],
                                              kernel_size=1,
                                              padding='same')
                                      )

        self.enc_blocks = nn.ModuleList()
        self.downsample_blocks = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        self.upsample_blocks = nn.ModuleList()

        for i in range(self.stages):
            if i < self.stages - 1:
                self.enc_blocks.append(
                    self.unet_block_cls_t2c(channels=self.inner_channels[i],
                                        dim=self.dim,
                                        feature_size=self.image_size / (2**i),
                                        attention_dim=self.attention_dim,
                                        attention_patch_size=self.patch_num,
                                        kwargs=self.kwargs)
                )
            else:
                self.enc_blocks.append(
                    self.unet_block_cls_c2t(channels=self.inner_channels[i],
                                        dim=self.dim,
                                        kwargs=self.kwargs)
                )
                
            if i < self.stages - 1:
                self.dec_blocks.append(
                    self.unet_block_cls_c2t(channels=self.inner_channels[-(i+2)],
                                        dim=self.dim,
                                        kwargs=self.kwargs)
                )
                
                self.upsample_blocks.append(
                    UnetUpsampleAndConcatShortcutBlock(
                        in_channels=self.inner_channels[-(i+1)],
                        in_shortcut_channels=self.inner_channels[-(i+2)],
                        out_channels=self.inner_channels[-(i+2)],
                        dim=self.dim,
                        context_filled=True,
                        target_filled=False,
                    )
                )
            if i < self.stages - 1:
                self.downsample_blocks.append(
                    UnetDownsampleAndCreateShortcutBlock(
                        in_channels=self.inner_channels[i],
                        out_channels=self.inner_channels[i+1],
                        dim=self.dim,
                        context_filled=True,
                        target_filled=False,
                    )
                )

    def forward(self,
                context_in: torch.Tensor,
                context_out: torch.Tensor,
                target_features: torch.Tensor,
                pos_embed_q: Optional[torch.Tensor] = None,
                pos_embed_k: Optional[torch.Tensor] = None,
                image_context_embedding: Optional[torch.Tensor] = None,
                ) -> torch.Tensor:
        """Performs the forward pass

        Args:
            context_in (torch.Tensor): Context input, shape BxCinxHxW or BxCinxHxWxD
            context_out (torch.Tensor): Context output, shape BxCoutxHxW or BxCoutxHxWxD
            target_in (torch.Tensor): Target input, shape BxCinxHxW or BxCinxHxWxD
            pos_embed_q (Optional[torch.Tensor]): Query position embedding for context
            pos_embed_k (Optional[torch.Tensor]): Key position embedding for target

        Returns:
            torch.Tensor: Target output, shape BxCoutxHxW or BxCoutxHxWxD
            
        """
        
        # concat context
        context = torch.cat([context_in, context_out],
                            dim=2)if self.context_filled else context_in  # BLCHW

        # apply input embeddings
        context = self.context_embedding(
            context) if self.context_filled else context  # BCHW1
        
        # reshape context embedding to (BxL,512,66)
        if len(pos_embed_q.shape) == 4:
            pos_embed_q = rearrange(pos_embed_q, 'b l p c -> (b l) p c')

        # feed through encoder
        shortcuts = []
        context_features = []
        
        for i in range(self.stages):
            if i < self.stages - 1:
                # run block
                context = self.enc_blocks[i](context, target_features[i], 
                                             pos_embed_q=pos_embed_q, 
                                             pos_embed_k=pos_embed_k)
                if i==0 and image_context_embedding is not None:
                    context += image_context_embedding.view([1,1,-1]+[1]*self.dim).expand(context.shape).to(context.device)
                # downsample
                context, _, shortcut = self.downsample_blocks[i](context, None)
                shortcuts.append(shortcut)
            else:
                context = self.enc_blocks[i](context)
                # save features
                context_features.append(context)
                
        # feed through decoder
        shortcuts = shortcuts[::-1]
        for i in range(self.stages - 1):
            # upsample and add shortcut
            context, _ = self.upsample_blocks[i](
                context, None, shortcuts[i])
            
            # run block
            context = self.dec_blocks[i](context)
            context_features.append(context)

            
        # apply output block
        return context_features
        



@dataclass(eq=False, repr=False)
class TargetDecoder(nn.Module):
    """
    Structure, channels, size
        in_channels, size H
    Embedding
        inner_channels[0], size H
    UnetStageBlock
        inner_channels[0], size H
    UnetDownsampleAndCreateShortcutBlock
        inner_channels[1], size H/2
    ...
        inner_channels[1], size H/2
    UnetUpsampleAndConcatShortcutBlock
        inner_channels[0], size H
    UnetStageBlock
        inner_channels[0], size H
    UnetOutputBlock
        out_channels, size H

    Args:
        nn (_type_): _description_
    """
    stages: int
    in_channels: int
    out_channels: int
    inner_channels: Union[int, List[int]]
    kwargs: Dict[str, Any] = field(default_factory=dict)
    dim: Literal[2, 3] = 2
    unet_block_cls_c2t: nn.Module = DefaultUnetStageBlock
    output_block_cls: nn.Module = DefaultUnetOutputBlock
    image_size: int = 128
    attention_dim: int = 66
    patch_num: int = 4

    def __post_init__(self):
        super().__init__()

        self.inner_channels = self.parse_channels(self.inner_channels)
        assert len(self.inner_channels) == self.stages
        self._build()

    @validate_arguments
    def parse_channels(self, inner_channels: Union[int, List[int]]) -> List[int]:
        if isinstance(inner_channels, int):
            # single int given. Expand to Iterable over stages.
            return [inner_channels] * self.stages
        elif isinstance(inner_channels, Iterable):
            if len(inner_channels) == 1:
                # single int given as Iterable. Expand to Iterable over stages.
                return inner_channels * self.stages
            else:
                return inner_channels

    def _build(self):
        conv_fn = getattr(nn, f"Conv{self.dim}d")

        self.enc_blocks = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        self.upsample_blocks = nn.ModuleList()
        self.output_block = self.output_block_cls(in_channels=self.inner_channels[0],
                                                  out_channels=self.out_channels,
                                                  dim=self.dim,
                                                  kwargs=self.kwargs)

        for i in range(self.stages):
            if i < self.stages - 1:
                pass
            else:
                self.enc_blocks.append(
                    self.unet_block_cls_c2t(channels=self.inner_channels[i],
                                        dim=self.dim,
                                        feature_size = self.image_size/(2**(self.stages-1)),
                                        attention_dim = self.attention_dim,
                                        attention_patch_size=self.patch_num,
                                        kwargs=self.kwargs)
                )
                
            if i < self.stages - 1:
                # Only use feature interaction at specific stage
                self.dec_blocks.append(
                    self.unet_block_cls_c2t(channels=self.inner_channels[-(i + 2)],
                                        dim=self.dim,
                                        feature_size=self.image_size / (2**(self.stages - 2 - i)),
                                        attention_dim=self.attention_dim,
                                        attention_patch_size=self.patch_num,
                                        kwargs=self.kwargs)
                )
                self.upsample_blocks.append(
                    UnetUpsampleAndConcatShortcutBlock(
                        in_channels=self.inner_channels[-(i+1)],
                        in_shortcut_channels=self.inner_channels[-(i+2)],
                        out_channels=self.inner_channels[-(i+2)],
                        dim=self.dim,
                        context_filled=False,
                        target_filled=True,
                    )
                )

    def forward(self,
                target: torch.Tensor,
                context_features_mean: list,
                shortcuts: list,
                pos_embed_q: Optional[torch.Tensor] = None,
                pos_embed_k: Optional[torch.Tensor] = None,
                ) -> torch.Tensor:
        """Performs the forward pass

        Args:
            target_in (torch.Tensor): Target input, shape BxCinxHxW or BxCinxHxWxD
            context_features_mean (list): List of context features
            shortcuts (list): List of shortcuts
            pos_embed_q (Optional[torch.Tensor]): Query position embedding for target
            pos_embed_k (Optional[torch.Tensor]): Key position embedding for context

        Returns:
            torch.Tensor: Target output, shape BxCoutxHxW or BxCoutxHxWxD
            
        """
        # reshape pos_embed_k to (BxL,512,66)
        if len(pos_embed_k.shape) == 4:
            pos_embed_k = rearrange(pos_embed_k, 'b l p c -> (b l) p c')
            
        # feed through encoder
        num_enc = len(self.enc_blocks)
        for i in range(num_enc):
            # run block
            target = self.enc_blocks[i](context_features_mean[i], target, pos_embed_q=pos_embed_q, pos_embed_k=pos_embed_k)
        # feed through decoder
        shortcuts = shortcuts[::-1]
        for i in range(self.stages - 1):
            # upsample and add shortcut
            _, target = self.upsample_blocks[i](
                None, target, shortcuts[i])

            # run block
            target = self.dec_blocks[i](context_features_mean[i+1], target, pos_embed_q=pos_embed_q, pos_embed_k=pos_embed_k)

        # apply output block
        return self.output_block(target)




if __name__ == '__main__':
    device = 'cuda'
    unet2d = UnetBackbone(dim=2, stages=4, in_channels=2,
                          out_channels=3, inner_channels=32).to(device)
    context_in = torch.rand(7, 8, 2, 64, 64).to(device)
    context_out = torch.rand(7, 8, 3, 64, 64).to(device)
    target_in = torch.rand(7, 2, 64, 64).to(device)
    target_out = unet2d(context_in, context_out, target_in)
    print(unet2d)
    print('2d ok')

    unet3d = UnetBackbone(dim=3, stages=4, in_channels=2,
                          out_channels=3, inner_channels=32).to(device)
    context_in = torch.rand(7, 8, 2, 64, 64, 32).to(device)
    context_out = torch.rand(7, 8, 3, 64, 64, 32).to(device)
    target_in = torch.rand(7, 2, 64, 64, 32).to(device)
    target_out = unet3d(context_in, context_out, target_in)
    print(unet3d)
    print('3d ok')

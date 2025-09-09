from dataclasses import dataclass
from typing import Literal, Optional, Tuple
from pydantic import validate_arguments
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import *
from .vmap import Vmap, vmap
from ..attention_utils.cross_attention import TargetToContextAttention3D, MultiContextSpatialCrossAttention3D
#     dim: Literal[2, 3]
@validate_arguments
@dataclass(eq=False, repr=False)
class ResidualUnit(nn.Module):
    channels: int
    dim: int
    conv_layers: int

    def __post_init__(self):
        super().__init__()
        conv_fn = getattr(nn, f'Conv{self.dim}d')
        layers = []
        for i in range(1, self.conv_layers):
            layers.append(conv_fn(in_channels=self.channels,
                                  out_channels=self.channels,
                                  kernel_size=3,
                                  padding='same'))
            layers.append(nn.GELU())
        layers.append(conv_fn(in_channels=self.channels,
                              out_channels=self.channels,
                              kernel_size=3,
                              padding='same'))
        self.layers = nn.Sequential(*layers)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        residual = self.layers(input)
        return F.gelu(input + residual)

    


@dataclass(eq=False, repr=False)
class DefaultUnetStageBlock(nn.Module):
    channels: int
    kwargs: Optional[Dict[str, Any]]
    dim: Literal[2, 3] = 2
    context_attention: bool = False

    def __post_init__(self):
        super().__init__()

    def forward(self,
                context: torch.Tensor,
                target: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """The forward pass of a unet-stage block.

        Args:
            context (torch.Tensor): the context embedding, shape BxLxCxHxWxD or BxLxCxHxW
            target (torch.Tensor): the target embedding, shape BxCxHxWxL or BxCxHxW

        Returns:
            context, target: the processed tensors, same shape as input.
        """
        return context, target
    
@dataclass(eq=False, repr=False)
class ConvBlock_target_encoder(DefaultUnetStageBlock):

    def __post_init__(self):
        super().__post_init__()
        self.target_conv = ResidualUnit(channels=self.channels,
                                        dim=self.dim,
                                        conv_layers=self.kwargs['conv_layers_per_stage'])

    def forward(self,
                target: torch.Tensor,
                verbose: bool = False
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """The forward pass of a unet-stage block.

        Args:
            target (torch.Tensor): the target embedding, shape BxCxHxWxL or BxCxHxW

        Returns:
            , target: the processed tensors, same shape as input.
        """

        # do single convs on input
        target = self.target_conv(target)  # B,C,...

        return target
    

    
@dataclass(eq=False, repr=False)
class ConvBlock_context_c2t(DefaultUnetStageBlock):

    def __post_init__(self):
        
        super().__post_init__()
        self.context_conv = Vmap(ResidualUnit(channels=self.channels,
                                              dim=self.dim,
                                              conv_layers=self.kwargs['conv_layers_per_stage']))
    def forward(self,
                context: torch.Tensor,
                verbose: bool=False,
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """The forward pass of a unet-stage block.

        Args:
            context (torch.Tensor): the context embedding, shape BxLxCxHxWxD or BxLxCxHxW
        Returns:
            context, target: the processed tensors, same shape as input.
        """
#         print('--'*10,'Decoder pairwise block','--'*10)
        # do single convs on input
        context = self.context_conv(context)  # B,L,C,...     

        # return augmented inputs
        return context
    
@dataclass(eq=False, repr=False)
class ConvBlock_context_t2c(DefaultUnetStageBlock):
    # Add parameters for attention configuration
    attention_num_heads: int = 1
    attention_dim: int = 66 # Dimension for Q/K projection in attention
    attention_patch_size: int = 8
    attention_dropout: float = 0.0
    feature_size: int = 32

    def __post_init__(self):
        super().__post_init__()
        # Keep the initial context convolution
        self.context_conv = Vmap(ResidualUnit(channels=self.channels,
                                              dim=self.dim,
                                              conv_layers=self.kwargs['conv_layers_per_stage']))

        conv_fn = getattr(nn, f'Conv{self.dim}d')



        # Instantiate TargetToContextAttention3D
        # Calculate input_dim based on channels and patch volume
        # Assuming split_into_patches flattens patch spatial dims into channels
        attention_input_dim = int(self.channels)
        self.cross_attention = TargetToContextAttention3D(
            input_dim=attention_input_dim,
            num_heads=self.attention_num_heads,
            attention_dim=self.attention_dim, # Q, K projection dim
            patch_size=self.attention_patch_size,
            expanded_channel =  int(self.channels* ((self.feature_size // self.attention_patch_size )** self.dim)),
            dropout=self.attention_dropout
        )

        # Instantiate the final 1x1 convolution, Vmapped for the L dimension
        self.final_conv = Vmap(conv_fn(in_channels=2*self.channels,
                                        out_channels=self.channels,
                                        kernel_size=1,
                                        padding='same'))


    def forward(self,
                context: torch.Tensor,
                target: torch.Tensor,
                # Add positional embeddings as arguments, default to None
                pos_embed_q: Optional[torch.Tensor] = None,
                pos_embed_k: Optional[torch.Tensor] = None,
                verbose: bool = False
                ) -> torch.Tensor: # Return only context
        """The forward pass of a unet-stage block using Context-to-Target attention.

        Args:
            context (torch.Tensor): the context embedding, shape BxLxCxHxWxD or BxLxCxHxW
            target (torch.Tensor): the target embedding, shape BxCxHxWxD or BxCxHxW
            pos_embed_q (Optional[torch.Tensor]): Positional embedding for context queries.
                                        If None, defaults to zeros. Shape (B*L, N, attention_dim)
                                        or broadcastable. N=patch_volume. Defaults to None.
            pos_embed_k (Optional[torch.Tensor]): Positional embedding for target keys.
                                        If None, defaults to zeros. Shape (B, N, attention_dim)
                                        or broadcastable. N=patch_volume. Defaults to None.
            verbose (bool): If True, print intermediate shapes/values.

        Returns:
            context: the processed tensors. Context is updated, target is unchanged.
        """
        # Process context features first
        context_processed = self.context_conv(context)  # B,L,C,...

        n_patches = self.attention_patch_size ** self.dim # N = patch_volume
        if verbose: print(f'----- cross block input: pos_embed_q: {pos_embed_q.shape}')
        if verbose: print(f'----- cross block input: pos_embed_k: {pos_embed_k.shape}')
            
        # Default positional embeddings to zeros if not provided
        if pos_embed_q is None:
            pos_embed_q = torch.zeros(1, n_patches, self.attention_dim,
                                      device=context.device, dtype=context.dtype)
        if pos_embed_k is None:
            pos_embed_k = torch.zeros(1, n_patches, self.attention_dim,
                                      device=target.device, dtype=target.dtype)

        # Perform Context-to-Target attention
        # context_processed provides Q, target provides K, V
        if verbose: print(f'target: {target.shape}')
        if verbose: print(f'context_processed: {context_processed.shape}')
        if verbose: print(f'----- cross block process: pos_embed_q: {pos_embed_q.shape}')
        if verbose: print(f'----- cross block process: pos_embed_k: {pos_embed_k.shape}')
        
        attn_output = self.cross_attention(
            target_feat=target,         # Shape: B, C, H, W, D (Provides K, V)
            context_feat=context_processed, # Shape: B, L, C, H, W, D (Provides Q)
            pos_embed_q=pos_embed_q,    # For context (Query)
            pos_embed_k=pos_embed_k     # For target (Key)
        ) # Output shape: B, L, C, H, W, D

        if verbose: print(f'Attention output shape: {attn_output.shape}')

        # Apply final 1x1 convolution to the attention output
        attn_output_concat = torch.concat(
            [context_processed, attn_output], dim=2)  # B,L,2C,...
        context_update = self.final_conv(attn_output_concat) # B, L, C, ...

        if verbose: print(f'Context update shape after final conv: {context_update.shape}')
        # if verbose: print('context_update[0,0,16,4,4,4]:',context_update[0,0,16,4,4,4])
        # if verbose: print('context_update[0,-1,16,4,4,4]:',context_update[0,-1,16,4,4,4])

        # Residual connection and activation (using the initially processed context)
        context = F.gelu(context_processed + context_update)

        # return augmented inputs
        return context

@dataclass(eq=False, repr=False)
class PairwiseConvAvgModelBlock_c2t(DefaultUnetStageBlock):
    # Add parameters for attention configuration from ConvBlock_context_t2c
    attention_num_heads: int = 1
    attention_dim: int = 66 # Dimension for Q/K projection in attention
    attention_patch_size: int = 8
    attention_dropout: float = 0.0
    feature_size: int = 32 # Assuming this corresponds to H, W, D dims

    def __post_init__(self):
        super().__post_init__()
        # Keep the initial target convolution
        self.target_conv = ResidualUnit(channels=self.channels,
                                        dim=self.dim,
                                        conv_layers=self.kwargs['conv_layers_per_stage'])
        
        conv_fn = getattr(nn, f'Conv{self.dim}d')
        
        # Instantiate MultiContextSpatialCrossAttention3D
        attention_input_dim = int(self.channels) # * ((self.feature_size // self.attention_patch_size )** self.dim)

        # Validate dimensions for attention heads
        if attention_input_dim % self.attention_num_heads != 0:
             pass # Allow potential mismatch if input_dim calculation needs review
             # raise ValueError(f"Attention input_dim ({attention_input_dim}) must be divisible by attention_num_heads ({self.attention_num_heads}) for multi-head value handling.")
        if self.attention_dim % self.attention_num_heads != 0:
            raise ValueError(f"Attention attention_dim ({self.attention_dim}) must be divisible by attention_num_heads ({self.attention_num_heads})")

        self.cross_attention = MultiContextSpatialCrossAttention3D(
            input_dim=attention_input_dim,
            num_heads=self.attention_num_heads,
            attention_dim=self.attention_dim, # Q, K projection dim
            patch_size=self.attention_patch_size,
            expanded_channel =  int(self.channels* ((self.feature_size // self.attention_patch_size )** self.dim)),
            dropout=self.attention_dropout
        )

        # Instantiate the final 1x1 convolution (non-Vmapped)
        self.final_conv = conv_fn(in_channels=2*self.channels,
                                   out_channels=self.channels,
                                   kernel_size=1,
                                   padding='same')

    def forward(self,
                context: torch.Tensor, # Renamed from context_mean
                target: torch.Tensor,
                # Add positional embeddings as arguments, default to None
                pos_embed_q: Optional[torch.Tensor] = None, # For target query
                pos_embed_k: Optional[torch.Tensor] = None, # For context key/value
                verbose: bool = False,
                ) -> torch.Tensor: # Return only target
        """The forward pass using Target-to-Multi-Context attention.

        Args:
            context (torch.Tensor): the context embedding, shape BxLxCxHxWxD or BxLxCxHxW
            target (torch.Tensor): the target embedding, shape BxCxHxWxD or BxCxHxW
            pos_embed_q (Optional[torch.Tensor]): Positional embedding for target queries.
                                        If None, defaults to zeros. Shape (B, N, attention_dim)
                                        or broadcastable. N=patch_volume. Defaults to None.
            pos_embed_k (Optional[torch.Tensor]): Positional embedding for context keys/values.
                                        If None, defaults to zeros. Shape (B*L, N, attention_dim)
                                        or broadcastable. N=patch_volume. Defaults to None.
            verbose (bool): If True, print intermediate shapes/values.

        Returns:
            target: the processed target tensor.
        """
        # Process target features first
        target_processed = self.target_conv(target)  # B,C,...
            
        b, c = target.shape[0], target.shape[2]
        n_patches = self.attention_patch_size ** self.dim # N = patch_volume
        attn_emb_dim = self.attention_dim

        # Default positional embeddings to zeros if not provided
        # Target Query Positional Embedding (Broadcastable B dim)
        if pos_embed_q is None:
            pos_embed_q = torch.zeros(1, n_patches, attn_emb_dim,
                                      device=target.device, dtype=target.dtype)
        # Context Key/Value Positional Embedding (Broadcastable B*L dim)
        if pos_embed_k is None:
            pos_embed_k = torch.zeros(1, n_patches, attn_emb_dim,
                                      device=context.device, dtype=context.dtype)
            
        # Perform Target-to-Multi-Context attention
        # target_processed provides Q, context provides K, V
        if verbose: print(f'----- Traget decoder Target_processed shape: {target_processed.shape}')
        if verbose: print(f'Context shape: {context.shape}')
        if verbose: print(f'Pos Embed Q shape: {pos_embed_q.shape}')
        if verbose: print(f'Pos Embed K shape: {pos_embed_k.shape}')

        attn_output = self.cross_attention(
            target_feat=target_processed, # Shape: B, C, H, W, D (Provides Q)
            context_feat=context,         # Shape: B, L, C, H, W, D (Provides K, V)
            pos_embed_q=pos_embed_q,    # For target (Query)
            pos_embed_k=pos_embed_k     # For context (Key/Value)
        ) # Output shape: B, C, H, W, D
        
        if verbose: print(f'Attention output shape: {attn_output.shape}')
        
        # Apply final 1x1 convolution to the attention output
        attn_output_concat = torch.concat([attn_output, target_processed], dim=1) # [B,C,H,W,D]
        target_update = self.final_conv(attn_output_concat) # B, C, ...
        if verbose: print(f'Target update shape after final conv: {target_update.shape}')
                
        # Residual connection and activation (using the initially processed target)
        target = F.gelu(target_processed + target_update)

        # return updated target only
        return target

@dataclass(eq=False, repr=False)
class DefaultUnetOutputBlock(nn.Module):
    """
    U-net output block. Reduces channels to out_channels. Can be used to apply additional smoothing.
    """

    in_channels: int
    out_channels: int
    kwargs: Optional[Dict[str, Any]]
    dim: Literal[2, 3] = 2

    def __post_init__(self):
        super().__init__()

        conv_fn = getattr(nn, f"Conv{self.dim}d")

        self.block = nn.Sequential(
            conv_fn(in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    kernel_size=1,
                    padding='same')
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input (torch.Tensor): input, shape BxCinxHxWxL or BxCinxHxW

        Returns:
            torch.Tensor: output, shape BxCinxHxWxL or BxCinxHxW
        """
        return self.block(input)
    

@dataclass(eq=False, repr=False)
class UnetDownsampleAndCreateShortcutBlock(nn.Module):
    in_channels: int
    out_channels: int
    dim: Literal[2, 3]
    context_filled: bool = True
    target_filled: bool = True   

    def __post_init__(self):
        super().__init__()
        self.needs_channel_asjustment = self.in_channels != self.out_channels
        if self.needs_channel_asjustment:
            conv_fn = getattr(nn, f"Conv{self.dim}d")
            self.context_linear_layer = Vmap(conv_fn(in_channels=self.in_channels,
                                                     out_channels=self.out_channels,
                                                     kernel_size=4,
                                                     stride=2,
                                                     padding=1)
                                             ) if self.context_filled else None
            self.target_linear_layer = conv_fn(in_channels=self.in_channels,
                                               out_channels=self.out_channels,
                                               kernel_size=4,
                                               stride=2,
                                               padding=1) if self.target_filled else None
        else:
            pool_fn = getattr(nn, f"MaxPool{self.dim}d")
            self.context_pooling_layer = Vmap(
                pool_fn(kernel_size=2)) if self.context_filled else None
            self.target_pooling_layer = pool_fn(kernel_size=2) if self.target_filled else None

    def forward(self,
                context: torch.Tensor,
                target: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        # make shortcut
        shortcut = (context, target)
        # downsample
        if self.needs_channel_asjustment:
            context = self.context_linear_layer(context) if self.context_filled else context
            target = self.target_linear_layer(target) if self.target_filled else target
        else:
            context = self.context_pooling_layer(context) if self.context_filled else context
            target = self.target_pooling_layer(target) if self.target_filled else target
            
        return context, target, shortcut
    
    

@dataclass(eq=False, repr=False)
class UnetUpsampleAndConcatShortcutBlock(nn.Module):
    in_channels: int
    in_shortcut_channels: int
    out_channels: int
    dim: Literal[2, 3]
    context_filled: bool = True
    target_filled: bool = True   

    def __post_init__(self):
        super().__init__()
        self.upsampling_layer = nn.Upsample(
            scale_factor=2, mode='trilinear' if self.dim == 3 else 'bilinear', align_corners=False)

        conv_fn = getattr(nn, f"Conv{self.dim}d")
        self.context_conv_layer = Vmap(conv_fn(in_channels=self.in_channels + self.in_shortcut_channels,
                                               out_channels=self.out_channels,
                                               kernel_size=1,
                                               padding='same')
                                       ) if self.context_filled else None
        self.target_conv_layer = conv_fn(in_channels=self.in_channels + self.in_shortcut_channels,
                                         out_channels=self.out_channels,
                                         kernel_size=1,
                                         padding='same') if self.target_filled else None

    def forward(self,
                context: torch.Tensor,
                target: torch.Tensor,
                shortcut: Tuple[torch.Tensor, torch.Tensor]
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        # upsample
        context = vmap(self.upsampling_layer,
                       context) if self.context_filled else context
        target = self.upsampling_layer(target) if self.target_filled else target

        # concat with shortcut
        ctx_short, tgt_short = shortcut
#         if self.context_filled: 
#             print('ctx_short:',ctx_short.shape)
#             print('context:',context.shape)
#         if self.target_filled: 
#             print('tgt_short:',tgt_short.shape)
#             print('target:',target.shape)
        # B L C ...
        context = torch.cat([context, ctx_short],
                            dim=2) if self.context_filled else context
        target = torch.cat([target, tgt_short], dim=1) if self.target_filled else target  # B C ...

        # reduce dim
        context = self.context_conv_layer(
            context) if self.context_filled else context
        target = self.target_conv_layer(target) if self.target_filled else target

        return context, target

@dataclass(eq=False, repr=False)
class PairwiseConvAvgModelOutput(DefaultUnetOutputBlock):
    """
    U-net output block. Reduces channels to out_channels. Can be used to apply additional smoothing.
    """

    def __post_init__(self):
        super().__post_init__()
        conv_fn = getattr(nn, f"Conv{self.dim}d")

        self.block = nn.Sequential(
            ResidualUnit(channels=self.in_channels,
                         dim=self.dim,
                         conv_layers=self.kwargs['conv_layers_per_stage']),
            conv_fn(in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    kernel_size=1,
                    padding='same')
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input (torch.Tensor): input, shape BxCinxHxWxL or BxCinxHxW

        Returns:
            torch.Tensor: output, shape BxCinxHxWxL or BxCinxHxW
        """
        return self.block(input)
    
    
    
'''
GPT o1 version
'''
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from .vmap import Vmap
from .layers import ResidualUnit

class GenericBlock(nn.Module):
    """
    通用积木模块，根据 context_target_mode 实现不同的上下文与目标特征处理逻辑。
    
    参数:
        channels: 通道数
        dim: 2D 或 3D 
        conv_layers_per_stage: 每个stage的卷积层数，用于构建 ResidualUnit
        use_vmap: 是否对 ResidualUnit 使用 Vmap（处理上下文为 BxL 维度的情形）
        context_target_mode: 控制处理逻辑的模式，可选:
            - 'none': 不进行任何处理，直接返回输入
            - 'target_only': 仅对 target 进行 ResidualUnit 处理
            - 'context_only': 仅对 context (BxLxC...) 进行 ResidualUnit 处理（可能使用 Vmap）
            - 'context_target_concat': 对 context 处理后与 target 拼接，再通过1x1卷积融合
            - 'context_target_concat_mean': 对 target 处理后与 context 拼接，通过1x1卷积融合并对 L 维度求平均
    """
    def __init__(self, 
                 channels: int, 
                 dim: int, 
                 conv_layers_per_stage: int, 
                 use_vmap: bool = False, 
                 context_target_mode: str = 'none'):
        super().__init__()
        self.channels = channels
        self.dim = dim
        self.use_vmap = use_vmap
        self.context_target_mode = context_target_mode
        
        # 创建 ResidualUnit
        self.res_unit = ResidualUnit(channels=self.channels, dim=self.dim, conv_layers=conv_layers_per_stage)
        if self.use_vmap:
            self.res_unit = Vmap(self.res_unit)

        conv_fn = getattr(nn, f'Conv{self.dim}d')
        
        # 当需要将context和target拼接时，需要一个1x1卷积来融合它们的特征
        self.combine_conv = None
        if self.context_target_mode in ['context_target_concat', 'context_target_concat_mean']:
            if self.use_vmap:
                self.combine_conv = Vmap(conv_fn(in_channels=2*self.channels,
                                                 out_channels=self.channels,
                                                 kernel_size=1,
                                                 padding='same'))
            else:
                self.combine_conv = conv_fn(in_channels=2*self.channels,
                                            out_channels=self.channels,
                                            kernel_size=1,
                                            padding='same')

    def forward(self, 
                context: Optional[torch.Tensor] = None, 
                target: Optional[torch.Tensor] = None, 
                verbose: bool = False
               ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        
        mode = self.context_target_mode

        if mode == 'none':
            # 不做处理
            return context, target

        elif mode == 'target_only':
            # 仅对target应用ResidualUnit
            target = self.res_unit(target)
            return context, target

        elif mode == 'context_only':
            # 仅对context应用ResidualUnit (可能有L维度, 因此使用vmap)
            context = self.res_unit(context)
            return context, target

        elif mode == 'context_target_concat':
            # 对context先处理，然后与target按通道拼接，再通过1x1卷积融合
            context = self.res_unit(context)
            # target需扩展至与context相同的L维度
            combined = torch.cat([context, target.unsqueeze(1).expand_as(context)], dim=2)
            combined_update = self.combine_conv(combined)
            context = F.gelu(context + combined_update)
            return context, target

        elif mode == 'context_target_concat_mean':
            # 对target处理，然后将context与target拼接，再融合，并对L维求平均
            target = self.res_unit(target)
            combined = torch.cat([context, target.unsqueeze(1).expand_as(context)], dim=2)
            combined_update = self.combine_conv(combined)
            combined_update = combined_update.mean(dim=1, keepdim=False)  # 沿L维平均
            target = F.gelu(target + combined_update)
            return context, target

        else:
            raise ValueError(f"Unknown context_target_mode: {mode}")

if __name__ == '__main__':
    # 利用GenericBlock来替换原先的重复类
    # 例如：
    ConvBlock_target_encoder = partial(GenericBlock,
                                       context_target_mode='target_only',
                                       use_vmap=False)

    ConvBlock_context_c2t = partial(GenericBlock,
                                    context_target_mode='context_only',
                                    use_vmap=True)

    ConvBlock_context_t2c = partial(GenericBlock,
                                    context_target_mode='context_target_concat',
                                    use_vmap=True)

    PairwiseConvAvgModelBlock_c2t = partial(GenericBlock,
                                            context_target_mode='context_target_concat_mean',
                                            use_vmap=False)


# 对于PairwiseConvAvgModelOutput，如需保留特殊逻辑可单独定义或使用相似结构再加一层residual+conv：
# 若希望统一，也可使用类似的逻辑，实现一个GenericOutputBlock
class GenericOutputBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dim: int, conv_layers_per_stage: int):
        super().__init__()
        conv_fn = getattr(nn, f"Conv{dim}d")
        self.block = nn.Sequential(
            ResidualUnit(channels=in_channels, dim=dim, conv_layers=conv_layers_per_stage),
            conv_fn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding='same')
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

if __name__ == '__main__':
    # PairwiseConvAvgModelOutput 可用 GenericOutputBlock 代替
    PairwiseConvAvgModelOutput = GenericOutputBlock

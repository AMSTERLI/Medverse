# attention_utils/cross_attention.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


from einops import rearrange
# Import patch utilities
from .patch_utils import split_into_patches, merge_patches, avg_into_patches



class SpatialCrossAttention3D(nn.Module):
    """
    Performs 3D spatial cross-attention between target and context feature maps
    operating on patches.

    Steps:
    1. Splits features into patches (e.g., 8x8x8).
    2. Flattens patches into sequences (B, N, E), where N=patch_volume.
    3. Adds positional embeddings.
    4. Projects target to Query (Q) and context to Key (K).
    5. Uses the *original* flattened context patches as Value (V).
    6. Computes multi-head scaled dot-product attention.
    7. Reshapes attention output back to patch format.
    8. Merges patches back to the original spatial dimensions.
    """
    def __init__(
        self,
        input_dim: int,
        num_heads: int = 2,
        attention_dim: int = 66, # Dimension for Q, K projections
        patch_size: int = 8,
        dropout: float = 0.0,
        expanded_channel = 0,
    ):
        """
        Args:
            input_dim (int): The embedding dimension (E) of the flattened patches
                             (output channels C_out from split_into_patches).
                             This is C * (H/pH) * (W/pW) * (D/pD).
            num_heads (int): Number of attention heads.
            attention_dim (int): The dimension to project Q and K to. Must be
                                 divisible by num_heads. Defaults to 66.
            patch_size (int): Spatial size of the cubic patches. Defaults to 8.
            dropout (float): Dropout probability for attention weights. Defaults to 0.0.

        Raises:
            ValueError: If dimensions are not divisible by num_heads.
        """
        super().__init__()

        if attention_dim % num_heads != 0:
            raise ValueError(f"attention_dim ({attention_dim}) must be divisible by num_heads ({num_heads})")
        if input_dim % num_heads != 0:
             raise ValueError(f"input_dim ({input_dim}) must be divisible by num_heads ({num_heads}) "
                              f"for multi-head value handling.")

        self.input_dim = input_dim
        self.num_heads = num_heads
        self.attention_dim = attention_dim
        self.patch_size = patch_size
        self.expanded_channel = expanded_channel
        self.head_dim_qk = attention_dim // num_heads
        self.head_dim_v = expanded_channel // num_heads # Value head dimension
        self.scale = self.head_dim_qk ** -0.5
        self.dropout_p = dropout

        self.q_proj = nn.Linear(input_dim, attention_dim, bias=True)
        self.k_proj = nn.Linear(input_dim, attention_dim, bias=True)
        # No V projection as per requirement
        # No output projection as per requirement

        self.patch_volume = patch_size ** 3

    def _separate_heads(self, x: torch.Tensor, is_value: bool = False) -> torch.Tensor:
        """ (B, N, C) -> (B, H, N, C/H) """
        b, n, c = x.shape
        head_dim = self.head_dim_v if is_value else self.head_dim_qk
        num_h = self.num_heads
        # Check if dimensions match expectation
        expected_dim = self.expanded_channel if is_value else self.attention_dim
        if c != expected_dim:
             raise ValueError(f"Input tensor channels {c} does not match expected dimension {expected_dim} for head separation.")

        x = x.reshape(b, n, num_h, head_dim)
        return x.permute(0, 2, 1, 3) # B, H, N, D

    def _recombine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """ (B, H, N, D) -> (B, N, H*D) """
        b, num_h, n, head_dim = x.shape
        # Expected head dim is for Value
        if head_dim != self.head_dim_v:
             raise ValueError(f"Input tensor head dimension {head_dim} does not match expected value head dimension {self.head_dim_v}.")
        if num_h != self.num_heads:
            raise ValueError(f"Input tensor num_heads {num_h} does not match expected num_heads {self.num_heads}.")

        return x.permute(0, 2, 1, 3).reshape(b, n, num_h * head_dim) # B, N, C

    def forward(
        self,
        target_feat: torch.Tensor,
        context_feat: torch.Tensor,
        pos_embed_q: torch.Tensor,
        pos_embed_k: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            target_feat (torch.Tensor): Target feature map (B, C, H, W, D).
            context_feat (torch.Tensor): Context feature map (B, C, H, W, D).
                                         Must have same B, H, W, D as target_feat.
            pos_embed_q (torch.Tensor): Positional embedding for target queries.
                                        Shape (B, N, attention_dim) or broadcastable,
                                        where N=patch_volume.
            pos_embed_k (torch.Tensor): Positional embedding for context keys.
                                        Shape (B, N, attention_dim) or broadcastable.

        Returns:
            torch.Tensor: Output feature map with the same shape as target_feat
                          (B, C, H, W, D).
        """
        # --- 1. Input Validation and Shape Extraction ---
        if target_feat.shape[0] != context_feat.shape[0] or \
           target_feat.shape[2:] != context_feat.shape[2:]:
            raise ValueError("Target and Context features must have the same Batch size and Spatial dimensions (H, W, D)."
                             f" Got Target: {target_feat.shape}, Context: {context_feat.shape}")
        if target_feat.ndim != 5 or context_feat.ndim != 5:
             raise ValueError("Input features must be 5D (B, C, H, W, D).")

        b, c_t, h, w, d = target_feat.shape
        _, c_c, _, _, _ = context_feat.shape # Original context channels might differ
        original_shape = (h, w, d)
        original_channels = c_t # Output will have same channels as target input

        # --- 2. Patch Splitting and Reshaping ---
        target_patches = split_into_patches(target_feat, self.patch_size)
        context_patches = split_into_patches(context_feat, self.patch_size)
        # Shape: (B, C_out, pH, pW, pD)
        b_p, c_out, ph, pw, pd = target_patches.shape
        if c_out != self.input_dim:
            raise ValueError(f"Channel dimension after patching ({c_out}) does not match module's input_dim ({self.input_dim}). "
                             f"Ensure input_dim is calculated correctly: C * (H/pH) * (W/pW) * (D/pD)")

        # Flatten patches into sequences: (B, C_out, pH, pW, pD) -> (B, N, C_out) where N = pH*pW*pD
        target_flat = rearrange(target_patches, 'b c ph pw pd -> b (ph pw pd) c',
                                ph=ph, pw=pw, pd=pd)
        context_flat = rearrange(context_patches, 'b c ph pw pd -> b (ph pw pd) c',
                                 ph=ph, pw=pw, pd=pd)
        # N = target_flat.shape[1]
        # E = target_flat.shape[2]
        # Check if sequence length matches patch volume
        if target_flat.shape[1] != self.patch_volume:
             raise RuntimeError(f"Sequence length {target_flat.shape[1]} does not match expected patch volume {self.patch_volume}")


        # --- 3. Assign Value ---
        # Value is the context sequence *without* pos embedding and *without* projection
        v = context_flat

        # --- 4. Project Q and K and Add Positional Embeddings ---
        # Project first
        q_proj = self.q_proj(target_flat) # (B, N, attention_dim)
        k_proj = self.k_proj(context_flat) # (B, N, attention_dim)

        # Ensure pos embeds have correct shape (B, N, attention_dim)
        if pos_embed_q.shape[-2] != q_proj.shape[-2] or pos_embed_q.shape[-1] != q_proj.shape[-1]:
             raise ValueError(f"pos_embed_q shape {pos_embed_q.shape} incompatible with projected q shape {q_proj.shape}")
        if pos_embed_k.shape[-2] != k_proj.shape[-2] or pos_embed_k.shape[-1] != k_proj.shape[-1]:
            raise ValueError(f"pos_embed_k shape {pos_embed_k.shape} incompatible with projected k shape {k_proj.shape}")

        # Add positional embeddings *after* projection
        q = q_proj + pos_embed_q
        k = k_proj + pos_embed_k

        # --- 5. Multi-Head Attention ---
        q_heads = self._separate_heads(q, is_value=False) # (B, H, N, head_dim_qk)
        k_heads = self._separate_heads(k, is_value=False) # (B, H, N, head_dim_qk)
        v_heads = self._separate_heads(v, is_value=True)  # (B, H, N, head_dim_v)

        # Scaled dot-product attention
        attn_output_heads = F.scaled_dot_product_attention(
            q_heads.contiguous(), k_heads.contiguous(), v_heads.contiguous(),
            dropout_p=self.dropout_p if self.training else 0.0,
            # scale=self.scale # Use scale if needed, PyTorch >= 2.0 handles it internally
        )
        # Shape: (B, H, N, head_dim_v)

        # Recombine heads
        attn_output = self._recombine_heads(attn_output_heads) # (B, N, input_dim)

        # --- 6. Reshape back to Patches and Merge ---
        # Reshape sequence back to patch grid: (B, N, E) -> (B, E, pH, pW, pD)
        # Note: E here is input_dim because V determines output dim
        output_patches_reshaped = rearrange(
            attn_output,
            'b (ph pw pd) c -> b c ph pw pd',
            ph=self.patch_size, pw=self.patch_size, pd=self.patch_size
        )

        # Merge patches back to original spatial dimensions
        output_feat = merge_patches(
            output_patches_reshaped,
            original_channels=original_channels, # Use original target channels
            original_shape=original_shape
        )
        # Shape: (B, C_t, H, W, D)

        return output_feat


# --- New Class Added Below --- (and updated __main__) ---
class MultiContextSpatialCrossAttention3D(SpatialCrossAttention3D):
    """
    Performs 3D spatial cross-attention where the target attends to multiple
    context feature maps simultaneously. Inherits from SpatialCrossAttention3D.

    The context feature map input shape is (B, L, C, H, W, D), where L is
    the number of context sources.
    """
    def forward(
        self,
        target_feat: torch.Tensor,
        context_feat: torch.Tensor, # Shape (B, L, C, H, W, D)
        pos_embed_q: torch.Tensor,
        pos_embed_k: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            target_feat (torch.Tensor): Target feature map (B, C, H, W, D).
            context_feat (torch.Tensor): Context feature maps (B, L, C, H, W, D).
                                         L is the number of contexts. B, C, H, W, D
                                         must match target_feat dimensions.
            pos_embed_q (torch.Tensor): Positional embedding for target queries.
                                        Shape (B, N, attention_dim) or (1, N, attention_dim)
                                        for broadcasting, where N=patch_volume.
            pos_embed_k (torch.Tensor): Positional embedding for context keys.
                                        Shape (B*L, N, attention_dim) or (1, N, attention_dim)
                                        for broadcasting.

        Returns:
            torch.Tensor: Output feature map with the same shape as target_feat
                          (B, C, H, W, D).
        """
        # --- 1. Input Validation and Shape Extraction ---
        if target_feat.ndim != 5:
             raise ValueError(f"Target features must be 5D (B, C, H, W, D), got {target_feat.ndim}D.")
        if context_feat.ndim != 6:
            raise ValueError(f"Context features must be 6D (B, L, C, H, W, D), got {context_feat.ndim}D.")

        b_t, c_t, h, w, d = target_feat.shape
        b_c, l, c_c, h_c, w_c, d_c = context_feat.shape

        if b_t != b_c or c_t != c_c or (h, w, d) != (h_c, w_c, d_c):
             raise ValueError("Target and Context features must have the same B, C, H, W, D dimensions."
                              f" Got Target: {target_feat.shape}, Context: {context_feat.shape}")

        original_shape = (h, w, d)
        original_channels = c_t
        b = b_t # Use b for batch size

        # --- Pre-Step: Reshape Context ---
        # Reshape context from (B, L, C, H, W, D) to (B*L, C, H, W, D)
        context_feat_reshaped = rearrange(context_feat, 'b l c h w d -> (b l) c h w d')

        # --- 2. Patch Splitting and Reshaping (Context reshaped before this) ---
        target_patches = avg_into_patches(target_feat, self.patch_size)
        context_patches = avg_into_patches(context_feat_reshaped, self.patch_size)
        context_patches_split = split_into_patches(context_feat_reshaped, self.patch_size)
        
        # target_patches Shape: (B, C_out, pH, pW, pD)
        # context_patches Shape: (B*L, C_out, pH, pW, pD)

        b_p, c_out, ph, pw, pd = target_patches.shape
        bl_p, c_out_c, _, _, _ = context_patches.shape

        if c_out != self.input_dim:
            raise ValueError(f"Channel dimension after patching ({c_out}) does not match module's input_dim ({self.input_dim}).")
        if bl_p != b * l:
            raise RuntimeError("Context batch dimension after patching doesn't match B*L.")

        # Flatten patches into sequences
        target_flat = rearrange(target_patches, 'b c ph pw pd -> b (ph pw pd) c', ph=ph, pw=pw, pd=pd)
        context_flat = rearrange(context_patches, 'bl c ph pw pd -> bl (ph pw pd) c', ph=ph, pw=pw, pd=pd)
        context_flat_split = rearrange(context_patches_split, 'b c ph pw pd -> b (ph pw pd) c', ph=ph, pw=pw, pd=pd)
        # target_flat Shape: (B, N, E)
        # context_flat Shape: (B*L, N, E) where N=patch_volume, E=input_dim

        n = target_flat.shape[1]
        if n != self.patch_volume:
             raise RuntimeError(f"Sequence length {n} does not match expected patch volume {self.patch_volume}")

        # --- 3. Assign Value ---
        v = context_flat_split # Shape: (B*L, N, E)

        # --- 4. Project Q and K ---
        q_proj = self.q_proj(target_flat) # Shape: (B, N, attention_dim)
        k_proj = self.k_proj(context_flat) # Shape: (B*L, N, attention_dim)

        # --- 4b. Handle Positional Embedding Broadcasting & Addition ---
        # Expand pos_embed_q if needed: (1, N, attn_dim) -> (B, N, attn_dim)
        if pos_embed_q.shape[0] == 1 and b > 1:
            pos_embed_q = pos_embed_q.expand(b, -1, -1)
        elif pos_embed_q.shape[0] != b:
             raise ValueError(f"pos_embed_q batch size {pos_embed_q.shape[0]} cannot broadcast to target batch size {b}")

        # Expand pos_embed_k if needed: (1, N, attn_dim) -> (B*L, N, attn_dim)
        bl = b * l
        if pos_embed_k.shape[0] == 1 and bl > 1:
            pos_embed_k = pos_embed_k.expand(bl, -1, -1)
        elif pos_embed_k.shape[0] != bl:
             raise ValueError(f"pos_embed_k batch size {pos_embed_k.shape[0]} cannot broadcast to context batch size {bl}")

        # Check compatibility after potential expansion
        if pos_embed_q.shape[-2:] != q_proj.shape[-2:]:
             raise ValueError(f"Expanded pos_embed_q shape {pos_embed_q.shape} incompatible with projected q shape {q_proj.shape}")
        if pos_embed_k.shape[-2:] != k_proj.shape[-2:]:
            raise ValueError(f"Expanded pos_embed_k shape {pos_embed_k.shape} incompatible with projected k shape {k_proj.shape}")

        # Add positional embeddings *after* projection
        q = q_proj + pos_embed_q # Shape: (B, N, attention_dim)
        k = k_proj + pos_embed_k # Shape: (B*L, N, attention_dim)

        # --- 4c. Reshape K and V for Attention Calculation ---
        # Reshape K from (B*L, N, attention_dim) -> (B, L*N, attention_dim)
        k_reshaped = rearrange(k, '(b l) n d -> b (l n) d', b=b, l=l)
        # Reshape V from (B*L, N, E) -> (B, L*N, E)
        v_reshaped = rearrange(v, '(b l) n e -> b (l n) e', b=b, l=l)
        

        # --- 5. Multi-Head Attention ---
        q_heads = self._separate_heads(q, is_value=False)       # Shape: (B, H, N, head_dim_qk)
        k_heads = self._separate_heads(k_reshaped, is_value=False) # Shape: (B, H, L*N, head_dim_qk)
        v_heads = self._separate_heads(v_reshaped, is_value=True)  # Shape: (B, H, L*N, head_dim_v)

        # Scaled dot-product attention
        # Query determines output seq length (N)
        attn_output_heads = F.scaled_dot_product_attention(
            q_heads.contiguous(), k_heads.contiguous(), v_heads.contiguous(),
            dropout_p=self.dropout_p if self.training else 0.0,
        ) # Shape: (B, H, N, head_dim_v)

        # Recombine heads
        attn_output = self._recombine_heads(attn_output_heads) # Shape: (B, N, input_dim)

        # --- 6. Reshape back to Patches and Merge ---
        # This part is identical to the parent class as attn_output has shape (B, N, E)
        output_patches_reshaped = rearrange(
            attn_output,
            'b (ph pw pd) c -> b c ph pw pd',
            ph=self.patch_size, pw=self.patch_size, pd=self.patch_size
        )
        output_feat = merge_patches(
            output_patches_reshaped,
            original_channels=original_channels,
            original_shape=original_shape
        )
        return output_feat


# --- New Class Added Below --- (and updated __main__) ---
class TargetToContextAttention3D(SpatialCrossAttention3D):
    """
    Performs 3D spatial cross-attention where multiple context feature maps
    attend to a single target feature map. Inherits from SpatialCrossAttention3D.

    Context acts as Query, Target acts as Key and Value.
    The context feature map input shape is (B, L, C, H, W, D).
    The output shape matches the input context feature map shape.
    """
    def forward(
        self,
        target_feat: torch.Tensor, # Shape (B, C, H, W, D) - Provides K, V
        context_feat: torch.Tensor, # Shape (B, L, C, H, W, D) - Provides Q
        pos_embed_q: torch.Tensor, # Positional embedding for context queries (Shape needs adjustment)
        pos_embed_k: torch.Tensor  # Positional embedding for target keys (Shape needs adjustment)
    ) -> torch.Tensor:
        """
        Args:
            target_feat (torch.Tensor): Target feature map (B, C, H, W, D). Provides K, V.
            context_feat (torch.Tensor): Context feature maps (B, L, C, H, W, D). Provides Q.
                                         L is the number of contexts. B, C, H, W, D
                                         must match target_feat dimensions.
            pos_embed_q (torch.Tensor): Positional embedding for context queries.
                                        Shape (B*L, N, attention_dim) or (1, N, attention_dim)
                                        for broadcasting, where N=patch_volume.
            pos_embed_k (torch.Tensor): Positional embedding for target keys.
                                        Shape (B, N, attention_dim) or (1, N, attention_dim)
                                        for broadcasting.

        Returns:
            torch.Tensor: Output feature map with the same shape as context_feat
                          (B, L, C, H, W, D).
        """
        # --- 1. Input Validation and Shape Extraction ---
        if target_feat.ndim != 5:
             raise ValueError(f"Target features must be 5D (B, C, H, W, D), got {target_feat.ndim}D.")
        if context_feat.ndim != 6:
            raise ValueError(f"Context features must be 6D (B, L, C, H, W, D), got {context_feat.ndim}D.")

        b_t, c_t, h, w, d = target_feat.shape
        b_c, l, c_c, h_c, w_c, d_c = context_feat.shape

        if b_t != b_c or c_t != c_c or (h, w, d) != (h_c, w_c, d_c):
             raise ValueError("Target and Context features must have the same B, C, H, W, D dimensions."
                              f" Got Target: {target_feat.shape}, Context: {context_feat.shape}")

        original_shape = (h, w, d)
        original_channels = c_c # Output matches context channels
        b = b_t # Use b for batch size

        # --- Pre-Step: Reshape Context ---
        context_feat_reshaped = rearrange(context_feat, 'b l c h w d -> (b l) c h w d') # Shape: (B*L, C, H, W, D)

        # --- 2. Patch Splitting and Reshaping ---
        target_patches = avg_into_patches(target_feat, self.patch_size) # Shape: (B, C_out, pH, pW, pD)
        target_patches_split = split_into_patches(target_feat, self.patch_size) # Shape: (B, C_out, pH, pW, pD)
        context_patches = avg_into_patches(context_feat_reshaped, self.patch_size) # Shape: (B*L, C_out, pH, pW, pD)
        b_p, c_out, ph, pw, pd = target_patches.shape
        bl_p, c_out_c, _, _, _ = context_patches.shape

        if c_out != self.input_dim:
            raise ValueError(f"Channel dimension after patching ({c_out}) does not match module's input_dim ({self.input_dim}).")
        if bl_p != b * l:
            raise RuntimeError("Context batch dimension after patching doesn't match B*L.")

        # Flatten patches into sequences
        target_flat = rearrange(target_patches, 'b c ph pw pd -> b (ph pw pd) c', ph=ph, pw=pw, pd=pd) # Shape: (B, N, E)
        target_flat_split = rearrange(target_patches_split, 'b c ph pw pd -> b (ph pw pd) c', ph=ph, pw=pw, pd=pd) # Shape: (B, N, E)
        context_flat = rearrange(context_patches, 'bl c ph pw pd -> bl (ph pw pd) c', ph=ph, pw=pw, pd=pd) # Shape: (B*L, N, E)

        n = target_flat.shape[1]
        if n != self.patch_volume:
             raise RuntimeError(f"Sequence length {n} does not match expected patch volume {self.patch_volume}")

        # --- 3. Assign Q, K, V ---
        # Context provides Query, Target provides Key and Value
        q_in = context_flat # Shape: (B*L, N, E)
        k_in = target_flat  # Shape: (B, N, E)
        v = target_flat_split     # Shape: (B, N, E)

        # --- 4. Project Q and K ---
        q_proj = self.q_proj(q_in) # Shape: (B*L, N, attention_dim)
        k_proj = self.k_proj(k_in) # Shape: (B, N, attention_dim)

        # --- 4b. Handle Positional Embedding Broadcasting & Addition ---
        bl = b * l
        # Expand pos_embed_q (for context) if needed: (1, N, attn_dim) -> (B*L, N, attn_dim)
        if pos_embed_q.shape[0] == 1 and bl > 1:
            pos_embed_q = pos_embed_q.expand(bl, -1, -1)
        elif pos_embed_q.shape[0] != bl:
             raise ValueError(f"pos_embed_q batch size {pos_embed_q.shape[0]} cannot broadcast/match context batch size {bl}")

        # Expand pos_embed_k (for target) if needed: (1, N, attn_dim) -> (B, N, attn_dim)
        if pos_embed_k.shape[0] == 1 and b > 1:
            pos_embed_k = pos_embed_k.expand(b, -1, -1)
        elif pos_embed_k.shape[0] != b:
            raise ValueError(f"pos_embed_k batch size {pos_embed_k.shape[0]} cannot broadcast/match target batch size {b}")

        # Check compatibility after potential expansion
        if pos_embed_q.shape[-2:] != q_proj.shape[-2:]:
             raise ValueError(f"Expanded pos_embed_q shape {pos_embed_q.shape} incompatible with projected q shape {q_proj.shape}")
        if pos_embed_k.shape[-2:] != k_proj.shape[-2:]:
            raise ValueError(f"Expanded pos_embed_k shape {pos_embed_k.shape} incompatible with projected k shape {k_proj.shape}")

        # Add positional embeddings *after* projection
        q = q_proj + pos_embed_q # Shape: (B*L, N, attention_dim)
        k = k_proj + pos_embed_k # Shape: (B, N, attention_dim)

        # --- 4c. Reshape Q, K, V for Attention Calculation ---
        # Reshape Q from (B*L, N, attention_dim) -> (B, L*N, attention_dim)
        q_reshaped = rearrange(q, '(b l) n d -> b (l n) d', b=b, l=l)
        # K and V are already correct: (B, N, ...)
        k_reshaped = k # Shape: (B, N, attention_dim)
        v_reshaped = v # Shape: (B, N, input_dim)

        # --- 5. Multi-Head Attention ---
        q_heads = self._separate_heads(q_reshaped, is_value=False) # Shape: (B, H, L*N, head_dim_qk)
        k_heads = self._separate_heads(k_reshaped, is_value=False) # Shape: (B, H, N, head_dim_qk)
        v_heads = self._separate_heads(v_reshaped, is_value=True)  # Shape: (B, H, N, head_dim_v)

        # Scaled dot-product attention
        # Query (Context) determines output seq length (L*N)
        attn_output_heads = F.scaled_dot_product_attention(
            q_heads.contiguous(), k_heads.contiguous(), v_heads.contiguous(),
            dropout_p=self.dropout_p if self.training else 0.0,
        ) # Shape: (B, H, L*N, head_dim_v)

        # Recombine heads
        attn_output = self._recombine_heads(attn_output_heads) # Shape: (B, L*N, input_dim)

        # --- 6. Reshape back to Patches and Merge ---
        # Reshape sequence from (B, L*N, E) back to (B*L, N, E)
        attn_output_bl = rearrange(attn_output, 'b (l n) e -> (b l) n e', l=l, n=n)

        # Reshape sequence back to patch grid: (B*L, N, E) -> (B*L, E, pH, pW, pD)
        output_patches_reshaped = rearrange(
            attn_output_bl,
            'bl (ph pw pd) c -> bl c ph pw pd',
            ph=self.patch_size, pw=self.patch_size, pd=self.patch_size
        )

        # Merge patches back to original spatial dimensions for the B*L batch
        merged_bl = merge_patches(
            output_patches_reshaped,
            original_channels=original_channels, # Use original context channels
            original_shape=original_shape
        ) # Shape: (B*L, C, H, W, D)

        # Reshape back to (B, L, C, H, W, D)
        output_feat = rearrange(merged_bl, '(b l) c h w d -> b l c h w d', b=b, l=l)

        return output_feat


if __name__ == '__main__':
    # --- Example for SpatialCrossAttention3D (Original) ---
    print("### Testing SpatialCrossAttention3D ###")
    batch_size = 2
    c_orig = 4
    h_orig = 32
    w_orig = 32
    d_orig = 16
    patch_s = 8
    nh = h_orig // patch_s
    nw = w_orig // patch_s
    nd = d_orig // patch_s
    patch_vol = patch_s ** 3
    input_dim_calc = c_orig * nh * nw * nd # 128
    heads = 8 # Divides 128
    attn_dim = 64 # Divides heads=8
    print(f"--- Config (Single Context) ---")
    print(f"Input Dim: {input_dim_calc}, Heads: {heads}, Attn Dim: {attn_dim}")
    target_f = torch.randn(batch_size, c_orig, h_orig, w_orig, d_orig)
    context_f_single = torch.randn(batch_size, c_orig, h_orig, w_orig, d_orig)
    pos_q_single = torch.randn(batch_size, patch_vol, attn_dim) # Q is target
    pos_k_single = torch.randn(batch_size, patch_vol, attn_dim) # K is context
    try:
        attention_layer_single = SpatialCrossAttention3D(
            input_dim=input_dim_calc, num_heads=heads, attention_dim=attn_dim, patch_size=patch_s)
        output_single = attention_layer_single(target_f, context_f_single, pos_q_single, pos_k_single)
        print(f"Single Context Input Target: {target_f.shape}")
        print(f"Single Context Input Context: {context_f_single.shape}")
        print(f"Single Context Output: {output_single.shape}")
        assert output_single.shape == target_f.shape
        print("Single Context Test Passed.")
    except Exception as e:
        print(f"Single Context Test Failed: {e}")
    print("\n" + "="*30 + "\n")


    # --- Example for MultiContextSpatialCrossAttention3D ---
    print("### Testing MultiContextSpatialCrossAttention3D ###")
    num_contexts = 3
    print(f"--- Config (Target to Multi Context, L={num_contexts}) ---")
    print(f"Input Dim: {input_dim_calc}, Heads: {heads}, Attn Dim: {attn_dim}")
    context_f_multi = torch.randn(batch_size, num_contexts, c_orig, h_orig, w_orig, d_orig)
    pos_q_multi_t2c = torch.randn(batch_size, patch_vol, attn_dim) # Q is target
    pos_k_multi_t2c_broadcast = torch.randn(1, patch_vol, attn_dim) # K is context (broadcast over B*L)
    # Pos Q can also be broadcast
    pos_q_multi_t2c_broadcast = torch.randn(1, patch_vol, attn_dim)
    try:
        attention_layer_multi_t2c = MultiContextSpatialCrossAttention3D(
            input_dim=input_dim_calc, num_heads=heads, attention_dim=attn_dim, patch_size=patch_s)
        print("Multi-Context (T->C) Attention module instantiated.")
        # Test with broadcasting pos_k and pos_q
        print("\nTesting with broadcasting pos_k and pos_q...")
        output_multi_broadcast_t2c = attention_layer_multi_t2c(target_f, context_f_multi, pos_q_multi_t2c_broadcast, pos_k_multi_t2c_broadcast)
        print(f"Multi Context (T->C) Input Target: {target_f.shape}")
        print(f"Multi Context (T->C) Input Context: {context_f_multi.shape}")
        print(f"Multi Context Input Pos Q (broadcast): {pos_q_multi_t2c_broadcast.shape}")
        print(f"Multi Context (T->C) Output: {output_multi_broadcast_t2c.shape}")
        assert output_multi_broadcast_t2c.shape == target_f.shape
        print("Multi Context Test (T->C, broadcast pos_k/q) Passed.")
    except Exception as e:
        print(f"Multi Context Test (T->C) Failed: {e}")
    print("\n" + "="*30 + "\n")


    # --- Example for TargetToContextAttention3D ---
    print("### Testing TargetToContextAttention3D ###")
    # Using same parameters as MultiContext test
    print(f"--- Config (Context to Target, L={num_contexts}) ---")
    print(f"Input Dim: {input_dim_calc}, Heads: {heads}, Attn Dim: {attn_dim}")

    # Positional embeddings for Q (context) K (target)
    pos_q_c2t_broadcast = torch.randn(1, patch_vol, attn_dim) # Broadcast over B*L queries
    pos_k_c2t = torch.randn(batch_size, patch_vol, attn_dim) # Explicit B for target keys

    try:
        attention_layer_c2t = TargetToContextAttention3D(
            input_dim=input_dim_calc, num_heads=heads, attention_dim=attn_dim, patch_size=patch_s)
        print("Context-to-Target Attention module instantiated.")

        # Test with broadcasting pos_q
        print("\nTesting with broadcasting pos_q...")
        output_c2t = attention_layer_c2t(target_f, context_f_multi, pos_q_c2t_broadcast, pos_k_c2t)
        print(f"Context-to-Target Input Target: {target_f.shape}")
        print(f"Context-to-Target Input Context: {context_f_multi.shape}")
        print(f"Context-to-Target Input Pos Q (broadcast): {pos_q_c2t_broadcast.shape}")
        print(f"Context-to-Target Input Pos K (explicit): {pos_k_c2t.shape}")
        print(f"Context-to-Target Output: {output_c2t.shape}")
        # Output shape should match context input shape
        assert output_c2t.shape == context_f_multi.shape
        print("Context-to-Target Test Passed.")

    except Exception as e:
        print(f"Context-to-Target Test Failed: {e}") 
import torch
from typing import Tuple
from einops import rearrange

import numpy as np
import torch.nn as nn
from typing import Sequence, Union
import warnings

def split_into_patches(feature_map: torch.Tensor, patch_size: int = 8) -> torch.Tensor:
    """
    Reshapes a 5D feature map into non-overlapping patches, combining the
    original channel dimension with the patch indices.

    Args:
        feature_map (torch.Tensor): The input feature map with shape (B, C, H, W, D).
        patch_size (int): The spatial size of the patches (assumed to be cubic).
                          Defaults to 8.

    Returns:
        torch.Tensor: The reshaped feature map with shape
                      (B, C * (H/patch_size) * (W/patch_size) * (D/patch_size), patch_size, patch_size, patch_size).

    Raises:
        ValueError: If the spatial dimensions (H, W, D) are not perfectly
                    divisible by patch_size.
        ImportError: If the 'einops' library is not installed.
    """
    if not isinstance(feature_map, torch.Tensor):
        raise TypeError(f"Input must be a torch.Tensor, got {type(feature_map)}")
    if feature_map.ndim != 5:
        raise ValueError(f"Input tensor must be 5D (B, C, H, W, D), got {feature_map.ndim}D")

    b, c, h, w, d = feature_map.shape
    ph, pw, pd = patch_size, patch_size, patch_size

    if h % ph != 0 or w % pw != 0 or d % pd != 0:
        raise ValueError(f"Spatial dimensions H({h}), W({w}), D({d}) must be divisible by patch_size ({patch_size})")

    # Rearrange: b c (nh ph) (nw pw) (nd pd) -> b (c nh nw nd) ph pw pd
    # where nh=h//ph, nw=w//pw, nd=d//pd
    reshaped_feature_map = rearrange(
        feature_map,
        'b c (nh ph) (nw pw) (nd pd) -> b (c nh nw nd) ph pw pd',
        ph=ph, pw=pw, pd=pd
    )

    # Verify the output shape calculation:
    # C_out = c * (h // ph) * (w // pw) * (d // pd)
    #       = c * h * w * d // (ph * pw * pd)
    #       = c * h * w * d // (patch_size**3)
    # For patch_size=8, C_out = c * h * w * d // 512
    # So the final shape is (b, C_out, ph, pw, pd)

    return reshaped_feature_map

def avg_into_patches(feature_map: torch.Tensor, patch_size: int = 8) -> torch.Tensor:
    """
    Reshapes a 5D feature map into non-overlapping patches, combining the
    original channel dimension with the patch indices.

    Args:
        feature_map (torch.Tensor): The input feature map with shape (B, C, H, W, D).
        patch_size (int): The spatial size of the patches (assumed to be cubic).
                          Defaults to 8.

    Returns:
        torch.Tensor: The reshaped feature map with shape
                      (B, C * (H/patch_size) * (W/patch_size) * (D/patch_size), patch_size, patch_size, patch_size).

    Raises:
        ValueError: If the spatial dimensions (H, W, D) are not perfectly
                    divisible by patch_size.
        ImportError: If the 'einops' library is not installed.
    """
    if not isinstance(feature_map, torch.Tensor):
        raise TypeError(f"Input must be a torch.Tensor, got {type(feature_map)}")
    if feature_map.ndim != 5:
        raise ValueError(f"Input tensor must be 5D (B, C, H, W, D), got {feature_map.ndim}D")

    b, c, h, w, d = feature_map.shape
    ph, pw, pd = patch_size, patch_size, patch_size

    if h % ph != 0 or w % pw != 0 or d % pd != 0:
        raise ValueError(f"Spatial dimensions H({h}), W({w}), D({d}) must be divisible by patch_size ({patch_size})")

    # Rearrange: b c (nh ph) (nw pw) (nd pd) -> b (c nh nw nd) ph pw pd
    # where nh=h//ph, nw=w//pw, nd=d//pd
    reshaped_feature_map = rearrange(
        feature_map,
        'b c (nh ph) (nw pw) (nd pd) -> b c (nh nw nd) ph pw pd',
        ph=ph, pw=pw, pd=pd
    )
    
    # compute mean instead of rearrange
    reshaped_feature_map = torch.mean(reshaped_feature_map, dim=2)
    return reshaped_feature_map

def merge_patches(patched_feature_map: torch.Tensor, original_channels: int, original_shape: Tuple[int, int, int]) -> torch.Tensor:
    """
    Reconstructs the original 5D feature map from its patched representation.
    This is the inverse operation of split_into_patches.

    Args:
        patched_feature_map (torch.Tensor): The input tensor with shape
            (B, C_out, pH, pW, pD), where C_out is C * num_patches.
        original_channels (int): The number of channels (C) in the original
                                 feature map.
        original_shape (Tuple[int, int, int]): The spatial dimensions (H, W, D)
                                              of the original feature map.

    Returns:
        torch.Tensor: The reconstructed feature map with shape (B, C, H, W, D).

    Raises:
        ValueError: If dimensions are inconsistent or not divisible.
        ImportError: If the 'einops' library is not installed.
    """
    if not isinstance(patched_feature_map, torch.Tensor):
        raise TypeError(f"Input must be a torch.Tensor, got {type(patched_feature_map)}")
    if patched_feature_map.ndim != 5:
        raise ValueError(f"Input patched tensor must be 5D (B, C_out, pH, pW, pD), got {patched_feature_map.ndim}D")
    if not isinstance(original_shape, tuple) or len(original_shape) != 3:
         raise ValueError(f"original_shape must be a tuple of 3 integers (H, W, D), got {original_shape}")
            
    # Modified check to handle both Python int and scalar torch.Tensor
    is_valid_int = isinstance(original_channels, int) and original_channels > 0
    is_valid_tensor = isinstance(original_channels, torch.Tensor) and original_channels.ndim == 0 and original_channels.item() > 0
    if not (is_valid_int or is_valid_tensor):
        raise ValueError(f"original_channels must be a positive integer or scalar tensor, got {original_channels} (type: {type(original_channels)})")

    # Ensure original_channels is an int for subsequent calculations if it was a tensor
    if isinstance(original_channels, torch.Tensor):
        original_channels = original_channels.item()

    b, c_out, ph, pw, pd = patched_feature_map.shape
    h, w, d = original_shape
    c = original_channels

    if h % ph != 0 or w % pw != 0 or d % pd != 0:
        raise ValueError(f"Original dimensions H({h}), W({w}), D({d}) must be divisible by patch dimensions ({ph}, {pw}, {pd})")

    nh = h // ph
    nw = w // pw
    nd = d // pd

    expected_c_out = c * nh * nw * nd
    if c_out != expected_c_out:
        raise ValueError(f"Input channel dimension C_out ({c_out}) does not match expected value based on original_channels ({c}) and grid size ({nh}x{nw}x{nd}): {expected_c_out}")

    # Rearrange: b (c nh nw nd) ph pw pd -> b c (nh ph) (nw pw) (nd pd)
    original_feature_map = rearrange(
        patched_feature_map,
        'b (c nh nw nd) ph pw pd -> b c (nh ph) (nw pw) (nd pd)',
        c=c, nh=nh, nw=nw, nd=nd, ph=ph, pw=pw, pd=pd # provide dimensions for decomposition and composition
    )

    # Verify final shape
    assert original_feature_map.shape == (b, c, h, w, d)

    return original_feature_map

def calculate_3d_intersection_volume(box1: Tuple[float, float, float, float, float, float],
                                     box2: Tuple[float, float, float, float, float, float]) -> float:
    """Calculates the intersection volume of two 3D boxes.

    Args:
        box1: Coordinates (min_h, min_w, min_d, max_h, max_w, max_d)
        box2: Coordinates (min_h, min_w, min_d, max_h, max_w, max_d)

    Returns:
        Intersection volume.
    """
    min_h1, min_w1, min_d1, max_h1, max_w1, max_d1 = box1
    min_h2, min_w2, min_d2, max_h2, max_w2, max_d2 = box2

    inter_min_h = max(min_h1, min_h2)
    inter_max_h = min(max_h1, max_h2)
    inter_len_h = max(0.0, inter_max_h - inter_min_h)

    inter_min_w = max(min_w1, min_w2)
    inter_max_w = min(max_w1, max_w2)
    inter_len_w = max(0.0, inter_max_w - inter_min_w)

    inter_min_d = max(min_d1, min_d2)
    inter_max_d = min(max_d1, max_d2)
    inter_len_d = max(0.0, inter_max_d - inter_min_d)

    return inter_len_h * inter_len_w * inter_len_d

def calculate_intersection_matrix(
    image_size: Sequence[int],
    window_coords: Tuple[Tuple[int, int, int], Tuple[int, int, int]],
    grid_division: Sequence[int] = (8, 8, 8)
) -> np.ndarray:
    """
    Calculates a position matrix based on intersection ratios between window patches and image patches.

    The image and the window are divided into a grid (e.g., 8x8x8 patches).
    The matrix element (i, j) represents the proportion of the i-th window patch
    that intersects with the j-th image patch.

    Args:
        image_size: Dimensions of the original 3D image [H, W, D].
        window_coords: Coordinates of the window within the image:
                       ((h_min, w_min, d_min), (h_max, w_max, d_max)).
        grid_division: Number of patches along each dimension [N_h, N_w, N_d].
                       Defaults to (8, 8, 8).

    Returns:
        A (N_h*N_w*N_d) x (N_h*N_w*N_d) numpy array representing the intersection ratios.
        Matrix[i, j] = intersection(window_patch_i, image_patch_j) / volume(window_patch_i).
    """
    H, W, D = image_size
    N_h, N_w, N_d = grid_division
    N_total = N_h * N_w * N_d

    (h_min, w_min, d_min), (h_max, w_max, d_max) = window_coords

    # --- Input Validation ---
    if not all(i_s % g_d == 0 for i_s, g_d in zip(image_size, grid_division)):
        raise ValueError(f"Image dimensions {image_size} must be divisible by grid divisions {grid_division}")
    if not (0 <= h_min < h_max <= H and 0 <= w_min < w_max <= W and 0 <= d_min < d_max <= D):
         warnings.warn(f"Window coordinates {window_coords} might be outside or invalid for image size {image_size}.")
    if h_max-h_min == 0 or w_max-w_min == 0 or d_max-d_min == 0:
         warnings.warn(f"Window {window_coords} has zero volume in at least one dimension.")


    # --- Calculate Image Patch Coordinates ---
    img_patch_dim = [H / N_h, W / N_w, D / N_d]
    image_patches_coords = []
    for k in range(N_d):
        for j in range(N_w):
            for i in range(N_h):
                p_min_h = i * img_patch_dim[0]
                p_max_h = (i + 1) * img_patch_dim[0]
                p_min_w = j * img_patch_dim[1]
                p_max_w = (j + 1) * img_patch_dim[1]
                p_min_d = k * img_patch_dim[2]
                p_max_d = (k + 1) * img_patch_dim[2]
                image_patches_coords.append((p_min_h, p_min_w, p_min_d, p_max_h, p_max_w, p_max_d))

    # --- Calculate Window Patch Coordinates and Volume ---
    win_size = [h_max - h_min, w_max - w_min, d_max - d_min]
    # Avoid division by zero if window dim is 0
    win_patch_dim = [
        win_size[0] / N_h if N_h > 0 else 0,
        win_size[1] / N_w if N_w > 0 else 0,
        win_size[2] / N_d if N_d > 0 else 0,
    ]
    window_patch_volume = win_patch_dim[0] * win_patch_dim[1] * win_patch_dim[2]
    window_patches_coords = []
    for k in range(N_d):
        for j in range(N_w):
            for i in range(N_h):
                wp_min_h = h_min + i * win_patch_dim[0]
                wp_max_h = h_min + (i + 1) * win_patch_dim[0]
                wp_min_w = w_min + j * win_patch_dim[1]
                wp_max_w = w_min + (j + 1) * win_patch_dim[1]
                wp_min_d = d_min + k * win_patch_dim[2]
                wp_max_d = d_min + (k + 1) * win_patch_dim[2]
                window_patches_coords.append((wp_min_h, wp_min_w, wp_min_d, wp_max_h, wp_max_w, wp_max_d))

    # --- Calculate Intersection Matrix ---
    intersection_matrix = np.zeros((N_total, N_total), dtype=np.float32)

    if window_patch_volume <= 1e-6: # Handle zero volume windows
        warnings.warn(f"Window patch volume is near zero ({window_patch_volume}). Intersection matrix will be all zeros.")
        return intersection_matrix

    for i in range(N_total): # Iterate through window patches
        win_patch_coords = window_patches_coords[i]
        for j in range(N_total): # Iterate through image patches
            img_patch_coords = image_patches_coords[j]
            intersection_vol = calculate_3d_intersection_volume(win_patch_coords, img_patch_coords)
            intersection_matrix[i, j] = intersection_vol / window_patch_volume

    # Clamp values to handle potential floating point issues slightly exceeding 1.0
    intersection_matrix = np.clip(intersection_matrix, 0.0, 1.0)

    return intersection_matrix

def calculate_intersection_matrix_spatial(
    image_size: Sequence[int],
    window_coords: Tuple[Tuple[int, int, int], Tuple[int, int, int]],
    grid_division: Sequence[int] = (8, 8, 8)
) -> np.ndarray:
    """
    Calculates a position matrix based on intersection ratios and returns it
    in a 6D spatial format.

    Calls `calculate_intersection_matrix` and reshapes the output.

    Args:
        image_size: Dimensions of the original 3D image [H, W, D].
        window_coords: Coordinates of the window within the image:
                       ((h_min, w_min, d_min), (h_max, w_max, d_max)).
        grid_division: Number of patches along each dimension [N_h, N_w, N_d].
                       Defaults to (8, 8, 8).

    Returns:
        A (N_h, N_w, N_d, N_h, N_w, N_d) numpy array representing the
        intersection ratios, preserving the spatial grid structure.
        Matrix[i_h, i_w, i_d, j_h, j_w, j_d] corresponds to the intersection ratio
        between the window patch at grid index (i_h, i_w, i_d) and the image
        patch at grid index (j_h, j_w, j_d).
    """
    # Calculate the flat intersection matrix
    flat_intersection_matrix = calculate_intersection_matrix(
        image_size=image_size,
        window_coords=window_coords,
        grid_division=grid_division
    )

    N_h, N_w, N_d = grid_division

    # Reshape the flat (N_total, N_total) matrix to 6D (Nh, Nw, Nd, Nh, Nw, Nd)
    # The default 'C' order corresponds to iterating through d, then w, then h last
    # which matches the loops in calculate_intersection_matrix
    spatial_intersection_matrix = flat_intersection_matrix.reshape(
        (N_h, N_w, N_d, N_h, N_w, N_d)
    )

    return spatial_intersection_matrix

if __name__ == '__main__':
    # Example Usage for split_into_patches
    batch_size = 2
    channels = 3
    height = 32
    width = 32
    depth = 16
    patch_s = 8
    original_spatial_shape = (height, width, depth)

    dummy_input = torch.randn(batch_size, channels, height, width, depth)
    print("--- Testing split_into_patches ---")
    print(f"Input shape: {dummy_input.shape}")

    try:
        output_patches = split_into_patches(dummy_input, patch_size=patch_s)
        print(f"Output shape after splitting (patch_size={patch_s}): {output_patches.shape}")
        # Expected output calculation
        nh = height // patch_s # 32 // 8 = 4
        nw = width // patch_s  # 32 // 8 = 4
        nd = depth // patch_s  # 16 // 8 = 2
        expected_c_out = channels * nh * nw * nd # 3 * 4 * 4 * 2 = 96
        expected_shape = (batch_size, expected_c_out, patch_s, patch_s, patch_s)
        print(f"Expected split output shape: {expected_shape}")
        assert output_patches.shape == expected_shape
        print("Split shape assertion passed.")

        # Example Usage for merge_patches
        print("\n--- Testing merge_patches ---")
        print(f"Input shape to merge: {output_patches.shape}")
        reconstructed_map = merge_patches(
            output_patches,
            original_channels=channels,
            original_shape=original_spatial_shape
        )
        print(f"Output shape after merging: {reconstructed_map.shape}")
        print(f"Expected reconstructed shape: {dummy_input.shape}")
        assert reconstructed_map.shape == dummy_input.shape
        print("Merge shape assertion passed.")

        # Test if reconstruction is (approximately) correct
        assert torch.allclose(reconstructed_map, dummy_input, atol=1e-6)
        print("Reconstruction value check passed.")

    except ValueError as e:
        print(f"Error: {e}")
    except ImportError as e:
        print(f"Import Error: {e}. Please install einops.")

    # Example with non-divisible dimensions for split
    print("\n--- Testing split_into_patches (invalid input) ---")
    dummy_invalid = torch.randn(batch_size, channels, 31, 32, 16)
    print(f"Input shape: {dummy_invalid.shape}")
    try:
        split_into_patches(dummy_invalid, patch_size=patch_s)
    except ValueError as e:
        print(f"Successfully caught split error: {e}")

    # Example with inconsistent dimensions for merge
    print("\n--- Testing merge_patches (invalid input) ---")
    # Create patches with correct split shape
    output_patches_valid = split_into_patches(dummy_input, patch_size=patch_s)
    invalid_original_channels = channels + 1
    print(f"Patched shape: {output_patches_valid.shape}, Trying to merge with original_channels={invalid_original_channels}")
    try:
         merge_patches(output_patches_valid, invalid_original_channels, original_spatial_shape)
    except ValueError as e:
        print(f"Successfully caught merge error (wrong C): {e}")

    invalid_original_shape = (height+1, width, depth)
    print(f"Patched shape: {output_patches_valid.shape}, Trying to merge with original_shape={invalid_original_shape}")
    try:
         merge_patches(output_patches_valid, channels, invalid_original_shape)
    except ValueError as e:
        print(f"Successfully caught merge error (wrong H): {e}")

    print("\n--- Intersection Matrix Example ---")
    img_size_example = (96, 96, 96)
    win_coords_example = ((10, 10, 10), (90, 90, 90)) # A window within the image
    grid_div_example = (8, 8, 8)
    total_patches = np.prod(grid_div_example)

    print(f"Image Size: {img_size_example}")
    print(f"Window Coords: {win_coords_example}")
    print(f"Grid Division: {grid_div_example}")
    print(f"Total Patches: {total_patches}")

    try:
        intersection_mat = calculate_intersection_matrix(
            image_size=img_size_example,
            window_coords=win_coords_example,
            grid_division=grid_div_example
        )
        print(f"Output Matrix Shape: {intersection_mat.shape}") # Expected: (512, 512)

        # Basic checks
        row_sums = np.sum(intersection_mat, axis=1)
        print(f"Range of row sums: [{np.min(row_sums):.4f}, {np.max(row_sums):.4f}]") # Should be close to 1.0
        if not np.allclose(row_sums, 1.0, atol=1e-5):
             warnings.warn("Row sums are not all close to 1.0, check calculation logic.")
        print(f"Matrix dtype: {intersection_mat.dtype}")
        print(f"Sample values (first 5x5):\\n{intersection_mat[:5, :5]}")

        # Test edge case: Full image window
        full_window_coords = ((0, 0, 0), (96, 96, 96))
        print("\\nTesting full image window...")
        intersection_mat_full = calculate_intersection_matrix(
            image_size=img_size_example,
            window_coords=full_window_coords,
            grid_division=grid_div_example
        )
        print(f"Full window matrix is diagonal: {np.allclose(intersection_mat_full, np.eye(total_patches))}")

        # Test edge case: Window completely outside
        outside_window_coords = ((100, 100, 100), (120, 120, 120))
        print("\\nTesting window outside image...")
        intersection_mat_outside = calculate_intersection_matrix(
            image_size=img_size_example,
            window_coords=outside_window_coords,
            grid_division=grid_div_example
        )
        print(f"Outside window matrix is zero: {np.all(intersection_mat_outside == 0.0)}")


    except ValueError as e:
        print(f"Error during calculation: {e}")

    print("\n--- Spatial Intersection Matrix Example ---")
    try:
        spatial_intersection_mat = calculate_intersection_matrix_spatial(
            image_size=img_size_example,
            window_coords=win_coords_example,
            grid_division=grid_div_example
        )
        print(f"Output Spatial Matrix Shape: {spatial_intersection_mat.shape}") # Expected: (8, 8, 8, 8, 8, 8)

        # Verify reshaping consistency
        N_h, N_w, N_d = grid_div_example
        flat_again = spatial_intersection_mat.reshape(total_patches, total_patches)
        assert np.allclose(flat_again, intersection_mat)
        print("Reshaping consistency check passed.")

        # Check value at a specific spatial index
        # e.g., window patch (1,1,1) intersection with image patch (1,1,1)
        # corresponding flat index is 1*8*8 + 1*8 + 1 = 73
        flat_idx = 1 * N_w * N_d + 1 * N_d + 1
        assert np.isclose(spatial_intersection_mat[1, 1, 1, 1, 1, 1], intersection_mat[flat_idx, flat_idx])
        print("Specific spatial index value check passed.")


    except ValueError as e:
        print(f"Error during spatial calculation: {e}") 
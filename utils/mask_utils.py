import cv2
import math
import numpy as np
import torch
import torch.nn.functional as F

def compute_gradient_mag(image, sobel_kernel_size=3, epsilon=0.001):
    if image.shape[0] == 3:
        # Use luminance weights for RGB to grayscale conversion
        gray = (0.299 * image[0] + 0.587 * image[1] + 0.114 * image[2])
    else:
        gray = image.squeeze(0)  # Remove channel dim for single-channel input
        
    device = image.device
    kernel_x = torch.tensor([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=torch.float32, device=device).view(1, 1, 3, 3)
    
    kernel_y = torch.tensor([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ], dtype=torch.float32, device=device).view(1, 1, 3, 3)
    
    # Add batch and channel dimensions for conv2d
    gray = gray.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    
    # Compute gradients
    Gx = F.conv2d(gray, kernel_x, padding=sobel_kernel_size//2)
    Gy = F.conv2d(gray, kernel_y, padding=sobel_kernel_size//2)
    
    # Calculate gradient magnitude
    G = torch.sqrt(Gx.pow(2) + Gy.pow(2))
    G = G.squeeze()  # Remove batch and channel dims -> (H, W)
    
    border_size = sobel_kernel_size // 2
    if border_size > 0:
        G[:border_size, :] = epsilon
        G[-border_size:, :] = epsilon
        G[:, :border_size] = epsilon
        G[:, -border_size:] = epsilon
    
    return G

def adaptive_random_sampling(
    image: torch.Tensor, 
    num_samples: int, 
    epsilon: float = 0.001,
    sobel_kernel_size: int = 3
) -> torch.Tensor:
    """
    Adaptive random sampling based on texture richness (gradient magnitude).
    Generates a binary mask with exactly num_samples pixels set to 1 (selected).
    
    Args:
        image: Input image tensor (C x H x W) where C=1 (grayscale) or 3 (RGB)
        num_samples: Total number of pixels to sample
        epsilon: Small constant to ensure non-zero sampling probability everywhere
        sobel_kernel_size: Size of Sobel kernel (3 or 5 recommended)
    
    Returns:
        Binary mask tensor (H x W) where 1 indicates sampled pixels
    """
    device = image.device
    G = compute_gradient_mag(image, sobel_kernel_size=sobel_kernel_size, epsilon=epsilon)
    
    # Step 3: Normalize gradient magnitudes to [0, 1]
    G_min = G.min()
    G_max = G.max()
    G_norm = (G - G_min) / (G_max - G_min + 1e-7)  # Avoid division by zero

    # Step 4: Create sampling probability map
    P = G_norm + epsilon
    P_flat = P.view(-1)  # Flatten to (H*W,)
    P_flat = P_flat / P_flat.sum()  # Normalize to sum to 1
    
    # Step 5: Sample indices according to probabilities
    # Create CDF (cumulative distribution function)
    cdf = torch.cumsum(P_flat, dim=0)
    cdf = cdf / cdf[-1]  # Ensure CDF ends at 1.0
    
    # Generate random numbers and find corresponding indices
    rand_vals = torch.rand(num_samples, device=device)
    sampled_indices = torch.searchsorted(cdf, rand_vals)
    
    # Step 6: Create output mask
    mask_flat = torch.zeros(P_flat.shape, dtype=torch.bool, device=device)
    mask_flat[sampled_indices] = 1
    mask = mask_flat.view(G.shape)  # Reshape to original dimensions
    
    return mask

def generate_random_mask(image_size, tile_size=16, device="cuda"):
    H, W = image_size
    tile_h, tile_w = (tile_size, tile_size) if isinstance(tile_size, int) else tile_size

    num_tiles_h = (H + tile_h - 1) // tile_h
    num_tiles_w = (W + tile_w - 1) // tile_w

    y_starts = torch.arange(0, num_tiles_h * tile_h, tile_h, device=device)
    x_starts = torch.arange(0, num_tiles_w * tile_w, tile_w, device=device)
    y_grid, x_grid = torch.meshgrid(y_starts, x_starts, indexing='ij')

    rand_h_offset = torch.randint(0, tile_h, (num_tiles_h, num_tiles_w), device=device)
    rand_w_offset = torch.randint(0, tile_w, (num_tiles_h, num_tiles_w), device=device)
    abs_h = (y_grid + rand_h_offset).clamp(max=H - 1)
    abs_w = (x_grid + rand_w_offset).clamp(max=W - 1)

    
    # row-major sorted coords
    sorted_coords = torch.stack([abs_w.flatten(), abs_h.flatten()], dim=1).to(torch.int32)

    # offsets record the starting index of each tile's pixels in sorted_coords
    offsets = torch.arange(num_tiles_h * num_tiles_w + 1, dtype=torch.int32, device=device)

    mask = torch.zeros((H, W), dtype=torch.bool, device=device)
    mask[abs_h.flatten(), abs_w.flatten()] = True

    return mask, offsets, sorted_coords

def get_pixel_info(mask, tile_size=16):
    assert len(mask.shape) == 2 or mask.shape[0] == 1, "Mask must be 2D or 3D with single channel"
    if len(mask.shape) != 2:
        mask = mask.squeeze(0)
    H, W = mask.shape
    device = mask.device
    
    # coord with format (row, col)
    coords = torch.nonzero(mask) # 形状为 [N, 2]
    if coords.shape[0] == 0:
        return None, None

    num_tiles_w = (W + tile_size - 1) // tile_size
    num_tiles_h = (H + tile_size - 1) // tile_size
    
    tile_row = coords[:, 0] // tile_size
    tile_col = coords[:, 1] // tile_size
    tile_indices = tile_row * num_tiles_w + tile_col
    
    # sort by row-major tile index 
    sort_idx = torch.argsort(tile_indices)
    sorted_coords = coords[sort_idx]
    sorted_tile_indices = tile_indices[sort_idx]
    
    num_total_tiles = num_tiles_h * num_tiles_w
    unique_tiles, counts = torch.unique_consecutive(sorted_tile_indices, return_counts=True)

    # Scatter counts of valid tiles into a full-length count array, with 0 for empty tiles
    counts_per_tile = torch.zeros((num_total_tiles), dtype=counts.dtype, device=device)
    counts_per_tile[unique_tiles] = counts

    # Compute offsets (prefix sum), length is num_total_tiles+1
    offsets = torch.zeros((num_total_tiles + 1), dtype=counts.dtype, device=device)
    offsets[1:] = torch.cumsum(counts_per_tile, dim=0)
    
    offsets = offsets.to(device=device)
    sorted_coords = sorted_coords.to(device=device)
    return offsets.to(torch.int32), sorted_coords[:, [1, 0]].to(torch.int32)
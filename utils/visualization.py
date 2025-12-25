# SPDX-License-Identifier: MIT
# Copyright (C) 2025 ComfyUI-Multiband Contributors

"""Visualization utilities for MULTIBAND_IMAGE."""

import math
from typing import Optional
import numpy as np
import torch


def apply_colormap(arr: np.ndarray, colormap: str = 'viridis') -> np.ndarray:
    """
    Apply a colormap to a single-channel array.

    Args:
        arr: 2D array (H, W) with values in [0, 1]
        colormap: Colormap name ('viridis', 'plasma', 'gray', 'jet')

    Returns:
        RGB array (H, W, 3) with values in [0, 1]
    """
    # Normalize to [0, 1]
    arr = np.clip(arr, 0, 1)

    if colormap == 'gray':
        return np.stack([arr, arr, arr], axis=-1)

    # Simple colormap implementations (avoiding matplotlib dependency)
    if colormap == 'viridis':
        # Simplified viridis approximation
        r = 0.267004 + arr * (0.993248 - 0.267004)
        g = 0.004874 + arr * (0.906157 - 0.004874)
        b = 0.329415 + arr * (0.143936 - 0.329415) * (1 - arr) + arr * 0.143936
        # Better viridis approximation
        r = np.clip(0.267 + arr * 0.73 - arr * arr * 0.5, 0, 1)
        g = np.clip(arr * 0.9, 0, 1)
        b = np.clip(0.33 + arr * 0.35 - arr * arr * 0.5, 0, 1)

    elif colormap == 'plasma':
        # Simplified plasma approximation
        r = np.clip(0.05 + arr * 0.95, 0, 1)
        g = np.clip(arr * arr * 0.8, 0, 1)
        b = np.clip(0.53 - arr * 0.5 + arr * arr * 0.5, 0, 1)

    elif colormap == 'jet':
        # Classic jet colormap
        r = np.clip(1.5 - np.abs(arr - 0.75) * 4, 0, 1)
        g = np.clip(1.5 - np.abs(arr - 0.5) * 4, 0, 1)
        b = np.clip(1.5 - np.abs(arr - 0.25) * 4, 0, 1)

    else:
        # Default to grayscale
        return np.stack([arr, arr, arr], axis=-1)

    return np.stack([r, g, b], axis=-1).astype(np.float32)


def preview_rgb_first3(samples: torch.Tensor) -> torch.Tensor:
    """
    Preview using first 3 channels as RGB.

    Args:
        samples: Tensor (B, C, H, W)

    Returns:
        IMAGE tensor (B, H, W, 3)
    """
    B, C, H, W = samples.shape

    if C >= 3:
        rgb = samples[:, :3, :, :]
    elif C == 2:
        # Pad with zeros for blue channel
        rgb = torch.cat([samples, torch.zeros(B, 1, H, W, device=samples.device)], dim=1)
    else:
        # Single channel - replicate to RGB
        rgb = samples.repeat(1, 3, 1, 1)

    # Convert from (B, C, H, W) to (B, H, W, C)
    rgb = rgb.permute(0, 2, 3, 1)

    # Clamp to [0, 1]
    rgb = torch.clamp(rgb, 0, 1)

    return rgb


def preview_single_channel(
    samples: torch.Tensor,
    channel_index: int = 0,
    colormap: str = 'viridis'
) -> torch.Tensor:
    """
    Preview a single channel with colormap.

    Args:
        samples: Tensor (B, C, H, W)
        channel_index: Which channel to show
        colormap: Colormap to apply

    Returns:
        IMAGE tensor (B, H, W, 3)
    """
    B, C, H, W = samples.shape

    # Clamp channel index
    channel_index = min(channel_index, C - 1)
    channel_index = max(channel_index, 0)

    # Extract channel
    channel = samples[:, channel_index, :, :].cpu().numpy()  # (B, H, W)

    # Apply colormap to each batch item
    result = []
    for i in range(B):
        arr = channel[i]
        # Normalize to [0, 1]
        if arr.max() > arr.min():
            arr = (arr - arr.min()) / (arr.max() - arr.min())
        rgb = apply_colormap(arr, colormap)
        result.append(rgb)

    return torch.from_numpy(np.stack(result, axis=0))


def preview_channel_grid(
    samples: torch.Tensor,
    grid_cols: int = 4,
    colormap: str = 'gray'
) -> torch.Tensor:
    """
    Preview all channels in a grid layout.

    Args:
        samples: Tensor (B, C, H, W)
        grid_cols: Number of columns in grid
        colormap: Colormap for each channel

    Returns:
        IMAGE tensor (B, H_grid, W_grid, 3)
    """
    B, C, H, W = samples.shape

    # Calculate grid dimensions
    grid_rows = math.ceil(C / grid_cols)

    # Create output array
    grid_h = grid_rows * H
    grid_w = grid_cols * W

    result = []
    for b in range(B):
        grid = np.zeros((grid_h, grid_w, 3), dtype=np.float32)

        for c in range(C):
            row = c // grid_cols
            col = c % grid_cols

            y_start = row * H
            x_start = col * W

            # Get channel data
            arr = samples[b, c, :, :].cpu().numpy()

            # Normalize
            if arr.max() > arr.min():
                arr = (arr - arr.min()) / (arr.max() - arr.min())

            # Apply colormap
            rgb = apply_colormap(arr, colormap)

            grid[y_start:y_start+H, x_start:x_start+W, :] = rgb

        result.append(grid)

    return torch.from_numpy(np.stack(result, axis=0))


def create_preview(
    samples: torch.Tensor,
    mode: str = 'rgb_first3',
    channel_index: int = 0,
    grid_cols: int = 4,
    colormap: str = 'viridis'
) -> torch.Tensor:
    """
    Create preview IMAGE from multiband samples.

    Args:
        samples: Tensor (B, C, H, W)
        mode: 'rgb_first3', 'single_channel', or 'channel_grid'
        channel_index: For single_channel mode
        grid_cols: For channel_grid mode
        colormap: Colormap for single_channel and channel_grid

    Returns:
        IMAGE tensor (B, H, W, 3)
    """
    if mode == 'rgb_first3':
        return preview_rgb_first3(samples)
    elif mode == 'single_channel':
        return preview_single_channel(samples, channel_index, colormap)
    elif mode == 'channel_grid':
        return preview_channel_grid(samples, grid_cols, colormap)
    else:
        raise ValueError(f"Unknown preview mode: {mode}")

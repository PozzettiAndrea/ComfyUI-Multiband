# SPDX-License-Identifier: MIT
# Copyright (C) 2025 ComfyUI-Multiband Contributors

"""
MULTIBAND_IMAGE type definition for ComfyUI.

A generic multi-channel 2D data type that can hold arbitrary number of channels
(segmentation masks, feature maps, spectral bands, etc.).
"""

from typing import List, Dict, Any, Optional
import torch
import numpy as np


# Custom type constant for ComfyUI node type system
MULTIBAND_IMAGE = "MULTIBAND_IMAGE"
"""
Dict containing multi-channel image data:
{
    'samples': torch.Tensor,      # (B, C, H, W) float32
    'channel_names': List[str],   # optional, length C, e.g. ["seg_0", "seg_1", ...]
    'metadata': dict,             # optional, arbitrary metadata
}

Shape convention: (B, C, H, W)
- B = batch (number of images/views)
- C = channels (arbitrary, not limited to 3)
- H = height
- W = width
"""


def create_multiband(
    samples: torch.Tensor,
    channel_names: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create a MULTIBAND_IMAGE dict from a tensor.

    Args:
        samples: Tensor of shape (B, C, H, W) or (C, H, W)
        channel_names: Optional list of channel names
        metadata: Optional metadata dict

    Returns:
        MULTIBAND_IMAGE dict
    """
    # Ensure 4D tensor
    if samples.ndim == 3:
        samples = samples.unsqueeze(0)

    if samples.ndim != 4:
        raise ValueError(f"Expected 3D or 4D tensor, got {samples.ndim}D")

    # Ensure float32
    if samples.dtype != torch.float32:
        samples = samples.float()

    B, C, H, W = samples.shape

    # Generate default channel names if not provided
    if channel_names is None:
        channel_names = [f"channel_{i}" for i in range(C)]
    elif len(channel_names) != C:
        raise ValueError(f"channel_names length ({len(channel_names)}) != channels ({C})")

    return {
        'samples': samples,
        'channel_names': list(channel_names),
        'metadata': metadata or {},
    }


def validate_multiband(data: Any) -> bool:
    """
    Validate that data is a valid MULTIBAND_IMAGE.

    Args:
        data: Data to validate

    Returns:
        True if valid, False otherwise
    """
    if not isinstance(data, dict):
        return False

    if 'samples' not in data:
        return False

    samples = data['samples']
    if not isinstance(samples, torch.Tensor):
        return False

    if samples.ndim != 4:
        return False

    return True


def get_shape(data: Dict[str, Any]) -> tuple:
    """Get (B, C, H, W) shape from MULTIBAND_IMAGE."""
    return tuple(data['samples'].shape)


def get_num_channels(data: Dict[str, Any]) -> int:
    """Get number of channels from MULTIBAND_IMAGE."""
    return data['samples'].shape[1]


def get_channel_names(data: Dict[str, Any]) -> List[str]:
    """Get channel names from MULTIBAND_IMAGE."""
    return data.get('channel_names', [f"channel_{i}" for i in range(get_num_channels(data))])


def get_spatial_size(data: Dict[str, Any]) -> tuple:
    """Get (H, W) spatial size from MULTIBAND_IMAGE."""
    return data['samples'].shape[2], data['samples'].shape[3]


def multiband_to_numpy(data: Dict[str, Any]) -> np.ndarray:
    """Convert MULTIBAND_IMAGE samples to numpy array."""
    return data['samples'].cpu().numpy()


def numpy_to_multiband(
    arr: np.ndarray,
    channel_names: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create MULTIBAND_IMAGE from numpy array.

    Args:
        arr: Array of shape (B, C, H, W), (C, H, W), or (H, W, C)
        channel_names: Optional channel names
        metadata: Optional metadata

    Returns:
        MULTIBAND_IMAGE dict
    """
    # Handle different input shapes
    if arr.ndim == 2:
        # (H, W) -> (1, 1, H, W)
        arr = arr[np.newaxis, np.newaxis, :, :]
    elif arr.ndim == 3:
        # Could be (C, H, W) or (H, W, C)
        # Assume (C, H, W) if C is small, otherwise (H, W, C)
        if arr.shape[0] <= 16:
            # Likely (C, H, W)
            arr = arr[np.newaxis, :, :, :]
        else:
            # Likely (H, W, C) - transpose to (C, H, W)
            arr = np.transpose(arr, (2, 0, 1))[np.newaxis, :, :, :]
    elif arr.ndim == 4:
        pass  # Already (B, C, H, W)
    else:
        raise ValueError(f"Unsupported array shape: {arr.shape}")

    samples = torch.from_numpy(arr.astype(np.float32))
    return create_multiband(samples, channel_names, metadata)

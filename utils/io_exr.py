# SPDX-License-Identifier: MIT
# Copyright (C) 2025 ComfyUI-Multiband Contributors

"""OpenEXR file I/O for MULTIBAND_IMAGE (optional)."""

import os
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

try:
    import OpenEXR
    import Imath
    HAS_OPENEXR = True
except ImportError:
    HAS_OPENEXR = False


def load_exr(path: str, normalize: bool = False) -> Tuple[np.ndarray, Optional[List[str]], Dict[str, Any]]:
    """
    Load an OpenEXR file with arbitrary channels.

    Args:
        path: Path to .exr file
        normalize: Whether to normalize to [0, 1] range (usually not needed for EXR)

    Returns:
        Tuple of (array, channel_names, metadata)
    """
    if not HAS_OPENEXR:
        raise ImportError("OpenEXR is required for EXR support. Install with: pip install OpenEXR")

    exr_file = OpenEXR.InputFile(path)
    header = exr_file.header()

    # Get data window
    dw = header['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    # Get channel names
    channel_names = sorted(header['channels'].keys())

    # Read all channels
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    channels = []

    for name in channel_names:
        raw_data = exr_file.channel(name, pt)
        arr = np.frombuffer(raw_data, dtype=np.float32)
        arr = arr.reshape(height, width)
        channels.append(arr)

    # Stack to (C, H, W)
    arr = np.stack(channels, axis=0)

    if normalize and arr.max() > 1.0:
        arr = arr / arr.max()

    # Get metadata from header
    metadata = {}
    for key, value in header.items():
        if key not in ('channels', 'dataWindow', 'displayWindow'):
            try:
                # Convert Imath types to Python types
                if hasattr(value, 'x'):
                    metadata[key] = {'x': value.x, 'y': value.y}
                else:
                    metadata[key] = value
            except Exception:
                pass

    exr_file.close()

    return arr, channel_names, metadata


def save_exr(
    path: str,
    arr: np.ndarray,
    channel_names: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """
    Save array to OpenEXR file.

    Args:
        path: Output path
        arr: Array of shape (C, H, W) or (B, C, H, W)
        channel_names: Optional list of channel names
        metadata: Optional metadata dict (limited support)

    Returns:
        Saved file path
    """
    if not HAS_OPENEXR:
        raise ImportError("OpenEXR is required for EXR support. Install with: pip install OpenEXR")

    # Ensure .exr extension
    if not path.endswith('.exr'):
        path = path + '.exr'

    # Create directory if needed
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)

    # Handle batch dimension
    if arr.ndim == 4:
        arr = arr[0]  # Take first batch item

    if arr.ndim != 3:
        raise ValueError(f"Expected 3D or 4D array, got {arr.ndim}D")

    C, H, W = arr.shape

    # Generate channel names if not provided
    if channel_names is None:
        if C == 1:
            channel_names = ['Y']
        elif C == 3:
            channel_names = ['R', 'G', 'B']
        elif C == 4:
            channel_names = ['R', 'G', 'B', 'A']
        else:
            channel_names = [f'channel_{i}' for i in range(C)]

    if len(channel_names) != C:
        raise ValueError(f"channel_names length ({len(channel_names)}) != channels ({C})")

    # Ensure float32
    arr = arr.astype(np.float32)

    # Create header
    header = OpenEXR.Header(W, H)

    # Add channels to header
    channel_type = Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
    header['channels'] = {name: channel_type for name in channel_names}

    # Create output file
    out = OpenEXR.OutputFile(path, header)

    # Write channels
    channel_data = {}
    for i, name in enumerate(channel_names):
        channel_data[name] = arr[i].tobytes()

    out.writePixels(channel_data)
    out.close()

    return path


def is_available() -> bool:
    """Check if OpenEXR support is available."""
    return HAS_OPENEXR

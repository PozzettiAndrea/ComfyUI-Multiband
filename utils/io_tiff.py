# SPDX-License-Identifier: MIT
# Copyright (C) 2025 ComfyUI-Multiband Contributors

"""TIFF file I/O for MULTIBAND_IMAGE."""

import os
import json
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False


def load_tiff(path: str, normalize: bool = True) -> Tuple[np.ndarray, Optional[List[str]], Dict[str, Any]]:
    """
    Load a multi-band TIFF file.

    Supports:
    - Multi-page TIFF (each page = 1 channel)
    - Multi-channel TIFF (RGB, RGBA, or arbitrary channels)
    - GeoTIFF metadata

    Channel names are read from ImageDescription tag if present as JSON.

    Args:
        path: Path to .tiff/.tif file
        normalize: Whether to normalize to [0, 1] range

    Returns:
        Tuple of (array, channel_names, metadata)
    """
    if not HAS_TIFFFILE:
        raise ImportError("tifffile is required for TIFF support. Install with: pip install tifffile")

    with tifffile.TiffFile(path) as tif:
        # Read image data
        arr = tif.asarray()

        # Get metadata from first page
        metadata = {}
        channel_names = None

        if tif.pages:
            page = tif.pages[0]

            # Try to get channel names from ImageDescription
            if page.description:
                try:
                    desc = json.loads(page.description)
                    if isinstance(desc, dict):
                        if 'channel_names' in desc:
                            channel_names = desc['channel_names']
                        metadata.update({k: v for k, v in desc.items() if k != 'channel_names'})
                except (json.JSONDecodeError, TypeError):
                    metadata['description'] = page.description

            # Get other TIFF tags as metadata
            for tag in page.tags.values():
                if tag.name not in ('ImageDescription', 'ImageWidth', 'ImageLength', 'BitsPerSample'):
                    try:
                        metadata[tag.name] = tag.value
                    except Exception:
                        pass

    # Handle different TIFF layouts
    # tifffile returns shape based on the file structure:
    # - Multi-page grayscale: (pages, H, W)
    # - Single page RGB: (H, W, 3)
    # - Multi-page RGB: (pages, H, W, 3)

    if arr.ndim == 2:
        # Single grayscale image -> (1, H, W)
        arr = arr[np.newaxis, :, :]
    elif arr.ndim == 3:
        if arr.shape[-1] in (3, 4):
            # Likely (H, W, C) format -> transpose to (C, H, W)
            arr = np.transpose(arr, (2, 0, 1))
        # else already (pages, H, W) which is (C, H, W)
    elif arr.ndim == 4:
        # (pages, H, W, C) -> (pages*C, H, W)? or keep as is?
        # For now, assume multi-page with channels and flatten
        pages, h, w, c = arr.shape
        arr = arr.transpose(0, 3, 1, 2).reshape(pages * c, h, w)

    if normalize and arr.max() > 1.0:
        if arr.dtype == np.uint8:
            arr = arr.astype(np.float32) / 255.0
        elif arr.dtype == np.uint16:
            arr = arr.astype(np.float32) / 65535.0
        else:
            arr = arr.astype(np.float32) / arr.max()

    return arr, channel_names, metadata


def save_tiff(
    path: str,
    arr: np.ndarray,
    channel_names: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    compress: bool = True
) -> str:
    """
    Save array to multi-band TIFF file.

    Channel names and metadata are stored in ImageDescription as JSON.

    Args:
        path: Output path
        arr: Array of shape (C, H, W) or (B, C, H, W)
        channel_names: Optional list of channel names
        metadata: Optional metadata dict
        compress: Whether to use compression

    Returns:
        Saved file path
    """
    if not HAS_TIFFFILE:
        raise ImportError("tifffile is required for TIFF support. Install with: pip install tifffile")

    # Ensure .tiff extension
    if not path.endswith(('.tiff', '.tif')):
        path = path + '.tiff'

    # Create directory if needed
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)

    # Handle batch dimension - for now, take first batch item
    if arr.ndim == 4:
        arr = arr[0]  # (C, H, W)

    # Ensure (C, H, W) format for multi-page TIFF
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D or 4D array, got {arr.ndim}D")

    # Build description JSON
    desc_dict = {}
    if channel_names is not None:
        desc_dict['channel_names'] = channel_names
    if metadata:
        desc_dict.update(metadata)

    description = json.dumps(desc_dict) if desc_dict else None

    # Convert to appropriate dtype for saving
    if arr.dtype == np.float32 or arr.dtype == np.float64:
        # Keep as float32 for precision
        arr = arr.astype(np.float32)
    elif arr.max() <= 1.0:
        # Scale up to uint16 for better precision
        arr = (arr * 65535).astype(np.uint16)

    # Save as multi-page TIFF
    compression = 'zlib' if compress else None
    tifffile.imwrite(
        path,
        arr,
        compression=compression,
        description=description,
        photometric='minisblack',
    )

    return path

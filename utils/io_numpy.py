# SPDX-License-Identifier: MIT
# Copyright (C) 2025 ComfyUI-Multiband Contributors

"""NumPy file I/O for MULTIBAND_IMAGE."""

import os
from typing import Dict, List, Any, Optional, Tuple
import numpy as np


def load_numpy(path: str, normalize: bool = True) -> Tuple[np.ndarray, Optional[List[str]], Dict[str, Any]]:
    """
    Load a .npy file.

    Args:
        path: Path to .npy file
        normalize: Whether to normalize to [0, 1] range

    Returns:
        Tuple of (array, channel_names, metadata)
        channel_names and metadata are None for .npy files
    """
    arr = np.load(path)

    if normalize and arr.max() > 1.0:
        arr = arr.astype(np.float32) / 255.0 if arr.max() <= 255 else arr.astype(np.float32) / arr.max()

    return arr, None, {}


def save_numpy(path: str, arr: np.ndarray) -> str:
    """
    Save array to .npy file.

    Args:
        path: Output path
        arr: Array to save

    Returns:
        Saved file path
    """
    # Ensure .npy extension
    if not path.endswith('.npy'):
        path = path + '.npy'

    # Create directory if needed
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)

    np.save(path, arr)
    return path


def load_npz(path: str, normalize: bool = True) -> Tuple[np.ndarray, Optional[List[str]], Dict[str, Any]]:
    """
    Load a .npz file with optional channel_names and metadata.

    Expected keys:
    - 'samples': the main array (required)
    - 'channel_names': string array of channel names (optional)
    - Other keys become metadata

    Args:
        path: Path to .npz file
        normalize: Whether to normalize to [0, 1] range

    Returns:
        Tuple of (array, channel_names, metadata)
    """
    data = np.load(path, allow_pickle=True)

    # Get samples array
    if 'samples' in data:
        arr = data['samples']
    elif 'arr_0' in data:
        # Fallback for simple npz files
        arr = data['arr_0']
    else:
        # Try first key
        keys = list(data.keys())
        if not keys:
            raise ValueError(f"Empty npz file: {path}")
        arr = data[keys[0]]

    if normalize and arr.max() > 1.0:
        arr = arr.astype(np.float32) / 255.0 if arr.max() <= 255 else arr.astype(np.float32) / arr.max()

    # Get channel names
    channel_names = None
    if 'channel_names' in data:
        cn = data['channel_names']
        if isinstance(cn, np.ndarray):
            channel_names = cn.tolist()
        else:
            channel_names = list(cn)

    # Get metadata (all other keys)
    metadata = {}
    exclude_keys = {'samples', 'arr_0', 'channel_names'}
    for key in data.keys():
        if key not in exclude_keys:
            val = data[key]
            # Handle numpy arrays that might contain pickled dicts
            if isinstance(val, np.ndarray) and val.ndim == 0:
                val = val.item()
            metadata[key] = val

    return arr, channel_names, metadata


def save_npz(
    path: str,
    arr: np.ndarray,
    channel_names: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    compressed: bool = True
) -> str:
    """
    Save array to .npz file with optional channel_names and metadata.

    Args:
        path: Output path
        arr: Array to save
        channel_names: Optional list of channel names
        metadata: Optional metadata dict
        compressed: Whether to use compression

    Returns:
        Saved file path
    """
    # Ensure .npz extension
    if not path.endswith('.npz'):
        path = path + '.npz'

    # Create directory if needed
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)

    # Build save dict
    save_dict = {'samples': arr}

    if channel_names is not None:
        save_dict['channel_names'] = np.array(channel_names, dtype=object)

    if metadata:
        for key, val in metadata.items():
            save_dict[key] = val

    # Save
    if compressed:
        np.savez_compressed(path, **save_dict)
    else:
        np.savez(path, **save_dict)

    return path

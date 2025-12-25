# SPDX-License-Identifier: MIT
# Copyright (C) 2025 ComfyUI-Multiband Contributors

"""Save Multiband Image node."""

import os
from ..multiband_types import MULTIBAND_IMAGE, multiband_to_numpy, get_channel_names
from ..utils.io_numpy import save_numpy, save_npz
from ..utils.io_tiff import save_tiff
from ..utils.io_exr import save_exr, is_available as exr_available


class SaveMultibandImage:
    """
    Save a multi-band image to disk.

    Formats:
    - npy: Simple numpy array (no metadata)
    - npz: Numpy with channel names and metadata
    - tiff: Multi-page TIFF with metadata in description
    - exr: OpenEXR with named channels
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "multiband": (MULTIBAND_IMAGE,),
                "file_path": ("STRING", {
                    "default": "output/multiband",
                    "tooltip": "Output file path (extension determines format if not specified)"
                }),
                "format": (["npz", "npy", "tiff", "exr"], {
                    "default": "npz",
                    "tooltip": "Output file format"
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("saved_path",)
    FUNCTION = "save"
    CATEGORY = "multiband/io"
    OUTPUT_NODE = True

    def save(self, multiband: dict, file_path: str, format: str = "npz"):
        # Get numpy array
        arr = multiband_to_numpy(multiband)
        channel_names = get_channel_names(multiband)
        metadata = multiband.get('metadata', {})

        # Ensure output directory exists
        os.makedirs(os.path.dirname(file_path) or '.', exist_ok=True)

        # Save based on format
        if format == 'npy':
            saved_path = save_numpy(file_path, arr)
        elif format == 'npz':
            saved_path = save_npz(file_path, arr, channel_names, metadata)
        elif format == 'tiff':
            saved_path = save_tiff(file_path, arr, channel_names, metadata)
        elif format == 'exr':
            if not exr_available():
                raise ImportError("OpenEXR not installed. Install with: pip install OpenEXR")
            saved_path = save_exr(file_path, arr, channel_names, metadata)
        else:
            raise ValueError(f"Unsupported format: {format}")

        print(f"SaveMultibandImage: Saved to {saved_path}")
        print(f"  Shape: {arr.shape}")
        print(f"  Format: {format}")

        return (saved_path,)

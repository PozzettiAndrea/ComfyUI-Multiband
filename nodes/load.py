# SPDX-License-Identifier: MIT
# Copyright (C) 2025 ComfyUI-Multiband Contributors

"""Load Multiband Image node."""

import os
from ..multiband_types import MULTIBAND_IMAGE, numpy_to_multiband
from ..utils.io_numpy import load_numpy, load_npz
from ..utils.io_tiff import load_tiff
from ..utils.io_exr import load_exr, is_available as exr_available


class LoadMultibandImage:
    """
    Load a multi-band image from disk.

    Supports: .npy, .npz, .tiff, .tif, .exr
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": ("STRING", {
                    "default": "",
                    "tooltip": "Path to multiband image file (.npy, .npz, .tiff, .tif, .exr)"
                }),
            },
            "optional": {
                "normalize": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Normalize values to [0, 1] range"
                }),
            }
        }

    RETURN_TYPES = (MULTIBAND_IMAGE, "INT", "STRING")
    RETURN_NAMES = ("multiband", "num_channels", "channel_names")
    FUNCTION = "load"
    CATEGORY = "multiband/io"

    def load(self, file_path: str, normalize: bool = True):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()

        # Load based on extension
        if ext == '.npy':
            arr, channel_names, metadata = load_numpy(file_path, normalize)
        elif ext == '.npz':
            arr, channel_names, metadata = load_npz(file_path, normalize)
        elif ext in ('.tiff', '.tif'):
            arr, channel_names, metadata = load_tiff(file_path, normalize)
        elif ext == '.exr':
            if not exr_available():
                raise ImportError("OpenEXR not installed. Install with: pip install OpenEXR")
            arr, channel_names, metadata = load_exr(file_path, normalize)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

        # Convert to MULTIBAND_IMAGE
        multiband = numpy_to_multiband(arr, channel_names, metadata)

        num_channels = multiband['samples'].shape[1]
        names_str = ",".join(multiband['channel_names'])

        print(f"LoadMultibandImage: Loaded {file_path}")
        print(f"  Shape: {tuple(multiband['samples'].shape)}")
        print(f"  Channels: {num_channels}")
        print(f"  Names: {names_str[:100]}{'...' if len(names_str) > 100 else ''}")

        return (multiband, num_channels, names_str)

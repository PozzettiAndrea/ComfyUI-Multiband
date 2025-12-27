# SPDX-License-Identifier: MIT
# Copyright (C) 2025 ComfyUI-Multiband Contributors

"""Load Multiband Image node."""

import os
import hashlib
import folder_paths
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
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        # Filter to supported multiband formats
        supported_ext = {'.tiff', '.tif', '.npz', '.npy', '.exr'}
        files = [f for f in files if os.path.splitext(f)[1].lower() in supported_ext]
        return {
            "required": {
                "image": (sorted(files), {"tooltip": "Multiband image file (.npy, .npz, .tiff, .tif, .exr)"}),
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

    def load(self, image: str, normalize: bool = True):
        file_path = folder_paths.get_annotated_filepath(image)
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

    @classmethod
    def IS_CHANGED(cls, image, normalize=True):
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(cls, image, normalize=True):
        if not folder_paths.exists_annotated_filepath(image):
            return f"Invalid image file: {image}"
        return True

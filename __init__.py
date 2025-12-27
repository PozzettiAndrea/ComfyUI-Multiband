# SPDX-License-Identifier: MIT
# Copyright (C) 2025 ComfyUI-Multiband Contributors

"""
ComfyUI-Multiband - Multi-channel image handling for ComfyUI.

This package provides a MULTIBAND_IMAGE type for handling images with arbitrary
number of channels (segmentation masks, feature maps, spectral bands, etc.)
along with nodes for loading, saving, previewing, and manipulating them.
"""

import os

from .multiband_types import MULTIBAND_IMAGE
from .nodes import (
    LoadMultibandImage,
    SaveMultibandImage,
    PreviewMultibandImage,
    ImageToMultiband,
    MultibandToImage,
    MaskToMultiband,
    MultibandToMask,
    BatchToMultiband,
    ComposeMultiband,
    DecomposeMultiband,
    SelectMultibandChannels,
    ConcatMultiband,
    RenameMultibandChannels,
)


NODE_CLASS_MAPPINGS = {
    # I/O
    "MultibandLoad": LoadMultibandImage,
    "MultibandSave": SaveMultibandImage,
    "MultibandPreview": PreviewMultibandImage,

    # Conversion
    "MultibandFromImage": ImageToMultiband,
    "MultibandToImage": MultibandToImage,
    "MultibandFromMask": MaskToMultiband,
    "MultibandToMask": MultibandToMask,
    "MultibandFromBatch": BatchToMultiband,

    # Compose/Decompose
    "MultibandCompose": ComposeMultiband,
    "MultibandDecompose": DecomposeMultiband,

    # Operations
    "MultibandSelectChannels": SelectMultibandChannels,
    "MultibandConcat": ConcatMultiband,
    "MultibandRenameChannels": RenameMultibandChannels,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # I/O
    "MultibandLoad": "Load Multiband Image",
    "MultibandSave": "Save Multiband Image",
    "MultibandPreview": "Preview Multiband Image",

    # Conversion
    "MultibandFromImage": "Image to Multiband",
    "MultibandToImage": "Multiband to Image",
    "MultibandFromMask": "Mask to Multiband",
    "MultibandToMask": "Multiband to Mask",
    "MultibandFromBatch": "Batch to Multiband",

    # Compose/Decompose
    "MultibandCompose": "Compose Multiband",
    "MultibandDecompose": "Decompose Multiband",

    # Operations
    "MultibandSelectChannels": "Select Multiband Channels",
    "MultibandConcat": "Concat Multiband",
    "MultibandRenameChannels": "Rename Multiband Channels",
}

__all__ = [
    'NODE_CLASS_MAPPINGS',
    'NODE_DISPLAY_NAME_MAPPINGS',
    'MULTIBAND_IMAGE',
]

WEB_DIRECTORY = os.path.join(os.path.dirname(__file__), "web")

print("ComfyUI-Multiband loaded: 13 nodes registered")

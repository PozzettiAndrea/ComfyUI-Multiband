# SPDX-License-Identifier: MIT
# Copyright (C) 2025 ComfyUI-Multiband Contributors

"""Conversion nodes between IMAGE/MASK and MULTIBAND_IMAGE."""

import torch
from ..multiband_types import MULTIBAND_IMAGE, create_multiband


class ImageToMultiband:
    """
    Convert a ComfyUI IMAGE (B, H, W, 3) to MULTIBAND_IMAGE.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "channel_names": ("STRING", {
                    "default": "R,G,B",
                    "tooltip": "Comma-separated channel names"
                }),
            }
        }

    RETURN_TYPES = (MULTIBAND_IMAGE,)
    RETURN_NAMES = ("multiband",)
    FUNCTION = "convert"
    CATEGORY = "multiband/convert"

    def convert(self, image: torch.Tensor, channel_names: str = "R,G,B"):
        # IMAGE is (B, H, W, C) -> convert to (B, C, H, W)
        samples = image.permute(0, 3, 1, 2)

        names = [n.strip() for n in channel_names.split(',')]

        # Pad or truncate names to match channels
        C = samples.shape[1]
        if len(names) < C:
            names.extend([f"channel_{i}" for i in range(len(names), C)])
        names = names[:C]

        return (create_multiband(samples, names),)


class MultibandToImage:
    """
    Convert MULTIBAND_IMAGE to ComfyUI IMAGE by selecting 3 channels.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "multiband": (MULTIBAND_IMAGE,),
            },
            "optional": {
                "channels": ("STRING", {
                    "default": "0,1,2",
                    "tooltip": "Comma-separated channel indices (e.g., '0,1,2' or '2,1,0')"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "convert"
    CATEGORY = "multiband/convert"

    def convert(self, multiband: dict, channels: str = "0,1,2"):
        samples = multiband['samples']  # (B, C, H, W)
        B, C, H, W = samples.shape

        # Parse channel indices
        indices = [int(idx.strip()) for idx in channels.split(',')]

        # Ensure exactly 3 channels for RGB output
        while len(indices) < 3:
            indices.append(indices[-1] if indices else 0)
        indices = indices[:3]

        # Clamp indices
        indices = [min(max(0, idx), C - 1) for idx in indices]

        # Select channels
        selected = samples[:, indices, :, :]  # (B, 3, H, W)

        # Convert to IMAGE format (B, H, W, C)
        image = selected.permute(0, 2, 3, 1)
        image = torch.clamp(image, 0, 1)

        return (image,)


class MaskToMultiband:
    """
    Convert a ComfyUI MASK (B, H, W) to single-channel MULTIBAND_IMAGE.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
            },
            "optional": {
                "channel_name": ("STRING", {
                    "default": "mask",
                    "tooltip": "Name for the mask channel"
                }),
            }
        }

    RETURN_TYPES = (MULTIBAND_IMAGE,)
    RETURN_NAMES = ("multiband",)
    FUNCTION = "convert"
    CATEGORY = "multiband/convert"

    def convert(self, mask: torch.Tensor, channel_name: str = "mask"):
        # MASK is (B, H, W) -> convert to (B, 1, H, W)
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        samples = mask.unsqueeze(1)

        return (create_multiband(samples, [channel_name]),)


class MultibandToMask:
    """
    Extract a single channel from MULTIBAND_IMAGE as MASK.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "multiband": (MULTIBAND_IMAGE,),
            },
            "optional": {
                "channel": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 1000,
                    "tooltip": "Channel index to extract"
                }),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "convert"
    CATEGORY = "multiband/convert"

    def convert(self, multiband: dict, channel: int = 0):
        samples = multiband['samples']  # (B, C, H, W)
        C = samples.shape[1]

        # Clamp channel index
        channel = min(max(0, channel), C - 1)

        # Extract channel as MASK (B, H, W)
        mask = samples[:, channel, :, :]

        return (mask,)

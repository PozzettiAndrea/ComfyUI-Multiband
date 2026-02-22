# SPDX-License-Identifier: MIT
# Copyright (C) 2025 ComfyUI-Multiband Contributors

"""Channel operation nodes for MULTIBAND_IMAGE."""

import torch
import comfy.utils
from ..multiband_types import MULTIBAND_IMAGE, create_multiband, get_channel_names, get_num_channels

MAX_RESOLUTION = 16384


class SelectMultibandChannels:
    """
    Select specific channels from a MULTIBAND_IMAGE by index or name.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "multiband": (MULTIBAND_IMAGE,),
                "channels": ("STRING", {
                    "default": "0,1,2",
                    "tooltip": "Comma-separated channel indices (0,1,2) or names (R,G,B)"
                }),
            },
        }

    RETURN_TYPES = (MULTIBAND_IMAGE,)
    RETURN_NAMES = ("multiband",)
    FUNCTION = "select"
    CATEGORY = "multiband/operations"

    def select(self, multiband: dict, channels: str):
        samples = multiband['samples']  # (B, C, H, W)
        channel_names = get_channel_names(multiband)
        C = get_num_channels(multiband)

        # Parse channel specifiers
        specs = [s.strip() for s in channels.split(',')]
        indices = []

        for spec in specs:
            if spec.isdigit():
                # Numeric index
                idx = int(spec)
                if 0 <= idx < C:
                    indices.append(idx)
            else:
                # Try to find by name
                if spec in channel_names:
                    indices.append(channel_names.index(spec))

        if not indices:
            raise ValueError(f"No valid channels found in: {channels}")

        # Select channels
        selected = samples[:, indices, :, :]
        selected_names = [channel_names[i] for i in indices]

        print(f"SelectMultibandChannels: Selected {len(indices)} channels: {selected_names}")

        return (create_multiband(selected, selected_names, multiband.get('metadata')),)


class ConcatMultiband:
    """
    Concatenate two MULTIBAND_IMAGEs along the channel dimension.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "multiband_1": (MULTIBAND_IMAGE,),
                "multiband_2": (MULTIBAND_IMAGE,),
            },
        }

    RETURN_TYPES = (MULTIBAND_IMAGE,)
    RETURN_NAMES = ("multiband",)
    FUNCTION = "concat"
    CATEGORY = "multiband/operations"

    def concat(self, multiband_1: dict, multiband_2: dict):
        samples_1 = multiband_1['samples']
        samples_2 = multiband_2['samples']

        # Verify compatible shapes
        if samples_1.shape[0] != samples_2.shape[0]:
            raise ValueError(f"Batch size mismatch: {samples_1.shape[0]} vs {samples_2.shape[0]}")
        if samples_1.shape[2:] != samples_2.shape[2:]:
            raise ValueError(f"Spatial size mismatch: {samples_1.shape[2:]} vs {samples_2.shape[2:]}")

        # Concatenate
        samples = torch.cat([samples_1, samples_2], dim=1)

        # Combine names
        names_1 = get_channel_names(multiband_1)
        names_2 = get_channel_names(multiband_2)
        names = names_1 + names_2

        # Combine metadata
        metadata = {}
        metadata.update(multiband_1.get('metadata', {}))
        metadata.update(multiband_2.get('metadata', {}))

        print(f"ConcatMultiband: {len(names_1)} + {len(names_2)} = {len(names)} channels")

        return (create_multiband(samples, names, metadata),)


class RenameMultibandChannels:
    """
    Rename channels in a MULTIBAND_IMAGE.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "multiband": (MULTIBAND_IMAGE,),
                "channel_names": ("STRING", {
                    "default": "",
                    "tooltip": "Comma-separated new channel names"
                }),
            },
        }

    RETURN_TYPES = (MULTIBAND_IMAGE,)
    RETURN_NAMES = ("multiband",)
    FUNCTION = "rename"
    CATEGORY = "multiband/operations"

    def rename(self, multiband: dict, channel_names: str):
        samples = multiband['samples']
        C = get_num_channels(multiband)

        # Parse new names
        new_names = [n.strip() for n in channel_names.split(',')]

        # Pad if needed
        while len(new_names) < C:
            new_names.append(f"channel_{len(new_names)}")

        # Truncate if too many
        new_names = new_names[:C]

        print(f"RenameMultibandChannels: Renamed to {new_names}")

        return (create_multiband(samples, new_names, multiband.get('metadata')),)


class ResizeMultiband:
    """
    Resize a MULTIBAND_IMAGE to specific dimensions.
    Mirrors ComfyUI's built-in ImageScale (Upscale Image) node.
    """

    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]
    crop_methods = ["disabled", "center"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "multiband": (MULTIBAND_IMAGE,),
                "upscale_method": (cls.upscale_methods,),
                "width": ("INT", {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "height": ("INT", {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "crop": (cls.crop_methods,),
            },
        }

    RETURN_TYPES = (MULTIBAND_IMAGE,)
    RETURN_NAMES = ("multiband",)
    FUNCTION = "upscale"
    CATEGORY = "multiband/operations"

    def upscale(self, multiband: dict, upscale_method: str, width: int, height: int, crop: str):
        samples = multiband['samples']  # (B, C, H, W)

        if width == 0 and height == 0:
            s = samples
        else:
            if width == 0:
                width = max(1, round(samples.shape[3] * height / samples.shape[2]))
            elif height == 0:
                height = max(1, round(samples.shape[2] * width / samples.shape[3]))

            s = comfy.utils.common_upscale(samples, width, height, upscale_method, crop)

        print(f"ResizeMultiband: {samples.shape[2]}x{samples.shape[3]} -> {s.shape[2]}x{s.shape[3]}")

        return (create_multiband(s, get_channel_names(multiband), multiband.get('metadata')),)

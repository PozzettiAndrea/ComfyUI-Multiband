# SPDX-License-Identifier: MIT
# Copyright (C) 2025 ComfyUI-Multiband Contributors

"""Preview Multiband Image node with JavaScript UI integration."""

import os
import numpy as np
from PIL import Image

import folder_paths

from ..multiband_types import MULTIBAND_IMAGE, get_num_channels, get_channel_names
from ..utils.visualization import create_preview


class PreviewMultibandImage:
    """
    Create a visual preview of a multi-band image with interactive channel selection.

    Modes:
    - rgb_first3: Show first 3 channels as RGB
    - single_channel: Show one channel with colormap
    - channel_grid: Show all channels in a grid

    The JavaScript UI provides a channel selector dropdown below the preview.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "multiband": (MULTIBAND_IMAGE,),
            },
            "optional": {
                "mode": (["rgb_first3", "single_channel"], {
                    "default": "rgb_first3",
                    "tooltip": "Visualization mode"
                }),
                "channel_index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 1000,
                    "tooltip": "Channel to show (for single_channel mode)"
                }),
                "colormap": (["viridis", "plasma", "gray", "jet"], {
                    "default": "gray",
                    "tooltip": "Colormap for single-channel visualization"
                }),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("preview",)
    FUNCTION = "preview"
    CATEGORY = "multiband/visualization"
    OUTPUT_NODE = True

    def preview(
        self,
        multiband: dict,
        mode: str = "rgb_first3",
        channel_index: int = 0,
        colormap: str = "gray",
        prompt=None,
        extra_pnginfo=None
    ):
        samples = multiband['samples']
        num_channels = get_num_channels(multiband)
        channel_names = get_channel_names(multiband)

        print(f"PreviewMultibandImage: Mode={mode}, Channels={num_channels}, Names={channel_names}")

        # Clamp channel_index
        channel_index = min(channel_index, num_channels - 1)

        # Create preview tensor (B, H, W, 3)
        preview_tensor = create_preview(
            samples,
            mode=mode,
            channel_index=channel_index,
            colormap=colormap
        )

        # Save preview images to temp folder for UI display
        results = []
        output_dir = folder_paths.get_temp_directory()

        for i in range(preview_tensor.shape[0]):
            # Convert tensor to PIL Image
            img_arr = (preview_tensor[i].numpy() * 255).astype(np.uint8)
            img = Image.fromarray(img_arr)

            # Generate unique filename
            filename = f"multiband_preview_{i:05d}.png"
            filepath = os.path.join(output_dir, filename)

            # Save image
            img.save(filepath, compress_level=4)

            results.append({
                "filename": filename,
                "subfolder": "",
                "type": "temp"
            })

        return {
            "ui": {
                "images": results,
                "channel_names": [channel_names],
            },
            "result": (preview_tensor,)
        }

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

    Shows a single channel at a time in grayscale. Use the channel_index selector
    below the preview to switch between channels.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "multiband": (MULTIBAND_IMAGE,),
            },
            "optional": {
                "channel_index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 1000,
                    "tooltip": "Channel to show"
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
        channel_index: int = 0,
        prompt=None,
        extra_pnginfo=None
    ):
        samples = multiband['samples']
        num_channels = get_num_channels(multiband)
        channel_names = get_channel_names(multiband)
        batch_size = samples.shape[0]

        print(f"PreviewMultibandImage: Batch={batch_size}, Channels={num_channels}, Names={channel_names}")

        # Clamp channel_index for the output tensor
        channel_index = min(channel_index, num_channels - 1)

        output_dir = folder_paths.get_temp_directory()

        # Generate unique prefix for this execution
        import uuid
        exec_id = uuid.uuid4().hex[:8]

        # Render ALL channels for ALL batch images for dynamic JS switching
        # Structure: all_channel_images[channel_idx] = [batch0_img, batch1_img, ...]
        all_channel_images = []
        for ch_idx in range(num_channels):
            preview_tensor = create_preview(
                samples,
                mode="single_channel",
                channel_index=ch_idx,
                colormap="gray"
            )

            # Save ALL batch images for this channel
            channel_batch_images = []
            for batch_idx in range(batch_size):
                img_arr = (preview_tensor[batch_idx].numpy() * 255).astype(np.uint8)
                img = Image.fromarray(img_arr)

                filename = f"multiband_{exec_id}_ch{ch_idx:03d}_b{batch_idx:03d}.png"
                filepath = os.path.join(output_dir, filename)
                img.save(filepath, compress_level=4)

                channel_batch_images.append({
                    "filename": filename,
                    "subfolder": "",
                    "type": "temp"
                })

            all_channel_images.append(channel_batch_images)

        # Return selected channel's images for initial display
        selected_preview = create_preview(
            samples,
            mode="single_channel",
            channel_index=channel_index,
            colormap="gray"
        )

        return {
            "ui": {
                "images": all_channel_images[channel_index],  # All batch images for selected channel
                "all_channel_images": [all_channel_images],   # [channel][batch] structure
                "channel_names": [channel_names],
                "current_channel": [channel_index],
                "batch_size": [batch_size],
            },
            "result": (selected_preview,)
        }

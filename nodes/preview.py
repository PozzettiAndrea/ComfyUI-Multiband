# SPDX-License-Identifier: MIT
# Copyright (C) 2025 ComfyUI-Multiband Contributors

"""Preview Multiband Image node."""

from ..multiband_types import MULTIBAND_IMAGE, get_num_channels
from ..utils.visualization import create_preview


class PreviewMultibandImage:
    """
    Create a visual preview of a multi-band image.

    Modes:
    - rgb_first3: Show first 3 channels as RGB
    - single_channel: Show one channel with colormap
    - channel_grid: Show all channels in a grid
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "multiband": (MULTIBAND_IMAGE,),
            },
            "optional": {
                "mode": (["rgb_first3", "single_channel", "channel_grid"], {
                    "default": "rgb_first3",
                    "tooltip": "Visualization mode"
                }),
                "channel_index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 1000,
                    "tooltip": "Channel to show (for single_channel mode)"
                }),
                "grid_cols": ("INT", {
                    "default": 4,
                    "min": 1,
                    "max": 16,
                    "tooltip": "Grid columns (for channel_grid mode)"
                }),
                "colormap": (["viridis", "plasma", "gray", "jet"], {
                    "default": "viridis",
                    "tooltip": "Colormap for single-channel visualization"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("preview",)
    FUNCTION = "preview"
    CATEGORY = "multiband/visualization"

    def preview(
        self,
        multiband: dict,
        mode: str = "rgb_first3",
        channel_index: int = 0,
        grid_cols: int = 4,
        colormap: str = "viridis"
    ):
        samples = multiband['samples']
        num_channels = get_num_channels(multiband)

        print(f"PreviewMultibandImage: Mode={mode}, Channels={num_channels}")

        # Clamp channel_index
        channel_index = min(channel_index, num_channels - 1)

        # Create preview
        preview = create_preview(
            samples,
            mode=mode,
            channel_index=channel_index,
            grid_cols=grid_cols,
            colormap=colormap
        )

        return (preview,)

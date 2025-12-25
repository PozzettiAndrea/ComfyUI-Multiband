# SPDX-License-Identifier: MIT
# Copyright (C) 2025 ComfyUI-Multiband Contributors

"""Compose Multiband node - stack multiple inputs into one multiband."""

import torch
from ..multiband_types import MULTIBAND_IMAGE, create_multiband


class ComposeMultiband:
    """
    Compose multiple masks/images/multiband inputs into a single MULTIBAND_IMAGE.

    Each input can be:
    - MASK (B, H, W) -> becomes 1 channel
    - IMAGE (B, H, W, 3) -> becomes 3 channels
    - MULTIBAND_IMAGE -> all channels are added
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "input_1": ("*", {"tooltip": "First input (MASK, IMAGE, or MULTIBAND_IMAGE)"}),
                "input_2": ("*", {"tooltip": "Second input (optional)"}),
                "input_3": ("*", {"tooltip": "Third input (optional)"}),
                "input_4": ("*", {"tooltip": "Fourth input (optional)"}),
                "input_5": ("*", {"tooltip": "Fifth input (optional)"}),
                "input_6": ("*", {"tooltip": "Sixth input (optional)"}),
                "input_7": ("*", {"tooltip": "Seventh input (optional)"}),
                "input_8": ("*", {"tooltip": "Eighth input (optional)"}),
                "channel_names": ("STRING", {
                    "default": "",
                    "tooltip": "Comma-separated channel names (optional, auto-generated if empty)"
                }),
            }
        }

    RETURN_TYPES = (MULTIBAND_IMAGE,)
    RETURN_NAMES = ("multiband",)
    FUNCTION = "compose"
    CATEGORY = "multiband/compose"

    def _to_channels(self, inp, name_prefix: str) -> tuple:
        """Convert input to (B, C, H, W) tensor and channel names."""
        if inp is None:
            return None, []

        if isinstance(inp, dict) and 'samples' in inp:
            # MULTIBAND_IMAGE
            samples = inp['samples']
            names = inp.get('channel_names', [f"{name_prefix}_{i}" for i in range(samples.shape[1])])
            return samples, names

        if isinstance(inp, torch.Tensor):
            if inp.ndim == 3:
                # MASK: (B, H, W) -> (B, 1, H, W)
                return inp.unsqueeze(1), [name_prefix]
            elif inp.ndim == 4:
                if inp.shape[-1] in (1, 3, 4):
                    # IMAGE: (B, H, W, C) -> (B, C, H, W)
                    samples = inp.permute(0, 3, 1, 2)
                    C = samples.shape[1]
                    if C == 3:
                        names = [f"{name_prefix}_R", f"{name_prefix}_G", f"{name_prefix}_B"]
                    elif C == 4:
                        names = [f"{name_prefix}_R", f"{name_prefix}_G", f"{name_prefix}_B", f"{name_prefix}_A"]
                    else:
                        names = [f"{name_prefix}_{i}" for i in range(C)]
                    return samples, names
                else:
                    # Assume already (B, C, H, W)
                    C = inp.shape[1]
                    return inp, [f"{name_prefix}_{i}" for i in range(C)]

        raise ValueError(f"Unsupported input type: {type(inp)}")

    def compose(
        self,
        input_1=None, input_2=None, input_3=None, input_4=None,
        input_5=None, input_6=None, input_7=None, input_8=None,
        channel_names: str = ""
    ):
        inputs = [input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8]

        all_channels = []
        all_names = []

        for i, inp in enumerate(inputs):
            if inp is None:
                continue
            channels, names = self._to_channels(inp, f"input_{i+1}")
            if channels is not None:
                all_channels.append(channels)
                all_names.extend(names)

        if not all_channels:
            raise ValueError("At least one input is required")

        # Verify all inputs have same batch size and spatial dimensions
        B, _, H, W = all_channels[0].shape
        for i, ch in enumerate(all_channels[1:], 2):
            if ch.shape[0] != B:
                raise ValueError(f"Batch size mismatch: input_1 has B={B}, input_{i} has B={ch.shape[0]}")
            if ch.shape[2] != H or ch.shape[3] != W:
                raise ValueError(f"Spatial size mismatch: input_1 has {H}x{W}, input_{i} has {ch.shape[2]}x{ch.shape[3]}")

        # Concatenate all channels
        samples = torch.cat(all_channels, dim=1)

        # Use custom channel names if provided
        if channel_names.strip():
            custom_names = [n.strip() for n in channel_names.split(',')]
            # Pad if needed
            while len(custom_names) < samples.shape[1]:
                custom_names.append(f"channel_{len(custom_names)}")
            all_names = custom_names[:samples.shape[1]]

        print(f"ComposeMultiband: Created {samples.shape[1]} channels from {len([i for i in inputs if i is not None])} inputs")

        return (create_multiband(samples, all_names),)

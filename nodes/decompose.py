# SPDX-License-Identifier: MIT
# Copyright (C) 2025 ComfyUI-Multiband Contributors

"""Decompose Multiband node - split multiband into individual channels."""

import torch
from ..multiband_types import MULTIBAND_IMAGE, get_num_channels, get_channel_names


class DecomposeMultiband:
    """
    Decompose a MULTIBAND_IMAGE into individual MASK channels.

    Outputs:
    - all_masks: All channels as a batched MASK (C, H, W)
    - channel_0 through channel_7: Individual channels (if they exist)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "multiband": (MULTIBAND_IMAGE,),
            },
        }

    RETURN_TYPES = ("MASK", "MASK", "MASK", "MASK", "MASK", "MASK", "MASK", "MASK", "MASK", "STRING")
    RETURN_NAMES = ("all_masks", "channel_0", "channel_1", "channel_2", "channel_3", "channel_4", "channel_5", "channel_6", "channel_7", "channel_names")
    FUNCTION = "decompose"
    CATEGORY = "multiband/compose"

    def decompose(self, multiband: dict):
        samples = multiband['samples']  # (B, C, H, W)
        B, C, H, W = samples.shape
        channel_names = get_channel_names(multiband)

        # For all_masks, reshape to (B*C, H, W) or just (C, H, W) if B=1
        if B == 1:
            all_masks = samples.squeeze(0)  # (C, H, W)
        else:
            all_masks = samples.reshape(B * C, H, W)

        # Extract individual channels (up to 8)
        channels = []
        for i in range(8):
            if i < C:
                # Get channel i across all batches
                ch = samples[:, i, :, :]  # (B, H, W)
                channels.append(ch)
            else:
                # Return empty mask for non-existent channels
                channels.append(torch.zeros(B, H, W, dtype=samples.dtype, device=samples.device))

        names_str = ",".join(channel_names)

        print(f"DecomposeMultiband: Split into {C} channels")

        return (all_masks, *channels, names_str)

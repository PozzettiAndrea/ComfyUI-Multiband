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


class MultibandToMasks:
    """
    Extract one or more channels from MULTIBAND_IMAGE as MASK batch.

    Supports single channel ("1") or multiple ("0,1,2,3,4,5,6").
    Can also auto-detect channels by name containing "mask".
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "multiband": (MULTIBAND_IMAGE,),
            },
            "optional": {
                "channels": ("STRING", {
                    "default": "0",
                    "tooltip": "Channel indices to extract. Single (e.g., '1') or comma-separated (e.g., '0,1,2,3'). Ignored if auto_detect_masks is enabled."
                }),
                "auto_detect_masks": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "When enabled, automatically finds all channels with 'mask' in their name and returns them. Overrides the 'channels' parameter."
                }),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("masks",)
    FUNCTION = "convert"
    CATEGORY = "multiband/convert"

    def convert(self, multiband: dict, channels: str = "0", auto_detect_masks: bool = False):
        samples = multiband['samples']  # (B, C, H, W)
        B, C, H, W = samples.shape
        channel_names = multiband.get('channel_names', [])

        if auto_detect_masks and channel_names:
            # Find all channels with "mask" in their name (case-insensitive)
            indices = [i for i, name in enumerate(channel_names) if 'mask' in name.lower()]
            if indices:
                print(f"MultibandToMasks: Auto-detected {len(indices)} mask channels: {[channel_names[i] for i in indices]}")
            else:
                print("MultibandToMasks: No channels with 'mask' in name found, falling back to channel indices")
                indices = [int(idx.strip()) for idx in channels.split(',') if idx.strip()]
        else:
            # Parse channel indices from comma-separated string
            indices = [int(idx.strip()) for idx in channels.split(',') if idx.strip()]

        if not indices:
            indices = [0]

        # Clamp indices to valid range
        indices = [min(max(0, idx), C - 1) for idx in indices]

        # Extract channels and stack into batch
        # For each requested channel, take all batch items
        # Output shape: (len(indices) * B, H, W) or just (len(indices), H, W) if B=1
        mask_list = []
        for idx in indices:
            mask_list.append(samples[:, idx, :, :])  # (B, H, W)

        # Stack along batch dimension: (N, H, W) where N = len(indices) * B
        masks = torch.cat(mask_list, dim=0)

        return (masks,)


class BatchToMultiband:
    """
    Convert image batch and/or mask batch into a single MULTIBAND_IMAGE.

    Flattens batch dimension into channels:
    - Images: img_01_r, img_01_g, img_01_b, img_02_r, ...
    - Masks: mask_01, mask_02, ...

    Output has batch=1 with all inputs as channels.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "images": ("IMAGE", {
                    "tooltip": "Batch of images (B, H, W, C). Each image becomes C channels named img_XX_r/g/b/a"
                }),
                "masks": ("MASK", {
                    "tooltip": "Batch of masks (B, H, W). Each mask becomes one channel named mask_XX"
                }),
            }
        }

    RETURN_TYPES = (MULTIBAND_IMAGE,)
    RETURN_NAMES = ("multiband",)
    FUNCTION = "convert"
    CATEGORY = "multiband/convert"

    def convert(self, images: torch.Tensor = None, masks: torch.Tensor = None):
        if images is None and masks is None:
            raise ValueError("BatchToMultiband: Must provide at least one of 'images' or 'masks'")

        all_channels = []
        all_names = []
        H, W = None, None

        # Process images
        if images is not None:
            # IMAGE is (B, H, W, C) -> permute to (B, C, H, W)
            img_bchw = images.permute(0, 3, 1, 2)
            B, C, H, W = img_bchw.shape

            # Channel suffix names
            channel_suffixes = ['r', 'g', 'b', 'a'] if C <= 4 else [str(i) for i in range(C)]

            # Flatten batch into channels: (B, C, H, W) -> (1, B*C, H, W)
            for b in range(B):
                for c in range(C):
                    channel = img_bchw[b, c, :, :]  # (H, W)
                    all_channels.append(channel)
                    all_names.append(f"img_{b+1:02d}_{channel_suffixes[c]}")

            print(f"BatchToMultiband: Processed {B} images with {C} channels each -> {B*C} channels")

        # Process masks
        if masks is not None:
            # MASK is (B, H, W) or (H, W)
            if masks.ndim == 2:
                masks = masks.unsqueeze(0)

            B_mask, H_mask, W_mask = masks.shape

            # Validate dimensions match if we have both
            if H is not None and W is not None:
                if H_mask != H or W_mask != W:
                    raise ValueError(f"BatchToMultiband: Dimension mismatch. Images are {H}x{W}, masks are {H_mask}x{W_mask}")
            else:
                H, W = H_mask, W_mask

            # Each mask becomes a channel
            for b in range(B_mask):
                channel = masks[b, :, :]  # (H, W)
                all_channels.append(channel)
                all_names.append(f"mask_{b+1:02d}")

            print(f"BatchToMultiband: Processed {B_mask} masks -> {B_mask} channels")

        # Stack all channels: list of (H, W) -> (1, total_channels, H, W)
        samples = torch.stack(all_channels, dim=0).unsqueeze(0)  # (1, C, H, W)

        print(f"BatchToMultiband: Output shape {samples.shape}, channels: {all_names}")

        return (create_multiband(samples, all_names),)

"""U-Net deep learning model architecture with ResNet-34 encoder for 6-band Sentinel-2 imagery.

Implements an explicit and documented initialization strategy for adapting ImageNet
pretrained weights to 6-band multispectral satellite input (B2, B3, B4, B8, B11, B12),
along with masked BCE and Dice segmentation loss functions that strictly ignore nodata pixels.
"""

from __future__ import annotations

from typing import Literal, Optional, Tuple

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Weight Adaptation for 6 Multispectral Bands
# ============================================================================

def adapt_resnet34_first_conv(
    model: nn.Module,
    in_channels: int = 6,
    init_strategy: Literal["channel_mapping_and_average", "repeat_rgb", "random"] = "channel_mapping_and_average",
) -> None:
    """Adapts a ResNet-34 encoder's first convolution (conv1) from 3 RGB channels to 6 Sentinel-2 channels.

    Band Ordering in Sentinel-2 Multispectral Rasters:
        Index 0: B2 (Blue, 490 nm)
        Index 1: B3 (Green, 560 nm)
        Index 2: B4 (Red, 665 nm)
        Index 3: B8 (Near-Infrared, 842 nm)
        Index 4: B11 (SWIR-1, 1610 nm)
        Index 5: B12 (SWIR-2, 2190 nm)

    ImageNet Pretrained conv1 Weight Ordering:
        Channel 0: Red
        Channel 1: Green
        Channel 2: Blue

    Initialization Strategy ('channel_mapping_and_average'):
        - Channel 0 (B2 Blue):  Transfers ImageNet Blue filter (Index 2).
        - Channel 1 (B3 Green): Transfers ImageNet Green filter (Index 1).
        - Channel 2 (B4 Red):   Transfers ImageNet Red filter (Index 0).
        - Channel 3 (B8 NIR):   Initializes from the mean of RGB filters (Index 0:3).
        - Channel 4 (B11 SWIR1): Initializes from the mean of RGB filters (Index 0:3).
        - Channel 5 (B12 SWIR2): Initializes from the mean of RGB filters (Index 0:3).
        - Energy Preservation:  The weights are scaled by (3.0 / in_channels) = 0.5 to
          preserve the overall activation variance entering the residual network.
    """
    old_conv = model.encoder.conv1
    old_weights = old_conv.weight.data  # Shape: (out_channels=64, in_channels=3, kernel=7, kernel=7)
    out_channels, _, k_h, k_w = old_weights.shape

    new_conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=old_conv.bias is not None,
    )

    new_weights = torch.zeros((out_channels, in_channels, k_h, k_w), dtype=old_weights.dtype)

    if init_strategy == "channel_mapping_and_average" and in_channels >= 3:
        # Scale factor to preserve filter energy variance: 3 / 6 = 0.5
        scale = 3.0 / float(in_channels)
        rgb_mean = old_weights.mean(dim=1, keepdim=True) * scale

        # Direct optical mapping: B2->Blue, B3->Green, B4->Red
        new_weights[:, 0:1, :, :] = old_weights[:, 2:3, :, :] * scale  # Blue
        new_weights[:, 1:2, :, :] = old_weights[:, 1:2, :, :] * scale  # Green
        new_weights[:, 2:3, :, :] = old_weights[:, 0:1, :, :] * scale  # Red

        # Infrared mapping: NIR, SWIR1, SWIR2 initialized with RGB mean
        for c in range(3, in_channels):
            new_weights[:, c : c + 1, :, :] = rgb_mean

    elif init_strategy == "repeat_rgb":
        scale = 3.0 / float(in_channels)
        for c in range(in_channels):
            new_weights[:, c : c + 1, :, :] = old_weights[:, (c % 3) : (c % 3) + 1, :, :] * scale

    else:
        # Random initialization
        nn.init.kaiming_normal_(new_weights, mode="fan_out", nonlinearity="relu")

    new_conv.weight.data = new_weights
    if old_conv.bias is not None:
        new_conv.bias.data = old_conv.bias.data

    model.encoder.conv1 = new_conv


# ============================================================================
# Model Factory
# ============================================================================

def build_unet_resnet34(
    in_channels: int = 6,
    num_classes: int = 1,
    encoder_weights: Optional[str] = "imagenet",
    first_conv_init: Literal["channel_mapping_and_average", "repeat_rgb", "random"] = "channel_mapping_and_average",
) -> smp.Unet:
    """Builds a U-Net model with a ResNet-34 encoder adapted to 6 Sentinel-2 bands.

    Args:
        in_channels: Number of input spectral bands (default 6).
        num_classes: Number of output segmentation classes (default 1 for binary forest).
        encoder_weights: Pretrained weights for ResNet-34 ('imagenet' or None).
        first_conv_init: Strategy for initializing the 6 input channels.

    Returns:
        Adapted smp.Unet module outputting raw logits.
    """
    if encoder_weights == "imagenet":
        # Instantiate with 3 channels to download/load ImageNet weights, then adapt conv1
        model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_classes,
            activation=None,  # Output raw logits
        )
        adapt_resnet34_first_conv(model, in_channels=in_channels, init_strategy=first_conv_init)
    else:
        # Without pretrained weights, instantiate directly
        model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=in_channels,
            classes=num_classes,
            activation=None,
        )

    return model


# ============================================================================
# Masked Loss Functions (Excludes Nodata Pixels)
# ============================================================================

class MaskedBCEWithLogitsLoss(nn.Module):
    """Binary Cross-Entropy loss that strictly ignores unobserved/nodata pixels."""

    def __init__(self, ignore_index: int = 255):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Shape (B, 1, H, W) or (B, H, W) raw model predictions.
            targets: Shape (B, 1, H, W) or (B, H, W) ground truth in {0, 1, ignore_index}.
        """
        if logits.dim() == 4 and logits.shape[1] == 1:
            logits = logits.squeeze(1)
        if targets.dim() == 4 and targets.shape[1] == 1:
            targets = targets.squeeze(1)

        valid_mask = (targets != self.ignore_index)
        if not valid_mask.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        valid_logits = logits[valid_mask]
        valid_targets = targets[valid_mask].float()

        return F.binary_cross_entropy_with_logits(valid_logits, valid_targets)


class MaskedDiceLoss(nn.Module):
    """Soft Dice loss computed strictly over valid observation pixels."""

    def __init__(self, ignore_index: int = 255, smooth: float = 1.0):
        super().__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.dim() == 4 and logits.shape[1] == 1:
            logits = logits.squeeze(1)
        if targets.dim() == 4 and targets.shape[1] == 1:
            targets = targets.squeeze(1)

        probs = torch.sigmoid(logits)

        valid_mask = (targets != self.ignore_index)
        if not valid_mask.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        valid_probs = probs[valid_mask]
        valid_targets = targets[valid_mask].float()

        intersection = (valid_probs * valid_targets).sum()
        cardinality = valid_probs.sum() + valid_targets.sum()

        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice


class CombinedMaskedLoss(nn.Module):
    """Combined Masked BCE + Masked Dice loss with configurable weights."""

    def __init__(self, bce_weight: float = 0.50, dice_weight: float = 0.50, ignore_index: int = 255):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce_loss = MaskedBCEWithLogitsLoss(ignore_index=ignore_index)
        self.dice_loss = MaskedDiceLoss(ignore_index=ignore_index)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = self.bce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)
        return self.bce_weight * bce + self.dice_weight * dice

import pytest
import torch
from src.model import build_unet_resnet34

def test_unet_six_channel_input_and_output_shape():
    # Build model without pretrained weights for speed
    model = build_unet_resnet34(in_channels=6, encoder_weights=None)
    # Verify first convolution has 6 input channels
    conv1 = model.encoder.conv1
    assert conv1.weight.shape[1] == 6, "First conv should have 6 input channels"
    # Forward pass with dummy data
    x = torch.randn(2, 6, 256, 256)
    logits = model(x)
    # Output should be (batch, 1, H, W)
    assert logits.shape == (2, 1, 256, 256), "U‑Net output shape mismatch"
    # Confirm encoder is ResNet‑34 (attribute exists)
    assert hasattr(model, "encoder"), "Model missing encoder"
    # SMP Unet stores encoder name in attribute _encoder_name (private) but we can check class name
    encoder_cls_name = model.encoder.__class__.__name__
    assert "ResNet" in encoder_cls_name, f"Encoder class should indicate ResNet, got {encoder_cls_name}"

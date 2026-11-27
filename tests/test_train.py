import os
import json
import shutil
import torch
from pathlib import Path
import pytest

from src.mock_data import generate_synthetic_dataset
from src.tiling import slice_raster_into_tiles, assign_spatial_block_splits, compute_training_normalization_stats
from src.dataset import create_dataloaders_from_tiles
from src.model import build_unet_resnet34
from src.train import train_model, seed_everything

@pytest.mark.parametrize("epochs", [1])
def test_training_cycle_creates_checkpoint_and_history(tmp_path: Path, epochs: int):
    # Prepare synthetic data
    synth_dir = tmp_path / "synthetic"
    files = generate_synthetic_dataset(
        output_dir=synth_dir,
        height=256,
        width=256,
        seed=123,
    )
    # Tile the raster (single tile)
    tiles = slice_raster_into_tiles(
        raster_path=files["t1_raster"],
        mask_path=files["t1_mask"],
        output_dir=tmp_path / "tiles",
        region_id="test_region",
        tile_size=256,
        drop_incomplete_edges=False,
    )
    # Assign splits (train only)
    tiles = assign_spatial_block_splits(tiles, train_ratio=1.0, val_ratio=0.0, test_ratio=0.0, buffer_tiles=0)
    # Compute normalization stats
    stats = compute_training_normalization_stats(tiles)
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders_from_tiles(
        tiles, normalization_stats=stats, batch_size=1
    )
    # Seed for reproducibility
    seed_everything(42)
    # Build model (no pretrained weights to keep test fast)
    model = build_unet_resnet34(in_channels=6, encoder_weights=None)
    # Train for a single epoch
    device = torch.device("cpu")
    output_dir = tmp_path / "model_out"
    model, summary = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=None,  # no validation for minimal test
        config=type('Cfg', (), {
            "training": type('T', (), {
                "num_epochs": epochs,
                "learning_rate": 0.001,
                "loss_weights": type('L', (), {"bce": 1.0, "dice": 1.0}),
                "weight_decay": 0.0,
                "scheduler": type('S', (), {"eta_min": 0.0}),
            }),
            "seed": 42,
            "data": type('D', (), {"mask_nodata_value": 255}),
        })(),
        device=device,
        output_dir=output_dir,
        num_epochs=epochs,
        is_synthetic=True,
        region_id="test_region",
    )
    # Verify checkpoint exists
    checkpoint_path = Path(summary["checkpoint_path"])
    assert checkpoint_path.is_file(), f"Checkpoint not found at {checkpoint_path}"
    # Verify training history JSON exists and contains expected keys
    history_path = Path(summary["history_path"])
    assert history_path.is_file(), f"Training history not found at {history_path}"
    with open(history_path, "r", encoding="utf-8") as f:
        history_json = json.load(f)
    assert "best_val_iou" in history_json
    assert "history" in history_json
    assert isinstance(history_json["history"], list)
    # Clean up temporary files automatically by pytest fixture

import os
import json
import torch
from pathlib import Path
import pytest
import rasterio

from src.mock_data import generate_synthetic_dataset
from src.tiling import slice_raster_into_tiles, assign_spatial_block_splits, compute_training_normalization_stats
from src.dataset import create_dataloaders_from_tiles
from src.model import build_unet_resnet34
from src.train import train_model, seed_everything
from src.predict import load_model_from_checkpoint, predict_raster

@pytest.mark.parametrize("epochs", [1])
def test_predict_writes_geotiff(tmp_path: Path, epochs: int):
    # 1️⃣ Generate synthetic data (single tile raster & mask)
    synth_dir = tmp_path / "synthetic"
    files = generate_synthetic_dataset(
        output_dir=synth_dir,
        height=256,
        width=256,
        seed=456,
    )

    # 2️⃣ Tile the raster (produces one tile)
    tiles = slice_raster_into_tiles(
        raster_path=files["t1_raster"],
        mask_path=files["t1_mask"],
        output_dir=tmp_path / "tiles",
        region_id="predict_region",
        tile_size=256,
        drop_incomplete_edges=False,
    )
    tiles = assign_spatial_block_splits(tiles, train_ratio=1.0, val_ratio=0.0, test_ratio=0.0, buffer_tiles=0)

    # 3️⃣ Compute per‑band normalization stats (training only)
    stats = compute_training_normalization_stats(tiles)

    # 4️⃣ Dataloaders (batch size 1 for speed)
    train_loader, _, _ = create_dataloaders_from_tiles(tiles, normalization_stats=stats, batch_size=1)

    # 5️⃣ Seed & model
    seed_everything(42)
    model = build_unet_resnet34(in_channels=6, encoder_weights=None)
    device = torch.device("cpu")
    out_dir = tmp_path / "model_out"

    # 6️⃣ Train a single epoch to generate a checkpoint
    model, summary = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=None,
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
        output_dir=out_dir,
        num_epochs=epochs,
        is_synthetic=True,
        region_id="predict_region",
    )

    # 7️⃣ Load model from the generated checkpoint
    checkpoint_path = Path(summary["checkpoint_path"])
    loaded_model = load_model_from_checkpoint(checkpoint_path, device=device)

    # 8️⃣ Run inference on the original synthetic raster
    output_mask_path = tmp_path / "predicted_mask.tif"
    predict_raster(
        model=loaded_model,
        input_raster_path=files["t1_raster"],
        output_mask_path=output_mask_path,
        normalization_stats=stats,
        tile_size=256,
        threshold=0.5,
        device=device,
    )

    # 9️⃣ Verify that the GeoTIFF exists and preserves key metadata
    assert output_mask_path.is_file(), "Prediction GeoTIFF was not created"
    with rasterio.open(output_mask_path) as src:
        assert src.count == 1, "Output should have a single band"
        assert src.dtypes[0] == "uint8", "Output dtype should be uint8"
        assert src.nodata == 255, "Nodata value should be 255"
        # Shape must match input raster dimensions
        in_height, in_width = rasterio.open(files["t1_raster"]).shape
        assert src.height == in_height and src.width == in_width, "Output dimensions differ from input"
        # CRS should be identical to the input raster's CRS
        with rasterio.open(files["t1_raster"]) as src_in:
            assert src.crs == src_in.crs, "CRS mismatch between input and output"
            assert src.transform == src_in.transform, "Affine transform mismatch"


from pathlib import Path

import numpy as np
import rasterio

from src.mock_data import generate_synthetic_dataset
from src.tiling import slice_raster_into_tiles, assign_spatial_block_splits
from src.dataset import (
    BAND_NAMES,
    create_dataloaders_from_tiles,
)
from src.tiling import compute_training_normalization_stats


def test_dataset_loading_and_normalization(tmp_path: Path):
    synthetic_dir = tmp_path / "synthetic"

    files = generate_synthetic_dataset(
        output_dir=synthetic_dir,
        height=256,
        width=256,
        seed=42,
    )

    tiles = slice_raster_into_tiles(
        raster_path=files["t1_raster"],
        mask_path=files["t1_mask"],
        output_dir=tmp_path / "tiles",
        region_id="test_region",
        tile_size=256,
        drop_incomplete_edges=False,
    )

    tiles = assign_spatial_block_splits(
        tiles,
        train_ratio=1.0,
        val_ratio=0.0,
        test_ratio=0.0,
        buffer_tiles=0,
    )

    stats = compute_training_normalization_stats(
        tiles,
        band_names=BAND_NAMES,
    )

    train_loader, val_loader, test_loader = create_dataloaders_from_tiles(
        tiles,
        normalization_stats=stats,
        batch_size=1,
        augment_train=False,
    )

    batch = next(iter(train_loader))
    images, masks, tile_ids = batch

    assert images.shape == (1, 6, 256, 256)
    assert masks.shape == (1, 256, 256)

    with rasterio.open(files["t1_raster"]) as src:
        nodata_val = src.nodata
        raster_arr = src.read(1)

    mask_arr = masks.squeeze(0).numpy()
    nodata_pixels = raster_arr == nodata_val

    if np.any(nodata_pixels):
        assert np.all(mask_arr[nodata_pixels] == 255)

    assert set(np.unique(mask_arr)).issubset({0, 1, 255})

"""Unit tests for raster tiling, spatial block splitting, and buffer enforcement."""

import shutil
from pathlib import Path

import numpy as np
import pytest
import rasterio

from src.mock_data import generate_synthetic_dataset
from src.tiling import (
    TileRecord,
    assign_leave_one_region_out_splits,
    assign_spatial_block_splits,
    compute_training_normalization_stats,
    load_tiles_index,
    save_tiles_index,
    slice_raster_into_tiles,
    verify_split_isolation,
)


@pytest.fixture(scope="module")
def shared_synthetic_data(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """Module-scoped fixture generating synthetic rasters once for tiling tests."""
    temp_dir = tmp_path_factory.mktemp("tiling_test_data")
    files = generate_synthetic_dataset(
        output_dir=temp_dir,
        region_id="test_region",
        height=512,
        width=768,
        resolution_meters=10.0,
        crs_epsg=32720,
        seed=42,
    )
    return {"files": files, "dir": temp_dir}


def test_tile_dimensions_and_transforms(shared_synthetic_data: dict, tmp_path: Path):
    """Verify that extracted tiles have exact 256x256 dimensions and valid windowed affine transforms."""
    files = shared_synthetic_data["files"]
    tiles_out = tmp_path / "tiles"

    tiles = slice_raster_into_tiles(
        raster_path=files["t1_raster"],
        mask_path=files["t1_mask"],
        output_dir=tiles_out,
        region_id="test_region",
        tile_size=256,
        drop_incomplete_edges=True,
    )

    # 512 x 768 raster -> (512 // 256) * (768 // 256) = 2 * 3 = 6 tiles
    assert len(tiles) == 6

    for t in tiles:
        assert t.height == 256
        assert t.width == 256
        assert Path(t.raster_tile_path).is_file()
        assert Path(t.mask_tile_path).is_file()

        with rasterio.open(t.raster_tile_path) as src:
            assert src.shape == (256, 256)
            assert src.count == 6
            assert src.crs.to_epsg() == 32720
            # Tile bounds must match tile record bounds
            b = src.bounds
            assert (b.left, b.bottom, b.right, b.top) == pytest.approx(t.bounds)

        with rasterio.open(t.mask_tile_path) as src:
            assert src.shape == (256, 256)
            assert src.count == 1
            assert src.crs.to_epsg() == 32720


def test_nodata_propagation(shared_synthetic_data: dict, tmp_path: Path):
    """Verify that nodata pixels in the source raster are preserved in the tiles and reflected in valid_fraction."""
    files = shared_synthetic_data["files"]
    tiles_out = tmp_path / "tiles_nodata"

    tiles = slice_raster_into_tiles(
        raster_path=files["t1_raster"],
        mask_path=files["t1_mask"],
        output_dir=tiles_out,
        region_id="test_region",
        tile_size=256,
        nodata_value=-9999.0,
        mask_nodata_value=255,
    )

    # In our synthetic generator, row 0 has nodata margin in top rows
    # Tile at r=0, c=0 should have valid_fraction < 1.0
    r0_c0 = next(t for t in tiles if t.row_idx == 0 and t.col_idx == 0)
    assert r0_c0.valid_fraction < 1.0

    # Read the tile and verify nodata values actually exist
    with rasterio.open(r0_c0.raster_tile_path) as src:
        d = src.read(1)
        assert np.any(d == -9999.0)

    with rasterio.open(r0_c0.mask_tile_path) as src:
        m = src.read(1)
        assert np.any(m == 255)


def test_zero_geographic_overlap_and_buffer_enforcement(shared_synthetic_data: dict, tmp_path: Path):
    """Verify zero overlap between splits and strict enforcement of the spatial exclusion buffer."""
    files = shared_synthetic_data["files"]
    tiles_out = tmp_path / "tiles_split"

    tiles = slice_raster_into_tiles(
        raster_path=files["t1_raster"],
        mask_path=files["t1_mask"],
        output_dir=tiles_out,
        region_id="test_region",
        tile_size=256,
    )

    # Assign block splits with 1-tile buffer
    tiles = assign_spatial_block_splits(tiles, buffer_tiles=1, split_axis="col")

    train_tiles = [t for t in tiles if t.split == "train"]
    test_tiles = [t for t in tiles if t.split == "test"]
    buffer_tiles = [t for t in tiles if t.split == "buffer"]

    assert len(train_tiles) > 0
    assert len(test_tiles) > 0
    assert len(buffer_tiles) > 0

    # 1. Assert zero tile intersection across splits
    train_ids = set(t.tile_id for t in train_tiles)
    test_ids = set(t.tile_id for t in test_tiles)
    buffer_ids = set(t.tile_id for t in buffer_tiles)
    assert train_ids.isdisjoint(test_ids)
    assert train_ids.isdisjoint(buffer_ids)
    assert test_ids.isdisjoint(buffer_ids)

    # 2. Check spatial buffer distance: 1 tile width at 10m/px = 256 * 10 = 2560m
    expected_buffer_m = 2560.0
    is_valid, msg = verify_split_isolation(tiles, expected_buffer_distance_m=expected_buffer_m)
    assert is_valid, msg


def test_leave_one_region_out_split():
    """Verify Leave-One-Region-Out partitioning holds out the test region entirely."""
    dummy_bounds = (0.0, 0.0, 2560.0, 2560.0)
    region_a = [
        TileRecord(f"regA_{i}", "regA", 0, i, "unassigned", dummy_bounds, "EPSG:32720", 256, 256, 1.0, True, "", "")
        for i in range(4)
    ]
    region_b = [
        TileRecord(f"regB_{i}", "regB", 0, i, "unassigned", dummy_bounds, "EPSG:32733", 256, 256, 1.0, True, "", "")
        for i in range(4)
    ]
    region_c = [
        TileRecord(f"regC_{i}", "regC", 0, i, "unassigned", dummy_bounds, "EPSG:32749", 256, 256, 1.0, True, "", "")
        for i in range(4)
    ]

    all_tiles = assign_leave_one_region_out_splits(
        tiles_by_region={"regA": region_a, "regB": region_b, "regC": region_c},
        test_region_id="regC",
        val_region_id="regB",
    )

    # Check that ALL region C tiles are test, region B are val, region A are train
    for t in all_tiles:
        if t.region_id == "regC":
            assert t.split == "test"
        elif t.region_id == "regB":
            assert t.split == "val"
        elif t.region_id == "regA":
            assert t.split == "train"


def test_training_normalization_stats_train_only(shared_synthetic_data: dict, tmp_path: Path):
    """Verify that normalization statistics are derived strictly from training tiles, ignoring nodata."""
    files = shared_synthetic_data["files"]
    tiles_out = tmp_path / "tiles_norm"

    tiles = slice_raster_into_tiles(
        raster_path=files["t1_raster"],
        mask_path=files["t1_mask"],
        output_dir=tiles_out,
        region_id="test_region",
        tile_size=256,
    )
    tiles = assign_spatial_block_splits(tiles, buffer_tiles=1)

    bands = ["B2", "B3", "B4", "B8", "B11", "B12"]
    stats = compute_training_normalization_stats(tiles, band_names=bands, nodata_value=-9999.0)

    for band in bands:
        assert band in stats
        assert "mean" in stats[band]
        assert "std" in stats[band]
        assert "pixel_count" in stats[band]
        assert stats[band]["std"] > 0.0
        assert stats[band]["pixel_count"] > 0
        # Realistic reflectance values must be in [0.0, 1.0]
        assert 0.0 < stats[band]["mean"] < 1.0


def test_save_and_load_tiles_index(shared_synthetic_data: dict, tmp_path: Path):
    """Verify serialization and deserialization of the tiles index JSON catalog."""
    files = shared_synthetic_data["files"]
    tiles = slice_raster_into_tiles(
        raster_path=files["t1_raster"],
        mask_path=files["t1_mask"],
        output_dir=tmp_path / "tiles_idx",
        region_id="test_region",
        tile_size=256,
    )
    tiles = assign_spatial_block_splits(tiles, buffer_tiles=1)

    index_file = tmp_path / "tiles_index.json"
    save_tiles_index(tiles, index_file)
    assert index_file.is_file()

    loaded = load_tiles_index(index_file)
    assert len(loaded) == len(tiles)
    for orig, load in zip(tiles, loaded):
        assert orig.tile_id == load.tile_id
        assert orig.split == load.split
        assert orig.bounds == pytest.approx(load.bounds)
        assert orig.valid_fraction == load.valid_fraction

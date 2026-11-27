"""Unit tests for NDVI forest thresholding and binary mask export."""

from pathlib import Path

import numpy as np
import pytest
import rasterio

from src.forest_mask import create_and_save_forest_mask, threshold_ndvi_to_forest_mask


def test_forest_mask_thresholding_logic():
    """Verify that NDVI values are correctly partitioned into forest, non-forest, and nodata."""
    # Threshold = 0.60
    ndvi = np.array([
        [0.75, 0.60, 0.59, 0.20, -9999.0, np.nan],
    ], dtype=np.float32)

    mask = threshold_ndvi_to_forest_mask(ndvi, threshold=0.60, ndvi_nodata=-9999.0, mask_nodata=255)

    assert mask[0, 0] == 1    # 0.75 >= 0.60 -> Forest
    assert mask[0, 1] == 1    # 0.60 >= 0.60 -> Forest (boundary case)
    assert mask[0, 2] == 0    # 0.59 < 0.60  -> Non-forest
    assert mask[0, 3] == 0    # 0.20 < 0.60  -> Non-forest
    assert mask[0, 4] == 255  # -9999.0      -> Nodata
    assert mask[0, 5] == 255  # NaN          -> Nodata


def test_forest_mask_configurable_threshold():
    """Verify that changing threshold dynamically alters classification without code modification."""
    ndvi = np.array([[0.65]], dtype=np.float32)

    # With threshold 0.60, 0.65 is Forest (1)
    mask_60 = threshold_ndvi_to_forest_mask(ndvi, threshold=0.60)
    assert mask_60[0, 0] == 1

    # With threshold 0.70, 0.65 is Non-forest (0)
    mask_70 = threshold_ndvi_to_forest_mask(ndvi, threshold=0.70)
    assert mask_70[0, 0] == 0


def test_forest_mask_file_export(tmp_path: Path):
    """Verify that create_and_save_forest_mask creates a valid single-band uint8 GeoTIFF."""
    # Write a dummy NDVI GeoTIFF
    ndvi_file = tmp_path / "dummy_ndvi.tif"
    mask_file = tmp_path / "dummy_mask.tif"

    dummy_ndvi = np.array([[0.8, 0.4], [-9999.0, 0.65]], dtype=np.float32)
    meta = {
        "driver": "GTiff",
        "height": 2,
        "width": 2,
        "count": 1,
        "dtype": np.float32,
        "crs": rasterio.crs.CRS.from_epsg(32720),
        "transform": rasterio.Affine(10.0, 0.0, 500000.0, 0.0, -10.0, 8900000.0),
        "nodata": -9999.0,
    }

    with rasterio.open(ndvi_file, "w", **meta) as dst:
        dst.write(dummy_ndvi, 1)

    create_and_save_forest_mask(ndvi_file, mask_file, threshold=0.60)
    assert mask_file.is_file()

    with rasterio.open(mask_file) as src:
        assert src.count == 1
        assert src.dtypes[0] == "uint8"
        assert src.nodata == 255
        assert src.crs.to_epsg() == 32720

        mask_data = src.read(1)
        assert mask_data[0, 0] == 1    # 0.8 -> Forest
        assert mask_data[0, 1] == 0    # 0.4 -> Non-forest
        assert mask_data[1, 0] == 255  # -9999 -> Nodata
        assert mask_data[1, 1] == 1    # 0.65 -> Forest

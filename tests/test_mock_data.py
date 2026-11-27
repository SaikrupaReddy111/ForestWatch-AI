"""Unit tests for synthetic satellite data generation and metadata preservation."""

import shutil
from pathlib import Path

import numpy as np
import pytest
import rasterio

from src.mock_data import (
    NODATA_VALUE_FLOAT,
    NODATA_VALUE_MASK,
    generate_synthetic_dataset,
    generate_landscape_classes,
    synthesize_multispectral_cube,
)


@pytest.fixture
def temp_synthetic_dir(tmp_path: Path) -> Path:
    """Fixture providing a temporary directory for synthetic datasets."""
    synthetic_dir = tmp_path / "synthetic_test"
    synthetic_dir.mkdir(parents=True, exist_ok=True)
    yield synthetic_dir
    shutil.rmtree(synthetic_dir, ignore_errors=True)


def test_synthetic_raster_generation(temp_synthetic_dir: Path):
    """Verify that synthetic rasters and masks have correct shapes, CRS, metadata tags, and nodata."""
    files = generate_synthetic_dataset(
        output_dir=temp_synthetic_dir,
        region_id="test_amazon",
        height=512,
        width=768,
        resolution_meters=10.0,
        crs_epsg=32720,
        seed=42,
    )

    assert "t1_raster" in files and files["t1_raster"].is_file()
    assert "t2_raster" in files and files["t2_raster"].is_file()
    assert "t1_mask" in files and files["t1_mask"].is_file()
    assert "t2_mask" in files and files["t2_mask"].is_file()

    # Check T1 6-band raster
    with rasterio.open(files["t1_raster"]) as src:
        assert src.count == 6
        assert src.shape == (512, 768)
        assert src.crs.to_epsg() == 32720
        assert src.nodata == NODATA_VALUE_FLOAT
        assert src.res == (10.0, 10.0)
        
        tags = src.tags()
        assert tags.get("IS_SYNTHETIC") == "TRUE"
        assert tags.get("DATA_SOURCE") == "DEMONSTRATION_SYNTHETIC"
        assert "B2,B3,B4,B8,B11,B12" in tags.get("BANDS", "")

        data = src.read()
        assert data.shape == (6, 512, 768)
        assert data.dtype == np.float32

        # Check that valid reflectance is within [0.0, 1.0]
        valid_mask = data != NODATA_VALUE_FLOAT
        assert np.all(data[valid_mask] >= 0.0)
        assert np.all(data[valid_mask] <= 1.0)
        # Check that nodata pixels exist
        assert np.any(~valid_mask)

    # Check T1 binary forest mask
    with rasterio.open(files["t1_mask"]) as src:
        assert src.count == 1
        assert src.shape == (512, 768)
        assert src.crs.to_epsg() == 32720
        assert src.nodata == NODATA_VALUE_MASK

        mask_data = src.read(1)
        unique_vals = set(np.unique(mask_data))
        # Mask must only contain 0 (non-forest), 1 (forest), and 255 (nodata)
        assert unique_vals.issubset({0, 1, NODATA_VALUE_MASK})


def test_reproducibility_with_fixed_seed(temp_synthetic_dir: Path):
    """Verify that generating synthetic datasets with the same seed produces identical arrays."""
    dir1 = temp_synthetic_dir / "run1"
    dir2 = temp_synthetic_dir / "run2"

    files1 = generate_synthetic_dataset(output_dir=dir1, seed=42)
    files2 = generate_synthetic_dataset(output_dir=dir2, seed=42)

    with rasterio.open(files1["t1_raster"]) as s1, rasterio.open(files2["t1_raster"]) as s2:
        d1 = s1.read()
        d2 = s2.read()
        np.testing.assert_array_equal(d1, d2)

    # Changing the seed must produce different values
    dir3 = temp_synthetic_dir / "run3"
    files3 = generate_synthetic_dataset(output_dir=dir3, seed=999)
    with rasterio.open(files1["t1_raster"]) as s1, rasterio.open(files3["t1_raster"]) as s3:
        d1 = s1.read()
        d3 = s3.read()
        assert not np.array_equal(d1, d3)


def test_spectral_profile_consistency():
    """Verify that simulated vegetation, water, and soil exhibit correct spectral behavior."""
    classes_t1, _, nodata = generate_landscape_classes(height=100, width=100, seed=42)
    cube = synthesize_multispectral_cube(classes_t1, nodata, seed=42)

    # Bands: 0:B2 (Blue), 1:B3 (Green), 2:B4 (Red), 3:B8 (NIR), 4:B11 (SWIR1), 5:B12 (SWIR2)
    forest_pixels = (classes_t1 == 1) & (~nodata)
    water_pixels = (classes_t1 == 3) & (~nodata)
    soil_pixels = (classes_t1 == 2) & (~nodata)

    # Forest: NIR (B8) must be significantly higher than Red (B4)
    forest_nir_mean = np.mean(cube[3, forest_pixels])
    forest_red_mean = np.mean(cube[2, forest_pixels])
    assert forest_nir_mean > 0.40
    assert forest_red_mean < 0.08
    # NDVI of dense forest should comfortably exceed 0.60
    forest_ndvi = (forest_nir_mean - forest_red_mean) / (forest_nir_mean + forest_red_mean + 1e-6)
    assert forest_ndvi > 0.70

    # Water: NIR must be very low (< 0.05)
    water_nir_mean = np.mean(cube[3, water_pixels])
    assert water_nir_mean < 0.05

    # Soil: SWIR1 (B11) and Red (B4) are relatively high
    soil_swir_mean = np.mean(cube[4, soil_pixels])
    assert soil_swir_mean > 0.25

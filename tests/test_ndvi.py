"""Unit tests for NDVI array calculation, zero denominator, nodata, and metadata preservation."""

from pathlib import Path

import numpy as np
import pytest
import rasterio

from src.mock_data import generate_synthetic_dataset
from src.ndvi import compute_and_save_ndvi, compute_ndvi_array


def test_ndvi_known_values():
    """Verify NDVI mathematical calculation with exact known values."""
    # Case 1: Standard high vegetation (NIR=0.8, Red=0.2) -> (0.8 - 0.2) / (0.8 + 0.2) = 0.60
    nir = np.array([[0.80]], dtype=np.float32)
    red = np.array([[0.20]], dtype=np.float32)
    ndvi = compute_ndvi_array(nir, red, epsilon=0.0)
    assert ndvi[0, 0] == pytest.approx(0.60, abs=1e-5)

    # Case 2: Water (NIR=0.02, Red=0.04) -> (0.02 - 0.04) / (0.02 + 0.04) = -0.02 / 0.06 = -0.3333
    nir = np.array([[0.02]], dtype=np.float32)
    red = np.array([[0.04]], dtype=np.float32)
    ndvi = compute_ndvi_array(nir, red, epsilon=0.0)
    assert ndvi[0, 0] == pytest.approx(-0.33333, abs=1e-4)

    # Case 3: Identical NIR and Red -> NDVI = 0.0
    nir = np.array([[0.25]], dtype=np.float32)
    red = np.array([[0.25]], dtype=np.float32)
    ndvi = compute_ndvi_array(nir, red, epsilon=0.0)
    assert ndvi[0, 0] == pytest.approx(0.0, abs=1e-5)


def test_ndvi_zero_denominator_handling():
    """Verify that zero radiance (NIR=0, Red=0) or negative sums yield nodata rather than crashing or Inf."""
    # Both zero
    nir = np.array([[0.0]], dtype=np.float32)
    red = np.array([[0.0]], dtype=np.float32)
    ndvi = compute_ndvi_array(nir, red, nodata_value=-9999.0)
    assert ndvi[0, 0] == -9999.0

    # Negative sum (unphysical/shadow sensor anomaly)
    nir = np.array([[-0.05]], dtype=np.float32)
    red = np.array([[-0.05]], dtype=np.float32)
    ndvi = compute_ndvi_array(nir, red, nodata_value=-9999.0)
    assert ndvi[0, 0] == -9999.0


def test_ndvi_nodata_propagation():
    """Verify that nodata pixels (-9999.0) or NaN values in either band result in nodata output."""
    nir = np.array([[0.5, -9999.0, np.nan, 0.4]], dtype=np.float32)
    red = np.array([[0.1, 0.1, 0.1, -9999.0]], dtype=np.float32)

    ndvi = compute_ndvi_array(nir, red, nodata_value=-9999.0)

    # First pixel is valid: (0.5 - 0.1) / (0.5 + 0.1) = 0.4 / 0.6 = 0.6667
    assert ndvi[0, 0] == pytest.approx(0.66667, abs=1e-4)
    # Remaining pixels have nodata in NIR, NaN in NIR, or nodata in Red
    assert ndvi[0, 1] == -9999.0
    assert ndvi[0, 2] == -9999.0
    assert ndvi[0, 3] == -9999.0


def test_ndvi_metadata_preservation(tmp_path: Path):
    """Verify that compute_and_save_ndvi preserves CRS, shape, transform, and tags."""
    files = generate_synthetic_dataset(output_dir=tmp_path / "synth", height=256, width=256, seed=42)
    raster_in = files["t1_raster"]
    ndvi_out = tmp_path / "output_ndvi.tif"

    compute_and_save_ndvi(raster_in, ndvi_out)
    assert ndvi_out.is_file()

    with rasterio.open(raster_in) as src_in, rasterio.open(ndvi_out) as src_out:
        assert src_out.count == 1
        assert src_out.crs == src_in.crs
        assert src_out.shape == src_in.shape
        assert src_out.transform == pytest.approx(src_in.transform)
        assert src_out.nodata == -9999.0

        tags = src_out.tags()
        assert "NDVI" in tags.get("DESCRIPTION", "")

"""Unit tests for raster alignment, reprojection, and valid-pixel mask generation."""

from pathlib import Path

import affine
import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS

from src.alignment import (
    align_raster_to_reference,
    generate_valid_pixel_mask,
    verify_raster_alignment,
)
from src.mock_data import generate_synthetic_dataset


@pytest.fixture(scope="module")
def sample_rasters(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """Fixture providing reference synthetic rasters."""
    d = tmp_path_factory.mktemp("alignment_data")
    files = generate_synthetic_dataset(output_dir=d, height=256, width=256, seed=42)
    return {"files": files, "dir": d}


def test_verify_raster_alignment_positive(sample_rasters: dict):
    """Verify that identical rasters pass alignment verification."""
    files = sample_rasters["files"]
    is_aligned, msg = verify_raster_alignment(files["t1_raster"], files["t1_mask"])
    assert is_aligned
    assert "aligned" in msg.lower()


def test_verify_raster_alignment_negative_crs(sample_rasters: dict, tmp_path: Path):
    """Verify that different CRS triggers alignment failure."""
    files = sample_rasters["files"]
    ref_raster = files["t1_raster"]

    # Create a dummy raster with different CRS (e.g. EPSG:4326)
    mismatched_path = tmp_path / "mismatched_crs.tif"
    with rasterio.open(ref_raster) as src:
        meta = src.meta.copy()
        meta.update({"crs": CRS.from_epsg(4326), "count": 1, "dtype": np.float32})
        with rasterio.open(mismatched_path, "w", **meta) as dst:
            dst.write(np.zeros((1, 256, 256), dtype=np.float32))

    is_aligned, msg = verify_raster_alignment(ref_raster, mismatched_path)
    assert not is_aligned
    assert "CRS mismatch" in msg


def test_align_raster_to_reference(sample_rasters: dict, tmp_path: Path):
    """Verify that reprojecting an unaligned raster brings it into exact alignment with reference."""
    files = sample_rasters["files"]
    ref_raster = files["t1_raster"]

    # Create an unaligned source raster (different shape, transform, and CRS)
    unaligned_path = tmp_path / "unaligned.tif"
    unaligned_transform = affine.Affine(0.001, 0.0, -62.5, 0.0, -0.001, -10.0)
    with rasterio.open(
        unaligned_path,
        "w",
        driver="GTiff",
        height=100,
        width=100,
        count=1,
        dtype=np.uint8,
        crs=CRS.from_epsg(4326),
        transform=unaligned_transform,
        nodata=255,
    ) as dst:
        dst.write(np.ones((1, 100, 100), dtype=np.uint8))

    aligned_path = tmp_path / "aligned_output.tif"
    align_raster_to_reference(
        input_path=unaligned_path,
        reference_path=ref_raster,
        output_path=aligned_path,
        resampling_method="nearest",
    )

    # Now verify alignment
    is_aligned, msg = verify_raster_alignment(ref_raster, aligned_path)
    assert is_aligned, msg

    with rasterio.open(aligned_path) as src, rasterio.open(ref_raster) as ref:
        assert src.crs == ref.crs
        assert src.shape == ref.shape
        assert src.transform == pytest.approx(ref.transform)


def test_generate_valid_pixel_mask(sample_rasters: dict, tmp_path: Path):
    """Verify that valid pixel mask is 1 for valid pixels and 0 for nodata pixels."""
    files = sample_rasters["files"]
    valid_mask_path = tmp_path / "valid_mask.tif"

    generate_valid_pixel_mask(
        image_path=files["t1_raster"],
        mask_path=files["t1_mask"],
        output_path=valid_mask_path,
        image_nodata=-9999.0,
        mask_nodata=255,
    )

    assert valid_mask_path.is_file()

    with (
        rasterio.open(valid_mask_path) as v_src,
        rasterio.open(files["t1_raster"]) as img_src,
        rasterio.open(files["t1_mask"]) as msk_src,
    ):
        v_data = v_src.read(1)
        img_data = img_src.read()
        msk_data = msk_src.read(1)

        # Check binary values
        unique_v = set(np.unique(v_data))
        assert unique_v.issubset({0, 1})

        # Check nodata alignment: where image is -9999 or mask is 255, valid mask MUST be 0
        invalid_locs = (img_data[0] == -9999.0) | (msk_data == 255)
        if np.any(invalid_locs):
            assert np.all(v_data[invalid_locs] == 0)

        # Where image is valid and mask is valid, valid mask MUST be 1
        valid_locs = (img_data[0] != -9999.0) & (msk_data != 255)
        if np.any(valid_locs):
            assert np.all(v_data[valid_locs] == 1)

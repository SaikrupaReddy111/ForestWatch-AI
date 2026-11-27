"""Unit tests for Earth Engine initialization, dry-run export manifests, and download unpacking."""

from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch
import zipfile

import pytest

from src.config import load_full_pipeline_config
from src.gee_export import (
    download_geotiff_from_url,
    fetch_and_export_region,
    initialize_earth_engine,
)


def test_initialize_earth_engine_unauthenticated():
    """Verify that initialize_earth_engine handles missing credentials gracefully."""
    # When unauthenticated or without project, it should return False and actionable message
    with patch("ee.Initialize", side_effect=Exception("Credentials not found")):
        is_init, msg = initialize_earth_engine()
        assert not is_init
        assert "initialization failed" in msg.lower()
        assert "earthengine authenticate" in msg


def test_fetch_and_export_region_dry_run():
    """Verify that fetch_and_export_region builds proper manifests from YAML configuration without downloading."""
    cfg, regions = load_full_pipeline_config("configs/config.yaml", "configs/regions.yaml")

    # Dry-run for Amazon region
    manifest = fetch_and_export_region(
        region_id="amazon_brazil",
        config=cfg,
        regions=regions,
        output_dir="data/raw",
        dry_run=True,
    )

    assert manifest["status"] == "dry_run_success"
    assert manifest["region_id"] == "amazon_brazil"
    assert manifest["target_years"]["t1"] == 2020
    assert manifest["target_years"]["t2"] == 2023
    assert manifest["projected_utm_crs"] == "EPSG:32720"
    assert "amazon_brazil_s2_2020.tif" in manifest["files"]["t1_raster"]
    assert "amazon_brazil_s2_2023.tif" in manifest["files"]["t2_raster"]
    assert "amazon_brazil_dw_mask_2020.tif" in manifest["files"]["t1_mask"]
    assert "amazon_brazil_dw_mask_2023.tif" in manifest["files"]["t2_mask"]

    # Unknown region ID must raise ValueError
    with pytest.raises(ValueError, match="Unknown region_id"):
        fetch_and_export_region("unknown_region_xyz", cfg, regions, dry_run=True)


def test_download_geotiff_from_mock_zip(tmp_path: Path):
    """Verify that download_geotiff_from_url unpacks zipped Earth Engine GeoTIFF downloads."""
    # Create an in-memory zip archive containing a dummy .tif file
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as z:
        z.writestr("downloaded_scene.tif", b"FAKE_GEOTIFF_DATA_BYTES")
    zip_bytes = zip_buffer.getvalue()

    # Mock urllib.request.urlopen
    mock_response = MagicMock()
    mock_response.read.return_value = zip_bytes

    out_file = tmp_path / "scene.tif"
    with patch("urllib.request.urlopen", return_value=mock_response):
        download_geotiff_from_url("http://mock-earth-engine.google.com/download", out_file)

    assert out_file.is_file()
    with open(out_file, "rb") as f:
        assert f.read() == b"FAKE_GEOTIFF_DATA_BYTES"

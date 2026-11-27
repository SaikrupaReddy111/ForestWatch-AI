"""Unit tests for multi-temporal change detection, noise filtering, and geodetic area calculations."""

from pathlib import Path

import affine
import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS

from src.change_detection import (
    CLASS_NODATA,
    CLASS_POTENTIAL_GAIN,
    CLASS_POTENTIAL_LOSS,
    CLASS_STABLE_FOREST,
    CLASS_STABLE_NON_FOREST,
    calculate_pixel_ground_area_m2,
    detect_tree_cover_change,
    filter_small_loss_patches,
    run_change_detection_workflow,
)
from src.mock_data import generate_synthetic_dataset


def test_detect_tree_cover_change_classes():
    """Verify categorical transition logic across dual observation dates."""
    m1 = np.array([
        [1, 1, 0, 0, 255, 1],
    ], dtype=np.uint8)
    m2 = np.array([
        [1, 0, 1, 0, 1, 255],
    ], dtype=np.uint8)

    change = detect_tree_cover_change(m1, m2, nodata_value=255)

    assert change[0, 0] == CLASS_STABLE_FOREST       # 1 -> 1
    assert change[0, 1] == CLASS_POTENTIAL_LOSS      # 1 -> 0 (Tree-cover loss)
    assert change[0, 2] == CLASS_POTENTIAL_GAIN      # 0 -> 1 (Regrowth)
    assert change[0, 3] == CLASS_STABLE_NON_FOREST   # 0 -> 0
    assert change[0, 4] == CLASS_NODATA              # 255 -> 1 (Invalid at T1)
    assert change[0, 5] == CLASS_NODATA              # 1 -> 255 (Invalid at T2)


def test_filter_small_loss_patches():
    """Verify that isolated loss components below minimum pixel size are filtered out."""
    change_map = np.zeros((20, 20), dtype=np.uint8)

    # Patch 1: Large component of 12 pixels (3 x 4)
    change_map[2:5, 2:6] = CLASS_POTENTIAL_LOSS

    # Patch 2: Tiny noise component of 3 pixels
    change_map[12:15, 12] = CLASS_POTENTIAL_LOSS

    # Filter with minimum patch size of 5 pixels
    filtered = filter_small_loss_patches(
        change_map,
        min_patch_pixels=5,
        loss_class_code=CLASS_POTENTIAL_LOSS,
        connectivity=8,
    )

    # Patch 1 (12 pixels >= 5) must be preserved
    assert np.all(filtered[2:5, 2:6] == CLASS_POTENTIAL_LOSS)

    # Patch 2 (3 pixels < 5) must be suppressed / reclassified to stable forest
    assert np.all(filtered[12:15, 12] == CLASS_STABLE_FOREST)


def test_calculate_pixel_ground_area_projected():
    """Verify ground area calculation for projected metric UTM systems."""
    crs = CRS.from_epsg(32720)  # UTM zone 20S
    # 10m x 10m resolution
    transform = affine.Affine(10.0, 0.0, 500000.0, 0.0, -10.0, 8900000.0)

    area_m2 = calculate_pixel_ground_area_m2(crs, transform, height=100, width=100)
    assert area_m2 == pytest.approx(100.0, abs=1e-3)

    # 100 pixels at 100 m^2 = 10,000 m^2 = 1.0 ha
    ha = (100 * area_m2) / 10000.0
    assert ha == pytest.approx(1.0, abs=1e-4)


def test_calculate_pixel_ground_area_geographic():
    """Verify ellipsoidal geodesic area calculation for geographic CRS (EPSG:4326)."""
    crs = CRS.from_epsg(4326)
    # 0.0001 deg ~ 11.1 meters at equator
    transform = affine.Affine(0.0001, 0.0, -60.0, 0.0, -0.0001, -10.0)

    area_m2 = calculate_pixel_ground_area_m2(crs, transform, height=100, width=100)
    # Area around 10 deg South for a 0.0001 deg square is ~ 121 m^2
    assert 100.0 < area_m2 < 140.0


def test_run_change_detection_workflow_end_to_end(tmp_path: Path):
    """Verify complete pipeline execution: NDVI calculation, masking, filtering, and JSON summary."""
    synth_dir = tmp_path / "synth"
    out_dir = tmp_path / "outputs"

    files = generate_synthetic_dataset(output_dir=synth_dir, height=256, width=256, seed=42)

    summary = run_change_detection_workflow(
        t1_raster_path=files["t1_raster"],
        t2_raster_path=files["t2_raster"],
        output_dir=out_dir,
        region_id="test_region",
        date_t1="2020",
        date_t2="2023",
        ndvi_threshold=0.60,
        min_patch_size_ha=0.10,
        is_synthetic=True,
    )

    assert summary["region_id"] == "test_region"
    assert summary["is_synthetic"] is True
    assert summary["min_patch_size_ha"] == 0.10
    assert summary["pixel_area_m2"] == pytest.approx(100.0)
    assert summary["pixel_counts"]["valid_pixels"] > 0
    assert summary["areas_hectares"]["monitored_valid_area_ha"] > 0.0

    # Verify that files were created
    for file_key, file_path_str in summary["output_files"].items():
        assert Path(file_path_str).is_file(), f"Output file missing: {file_key}"

    # Verify change GeoTIFF metadata
    with rasterio.open(summary["output_files"]["change_map"]) as src:
        assert src.count == 1
        assert src.dtypes[0] == "uint8"
        assert src.nodata == CLASS_NODATA
        assert src.crs.to_epsg() == 32720
        tags = src.tags()
        assert "tree-cover change" in tags.get("DESCRIPTION", "").lower()
        assert tags.get("IS_SYNTHETIC") == "TRUE"

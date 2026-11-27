"""Synthetic multispectral satellite data generator for ForestWatch AI.

Generates reproducible, georeferenced six-band Sentinel-2 Level-2A GeoTIFFs
and corresponding Dynamic World-style forest masks for two observation dates (T1 and T2).
Strictly used for local offline development, unit tests, and CI/CD verification without
requiring Google Earth Engine credentials or heavy downloads.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import affine
import numpy as np
import rasterio
from rasterio.crs import CRS


# Realistic Sentinel-2 surface reflectance profiles (mean, std) per land cover type
# Bands: [B2 (Blue), B3 (Green), B4 (Red), B8 (NIR), B11 (SWIR1), B12 (SWIR2)]
SPECTRAL_PROFILES: Dict[str, Dict[str, np.ndarray]] = {
    "dense_forest": {
        "mean": np.array([0.025, 0.045, 0.028, 0.520, 0.160, 0.070], dtype=np.float32),
        "std": np.array([0.005, 0.008, 0.006, 0.040, 0.020, 0.012], dtype=np.float32),
    },
    "bare_soil": {
        "mean": np.array([0.140, 0.180, 0.220, 0.270, 0.380, 0.320], dtype=np.float32),
        "std": np.array([0.020, 0.025, 0.030, 0.035, 0.040, 0.035], dtype=np.float32),
    },
    "water_river": {
        "mean": np.array([0.060, 0.050, 0.030, 0.015, 0.010, 0.008], dtype=np.float32),
        "std": np.array([0.008, 0.007, 0.005, 0.003, 0.002, 0.002], dtype=np.float32),
    },
}

NODATA_VALUE_FLOAT = -9999.0
NODATA_VALUE_MASK = 255


def generate_landscape_classes(
    height: int = 512,
    width: int = 768,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generates synthetic land cover classification maps for T1 and T2.

    Returns:
        classes_t1: Array with class codes for T1 (1: Forest, 2: Bare Soil, 3: Water, 0: Nodata).
        classes_t2: Array with class codes for T2 (simulates tree-cover loss).
        nodata_mask: Boolean array where True indicates invalid / nodata pixels.
    """
    rng = np.random.default_rng(seed)

    # 1. Start with pervasive forest cover (code 1)
    classes_t1 = np.ones((height, width), dtype=np.uint8)
    classes_t2 = np.ones((height, width), dtype=np.uint8)

    # 2. Add an established river winding from top to bottom (code 3)
    river_x_center = int(width * 0.45)
    for y in range(height):
        # Sine wave river meander
        meander = int(25 * np.sin(y / 40.0) + 10 * np.sin(y / 15.0))
        rx = np.clip(river_x_center + meander, 15, width - 15)
        river_width = 8 + int(3 * np.sin(y / 30.0))
        classes_t1[y, max(0, rx - river_width) : min(width, rx + river_width)] = 3
        classes_t2[y, max(0, rx - river_width) : min(width, rx + river_width)] = 3

    # 3. Add existing historical bare soil / agricultural clearings at T1 (code 2)
    # E.g., Southwestern corner
    classes_t1[int(height * 0.6) :, : int(width * 0.25)] = 2
    classes_t2[int(height * 0.6) :, : int(width * 0.25)] = 2

    # 4. Simulate tree-cover loss between T1 and T2:
    # Large contiguous fishbone / logging blocks in Central-East that were forest at T1, cleared at T2
    # Deforestation patch 1 (approx 120 x 140 pixels = 16,800 pixels = 168 ha at 10m)
    y1_start, y1_end = int(height * 0.2), int(height * 0.45)
    x1_start, x1_end = int(width * 0.6), int(width * 0.8)
    classes_t2[y1_start:y1_end, x1_start:x1_end] = 2

    # Deforestation patch 2 (smaller road corridor, approx 20 x 180 pixels = 3,600 pixels = 36 ha)
    y2_start, y2_end = int(height * 0.7), int(height * 0.75)
    x2_start, x2_end = int(width * 0.55), int(width * 0.9)
    classes_t2[y2_start:y2_end, x2_start:x2_end] = 2

    # 5. Add small isolated clearing patches (>= 50 pixels)
    classes_t2[int(height * 0.15) : int(height * 0.15) + 12, int(width * 0.15) : int(width * 0.15) + 10] = 2

    # 6. Add a tiny 4-pixel noise patch (< 50 pixels) to test connected-component area filtering
    classes_t2[10:12, 10:12] = 2

    # 7. Add invalid / nodata border region (e.g., top-left triangle or border stripe)
    nodata_mask = np.zeros((height, width), dtype=bool)
    # Margin strip (top 10 rows and rightmost 10 columns)
    nodata_mask[:10, :] = True
    nodata_mask[:, -10:] = True

    # Cloud shadow spot in T1
    nodata_mask[50:70, 80:100] = True

    classes_t1[nodata_mask] = 0
    classes_t2[nodata_mask] = 0

    return classes_t1, classes_t2, nodata_mask


def synthesize_multispectral_cube(
    classes: np.ndarray,
    nodata_mask: np.ndarray,
    seed: int = 42,
) -> np.ndarray:
    """Synthesizes a 6-band reflectance cube from a land-cover classification map.

    Returns:
        cube: Shape (6, height, width), float32 in [0, 1] with -9999.0 for nodata.
    """
    height, width = classes.shape
    num_bands = 6
    cube = np.zeros((num_bands, height, width), dtype=np.float32)

    rng = np.random.default_rng(seed)

    # Populate band values based on class spectral signatures
    for band_idx in range(num_bands):
        # 1: Dense forest
        forest_mask = (classes == 1)
        mean_f = SPECTRAL_PROFILES["dense_forest"]["mean"][band_idx]
        std_f = SPECTRAL_PROFILES["dense_forest"]["std"][band_idx]
        cube[band_idx, forest_mask] = rng.normal(mean_f, std_f, size=np.count_nonzero(forest_mask))

        # 2: Bare soil / cleared
        soil_mask = (classes == 2)
        mean_s = SPECTRAL_PROFILES["bare_soil"]["mean"][band_idx]
        std_s = SPECTRAL_PROFILES["bare_soil"]["std"][band_idx]
        cube[band_idx, soil_mask] = rng.normal(mean_s, std_s, size=np.count_nonzero(soil_mask))

        # 3: Water river
        water_mask = (classes == 3)
        mean_w = SPECTRAL_PROFILES["water_river"]["mean"][band_idx]
        std_w = SPECTRAL_PROFILES["water_river"]["std"][band_idx]
        cube[band_idx, water_mask] = rng.normal(mean_w, std_w, size=np.count_nonzero(water_mask))

        # Clip valid reflectance to physical range [0.001, 1.0]
        valid_pixels = ~nodata_mask
        cube[band_idx, valid_pixels] = np.clip(cube[band_idx, valid_pixels], 0.001, 1.0)

        # Set nodata pixels
        cube[band_idx, nodata_mask] = NODATA_VALUE_FLOAT

    return cube


def create_binary_forest_mask(classes: np.ndarray, nodata_mask: np.ndarray) -> np.ndarray:
    """Creates a Dynamic World-style binary forest mask.

    Returns:
        mask: uint8 array (1 = Forest, 0 = Non-forest, 255 = Nodata).
    """
    mask = np.zeros(classes.shape, dtype=np.uint8)
    mask[classes == 1] = 1
    mask[nodata_mask] = NODATA_VALUE_MASK
    return mask


def generate_synthetic_dataset(
    output_dir: str | Path = "data/synthetic",
    region_id: str = "synthetic_amazon_demo",
    height: int = 512,
    width: int = 768,
    resolution_meters: float = 10.0,
    origin_x: float = 500000.0,
    origin_y: float = 8900000.0,
    crs_epsg: int = 32720,
    seed: int = 42,
) -> Dict[str, Path]:
    """Generates complete synthetic GeoTIFFs (T1, T2 rasters and masks) with geospatial metadata.

    Args:
        output_dir: Destination directory.
        region_id: Identifier for the synthetic region.
        height: Raster height in pixels (default 512 = 2 tiles of 256).
        width: Raster width in pixels (default 768 = 3 tiles of 256).
        resolution_meters: Pixel ground resolution (default 10m).
        origin_x: UTM Easting in meters.
        origin_y: UTM Northing in meters.
        crs_epsg: Projected CRS EPSG code (default UTM Zone 20S).
        seed: Random seed for exact reproducibility.

    Returns:
        Dictionary mapping file labels to their Path locations.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Generate landscape structures
    classes_t1, classes_t2, nodata_mask = generate_landscape_classes(height, width, seed=seed)

    # 2. Synthesize 6-band reflectance cubes
    cube_t1 = synthesize_multispectral_cube(classes_t1, nodata_mask, seed=seed)
    cube_t2 = synthesize_multispectral_cube(classes_t2, nodata_mask, seed=seed + 1)

    # 3. Create binary forest ground-truth masks
    mask_t1 = create_binary_forest_mask(classes_t1, nodata_mask)
    mask_t2 = create_binary_forest_mask(classes_t2, nodata_mask)

    # 4. Define geospatial affine transform and CRS
    # Affine: pixel width = resolution, pixel height = -resolution (north up)
    transform = affine.Affine(
        resolution_meters, 0.0, origin_x,
        0.0, -resolution_meters, origin_y,
    )
    crs = CRS.from_epsg(crs_epsg)

    # Standard tags marking synthetic demonstration data
    metadata_tags = {
        "IS_SYNTHETIC": "TRUE",
        "DATA_SOURCE": "DEMONSTRATION_SYNTHETIC",
        "DESCRIPTION": "Synthetic multi-band demonstration raster for ForestWatch AI pipeline verification",
        "REGION_ID": region_id,
        "RESOLUTION_METERS": str(resolution_meters),
        "BANDS": "B2,B3,B4,B8,B11,B12",
    }

    # 5. Write GeoTIFF files
    file_paths: Dict[str, Path] = {}

    # File 1: T1 Sentinel-2 6-band raster
    t1_raster_path = out_path / f"{region_id}_s2_t1.tif"
    with rasterio.open(
        t1_raster_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=6,
        dtype=np.float32,
        crs=crs,
        transform=transform,
        nodata=NODATA_VALUE_FLOAT,
    ) as dst:
        dst.write(cube_t1)
        dst.update_tags(**metadata_tags)
        for i, band_name in enumerate(["B2", "B3", "B4", "B8", "B11", "B12"], start=1):
            dst.set_band_description(i, band_name)
    file_paths["t1_raster"] = t1_raster_path

    # File 2: T2 Sentinel-2 6-band raster
    t2_raster_path = out_path / f"{region_id}_s2_t2.tif"
    with rasterio.open(
        t2_raster_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=6,
        dtype=np.float32,
        crs=crs,
        transform=transform,
        nodata=NODATA_VALUE_FLOAT,
    ) as dst:
        dst.write(cube_t2)
        dst.update_tags(**metadata_tags)
        for i, band_name in enumerate(["B2", "B3", "B4", "B8", "B11", "B12"], start=1):
            dst.set_band_description(i, band_name)
    file_paths["t2_raster"] = t2_raster_path

    # File 3: T1 Forest Mask
    t1_mask_path = out_path / f"{region_id}_mask_t1.tif"
    with rasterio.open(
        t1_mask_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=np.uint8,
        crs=crs,
        transform=transform,
        nodata=NODATA_VALUE_MASK,
    ) as dst:
        dst.write(mask_t1, 1)
        dst.update_tags(**metadata_tags)
        dst.set_band_description(1, "Dynamic World Tree Cover Mask (1=Forest, 0=Non-forest, 255=Nodata)")
    file_paths["t1_mask"] = t1_mask_path

    # File 4: T2 Forest Mask
    t2_mask_path = out_path / f"{region_id}_mask_t2.tif"
    with rasterio.open(
        t2_mask_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=np.uint8,
        crs=crs,
        transform=transform,
        nodata=NODATA_VALUE_MASK,
    ) as dst:
        dst.write(mask_t2, 1)
        dst.update_tags(**metadata_tags)
        dst.set_band_description(1, "Dynamic World Tree Cover Mask (1=Forest, 0=Non-forest, 255=Nodata)")
    file_paths["t2_mask"] = t2_mask_path

    return file_paths

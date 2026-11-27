"""Multi-temporal forest change detection, noise filtering, and geodetic area calculation.

Compares dual-date forest masks to identify potential tree-cover loss, stable forest,
and forest gain. Excludes invalid/cloudy observations, applies connected-component
minimum patch-size filtering, and computes true ground area in hectares.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import affine
import numpy as np
import pyproj
import rasterio
from rasterio.crs import CRS
from scipy import ndimage
import shapely.geometry

from src.forest_mask import create_and_save_forest_mask
from src.ndvi import compute_and_save_ndvi


# Classification codes for multi-temporal change map
CLASS_STABLE_NON_FOREST = 0
CLASS_STABLE_FOREST = 1
CLASS_POTENTIAL_LOSS = 2   # Tree-cover loss
CLASS_POTENTIAL_GAIN = 3   # Vegetation regrowth / canopy expansion
CLASS_NODATA = 255


def calculate_pixel_ground_area_m2(
    crs: CRS,
    transform: affine.Affine,
    height: int,
    width: int,
) -> float:
    """Calculates ground area of a single pixel in square meters.

    For projected CRS (e.g. UTM meters):
        Uses |det(transform)| = |dx * dy|.

    For geographic CRS (e.g. EPSG:4326 degrees):
        Computes the ellipsoidal geodesic surface area of the bounding box
        divided by total pixel count using the WGS84 ellipsoid.
    """
    if crs.is_projected:
        # Affine transform: a=dx, e=dy, b=shear_x, d=shear_y
        det = abs(transform.a * transform.e - transform.b * transform.d)
        if det > 0.0:
            return float(det)

    # Geographic or fallback: Ellipsoidal geodesic calculation via PyProj
    west = transform.c
    north = transform.f
    east = west + width * transform.a
    south = north + height * transform.e

    poly = shapely.geometry.box(min(west, east), min(south, north), max(west, east), max(south, north))
    geod = pyproj.Geod(ellps="WGS84")
    total_area_m2, _ = geod.geometry_area_perimeter(poly)
    total_pixels = height * width

    if total_pixels > 0:
        return float(abs(total_area_m2) / total_pixels)

    # Fallback to standard 10m x 10m = 100 m^2
    return 100.0


def detect_tree_cover_change(
    mask_t1: np.ndarray,
    mask_t2: np.ndarray,
    nodata_value: int = 255,
) -> np.ndarray:
    """Compares dual-date binary forest masks to produce a categorical change raster.

    Classes:
        0: Stable Non-Forest (0 at T1, 0 at T2)
        1: Stable Forest (1 at T1, 1 at T2)
        2: Potential Tree-Cover Loss (1 at T1, 0 at T2)
        3: Potential Forest Gain (0 at T1, 1 at T2)
        255: Nodata / Invalid (nodata in either T1 or T2)
    """
    if mask_t1.shape != mask_t2.shape:
        raise ValueError(f"Mask T1 shape {mask_t1.shape} must match Mask T2 shape {mask_t2.shape}")

    change_mask = np.full(mask_t1.shape, nodata_value, dtype=np.uint8)

    # Both dates must have valid observation
    valid_dual = (mask_t1 != nodata_value) & (mask_t2 != nodata_value)

    f1 = (mask_t1 == 1)
    f2 = (mask_t2 == 1)
    nf1 = (mask_t1 == 0)
    nf2 = (mask_t2 == 0)

    # Assign change classes
    change_mask[valid_dual & nf1 & nf2] = CLASS_STABLE_NON_FOREST
    change_mask[valid_dual & f1 & f2] = CLASS_STABLE_FOREST
    change_mask[valid_dual & f1 & nf2] = CLASS_POTENTIAL_LOSS
    change_mask[valid_dual & nf1 & f2] = CLASS_POTENTIAL_GAIN

    return change_mask


def filter_small_loss_patches(
    change_mask: np.ndarray,
    min_patch_pixels: int,
    loss_class_code: int = CLASS_POTENTIAL_LOSS,
    connectivity: int = 8,
) -> np.ndarray:
    """Removes isolated loss components smaller than a pixel threshold using connected component labeling.

    Suppressed tiny patches are reclassified to stable forest to eliminate edge misregistration noise.
    """
    if min_patch_pixels <= 1:
        return change_mask.copy()

    filtered = change_mask.copy()
    loss_binary = (filtered == loss_class_code)

    if not np.any(loss_binary):
        return filtered

    # Structure for 4-connectivity or 8-connectivity
    struct = ndimage.generate_binary_structure(2, 2 if connectivity == 8 else 1)
    labeled, num_features = ndimage.label(loss_binary, structure=struct)

    if num_features == 0:
        return filtered

    # Count component sizes
    component_sizes = np.bincount(labeled.ravel())
    # Identify components strictly smaller than threshold (ignoring background index 0)
    too_small_mask = component_sizes < min_patch_pixels
    too_small_mask[0] = False  # background is not a component

    remove_pixels = too_small_mask[labeled]
    # Reclassify tiny false-positive loss patches to stable forest
    filtered[remove_pixels] = CLASS_STABLE_FOREST

    return filtered


def run_change_detection_workflow(
    t1_raster_path: str | Path,
    t2_raster_path: str | Path,
    output_dir: str | Path,
    region_id: str = "region",
    date_t1: str = "T1",
    date_t2: str = "T2",
    ndvi_threshold: float = 0.60,
    min_patch_size_ha: float = 0.50,
    connectivity: int = 8,
    is_synthetic: bool = False,
) -> Dict[str, Any]:
    """Complete end-to-end change detection workflow from Sentinel-2 rasters to change maps and metrics.

    Steps:
        1. Compute and save NDVI for T1 and T2.
        2. Threshold NDVI into binary forest masks for T1 and T2.
        3. Detect multi-temporal change categories.
        4. Calculate pixel ground area from CRS and affine transform.
        5. Convert min_patch_size_ha to pixel count and filter small noise patches.
        6. Compute area metrics in hectares.
        7. Save categorical change GeoTIFF and write JSON summary report.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Output file paths
    t1_ndvi_path = out_dir / f"{region_id}_{date_t1}_ndvi.tif"
    t2_ndvi_path = out_dir / f"{region_id}_{date_t2}_ndvi.tif"
    t1_mask_path = out_dir / f"{region_id}_{date_t1}_forest_mask.tif"
    t2_mask_path = out_dir / f"{region_id}_{date_t2}_forest_mask.tif"
    change_map_path = out_dir / f"{region_id}_change_{date_t1}_{date_t2}.tif"
    summary_json_path = out_dir / f"{region_id}_change_summary_{date_t1}_{date_t2}.json"

    # 2. Compute NDVI rasters
    compute_and_save_ndvi(t1_raster_path, t1_ndvi_path)
    compute_and_save_ndvi(t2_raster_path, t2_ndvi_path)

    # 3. Create Forest Masks
    create_and_save_forest_mask(t1_ndvi_path, t1_mask_path, threshold=ndvi_threshold)
    create_and_save_forest_mask(t2_ndvi_path, t2_mask_path, threshold=ndvi_threshold)

    # 4. Read masks and CRS metadata
    with rasterio.open(t1_mask_path) as src_m1, rasterio.open(t2_mask_path) as src_m2:
        mask1 = src_m1.read(1)
        mask2 = src_m2.read(1)
        crs = src_m1.crs
        transform = src_m1.transform
        height, width = src_m1.shape
        meta = src_m1.meta.copy()

    # 5. Detect raw change
    raw_change = detect_tree_cover_change(mask1, mask2, nodata_value=CLASS_NODATA)

    # 6. Calculate ground pixel area (m^2) and minimum patch size in pixels
    pixel_area_m2 = calculate_pixel_ground_area_m2(crs, transform, height, width)
    # 1 ha = 10,000 m^2
    min_patch_pixels = max(1, int(np.ceil((min_patch_size_ha * 10000.0) / pixel_area_m2)))

    # 7. Filter small noise patches
    filtered_change = filter_small_loss_patches(
        raw_change,
        min_patch_pixels=min_patch_pixels,
        loss_class_code=CLASS_POTENTIAL_LOSS,
        connectivity=connectivity,
    )

    # 8. Compute pixel counts and area in hectares
    valid_mask = (filtered_change != CLASS_NODATA)
    valid_count = int(np.count_nonzero(valid_mask))

    stable_forest_count = int(np.count_nonzero(filtered_change == CLASS_STABLE_FOREST))
    stable_non_forest_count = int(np.count_nonzero(filtered_change == CLASS_STABLE_NON_FOREST))
    loss_count = int(np.count_nonzero(filtered_change == CLASS_POTENTIAL_LOSS))
    gain_count = int(np.count_nonzero(filtered_change == CLASS_POTENTIAL_GAIN))

    # Convert to hectares
    ha_per_pixel = pixel_area_m2 / 10000.0
    loss_area_ha = round(loss_count * ha_per_pixel, 3)
    stable_forest_area_ha = round(stable_forest_count * ha_per_pixel, 3)
    gain_area_ha = round(gain_count * ha_per_pixel, 3)
    total_valid_area_ha = round(valid_count * ha_per_pixel, 3)

    # 9. Write change GeoTIFF
    meta.update({
        "count": 1,
        "dtype": np.uint8,
        "nodata": CLASS_NODATA,
    })
    with rasterio.open(change_map_path, "w", **meta) as dst:
        dst.write(filtered_change, 1)
        dst.update_tags(
            DESCRIPTION="Multi-temporal tree-cover change detection map",
            DATE_T1=str(date_t1),
            DATE_T2=str(date_t2),
            NDVI_THRESHOLD=str(ndvi_threshold),
            MIN_PATCH_SIZE_HA=str(min_patch_size_ha),
            CLASSES="0=Stable Non-Forest, 1=Stable Forest, 2=Potential Tree-Cover Loss, 3=Potential Forest Gain, 255=Nodata",
            IS_SYNTHETIC="TRUE" if is_synthetic else "FALSE",
        )
        dst.set_band_description(1, "Forest Cover Change")

    # 10. Generate JSON Summary
    disclaimer = (
        "Potential tree-cover loss reflects canopy reduction between observation dates. "
        "It does not distinguish definitive agricultural land-use conversion from selective logging, "
        "forest degradation, windfall, fire, or rotational shifting cultivation without independent ground truth."
    )
    if is_synthetic:
        disclaimer = "DEMONSTRATION RUN ONLY: Metrics derived from synthetic demonstration rasters."

    summary: Dict[str, Any] = {
        "region_id": region_id,
        "dates": {"t1": str(date_t1), "t2": str(date_t2)},
        "is_synthetic": is_synthetic,
        "ndvi_threshold": ndvi_threshold,
        "min_patch_size_ha": min_patch_size_ha,
        "min_patch_pixels": min_patch_pixels,
        "pixel_area_m2": round(pixel_area_m2, 4),
        "pixel_counts": {
            "valid_pixels": valid_count,
            "stable_forest": stable_forest_count,
            "stable_non_forest": stable_non_forest_count,
            "potential_tree_cover_loss": loss_count,
            "potential_forest_gain": gain_count,
        },
        "areas_hectares": {
            "monitored_valid_area_ha": total_valid_area_ha,
            "stable_forest_ha": stable_forest_area_ha,
            "potential_tree_cover_loss_ha": loss_area_ha,
            "potential_forest_gain_ha": gain_area_ha,
        },
        "output_files": {
            "t1_ndvi": str(t1_ndvi_path),
            "t2_ndvi": str(t2_ndvi_path),
            "t1_forest_mask": str(t1_mask_path),
            "t2_forest_mask": str(t2_mask_path),
            "change_map": str(change_map_path),
        },
        "scientific_disclaimer": disclaimer,
    }

    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


# ============================================================================
# Command-Line Demo Entry Point
# ============================================================================

if __name__ == "__main__":
    import argparse
    from src.mock_data import generate_synthetic_dataset

    parser = argparse.ArgumentParser(description="ForestWatch AI: NDVI Forest Change Detection")
    parser.add_argument("--demo", action="store_true", help="Run end-to-end demo on synthetic demonstration data")
    parser.add_argument("--t1-raster", type=str, help="Path to T1 6-band Sentinel-2 GeoTIFF")
    parser.add_argument("--t2-raster", type=str, help="Path to T2 6-band Sentinel-2 GeoTIFF")
    parser.add_argument("--output-dir", type=str, default="data/outputs", help="Output directory")
    parser.add_argument("--threshold", type=float, default=0.60, help="NDVI forest classification threshold")
    parser.add_argument("--min-patch-ha", type=float, default=0.50, help="Minimum patch size in hectares")
    parser.add_argument("--region-id", type=str, default="synthetic_amazon_demo", help="Region identifier")
    args = parser.parse_args()

    if args.demo or (not args.t1_raster and not args.t2_raster):
        print("=== Running ForestWatch AI Change Detection Demo (Synthetic Data) ===")
        print("Generating synthetic multi-temporal demonstration rasters...")
        demo_files = generate_synthetic_dataset(
            output_dir="data/synthetic",
            region_id="synthetic_amazon_demo",
            height=512,
            width=768,
            seed=42,
        )
        t1_path = demo_files["t1_raster"]
        t2_path = demo_files["t2_raster"]
        is_synth = True
    else:
        t1_path = Path(args.t1_raster)
        t2_path = Path(args.t2_raster)
        is_synth = False

    print(f"Executing change detection workflow with NDVI threshold={args.threshold}, min_patch={args.min_patch_ha} ha...")
    result_summary = run_change_detection_workflow(
        t1_raster_path=t1_path,
        t2_raster_path=t2_path,
        output_dir=args.output_dir,
        region_id=args.region_id,
        date_t1="2020",
        date_t2="2023",
        ndvi_threshold=args.threshold,
        min_patch_size_ha=args.min_patch_ha,
        is_synthetic=is_synth,
    )

    print("\n--- Change Detection Results Summary ---")
    print(f"Region: {result_summary['region_id']} (Synthetic: {result_summary['is_synthetic']})")
    print(f"Pixel Ground Area: {result_summary['pixel_area_m2']} m^2 (Min patch: {result_summary['min_patch_pixels']} pixels)")
    print(f"Monitored Valid Area: {result_summary['areas_hectares']['monitored_valid_area_ha']} ha")
    print(f"Stable Forest: {result_summary['areas_hectares']['stable_forest_ha']} ha")
    print(f"Potential Tree-Cover Loss: {result_summary['areas_hectares']['potential_tree_cover_loss_ha']} ha ({result_summary['pixel_counts']['potential_tree_cover_loss']} pixels)")
    print(f"Potential Forest Gain: {result_summary['areas_hectares']['potential_forest_gain_ha']} ha")
    print(f"Outputs written to: {args.output_dir}")
    print(f"Summary JSON: {result_summary['output_files']['change_map']}")

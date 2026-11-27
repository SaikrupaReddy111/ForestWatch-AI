
import json
from pathlib import Path

import numpy as np
import rasterio

from src.change_detection import (
    detect_tree_cover_change,
    filter_small_loss_patches,
    calculate_pixel_ground_area_m2,
)

EARLY_MASK = Path("data/outputs/western_ghats_pilot_forest_2020.tif")
LATE_MASK = Path("data/outputs/western_ghats_pilot_forest_2023.tif")

OUTPUT_DIR = Path("data/outputs")
CHANGE_MAP = OUTPUT_DIR / "western_ghats_pilot_change_2020_2023.tif"
SUMMARY_FILE = OUTPUT_DIR / "western_ghats_pilot_summary_2020_2023.json"

MIN_PATCH_HA = 0.50


def main():
    print("=== ForestWatch AI: Western Ghats Pilot Change Detection ===")

    with rasterio.open(EARLY_MASK) as src1, rasterio.open(LATE_MASK) as src2:
        if src1.shape != src2.shape:
            raise ValueError("Prediction dimensions do not match.")

        if src1.crs != src2.crs:
            raise ValueError("Prediction CRS does not match.")

        if src1.transform != src2.transform:
            raise ValueError("Prediction transforms do not match.")

        mask_2020 = src1.read(1)
        mask_2023 = src2.read(1)

        profile = src1.profile.copy()
        crs = src1.crs
        transform = src1.transform
        height, width = src1.shape

    print(f"Raster dimensions: {width} x {height}")
    print(f"CRS: {crs}")

    change = detect_tree_cover_change(
        mask_2020,
        mask_2023,
        nodata_value=255,
    )

    pixel_area = calculate_pixel_ground_area_m2(
        crs, transform, height, width
    )

    min_patch_pixels = max(
        1, int(np.ceil(MIN_PATCH_HA * 10000 / pixel_area))
    )

    print(f"Pixel area: {pixel_area:.2f} m²")
    print(f"Minimum loss patch: {min_patch_pixels} pixels")

    change = filter_small_loss_patches(
        change,
        min_patch_pixels=min_patch_pixels,
        loss_class_code=2,
        connectivity=8,
    )

    valid = change != 255

    counts = {
        "stable_non_forest": int(np.count_nonzero(change == 0)),
        "stable_forest": int(np.count_nonzero(change == 1)),
        "potential_tree_cover_loss": int(np.count_nonzero(change == 2)),
        "potential_forest_gain": int(np.count_nonzero(change == 3)),
        "invalid_pixels": int(np.count_nonzero(~valid)),
    }

    hectares_per_pixel = pixel_area / 10000

    areas = {
        "stable_forest_ha": round(
            counts["stable_forest"] * hectares_per_pixel, 3
        ),
        "potential_tree_cover_loss_ha": round(
            counts["potential_tree_cover_loss"] * hectares_per_pixel, 3
        ),
        "potential_forest_gain_ha": round(
            counts["potential_forest_gain"] * hectares_per_pixel, 3
        ),
        "valid_monitored_area_ha": round(
            np.count_nonzero(valid) * hectares_per_pixel, 3
        ),
    }

    profile.update(
        count=1,
        dtype="uint8",
        nodata=255,
        compress="deflate",
    )

    with rasterio.open(CHANGE_MAP, "w", **profile) as dst:
        dst.write(change.astype(np.uint8), 1)
        dst.set_band_description(1, "Western Ghats Pilot Forest Cover Change")
        dst.update_tags(
            DATE_T1="2020",
            DATE_T2="2023",
            CLASSES="0=Stable Non-Forest,1=Stable Forest,2=Potential Loss,3=Potential Gain,255=NoData",
            SOURCE="Experimental U-Net predictions trained with Amazon Dynamic World pseudo-labels",
        )

    summary = {
        "region": "Western Ghats pilot area",
        "period": "2020-2023",
        "method": "Experimental U-Net prediction comparison",
        "pixel_area_m2": pixel_area,
        "minimum_patch_ha": MIN_PATCH_HA,
        "pixel_counts": counts,
        "areas_hectares": areas,
        "change_map": str(CHANGE_MAP),
        "scientific_note": (
            "Experimental model-based change estimates. The U-Net was trained "
            "on Amazon Brazil data using Dynamic World pseudo-labels and applied "
            "to a Western Ghats pilot area. These results are not independently "
            "validated ground-truth deforestation measurements."
        ),
    }

    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== WESTERN GHATS CHANGE DETECTION COMPLETE ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

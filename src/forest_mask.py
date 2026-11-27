"""NDVI thresholding for forest cover classification and binary mask export.

Generates binary forest masks using a configurable threshold while strictly
preserving nodata and unobserved pixels separately from classified non-forest pixels.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import rasterio


def threshold_ndvi_to_forest_mask(
    ndvi: np.ndarray,
    threshold: float = 0.60,
    ndvi_nodata: float = -9999.0,
    mask_nodata: int = 255,
) -> np.ndarray:
    """Classifies an NDVI array into forest, non-forest, and nodata.

    Output Values:
        1: Forest (NDVI >= threshold and valid)
        0: Non-forest (NDVI < threshold and valid)
        255: Nodata / invalid (NDVI == ndvi_nodata or NaN)

    Args:
        ndvi: 2D array of NDVI values.
        threshold: Configurable minimum NDVI to classify as forest cover (e.g. 0.60).
        ndvi_nodata: Sentinel value representing invalid NDVI pixels.
        mask_nodata: Output integer sentinel value for invalid pixels (default 255).

    Returns:
        uint8 array with values in {0, 1, mask_nodata}.
    """
    mask = np.full(ndvi.shape, mask_nodata, dtype=np.uint8)

    valid_pixels = (ndvi != ndvi_nodata) & ~np.isnan(ndvi)

    # Classify valid pixels
    forest_pixels = valid_pixels & (ndvi >= threshold)
    non_forest_pixels = valid_pixels & (ndvi < threshold)

    mask[non_forest_pixels] = 0
    mask[forest_pixels] = 1

    return mask


def create_and_save_forest_mask(
    ndvi_path: str | Path,
    output_path: str | Path,
    threshold: float = 0.60,
    ndvi_nodata: float = -9999.0,
    mask_nodata: int = 255,
) -> Path:
    """Thresholds an NDVI GeoTIFF and writes a single-band uint8 forest mask GeoTIFF.

    Preserves exact CRS, dimensions, resolution, and affine transform from the input.
    """
    in_p = Path(ndvi_path)
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(in_p) as src:
        ndvi = src.read(1)
        src_nodata = src.nodata if src.nodata is not None else ndvi_nodata

        mask = threshold_ndvi_to_forest_mask(
            ndvi=ndvi,
            threshold=threshold,
            ndvi_nodata=src_nodata,
            mask_nodata=mask_nodata,
        )

        out_meta = src.meta.copy()
        out_meta.update({
            "count": 1,
            "dtype": np.uint8,
            "nodata": mask_nodata,
        })

        with rasterio.open(out_p, "w", **out_meta) as dst:
            dst.write(mask, 1)
            dst.update_tags(
                DESCRIPTION="NDVI-derived binary forest mask",
                NDVI_THRESHOLD=str(threshold),
                CLASSES="1=Forest, 0=Non-forest, 255=Nodata",
            )
            dst.set_band_description(1, f"Forest Mask (NDVI >= {threshold})")

    return out_p

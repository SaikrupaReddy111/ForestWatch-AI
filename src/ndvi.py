"""Normalized Difference Vegetation Index (NDVI) computation and raster generation.

Computes NDVI = (NIR - Red) / (NIR + Red) safely with explicit nodata,
invalid pixel, and zero-denominator handling, preserving complete geospatial metadata.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import rasterio


def compute_ndvi_array(
    nir: np.ndarray,
    red: np.ndarray,
    nodata_value: float = -9999.0,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """Computes Normalized Difference Vegetation Index (NDVI) for NumPy arrays.

    Formula:
        NDVI = (NIR - Red) / (NIR + Red + epsilon)

    Safety & Validation Rules:
        - If either NIR or Red is nodata (-9999.0 or user-specified), output is nodata.
        - If either NIR or Red is NaN, output is nodata.
        - If (NIR + Red) <= 0.0 (e.g. invalid shadow or water zero radiance), output is nodata.
        - Valid output values are clipped to [-1.0, 1.0].

    Args:
        nir: Near-infrared reflectance array (e.g. Sentinel-2 B8).
        red: Red reflectance array (e.g. Sentinel-2 B4).
        nodata_value: Sentinal value representing invalid pixels (default -9999.0).
        epsilon: Small numerical stabilizer to prevent division by exact zero.

    Returns:
        float32 array containing NDVI in [-1.0, 1.0] and nodata_value for invalid pixels.
    """
    if nir.shape != red.shape:
        raise ValueError(f"NIR shape {nir.shape} must match Red shape {red.shape}")

    ndvi = np.full(nir.shape, nodata_value, dtype=np.float32)

    # Detect valid input pixels
    valid_nir = (nir != nodata_value) & ~np.isnan(nir)
    valid_red = (red != nodata_value) & ~np.isnan(red)
    both_valid = valid_nir & valid_red

    denom = nir + red
    # Denominator must be strictly positive to represent physical vegetation reflectance
    compute_mask = both_valid & (denom > 0.0)

    if np.any(compute_mask):
        numerator = nir[compute_mask].astype(np.float32) - red[compute_mask].astype(np.float32)
        denominator = denom[compute_mask].astype(np.float32) + np.float32(epsilon)
        ratio = numerator / denominator
        ndvi[compute_mask] = np.clip(ratio, -1.0, 1.0)

    return ndvi


def resolve_band_indices(
    src: rasterio.io.DatasetReader,
    default_nir_idx: int = 4,
    default_red_idx: int = 3,
) -> Tuple[int, int]:
    """Resolves NIR and Red 1-based band indices from raster band descriptions or defaults."""
    descriptions = [d.upper() if d else "" for d in src.descriptions]
    nir_idx = default_nir_idx
    red_idx = default_red_idx

    for i, desc in enumerate(descriptions, start=1):
        if desc == "B8" or "NIR" in desc:
            nir_idx = i
        elif desc == "B4" or "RED" in desc:
            red_idx = i

    return nir_idx, red_idx


def compute_and_save_ndvi(
    raster_path: str | Path,
    output_path: str | Path,
    nir_band_idx: Optional[int] = None,
    red_band_idx: Optional[int] = None,
    nodata_value: float = -9999.0,
    epsilon: float = 1e-6,
) -> Path:
    """Computes NDVI from a multispectral GeoTIFF and writes a single-band GeoTIFF.

    Preserves exact CRS, dimensions, resolution, and affine transform from the input.
    """
    in_p = Path(raster_path)
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(in_p) as src:
        # Resolve band indices if not explicitly provided
        nir_idx, red_idx = (
            (nir_band_idx, red_band_idx)
            if (nir_band_idx is not None and red_band_idx is not None)
            else resolve_band_indices(src)
        )

        if nir_idx > src.count or red_idx > src.count:
            raise ValueError(
                f"Requested band indices (NIR={nir_idx}, Red={red_idx}) exceed "
                f"raster band count ({src.count})."
            )

        nir = src.read(nir_idx)
        red = src.read(red_idx)

        # In case the source raster uses a specific nodata value
        src_nodata = src.nodata if src.nodata is not None else nodata_value
        if src_nodata != nodata_value:
            nir[nir == src_nodata] = nodata_value
            red[red == src_nodata] = nodata_value

        ndvi_array = compute_ndvi_array(
            nir=nir,
            red=red,
            nodata_value=nodata_value,
            epsilon=epsilon,
        )

        out_meta = src.meta.copy()
        out_meta.update({
            "count": 1,
            "dtype": np.float32,
            "nodata": nodata_value,
        })

        with rasterio.open(out_p, "w", **out_meta) as dst:
            dst.write(ndvi_array, 1)
            dst.update_tags(
                DESCRIPTION="Normalized Difference Vegetation Index (NDVI)",
                FORMULA="(NIR - Red) / (NIR + Red)",
                NIR_BAND=f"Band {nir_idx}",
                RED_BAND=f"Band {red_idx}",
            )
            dst.set_band_description(1, "NDVI")

    return out_p

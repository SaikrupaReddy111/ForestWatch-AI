"""Geospatial raster alignment, reprojection, and nodata harmonization.

Ensures Sentinel-2 imagery and Dynamic World land cover masks share the exact
same Coordinate Reference System (CRS), spatial resolution, affine transform,
and pixel bounding grid.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject


def verify_raster_alignment(
    raster_path_a: str | Path,
    raster_path_b: str | Path,
    tolerance: float = 1e-4,
) -> Tuple[bool, str]:
    """Verifies that two rasters have identical CRS, dimensions, resolution, and affine grid.

    Returns:
        (is_aligned, message)
    """
    path_a = Path(raster_path_a)
    path_b = Path(raster_path_b)

    if not path_a.is_file():
        return False, f"File A does not exist: {path_a.resolve()}"
    if not path_b.is_file():
        return False, f"File B does not exist: {path_b.resolve()}"

    with rasterio.open(path_a) as src_a, rasterio.open(path_b) as src_b:
        # Check CRS
        if src_a.crs != src_b.crs:
            return False, f"CRS mismatch: {src_a.crs} vs {src_b.crs}"

        # Check Shape (Height, Width)
        if src_a.shape != src_b.shape:
            return False, f"Shape mismatch: {src_a.shape} vs {src_b.shape}"

        # Check Resolution
        if (
            abs(src_a.res[0] - src_b.res[0]) > tolerance
            or abs(src_a.res[1] - src_b.res[1]) > tolerance
        ):
            return False, f"Resolution mismatch: {src_a.res} vs {src_b.res}"

        # Check Affine Transform
        diffs = [abs(a - b) for a, b in zip(src_a.transform, src_b.transform)]
        if any(d > tolerance for d in diffs):
            return False, f"Affine transform mismatch: {src_a.transform} vs {src_b.transform}"

    return True, "Rasters are perfectly aligned (identical CRS, shape, resolution, and affine grid)."


def align_raster_to_reference(
    input_path: str | Path,
    reference_path: str | Path,
    output_path: str | Path,
    resampling_method: str = "nearest",
    dst_nodata: Optional[float | int] = None,
) -> Path:
    """Reprojects and resamples input raster to match the exact grid of reference raster.

    Args:
        input_path: Path to source raster to be aligned.
        reference_path: Path to reference raster whose CRS and grid must be matched.
        output_path: Path for aligned raster.
        resampling_method: 'nearest' (categorical/masks) or 'bilinear'/'cubic' (reflectance).
        dst_nodata: Nodata value for destination raster.

    Returns:
        Path to output aligned raster.
    """
    in_p = Path(input_path)
    ref_p = Path(reference_path)
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    resample_enum = (
        Resampling.nearest if resampling_method == "nearest"
        else Resampling.bilinear if resampling_method == "bilinear"
        else Resampling.cubic
    )

    with rasterio.open(ref_p) as ref_src, rasterio.open(in_p) as in_src:
        dst_crs = ref_src.crs
        dst_transform = ref_src.transform
        dst_width = ref_src.width
        dst_height = ref_src.height
        dst_count = in_src.count
        dst_dtype = in_src.dtypes[0]

        nodata_val = dst_nodata if dst_nodata is not None else in_src.nodata

        out_meta = in_src.meta.copy()
        out_meta.update({
            "crs": dst_crs,
            "transform": dst_transform,
            "width": dst_width,
            "height": dst_height,
            "nodata": nodata_val,
        })

        with rasterio.open(out_p, "w", **out_meta) as dst:
            for i in range(1, dst_count + 1):
                reproject(
                    source=rasterio.band(in_src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=in_src.transform,
                    src_crs=in_src.crs,
                    src_nodata=in_src.nodata,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    dst_nodata=nodata_val,
                    resampling=resample_enum,
                )

    return out_p


def generate_valid_pixel_mask(
    image_path: str | Path,
    mask_path: str | Path,
    output_path: str | Path,
    image_nodata: float = -9999.0,
    mask_nodata: int = 255,
) -> Path:
    """Generates an explicit boolean validity mask combining image and label validity.

    Valid pixels (1) must have non-nodata reflectance across all bands and non-nodata label.
    Invalid pixels (0) represent cloud shadows, clouds, cirrus, or unobserved margins.
    """
    img_p = Path(image_path)
    msk_p = Path(mask_path)
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(img_p) as img_src, rasterio.open(msk_p) as msk_src:
        img_data = img_src.read()  # (B, H, W)
        msk_data = msk_src.read(1)  # (H, W)

        # Image validity across all bands
        img_valid = np.all(img_data != image_nodata, axis=0) & ~np.isnan(img_data).any(axis=0)
        # Mask validity
        msk_valid = (msk_data != mask_nodata)

        combined_valid = (img_valid & msk_valid).astype(np.uint8)

        out_meta = msk_src.meta.copy()
        out_meta.update({
            "count": 1,
            "dtype": np.uint8,
            "nodata": 0,
        })

        with rasterio.open(out_p, "w", **out_meta) as dst:
            dst.write(combined_valid, 1)
            dst.update_tags(
                DESCRIPTION="Boolean observation validity mask (1=Valid, 0=Invalid/Cloud/Nodata)"
            )

    return out_p

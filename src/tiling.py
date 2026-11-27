"""Raster tiling and spatial split utilities for Sentinel-2 and Dynamic World data."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import rasterio
from rasterio.windows import Window, bounds as window_bounds, transform as window_transform


@dataclass
class TileRecord:
    tile_id: str
    region_id: str
    row_idx: int
    col_idx: int
    split: Literal["train", "val", "test", "buffer", "unassigned"]
    bounds: Tuple[float, float, float, float]
    crs: str
    height: int
    width: int
    valid_fraction: float
    is_usable: bool
    raster_tile_path: str
    mask_tile_path: str

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["bounds"] = list(self.bounds)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TileRecord":
        item = data.copy()
        item["bounds"] = tuple(item["bounds"])
        return cls(**item)


def _assert_matching_grids(image, mask) -> None:
    if image.count != 6:
        raise ValueError(f"Expected 6 image bands, found {image.count}.")
    if mask.count != 1:
        raise ValueError(f"Expected 1 mask band, found {mask.count}.")
    if image.crs != mask.crs:
        raise ValueError(f"CRS mismatch: {image.crs} vs {mask.crs}.")
    if (image.width, image.height) != (mask.width, mask.height):
        raise ValueError("Image and mask dimensions do not match.")
    if not image.transform.almost_equals(mask.transform):
        raise ValueError("Image and mask affine transforms do not match.")


def slice_raster_into_tiles(
    raster_path: str | Path,
    mask_path: str | Path,
    output_dir: str | Path,
    region_id: str,
    tile_size: int = 256,
    nodata_value: float = -9999.0,
    mask_nodata_value: int = 255,
    min_valid_fraction: float = 0.05,
    drop_incomplete_edges: bool = True,
) -> List[TileRecord]:
    """Write aligned image/mask tiles; invalid pixels become -9999/255.

    Source NoData metadata is not required. Rasterio validity masks, finite
    checks, and expected binary label values (0 and 1) determine validity.
    """
    if tile_size <= 0:
        raise ValueError("tile_size must be positive.")
    if not 0 <= min_valid_fraction <= 1:
        raise ValueError("min_valid_fraction must be between 0 and 1.")
    if mask_nodata_value in (0, 1):
        raise ValueError("mask_nodata_value must differ from valid labels 0 and 1.")

    out_dir = Path(output_dir) / region_id
    out_dir.mkdir(parents=True, exist_ok=True)
    records: List[TileRecord] = []

    with rasterio.open(raster_path) as image, rasterio.open(mask_path) as mask:
        _assert_matching_grids(image, mask)
        height, width = image.height, image.width
        rows = height // tile_size if drop_incomplete_edges else int(np.ceil(height / tile_size))
        cols = width // tile_size if drop_incomplete_edges else int(np.ceil(width / tile_size))

        for row in range(rows):
            for col in range(cols):
                row_off, col_off = row * tile_size, col * tile_size
                win_h = min(tile_size, height - row_off)
                win_w = min(tile_size, width - col_off)
                if drop_incomplete_edges and (win_h != tile_size or win_w != tile_size):
                    continue

                window = Window(col_off, row_off, win_w, win_h)
                tile_transform = window_transform(window, image.transform)
                bbox = window_bounds(window, image.transform)

                img = image.read(window=window).astype(np.float32, copy=False)
                labels = mask.read(1, window=window)
                image_valid = (image.read_masks(window=window) > 0).all(axis=0)
                image_valid &= np.isfinite(img).all(axis=0)
                mask_valid = mask.read_masks(1, window=window) > 0
                mask_valid &= np.isfinite(labels)
                mask_valid &= (labels == 0) | (labels == 1)
                valid = image_valid & mask_valid

                img_out = img.copy()
                img_out[:, ~valid] = nodata_value
                mask_out = labels.astype(np.uint8, copy=True)
                mask_out[~valid] = mask_nodata_value
                valid_fraction = float(valid.mean())
                usable = valid_fraction >= min_valid_fraction

                tile_id = f"{region_id}_r{row:03d}_c{col:03d}"
                image_tile_path = out_dir / f"{tile_id}_raster.tif"
                mask_tile_path = out_dir / f"{tile_id}_mask.tif"

                image_meta = image.meta.copy()
                image_meta.update(
                    driver="GTiff",
                    height=win_h,
                    width=win_w,
                    count=6,
                    dtype="float32",
                    transform=tile_transform,
                    nodata=nodata_value,
                    compress="deflate",
                )
                with rasterio.open(image_tile_path, "w", **image_meta) as dst:
                    dst.write(img_out)
                    for band, name in enumerate(("B2", "B3", "B4", "B8", "B11", "B12"), 1):
                        dst.set_band_description(band, name)

                mask_meta = mask.meta.copy()
                mask_meta.update(
                    driver="GTiff",
                    height=win_h,
                    width=win_w,
                    count=1,
                    dtype="uint8",
                    transform=tile_transform,
                    nodata=mask_nodata_value,
                    compress="deflate",
                )
                with rasterio.open(mask_tile_path, "w", **mask_meta) as dst:
                    dst.write(mask_out, 1)
                    dst.set_band_description(1, "Dynamic World tree-cover pseudo-label")
                    dst.update_tags(CLASSES="1=Trees, 0=Other, 255=Invalid")

                records.append(
                    TileRecord(
                        tile_id=tile_id,
                        region_id=region_id,
                        row_idx=row,
                        col_idx=col,
                        split="unassigned",
                        bounds=tuple(float(v) for v in bbox),
                        crs=image.crs.to_string(),
                        height=win_h,
                        width=win_w,
                        valid_fraction=round(valid_fraction, 4),
                        is_usable=usable,
                        raster_tile_path=str(image_tile_path),
                        mask_tile_path=str(mask_tile_path),
                    )
                )

    return records


def assign_spatial_block_splits(
    tiles: List[TileRecord],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    buffer_tiles: int = 1,
    split_axis: Literal["col", "row"] = "col",
) -> List[TileRecord]:
    """Assign contiguous spatial splits, using buffer blocks where feasible.

    For small datasets, validation is omitted before sacrificing a requested
    train/test separation buffer. If only one split has a positive ratio, all
    tiles are assigned to that split.
    """
    if not tiles:
        return []
    if split_axis not in ("col", "row"):
        raise ValueError("split_axis must be 'col' or 'row'.")

    ratios = [float(train_ratio), float(val_ratio), float(test_ratio)]
    if any(r < 0 for r in ratios) or not np.isclose(sum(ratios), 1.0):
        raise ValueError("Split ratios must be nonnegative and sum to 1.")
    if buffer_tiles < 0:
        raise ValueError("buffer_tiles cannot be negative.")

    split_names = ["train", "val", "test"]
    active = [i for i, ratio in enumerate(ratios) if ratio > 0]
    if not active:
        raise ValueError("At least one split ratio must be positive.")

    if len(active) == 1:
        name = split_names[active[0]]
        for tile in tiles:
            tile.split = name
        return tiles

    indices = sorted({
        tile.col_idx if split_axis == "col" else tile.row_idx
        for tile in tiles
    })
    n_blocks = len(indices)

    # Preserve train and test for evaluation; omit validation first when the
    # available spatial blocks cannot support all requested splits and buffers.
    if n_blocks < len(active) + buffer_tiles * (len(active) - 1):
        if 1 in active and 0 in active and 2 in active:
            active = [0, 2]
        elif len(active) == 3:
            active = [i for i in active if i != 1]

    if n_blocks < len(active):
        raise ValueError(
            f"Only {n_blocks} spatial blocks are available for "
            f"{len(active)} nonempty splits."
        )

    if len(active) == 1:
        name = split_names[active[0]]
        for tile in tiles:
            tile.split = name
        return tiles

    # Reduce the buffer only if the remaining requested splits still cannot fit.
    max_buffer = (n_blocks - len(active)) // (len(active) - 1)
    effective_buffer = min(buffer_tiles, max_buffer)
    usable_blocks = n_blocks - effective_buffer * (len(active) - 1)

    active_weights = np.array([ratios[i] for i in active], dtype=np.float64)
    active_weights /= active_weights.sum()

    counts = np.ones(len(active), dtype=int)
    remaining = usable_blocks - len(active)
    if remaining:
        raw_extra = active_weights * remaining
        extra = np.floor(raw_extra).astype(int)
        counts += extra
        remainder = remaining - int(extra.sum())
        order = np.argsort(-(raw_extra - extra))
        for position in order[:remainder]:
            counts[position] += 1

    assignments: Dict[int, str] = {}
    cursor = 0

    for position, split_index in enumerate(active):
        count = int(counts[position])
        for block_index in indices[cursor:cursor + count]:
            assignments[block_index] = split_names[split_index]
        cursor += count

        if position < len(active) - 1:
            for block_index in indices[cursor:cursor + effective_buffer]:
                assignments[block_index] = "buffer"
            cursor += effective_buffer

    for tile in tiles:
        block_index = tile.col_idx if split_axis == "col" else tile.row_idx
        tile.split = assignments.get(block_index, "buffer")

    return tiles


def assign_leave_one_region_out_splits(
    tiles_by_region: Dict[str, List[TileRecord]],
    test_region_id: str,
    val_region_id: Optional[str] = None,
) -> List[TileRecord]:
    """Assign whole regions to train, validation, and held-out test partitions."""
    if test_region_id not in tiles_by_region:
        raise ValueError(f"test_region_id '{test_region_id}' not found.")
    if val_region_id is not None and val_region_id not in tiles_by_region:
        raise ValueError(f"val_region_id '{val_region_id}' not found.")
    if val_region_id == test_region_id:
        raise ValueError("Test and validation regions must differ.")

    result: List[TileRecord] = []
    for region_id, region_tiles in tiles_by_region.items():
        split = "test" if region_id == test_region_id else (
            "val" if region_id == val_region_id else "train"
        )
        for tile in region_tiles:
            tile.split = split
        result.extend(region_tiles)
    return result


def compute_training_normalization_stats(
    tiles: List[TileRecord],
    band_names: Optional[List[str]] = None,
    nodata_value: float = -9999.0,
) -> Dict[str, Dict[str, float]]:
    """Compute per-band mean/std using valid pixels from training tiles only."""
    training = [tile for tile in tiles if tile.split == "train" and tile.is_usable]
    if not training:
        raise ValueError("No usable training tiles found.")
    if band_names is None:
        band_names = ["B2", "B3", "B4", "B8", "B11", "B12"]
    if len(band_names) != 6:
        raise ValueError("Exactly six band names are required.")

    sums = np.zeros(6, dtype=np.float64)
    squares = np.zeros(6, dtype=np.float64)
    counts = np.zeros(6, dtype=np.int64)

    for tile in training:
        with rasterio.open(tile.raster_tile_path) as src:
            data = src.read().astype(np.float64)
            valid = (src.read_masks() > 0).all(axis=0)
            valid &= np.isfinite(data).all(axis=0)
            valid &= (data != nodata_value).all(axis=0)

            for band in range(6):
                values = data[band][valid]
                sums[band] += values.sum()
                squares[band] += np.square(values).sum()
                counts[band] += values.size

    stats: Dict[str, Dict[str, float]] = {}
    for index, name in enumerate(band_names):
        if counts[index] == 0:
            raise ValueError(f"No valid training pixels found for {name}.")
        mean = sums[index] / counts[index]
        variance = max(1e-12, squares[index] / counts[index] - mean ** 2)
        stats[name] = {
            "mean": float(mean),
            "std": float(np.sqrt(variance)),
            "pixel_count": int(counts[index]),
        }
    return stats


def save_tiles_index(tiles: List[TileRecord], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump([tile.to_dict() for tile in tiles], file, indent=2)


def load_tiles_index(index_path: str | Path) -> List[TileRecord]:
    path = Path(index_path)
    if not path.is_file():
        raise FileNotFoundError(f"Tile index not found: {path.resolve()}")
    with path.open("r", encoding="utf-8") as file:
        return [TileRecord.from_dict(item) for item in json.load(file)]


def verify_split_isolation(
    tiles: List[TileRecord],
    expected_buffer_distance_m: float,
) -> Tuple[bool, str]:
    """Verify no train/test overlap and check minimum separation distance."""
    train = [tile for tile in tiles if tile.split == "train"]
    test = [tile for tile in tiles if tile.split == "test"]

    if not train or not test:
        return False, "Cannot verify isolation because train or test split is empty."

    minimum = float("inf")
    for train_tile in train:
        x1, y1, x2, y2 = train_tile.bounds
        for test_tile in test:
            a1, b1, a2, b2 = test_tile.bounds
            overlap_x = max(0.0, min(x2, a2) - max(x1, a1))
            overlap_y = max(0.0, min(y2, b2) - max(y1, b1))
            if overlap_x > 0 and overlap_y > 0:
                return False, (
                    f"Train/test overlap: {train_tile.tile_id} "
                    f"and {test_tile.tile_id}."
                )

            dx = max(0.0, max(x1 - a2, a1 - x2))
            dy = max(0.0, max(y1 - b2, b1 - y2))
            minimum = min(minimum, float(np.hypot(dx, dy)))

    if minimum < expected_buffer_distance_m:
        return False, (
            f"Buffer violation: {minimum:.1f}m is below "
            f"{expected_buffer_distance_m:.1f}m."
        )

    return True, (
        f"Split isolation verified: minimum train/test distance "
        f"{minimum:.1f}m >= {expected_buffer_distance_m:.1f}m."
    )

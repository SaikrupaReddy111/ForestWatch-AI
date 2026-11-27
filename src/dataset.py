
"""
PyTorch Dataset, spatial augmentations, and DataLoader factory
for 6-band Sentinel-2 tiles.

Strictly preserves geographic block split isolation, applies
training-only normalization, and handles nodata ignore-labels.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import rasterio
import torch
from torch.utils.data import DataLoader, Dataset

from src.tiling import TileRecord

BAND_NAMES = ["B2", "B3", "B4", "B8", "B11", "B12"]


class ForestTileDataset(Dataset):
    """PyTorch Dataset loading 6-band satellite tiles and forest masks."""

    def __init__(
        self,
        tiles: List[TileRecord],
        normalization_stats: Optional[Dict[str, Dict[str, float]]] = None,
        augment: bool = False,
        nodata_value: float = -9999.0,
        mask_nodata_value: int = 255,
    ):
        self.tiles = [t for t in tiles if t.is_usable]
        self.normalization_stats = normalization_stats
        self.augment = augment
        self.nodata_value = nodata_value
        self.mask_nodata_value = mask_nodata_value

        if normalization_stats is not None:
            self.means = np.array(
                [
                    normalization_stats.get(b, {}).get("mean", 0.0)
                    for b in BAND_NAMES
                ],
                dtype=np.float32,
            ).reshape(6, 1, 1)

            self.stds = np.array(
                [
                    normalization_stats.get(b, {}).get("std", 1.0)
                    for b in BAND_NAMES
                ],
                dtype=np.float32,
            ).reshape(6, 1, 1)

            self.stds = np.where(self.stds <= 0.0, 1.0, self.stds)
        else:
            self.means = None
            self.stds = None

    def __len__(self) -> int:
        return len(self.tiles)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, str]:

        record = self.tiles[idx]

        # 1. Read multispectral raster tile
        with rasterio.open(record.raster_tile_path) as src_r:
            raster_data = src_r.read().astype(np.float32)
            raster_nodata = (
                src_r.nodata
                if src_r.nodata is not None
                else self.nodata_value
            )

        # 2. Read forest mask tile
        with rasterio.open(record.mask_tile_path) as src_m:
            mask_data = src_m.read(1).astype(np.int64)
            mask_file_nodata = src_m.nodata

        # 3. Detect invalid raster pixels across all bands
        if raster_nodata is not None and np.isnan(raster_nodata):
            nodata_pixels = np.isnan(raster_data)
        elif raster_nodata is not None:
            nodata_pixels = np.isclose(
                raster_data, raster_nodata, atol=1e-3
            )
        else:
            nodata_pixels = np.zeros_like(raster_data, dtype=bool)

        raster_invalid = (
            nodata_pixels
            | np.isnan(raster_data)
            | np.isinf(raster_data)
        ).any(axis=0)

        # Preserve existing mask nodata values
        mask_invalid = mask_data == self.mask_nodata_value

        if mask_file_nodata is not None:
            if np.isnan(mask_file_nodata):
                mask_invalid |= np.isnan(mask_data)
            else:
                mask_invalid |= mask_data == mask_file_nodata

        invalid_pixels = raster_invalid | mask_invalid

        # Mark all invalid pixels with ignore label 255
        mask_data[invalid_pixels] = self.mask_nodata_value

        # 4. Normalize using training-only statistics
        if self.means is not None and self.stds is not None:
            normalized = (raster_data - self.means) / self.stds
        else:
            normalized = raster_data.copy()

        # Prevent invalid values from entering the network
        normalized[:, invalid_pixels] = 0.0
        normalized = np.nan_to_num(
            normalized,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        # 5. Spatial-only augmentations
        if self.augment:
            if random.random() > 0.5:
                normalized = np.flip(normalized, axis=2)
                mask_data = np.flip(mask_data, axis=1)

            if random.random() > 0.5:
                normalized = np.flip(normalized, axis=1)
                mask_data = np.flip(mask_data, axis=0)

            k = random.randint(0, 3)

            if k > 0:
                normalized = np.rot90(normalized, k=k, axes=(1, 2))
                mask_data = np.rot90(mask_data, k=k, axes=(0, 1))

        # 6. Convert to PyTorch tensors
        image_tensor = torch.from_numpy(
            np.ascontiguousarray(normalized)
        ).float()

        target_tensor = torch.from_numpy(
            np.ascontiguousarray(mask_data)
        ).long()

        return image_tensor, target_tensor, record.tile_id


def create_dataloaders_from_tiles(
    tiles: List[TileRecord],
    normalization_stats: Optional[Dict[str, Dict[str, float]]] = None,
    batch_size: int = 16,
    num_workers: int = 0,
    augment_train: bool = True,
) -> Tuple[
    DataLoader,
    Optional[DataLoader],
    Optional[DataLoader],
]:
    """Creates isolated DataLoaders for train, validation and test."""

    train_tiles = [
        t for t in tiles
        if t.split == "train" and t.is_usable
    ]

    val_tiles = [
        t for t in tiles
        if t.split == "val" and t.is_usable
    ]

    test_tiles = [
        t for t in tiles
        if t.split == "test" and t.is_usable
    ]

    if not train_tiles:
        raise ValueError(
            "No usable training tiles found for DataLoader."
        )

    train_ds = ForestTileDataset(
        train_tiles,
        normalization_stats=normalization_stats,
        augment=augment_train,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=(len(train_ds) > batch_size),
    )

    val_loader = None

    if val_tiles:
        val_ds = ForestTileDataset(
            val_tiles,
            normalization_stats=normalization_stats,
            augment=False,
        )

        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
        )

    test_loader = None

    if test_tiles:
        test_ds = ForestTileDataset(
            test_tiles,
            normalization_stats=normalization_stats,
            augment=False,
        )

        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
        )

    return train_loader, val_loader, test_loader

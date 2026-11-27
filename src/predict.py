
"""Overlapping tiled U-Net inference with probability blending."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import rasterio
import torch

from rasterio.windows import Window

from src.dataset import BAND_NAMES
from src.model import build_unet_resnet34


def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device | str = "cpu",
    in_channels: int = 6,
    num_classes: int = 1,
) -> torch.nn.Module:

    path = Path(checkpoint_path)

    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path.resolve()}")

    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )

    model = build_unet_resnet34(
        in_channels=in_channels,
        num_classes=num_classes,
        encoder_weights=None,
    )

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    return model


def get_tile_starts(length: int, tile_size: int, stride: int):
    """Generate overlapping tile positions covering the entire dimension."""

    if length <= tile_size:
        return [0]

    starts = list(range(0, length - tile_size + 1, stride))

    final_start = length - tile_size

    if starts[-1] != final_start:
        starts.append(final_start)

    return starts


def create_blend_weights(tile_size: int) -> np.ndarray:
    """Create smooth weights that reduce tile-edge influence."""

    hann = np.hanning(tile_size).astype(np.float32)

    weights = np.outer(hann, hann)

    # Prevent zero weights at outer edges.
    weights = np.maximum(weights, 0.05)

    return weights.astype(np.float32)


def predict_raster(
    model: torch.nn.Module,
    input_raster_path: str | Path,
    output_mask_path: str | Path,
    normalization_stats: Optional[Dict[str, Dict[str, float]]] = None,
    tile_size: int = 256,
    stride: int = 192,
    threshold: float = 0.50,
    device: torch.device | str = "cpu",
    nodata_value: float = -9999.0,
    mask_nodata_value: int = 255,
) -> Path:

    if stride <= 0 or stride > tile_size:
        raise ValueError("Stride must be between 1 and tile_size.")

    if normalization_stats is None:
        raise ValueError("Training normalization statistics are required.")

    in_path = Path(input_raster_path)
    out_path = Path(output_mask_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    dev = torch.device(device)

    model.to(dev)
    model.eval()

    means = np.array(
        [normalization_stats[b]["mean"] for b in BAND_NAMES],
        dtype=np.float32,
    ).reshape(6, 1, 1)

    stds = np.array(
        [normalization_stats[b]["std"] for b in BAND_NAMES],
        dtype=np.float32,
    ).reshape(6, 1, 1)

    stds = np.where(stds <= 0.0, 1.0, stds)

    blend_weights = create_blend_weights(tile_size)

    with rasterio.open(in_path) as src:

        if src.count != 6:
            raise ValueError(f"Expected 6 bands, found {src.count}")

        height, width = src.shape

        profile = src.profile.copy()

        profile.update(
            count=1,
            dtype="uint8",
            nodata=mask_nodata_value,
            compress="deflate",
            BIGTIFF="IF_SAFER",
        )

        row_starts = get_tile_starts(height, tile_size, stride)
        col_starts = get_tile_starts(width, tile_size, stride)

        total_tiles = len(row_starts) * len(col_starts)

        print(f"Raster dimensions: {width} x {height}")
        print(f"Tile size: {tile_size}")
        print(f"Stride: {stride}")
        print(f"Overlap: {tile_size - stride} pixels")
        print(f"Total inference tiles: {total_tiles}")

        # Disk-backed arrays avoid allocating large accumulators in RAM.
        with tempfile.TemporaryDirectory(
            prefix="forestwatch_blending_",
            dir=str(out_path.parent),
        ) as temp_dir:

            sum_path = Path(temp_dir) / "probability_sum.dat"
            weight_path = Path(temp_dir) / "weight_sum.dat"

            probability_sum = np.memmap(
                sum_path,
                dtype="float32",
                mode="w+",
                shape=(height, width),
            )

            weight_sum = np.memmap(
                weight_path,
                dtype="float32",
                mode="w+",
                shape=(height, width),
            )

            probability_sum[:] = 0
            weight_sum[:] = 0

            processed = 0

            with torch.no_grad():

                for row in row_starts:
                    for col in col_starts:

                        window = Window(
                            col_off=col,
                            row_off=row,
                            width=tile_size,
                            height=tile_size,
                        )

                        tile_data = src.read(
                            window=window,
                            boundless=True,
                            fill_value=nodata_value,
                        ).astype(np.float32)

                        valid = (
                            (tile_data[0] != nodata_value)
                            & np.isfinite(tile_data).all(axis=0)
                        )

                        normalized = (tile_data - means) / stds
                        normalized[:, ~valid] = 0.0

                        tensor = (
                            torch.from_numpy(normalized)
                            .unsqueeze(0)
                            .float()
                            .to(dev)
                        )

                        logits = model(tensor)

                        probabilities = (
                            torch.sigmoid(logits)
                            .squeeze()
                            .cpu()
                            .numpy()
                            .astype(np.float32)
                        )

                        tile_weights = blend_weights.copy()
                        tile_weights[~valid] = 0.0

                        row_end = min(row + tile_size, height)
                        col_end = min(col + tile_size, width)

                        actual_h = row_end - row
                        actual_w = col_end - col

                        probabilities = probabilities[:actual_h, :actual_w]
                        tile_weights = tile_weights[:actual_h, :actual_w]

                        probability_sum[
                            row:row_end,
                            col:col_end,
                        ] += probabilities * tile_weights

                        weight_sum[
                            row:row_end,
                            col:col_end,
                        ] += tile_weights

                        processed += 1

                        if processed % 100 == 0 or processed == total_tiles:
                            print(
                                f"Processed {processed}/{total_tiles} tiles"
                            )

            probability_sum.flush()
            weight_sum.flush()

            print("Blending overlapping predictions...")
            print("Writing final forest mask...")

            with rasterio.open(out_path, "w", **profile) as dst:

                for row in range(0, height, 512):

                    row_end = min(row + 512, height)

                    sums = np.asarray(
                        probability_sum[row:row_end, :]
                    )

                    weights = np.asarray(
                        weight_sum[row:row_end, :]
                    )

                    valid = weights > 0

                    averaged = np.zeros_like(sums)

                    np.divide(
                        sums,
                        weights,
                        out=averaged,
                        where=valid,
                    )

                    result = np.full(
                        averaged.shape,
                        mask_nodata_value,
                        dtype=np.uint8,
                    )

                    result[valid] = (
                        averaged[valid] >= threshold
                    ).astype(np.uint8)

                    dst.write(
                        result,
                        1,
                        window=Window(
                            0,
                            row,
                            width,
                            row_end - row,
                        ),
                    )

                dst.update_tags(
                    DESCRIPTION="U-Net forest mask with overlapping probability blending",
                    PROBABILITY_THRESHOLD=str(threshold),
                    TILE_SIZE=str(tile_size),
                    TILE_STRIDE=str(stride),
                    CLASSES="1=Forest,0=Non-forest,255=Nodata",
                )

                dst.set_band_description(
                    1,
                    "Blended U-Net Forest Mask",
                )

            del probability_sum
            del weight_sum

    return out_path


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="ForestWatch AI: Overlapping U-Net Inference"
    )

    parser.add_argument("--model-path", required=True)
    parser.add_argument("--input-raster", required=True)
    parser.add_argument("--output-mask", required=True)

    parser.add_argument("--threshold", type=float, default=0.50)

    parser.add_argument(
        "--normalization-stats",
        default="data/outputs/models/real_amazon/normalization_stats.json",
    )

    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=192)

    args = parser.parse_args()

    with open(args.normalization_stats, "r", encoding="utf-8") as f:
        stats = json.load(f)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    model = load_model_from_checkpoint(
        args.model_path,
        device=device,
    )

    predict_raster(
        model=model,
        input_raster_path=args.input_raster,
        output_mask_path=args.output_mask,
        normalization_stats=stats,
        tile_size=args.tile_size,
        stride=args.stride,
        threshold=args.threshold,
        device=device,
    )

    print(f"Prediction saved successfully: {args.output_mask}")

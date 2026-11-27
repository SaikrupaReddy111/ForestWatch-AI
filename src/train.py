
"""ForestWatch AI training pipeline: synthetic demo and real satellite training."""

from __future__ import annotations

import argparse
import json
import random
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.config import MasterConfig, load_config
from src.dataset import create_dataloaders_from_tiles
from src.evaluate import evaluate_model
from src.mock_data import generate_synthetic_dataset
from src.model import CombinedMaskedLoss, build_unet_resnet34
from src.tiling import (
    TileRecord,
    assign_spatial_block_splits,
    compute_training_normalization_stats,
    load_tiles_index,
    slice_raster_into_tiles,
)


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    num_batches = 0

    for images, targets, _ in dataloader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, targets)

        loss.backward()
        optimizer.step()

        running_loss += float(loss.item())
        num_batches += 1

    return running_loss / max(1, num_batches)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    config: MasterConfig,
    device: torch.device,
    output_dir: str | Path = "data/outputs/models",
    num_epochs: Optional[int] = None,
    learning_rate: Optional[float] = None,
    is_synthetic: bool = False,
    region_id: str = "region",
) -> Tuple[nn.Module, Dict[str, Any]]:

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    epochs = num_epochs or config.training.num_epochs
    lr = learning_rate or config.training.learning_rate

    criterion = CombinedMaskedLoss(
        bce_weight=config.training.loss_weights.bce,
        dice_weight=config.training.loss_weights.dice,
        ignore_index=config.data.mask_nodata_value,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=config.training.weight_decay,
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=config.training.scheduler.eta_min,
    )

    best_val_iou = -1.0
    checkpoint_path = out_dir / f"{region_id}_best_model.pth"
    history = []

    model.to(device)

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        scheduler.step()

        epoch_stats = {
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "learning_rate": float(
                optimizer.param_groups[0]["lr"]
            ),
        }

        print(
            f"Epoch {epoch}/{epochs} | "
            f"Training loss: {train_loss:.5f}"
        )

        if val_loader is not None:
            val_metrics = evaluate_model(
                model=model,
                dataloader=val_loader,
                device=device,
                ignore_index=config.data.mask_nodata_value,
                is_synthetic=is_synthetic,
                region_id=region_id,
            )

            val_iou = val_metrics["iou"]

            epoch_stats.update({
                "val_iou": val_iou,
                "val_dice": val_metrics["dice"],
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
            })

            print(f"Validation IoU: {val_iou:.4f}")

            if val_iou > best_val_iou:
                best_val_iou = val_iou

                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_iou": val_iou,
                        "val_metrics": val_metrics,
                        "config": getattr(
                            config, "model_dump", lambda: {}
                        )(),
                        "is_synthetic": is_synthetic,
                    },
                    checkpoint_path,
                )

                epoch_stats["checkpoint_saved"] = True
                print("Best model checkpoint saved.")

        history.append(epoch_stats)

    if val_loader is None:
        torch.save(
            {
                "epoch": epochs,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": getattr(
                    config, "model_dump", lambda: {}
                )(),
                "is_synthetic": is_synthetic,
            },
            checkpoint_path,
        )

    history_path = out_dir / f"{region_id}_training_history.json"

    with history_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "region_id": region_id,
                "is_synthetic": is_synthetic,
                "best_val_iou": best_val_iou,
                "history": history,
            },
            file,
            indent=2,
        )

    return model, {
        "best_val_iou": best_val_iou,
        "checkpoint_path": str(checkpoint_path),
        "history_path": str(history_path),
        "total_epochs": epochs,
    }


def run_demo(args, config, device):
    print("Running synthetic demonstration training...")

    demo_files = generate_synthetic_dataset(
        output_dir="data/synthetic",
        region_id="synthetic_demo_train",
        height=512,
        width=768,
        seed=42,
    )

    tiles = slice_raster_into_tiles(
        raster_path=demo_files["t1_raster"],
        mask_path=demo_files["t1_mask"],
        output_dir="data/tiles",
        region_id="synthetic_demo_train",
        tile_size=256,
    )

    tiles = assign_spatial_block_splits(
        tiles,
        buffer_tiles=1,
    )

    stats = compute_training_normalization_stats(tiles)

    train_loader, val_loader, _ = create_dataloaders_from_tiles(
        tiles=tiles,
        normalization_stats=stats,
        batch_size=args.batch_size,
    )

    model = build_unet_resnet34(
        in_channels=6,
        num_classes=1,
        encoder_weights=None,
    )

    model, summary = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader or train_loader,
        config=config,
        device=device,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        is_synthetic=True,
        region_id="synthetic_demo_train",
    )

    print("\nSynthetic demo completed.")
    print(json.dumps(summary, indent=2))


def run_real_training(args, config, device):
    index_path = Path(args.index)

    print(f"Loading real satellite tile index: {index_path}")

    tiles = load_tiles_index(index_path)

    usable = [
        tile for tile in tiles
        if tile.is_usable
    ]

    if not usable:
        raise ValueError("No usable real satellite tiles found.")

    if not any(t.split == "train" for t in usable):
        raise ValueError("No training tiles found.")

    print(f"Total tiles: {len(tiles)}")
    print(f"Usable tiles: {len(usable)}")

    for split in ("train", "val", "test", "buffer"):
        count = sum(
            t.is_usable and t.split == split
            for t in tiles
        )
        print(f"{split}: {count}")

    stats = compute_training_normalization_stats(tiles)

    output_dir = Path("data/outputs/models/real_amazon")
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "normalization_stats.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(stats, file, indent=2)

    train_loader, val_loader, test_loader = (
        create_dataloaders_from_tiles(
            tiles=tiles,
            normalization_stats=stats,
            batch_size=args.batch_size,
            num_workers=0,
            augment_train=True,
        )
    )

    print(f"Training batches: {len(train_loader)}")
    print(
        f"Validation batches: "
        f"{len(val_loader) if val_loader else 0}"
    )
    print(
        f"Test batches: "
        f"{len(test_loader) if test_loader else 0}"
    )

    model = build_unet_resnet34(
        in_channels=6,
        num_classes=1,
        encoder_weights=None,
    )

    model, summary = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        output_dir=output_dir,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        is_synthetic=False,
        region_id="amazon_brazil_real",
    )

    test_metrics = None

    if test_loader is not None:
        print("\nEvaluating on held-out test tiles...")

        test_metrics = evaluate_model(
            model=model,
            dataloader=test_loader,
            device=device,
            ignore_index=config.data.mask_nodata_value,
            is_synthetic=False,
            region_id="amazon_brazil_real_test",
        )

    result = {
        "training": summary,
        "test_metrics": test_metrics,
        "dataset": "Real Sentinel-2 and Dynamic World pseudo-labels",
    }

    with (output_dir / "real_training_summary.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(result, file, indent=2)

    print("\nREAL DATA TRAINING COMPLETE")
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(
        description="ForestWatch AI U-Net Training Pipeline"
    )

    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run synthetic demonstration training",
    )

    parser.add_argument(
        "--real",
        action="store_true",
        help="Train using real satellite tiles",
    )

    parser.add_argument(
        "--index",
        type=str,
        default="data/tiles/real_amazon/real_amazon_tiles.json",
        help="Real satellite tile index",
    )

    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
    )

    args = parser.parse_args()

    if args.demo == args.real:
        parser.error("Choose exactly one mode: --demo or --real.")

    config = load_config(args.config)
    seed_everything(config.seed)

    torch.set_num_threads(8)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=== ForestWatch AI Training ===")
    print(f"Device: {device}")

    if args.real:
        run_real_training(args, config, device)
    else:
        run_demo(args, config, device)


if __name__ == "__main__":
    main()

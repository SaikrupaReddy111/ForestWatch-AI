
from pathlib import Path
import json

from src.tiling import (
    slice_raster_into_tiles,
    assign_spatial_block_splits,
    save_tiles_index,
    verify_split_isolation,
)

RAW = Path("data/raw")
OUTPUT = Path("data/tiles/real_amazon")
TILE_SIZE = 256

YEARS = [2020, 2023]


def main():
    all_records = {}

    for year in YEARS:
        image_path = RAW / f"amazon_brazil_s2_{year}.tif"
        mask_path = RAW / f"amazon_brazil_dw_mask_{year}.tif"

        if not image_path.exists() or not mask_path.exists():
            raise FileNotFoundError(
                f"Missing image or mask for {year}. "
                "Wait for the Earth Engine exports."
            )

        print(f"\nPreparing {year}...")

        records = slice_raster_into_tiles(
            raster_path=image_path,
            mask_path=mask_path,
            output_dir=OUTPUT,
            region_id=f"amazon_brazil_{year}",
            tile_size=TILE_SIZE,
            nodata_value=-9999.0,
            mask_nodata_value=255,
            min_valid_fraction=0.05,
        )

        print(f"Tiles generated: {len(records)}")
        all_records[year] = records

    # Assign the same spatial split to matching geographic tiles.
    reference = all_records[2020]
    split_reference = assign_spatial_block_splits(
        reference,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        buffer_tiles=1,
        split_axis="col",
    )

    split_lookup = {
        (tile.row_idx, tile.col_idx): tile.split
        for tile in split_reference
    }

    combined = []

    for year in YEARS:
        for tile in all_records[year]:
            tile.split = split_lookup[
                (tile.row_idx, tile.col_idx)
            ]
            combined.append(tile)

    index_path = OUTPUT / "real_amazon_tiles.json"
    OUTPUT.mkdir(parents=True, exist_ok=True)

    save_tiles_index(combined, index_path)

    isolation, isolation_message = verify_split_isolation(
        combined,
        expected_buffer_distance_m=2560.0,
    )
    summary = {
        "years": YEARS,
        "total_tiles": len(combined),
        "usable_tiles": sum(t.is_usable for t in combined),
        "splits": {},
        "index": str(index_path),
        "split_isolation": isolation_message,
    }

    for split in ("train", "val", "test", "buffer"):
        summary["splits"][split] = sum(
            t.split == split for t in combined
        )

    print("\nREAL DATA PREPARATION SUMMARY")
    print(json.dumps(summary, indent=2))

    if isolation is False:
        raise RuntimeError("Spatial split isolation check failed.")


if __name__ == "__main__":
    main()

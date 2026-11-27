
from pathlib import Path

import numpy as np
import rasterio

PREDICTIONS = {
    "2020": Path("data/outputs/amazon_brazil_forest_2020.tif"),
    "2023": Path("data/outputs/amazon_brazil_forest_2023.tif"),
}

TILE_SIZE = 256
NODATA = 255


def inspect_prediction(year, path):
    print(f"\n{'=' * 55}")
    print(f"PREDICTION QUALITY CHECK: {year}")
    print("=" * 55)

    if not path.exists():
        print(f"ERROR: File not found: {path}")
        return

    with rasterio.open(path) as src:
        mask = src.read(1)
        print(f"Dimensions: {src.width} x {src.height}")
        print(f"CRS: {src.crs}")
        print(f"Transform: {src.transform}")

    valid = mask != NODATA
    total = int(valid.sum())

    if total == 0:
        print("ERROR: No valid pixels.")
        return

    forest = int(np.count_nonzero(mask[valid] == 1))
    non_forest = int(np.count_nonzero(mask[valid] == 0))
    invalid = int(mask.size - total)

    print("\nCLASS DISTRIBUTION")
    print(f"Forest: {forest:,} ({forest / total * 100:.2f}%)")
    print(f"Non-forest: {non_forest:,} ({non_forest / total * 100:.2f}%)")
    print(f"NoData: {invalid:,}")

    # Compare neighbouring pixels across tile boundaries
    # against neighbouring pixels inside tiles.

    horizontal_diff = mask[:, :-1] != mask[:, 1:]
    horizontal_valid = valid[:, :-1] & valid[:, 1:]

    vertical_diff = mask[:-1, :] != mask[1:, :]
    vertical_valid = valid[:-1, :] & valid[1:, :]

    cols = np.arange(1, mask.shape[1])
    rows = np.arange(1, mask.shape[0])

    horizontal_boundary = cols % TILE_SIZE == 0
    vertical_boundary = rows % TILE_SIZE == 0

    def transition_rate(diff, valid_pairs):
        count = int(valid_pairs.sum())
        changes = int((diff & valid_pairs).sum())

        if count == 0:
            return 0.0, 0

        return changes / count * 100, count

    h_boundary = np.broadcast_to(
        horizontal_boundary[None, :],
        horizontal_diff.shape,
    )
    v_boundary = np.broadcast_to(
        vertical_boundary[:, None],
        vertical_diff.shape,
    )

    h_boundary_rate, h_boundary_count = transition_rate(
        horizontal_diff[:, h_boundary[0]],
        horizontal_valid[:, h_boundary[0]],
    )

    h_interior_rate, h_interior_count = transition_rate(
        horizontal_diff[:, ~h_boundary[0]],
        horizontal_valid[:, ~h_boundary[0]],
    )

    v_boundary_rate, v_boundary_count = transition_rate(
        vertical_diff[v_boundary[:, 0], :],
        vertical_valid[v_boundary[:, 0], :],
    )

    v_interior_rate, v_interior_count = transition_rate(
        vertical_diff[~v_boundary[:, 0], :],
        vertical_valid[~v_boundary[:, 0], :],
    )

    print("\nTILE-BOUNDARY ANALYSIS")
    print("Horizontal boundaries:")
    print(f"  Boundary transitions: {h_boundary_rate:.3f}%")
    print(f"  Interior transitions: {h_interior_rate:.3f}%")

    print("Vertical boundaries:")
    print(f"  Boundary transitions: {v_boundary_rate:.3f}%")
    print(f"  Interior transitions: {v_interior_rate:.3f}%")

    print("\nINTERPRETATION")

    ratios = []

    if h_interior_rate > 0:
        ratios.append(h_boundary_rate / h_interior_rate)

    if v_interior_rate > 0:
        ratios.append(v_boundary_rate / v_interior_rate)

    if ratios:
        max_ratio = max(ratios)
        print(f"Maximum boundary/interior transition ratio: {max_ratio:.2f}")

        if max_ratio > 1.5:
            print("NOTICE: Possible tile-boundary artefacts. Visual inspection recommended.")
        else:
            print("No strong tile-boundary signal detected by this test.")
    else:
        print("Insufficient transitions for a reliable comparison.")

    print("\nCheck complete.")


def main():
    print("FORESTWATCH AI - PREDICTION QUALITY DIAGNOSTICS")
    print("Read-only analysis. Original files will not be changed.")

    for year, path in PREDICTIONS.items():
        inspect_prediction(year, path)

    print("\nALL DIAGNOSTICS COMPLETED.")


if __name__ == "__main__":
    main()

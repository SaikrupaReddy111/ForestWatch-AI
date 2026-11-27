
import rasterio
import numpy as np
from pathlib import Path

folder = Path("data/raw")

files = [
    "amazon_brazil_s2_2020.tif",
    "amazon_brazil_s2_2023.tif",
    "amazon_brazil_dw_mask_2020.tif",
    "amazon_brazil_dw_mask_2023.tif",
]

for filename in files:
    path = folder / filename

    with rasterio.open(path) as src:
        print(f"\nFile: {filename}")
        print(f"Dimensions: {src.width} x {src.height}")
        print(f"Bands: {src.count}")
        print(f"CRS: {src.crs}")
        print(f"NoData: {src.nodata}")

        if src.count == 1:
            values, counts = np.unique(
                src.read(1),
                return_counts=True
            )
            print("Classes:", dict(zip(values.tolist(), counts.tolist())))

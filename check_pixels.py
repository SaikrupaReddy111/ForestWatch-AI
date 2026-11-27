
import rasterio
import numpy as np

path = "data/raw/amazon_brazil_s2_2020.tif"

with rasterio.open(path) as src:
    data = src.read()

    print("Sentinel-2 2020 pixel inspection")
    for i in range(data.shape[0]):
        band = data[i]
        print(
            f"Band {i + 1}: "
            f"zeros={(band == 0).sum():,}, "
            f"min={band.min()}, "
            f"max={band.max()}, "
            f"mean={band.mean():.2f}"
        )

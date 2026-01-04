from typing import Iterator
from pathlib import Path

import torch
import random

import rasterio
from rasterio.merge import merge

class DataWithLabels:
    data: torch.Tensor
    labels: torch.Tensor

    def __init__(self, data: torch.Tensor, labels: torch.Tensor):
        assert data.shape == labels.shape, "labels' shape do not match the given dataset's shape"
        self.data: torch.Tensor = data
        self.labels: torch.Tensor = labels

    def get_iterable(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        return zip(self.data, self.labels)
    
    def get_hacky_fold_iterable(self, fold_size = 20):
        window_start = random.randint(0, len(self.data))
        window_end = window_start + fold_size
        return zip(self.data[window_start:window_end], self.labels[window_start:window_end])


from pathlib import Path
from osgeo import gdal

def merge_tiffs(
    target_dir,
    save_dir,
    out_name="merged.tif",
    glob="*.tif",
    dst_srs="EPSG:7856",
    resample_alg="bilinear",
    nodata=-9999,
):
    target_dir = Path(target_dir)
    files = sorted(target_dir.glob(glob))
    if not files:
        raise FileNotFoundError(f"No GeoTIFFs found in {target_dir}")

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    out_path = save_dir / out_name
    if out_path.suffix.lower() not in {".tif", ".tiff"}:
        out_path = out_path.with_suffix(".tif")

    gdal.Warp(
        destNameOrDestDS=str(out_path),
        srcDSOrSrcDSTab=[str(f) for f in files],
        dstSRS=dst_srs,
        resampleAlg=resample_alg,
        dstNodata=nodata,
        creationOptions=[
            "TILED=YES",
            "COMPRESS=DEFLATE",
            "BIGTIFF=IF_SAFER",
        ],
        multithread=True,
    )

    return out_path

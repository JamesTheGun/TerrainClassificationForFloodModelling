from typing import Iterator
from pathlib import Path

import torch
import random

from osgeo import gdal
from common.safety_first.dwl_checks import (
    check_data_dwl,
    check_labels_dwl,
    check_params_dwl,
)
from common.safety_first.sdwl_checks import (
    check_data_sdwl,
    check_labels_sdwl,
    check_params_sdwl,
)

# note to future self: We dont even need to track the offsets in the segmented data because it is only used for training the model...
# However, we should track the offsets and EPSG of the un-segmented datasets with labels because we will generate these labels and want to be able to project them onto the globe going forwards, because we probably want to use
# basically this entire area needs a rethink, probs revert to earler version


class SegmentedDataWithLabels:
    data_and_labels: torch.Tensor

    def __init__(self, data: torch.Tensor, labels: torch.Tensor):
        check_params_sdwl(data, labels)
        # data: (N, C, H, W), labels: (N, C, H, W)
        # stack so dim 1 selects data (0) vs labels (1)
        self.data_and_labels = torch.stack([data, labels], dim=1)

    def get_iterable(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        return zip(self.data, self.labels)

    def get_hacky_fold_iterable(
        self, fold_size=20
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        window_start = random.randint(0, len(self.data) - fold_size)
        window_end = window_start + fold_size
        return zip(
            self.data[window_start:window_end], self.labels[window_start:window_end]
        )

    @property
    def data(self) -> torch.Tensor:
        return self.data_and_labels[:, 0, :, :, :]

    @property
    def labels(self) -> torch.Tensor:
        return self.data_and_labels[:, 1, :, :, :]

    @data.setter
    def data(self, data: torch.Tensor):
        check_data_sdwl(data)
        self.data_and_labels[:, 0, :, :, :] = data

    @labels.setter
    def labels(self, labels: torch.Tensor):
        check_labels_sdwl(labels)
        self.data_and_labels[:, 1, :, :, :] = labels


class DataWithLabels:
    data_and_labels: torch.Tensor

    def __init__(
        self,
        data: torch.Tensor,
        labels: torch.Tensor,
    ):
        check_params_dwl(data, labels)
        # stack so dim 0 selects data (0) vs labels (1)
        self.data_and_labels = torch.stack([data, labels], dim=0)

    @property
    def data(self) -> torch.Tensor:
        return self.data_and_labels[0]

    @property
    def labels(self) -> torch.Tensor:
        return self.data_and_labels[1]

    @labels.setter
    def labels(self, labels: torch.Tensor):
        check_labels_dwl(labels)
        self.data_and_labels[1] = labels

    @data.setter
    def data(self, data: torch.Tensor):
        check_data_dwl(data)
        self.data_and_labels[0] = data


class DataWithLabelsGeoTethered(DataWithLabels):
    """dwl but tethered to geology..."""

    data_and_labels: torch.Tensor
    epsg: int
    offset: float
    res: float

    def __init__(
        self,
        data: torch.Tensor,
        labels: torch.Tensor,
        epsg: int = None,
        offset: float = None,
        res: float = None,
    ):
        super().__init__(data, labels)
        self.epsg = epsg
        self.offset = offset
        self.res = res


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

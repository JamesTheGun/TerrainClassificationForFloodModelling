
import torch
from typing import List, Tuple
from structured_data_utils.config.constants import ESPSG, GEOTIFF_LOCATIONS_TO_CORRESPONDING_STANDARDISED_LOCATION, RES, EMPTY_VAL
from structured_data_utils.structuring import get_negative_geotiff_tensor_and_offset, get_positive_geotiff_tensor_and_offset, get_combined_geotiff_tensor_and_offset
from common.data_managment import DataWithLabels
import subprocess
import torch.nn.functional as F
from typing import TYPE_CHECKING

def standardise_geotiff(target_dir: str, write_dir: str):
    subprocess.run([
        "gdalwarp",
        "-t_srs", ESPSG,
        "-tr", RES, RES,
        "-r", "bilinear",
        "-overwrite",
        target_dir,
        write_dir,
    ], check=False,
    capture_output=True,
    text=True)

def standardise_core_geotiffs():
    for target, write_dir in GEOTIFF_LOCATIONS_TO_CORRESPONDING_STANDARDISED_LOCATION.items():
        standardise_geotiff(target, write_dir)

def offset_meters_to_offset_pixels(offset):
    print(offset[0]/float(RES))
    offset_x_corrected = int(offset[0]/float(RES))
    offset_y_corrected = int(offset[1]/float(RES))
    return (offset_x_corrected, offset_y_corrected)

def pad_pos_mask_to_match(pos_tensor: torch.Tensor, other_tensor: torch.Tensor, offset: Tuple[int, int]):
    pad_x = other_tensor.shape[-1] - pos_tensor.shape[-1]
    pad_y = other_tensor.shape[-2] - pos_tensor.shape[-2]

    padded = F.pad(pos_tensor, (0, pad_x, 0, pad_y), mode="constant", value=EMPTY_VAL)
    pixel_offset = offset_meters_to_offset_pixels(offset)
    #padded = torch.roll(padded, shifts = pixel_offset, dims=(-2,-1))
    padded = torch.roll(padded, shifts = pixel_offset[0], dims=1)
    padded = torch.roll(padded, shifts = pixel_offset[1], dims=0)
    return padded

def put_nans_in_neggative_positions(data: torch.Tensor):
    outlier_mask = data < 0

    data[outlier_mask] = torch.nan

    print(outlier_mask)

    return data

def load_combined_pos_neg_df_structured() -> DataWithLabels:
    positive, offset_positive = get_positive_geotiff_tensor_and_offset()
    combined, offset_combined = get_combined_geotiff_tensor_and_offset()

    offset = (
        offset_positive[0] - offset_combined[0],
        offset_combined[1] - offset_positive[1],
    )

    positive = pad_pos_mask_to_match(positive, combined, offset)
    positive = positive.unsqueeze(0)
    combined = combined.unsqueeze(0)

    labels = torch.zeros_like(combined[:1])

    pos_mask = positive[0] != EMPTY_VAL

    labels[0, pos_mask] = 1

    data = combined.clone()
    data[0, pos_mask] = positive[0, pos_mask]

    return DataWithLabels(data, labels)

def get_segments_with_sliding_window(data_with_labels: DataWithLabels, window_size=400, stride=200) -> DataWithLabels:
    patches = (
        data_with_labels.data.unfold(1, window_size, stride)
         .unfold(2, window_size, stride)
    )
    patch_labels = (
        data_with_labels.labels.unfold(1, window_size, stride)
         .unfold(2, window_size, stride)
    )
    patches = patches.contiguous().view(-1, window_size, window_size)
    patch_labels = patch_labels.contiguous().view(-1, window_size, window_size)
    data_with_labels_out = DataWithLabels(patches, patch_labels)
    return data_with_labels_out

def remove_empty_segments(data_with_labels: DataWithLabels) -> DataWithLabels:
    not_empty = data_with_labels.data != EMPTY_VAL
    mean_occupied = not_empty.float().mean(dim=(1,2))
    mask = mean_occupied > 0.99
    data_with_labels = DataWithLabels(data_with_labels.data[mask], data_with_labels.labels[mask])
    return data_with_labels

def generate_train_test_sets(labeled_tensor: torch.Tensor):
    pass

def generate_folds(labeled_tensor: torch.Tensor) -> List[torch.Tensor]:
    pass
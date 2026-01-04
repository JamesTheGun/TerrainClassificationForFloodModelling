
import torch
from typing import List, Tuple
from structured_data_utils.config.constants import ESPSG, GEOTIFF_LOCATIONS_TO_CORRESPONDING_STANDARDISED_LOCATION, RES, EMPTY_VAL
from structured_data_utils.structuring import get_positive_geotiff_tensor_and_offset, get_combined_geotiff_tensor_and_offset
from common.data_managment import DataWithLabels
from structured_data_utils.data import ModelData
import subprocess
import torch.nn.functional as F
from typing import TYPE_CHECKING

import random

def standardise_geotiff_with_res_noise(target_dir: str, write_dir: str, noise_frac: float):
    base = float(RES)
    jitter = random.uniform(-noise_frac, noise_frac)
    res = base * (1.0 + jitter)

    p = subprocess.run(
        [
            "gdalwarp",
            "-t_srs", ESPSG,
            "-tr", f"{res}", f"{res}",   # noise here
            "-tap",                      # optional; aligns to this file’s res grid
            "-r", "bilinear",
            "-of", "GTiff",
            "-overwrite",
            target_dir,
            write_dir,
        ],
        capture_output=True,
        text=True,
    )
    if p.returncode != 0:
        raise RuntimeError(p.stderr)

def standardise_geotiff(target_dir: str, write_dir: str):
    p = subprocess.run(
        [
            "gdalwarp",
            "-t_srs", ESPSG,        # e.g. "EPSG:7856"
            "-tr", RES, RES,        # e.g. "1", "1"
            "-tap",                # CRITICAL: align pixel grid
            "-r", "bilinear",      # use "near" for masks/classes
            "-of", "GTiff",
            "-overwrite",
            target_dir,
            write_dir,
        ],
        capture_output=True,
        text=True,
    )

    if p.returncode != 0:
        raise RuntimeError(p.stderr)

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

def load_data_with_labeles(folder_name: str, test = False) -> DataWithLabels:
    positive, offset_positive = get_positive_geotiff_tensor_and_offset(test, folder_name)
    combined, offset_combined = get_combined_geotiff_tensor_and_offset(test, folder_name)

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

def get_segments_with_sliding_window(data_with_labels: DataWithLabels, window_size=300, stride=300) -> DataWithLabels:
    print("starting")
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
    print("ending")
    return data_with_labels_out

def remove_empty_segments(data_with_labels: DataWithLabels) -> DataWithLabels:
    print(data_with_labels.data.shape)
    not_empty = ~torch.isnan(data_with_labels.data)
    mean_occupied = not_empty.float().mean(dim=(1,2))
    print(mean_occupied)
    mask = mean_occupied > 0.8
    data_with_labels = DataWithLabels(data_with_labels.data[mask], data_with_labels.labels[mask])
    print(data_with_labels.data.shape)
    return data_with_labels

def remove_segments_missing_positive(data_with_labels: DataWithLabels) -> DataWithLabels:
    y = data_with_labels.labels

    # Accept (N,H,W) or (N,1,H,W) or (N,C,H,W)
    # Define "has any positive" per segment:
    has_pos = (y > 0)
    has_pos = has_pos.flatten(start_dim=1).any(dim=1)  # (N,)

    return DataWithLabels(data_with_labels.data[has_pos], y[has_pos])

def infer_nans(dwl: DataWithLabels, max_iters: int = 50) -> DataWithLabels:
    """
    Fill NaNs by iteratively averaging valid 4-neighbors (N/S/E/W).
    Uses conv2d + a validity mask for speed (GPU-friendly).
    Handles contiguous NaN regions by iterating until they get a boundary.
    """
    x0 = dwl.data
    if not x0.is_floating_point():
        x0 = x0.float()
    x = x0.clone()

    # Support (H,W), (C,H,W), (B,C,H,W)
    if x.ndim == 2:
        x4 = x[None, None]          # (1,1,H,W)
        squeeze_mode = "HW"
    elif x.ndim == 3:
        x4 = x[None]                 # (1,C,H,W)
        squeeze_mode = "CHW"
    elif x.ndim == 4:
        x4 = x                       # (B,C,H,W)
        squeeze_mode = None
    else:
        raise ValueError(f"Expected 2D/3D/4D tensor, got {tuple(x.shape)}")

    device, dtype = x4.device, x4.dtype
    C = x4.shape[1]

    # 4-neighbor kernel (no wrap-around, unlike torch.roll)
    k = torch.tensor([[0, 1, 0],
                      [1, 0, 1],
                      [0, 1, 0]], device=device, dtype=dtype)[None, None]
    k = k.repeat(C, 1, 1, 1)  # depthwise: (C,1,3,3)

    for _ in range(max_iters):
        nan_mask = torch.isnan(x4)
        if not nan_mask.any():
            break

        valid = (~nan_mask).to(dtype)
        x_filled = torch.nan_to_num(x4, nan=0.0)

        # Sum/count of valid neighbors
        summed = F.conv2d(x_filled, k, padding=1, groups=C)
        count  = F.conv2d(valid,    k, padding=1, groups=C)

        can_fill = nan_mask & (count > 0)
        if not can_fill.any():
            break

        fill = summed / count.clamp_min(1.0)
        x4 = torch.where(can_fill, fill, x4)

    # Restore original shape
    if squeeze_mode == "HW":
        out = x4[0, 0]
    elif squeeze_mode == "CHW":
        out = x4[0]
    else:
        out = x4

    return DataWithLabels(out, dwl.labels)

def splice_tensors(tensors: List[torch.Tensor]):
    torch.stack(tensors, dim=1).reshape(-1, *tensors[0].shape[1:])

def splice_datas(datas: List[ModelData]):

    data_with_labels: DataWithLabels = ModelData.data_with_labels

    data = data_with_labels.data

    labels = data_with_labels.labels

    out_data_segmented = splice_tensors()
    out_labels_segmented = splice_tensors()

    out_data = splice_tensors()
    out_labels = splice_tensors()

    out_data_with_labels_segmented = DataWithLabels(out_data_segmented, out_labels_segmented)

    out_model_data = ModelData()

    return out


def generate_train_test_sets(labeled_tensor: torch.Tensor):
    pass

def generate_folds(labeled_tensor: torch.Tensor) -> List[torch.Tensor]:
    pass
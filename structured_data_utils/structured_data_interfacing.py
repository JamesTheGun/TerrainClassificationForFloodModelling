import os
import torch
from typing import List, Tuple
from common.data_managment import (
    DataWithLabelsGeoTethered,
    SegmentedDataWithLabels,
    DataWithLabels,
)
import subprocess
from torchvision.transforms import v2 as trch_trns
from torchvision import tv_tensors
import torch.nn.functional as F
from typing import TYPE_CHECKING
import rasterio
from affine import Affine
import cv2

import random
import numpy as np
import json

VERBOSE = False
NEGATIVE_LABEL = 0
POSITIVE_LABEL = 1


def load_config() -> dict:
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    with open(config_path, "r") as f:
        return json.load(f)


CONFIG = load_config()


def standardise_geotiff(target_path: str, write_path: str, force: bool = False):
    if os.path.exists(write_path) and not force:
        if VERBOSE:
            print(
                f"Standardised file already exists at {write_path}. Skipping. Use force=True to override."
            )
        return

    p = subprocess.run(
        [
            "gdalwarp",
            "-t_srs",
            CONFIG["ESPSG"],  # e.g. "EPSG:7856"
            "-tr",
            CONFIG["RES"],
            CONFIG["RES"],  # e.g. "1", "1"
            "-tap",  # CRITICAL: align pixel grid
            "-r",
            "bilinear",  # use "near" for masks/classes
            "-of",
            "GTiff",
            "-overwrite",
            target_path,
            write_path,
        ],
        capture_output=True,
        text=True,
    )

    if p.returncode != 0:
        raise RuntimeError(p.stderr)


def standardise_dataset(
    dataset_name: str, force: bool = False, target_files: list = None
):
    standardise_folder(
        os.path.join("data", dataset_name), force=force, target_files=target_files
    )


def standardise_folder(dir: str, force: bool = False, target_files: list = None):
    files_to_process = (
        target_files
        if target_files is not None
        else CONFIG["STANDARDISATION_TARGET_TIFFS"]
    )

    for tiff in files_to_process:
        tiff_path = os.path.join(dir, tiff)
        write_path = tiff_path.replace(".tif", "_STANDARDISED.tif")
        standardise_geotiff(tiff_path, write_path, force=force)


def offset_meters_to_offset_pixels(offset):
    offset_x_corrected = int(offset[0] / float(CONFIG["RES"]))
    offset_y_corrected = int(offset[1] / float(CONFIG["RES"]))
    return (offset_x_corrected, offset_y_corrected)


def pad_pos_mask_to_match(
    pos_tensor: torch.Tensor, other_tensor: torch.Tensor, offset: Tuple[int, int]
):
    pad_x = other_tensor.shape[-1] - pos_tensor.shape[-1]
    pad_y = other_tensor.shape[-2] - pos_tensor.shape[-2]

    padded = F.pad(
        pos_tensor,
        (0, pad_x, 0, pad_y),
        mode="constant",
        value=CONFIG["EMPTY_VAL_GEOTIFF"],
    )
    pixel_offset = offset_meters_to_offset_pixels(offset)
    # padded = torch.roll(padded, shifts = pixel_offset, dims=(-2,-1))
    assert (
        padded.shape == other_tensor.shape
    ), f"padded shape {padded.shape} does not match other tensor shape {other_tensor.shape}"
    padded = torch.roll(padded, shifts=pixel_offset[0], dims=1)
    padded = torch.roll(padded, shifts=pixel_offset[1], dims=0)
    return padded


def put_nans_in_neggative_positions(data: torch.Tensor) -> torch.Tensor:
    outlier_mask = data < 0

    data[outlier_mask] = torch.nan

    return data


def load_data_with_labels(folder_name: str) -> DataWithLabelsGeoTethered:
    positive, offset_positive = get_positive_geotiff_tensor_and_offset(folder_name)
    combined, offset_combined = get_combined_geotiff_tensor_and_offset(folder_name)

    offset = (
        offset_positive[0] - offset_combined[0],
        offset_combined[1] - offset_positive[1],
    )

    positive = pad_pos_mask_to_match(positive, combined, offset)
    positive = positive.unsqueeze(0)
    combined = combined.unsqueeze(0)

    labels = torch.zeros_like(combined[:1])

    pos_mask = positive[0] != CONFIG["EMPTY_VAL_GEOTIFF"]

    labels[0, pos_mask] = POSITIVE_LABEL
    labels[0, ~pos_mask] = NEGATIVE_LABEL

    epsg = retrieve_dataset_EPSG(folder_name)

    return DataWithLabelsGeoTethered(
        combined, labels, epsg, offset_combined, CONFIG["RES"]
    )


ROTATION_PER_PATCH = CONFIG["ROTATION_PER_PATCH"]


def _apply_rotation_to_patch(
    dwl_segment: DataWithLabels,
    angle: float,
    image_fill: float = None,
    mask_fill: int = None,
) -> DataWithLabels:

    if not image_fill:
        image_fill = torch.mean(dwl_segment.data).item()
    if not mask_fill:
        mask_fill = 0

    data = tv_tensors.Image(dwl_segment.data)
    labels = tv_tensors.Mask(dwl_segment.labels)

    data_rotated = trch_trns.functional.rotate(
        data,
        angle=angle,
        interpolation=trch_trns.InterpolationMode.NEAREST,
        expand=False,
        fill=image_fill,
    )

    labels_rotated = trch_trns.functional.rotate(
        labels,
        angle=angle,
        interpolation=trch_trns.InterpolationMode.NEAREST,
        expand=False,
        fill=mask_fill,
    )

    return DataWithLabels(torch.Tensor(data_rotated), torch.Tensor(labels_rotated))


def _apply_rotation_to_sdwl(
    sdwl: SegmentedDataWithLabels,
) -> SegmentedDataWithLabels:

    rotated_data_segments = []
    rotated_label_segments = []

    for _ in range(ROTATION_PER_PATCH):
        angles = torch.empty(sdwl.data.shape[0]).uniform_(0, 360)
        for idx in range(sdwl.data.shape[0]):
            patch = DataWithLabels(sdwl.data[idx], sdwl.labels[idx])
            rotat_dwl = _apply_rotation_to_patch(
                patch,
                angle=angles[idx].item(),
            )
            rotated_data_segments.append(rotat_dwl.data)
            rotated_label_segments.append(rotat_dwl.labels)

    stacked_data = torch.stack(rotated_data_segments, dim=0)
    stacked_labels = torch.stack(rotated_label_segments, dim=0)

    result_sdwl = SegmentedDataWithLabels(
        stacked_data,
        stacked_labels,
    )

    return result_sdwl


def _check_dwl_size(
    sdwl: SegmentedDataWithLabels,
    critical_threshold_gb: float = CONFIG["CRITICAL_THRESHOLD_GB"],
) -> float:
    sdwl_data_size_bytes = sdwl.data.element_size() * sdwl.data.nelement()
    sdwl_labels_size_bytes = sdwl.labels.element_size() * sdwl.labels.nelement()
    sdwl_data_size_gb = (sdwl_data_size_bytes) / (1024**3)
    sdwl_labels_size_gb = (sdwl_labels_size_bytes) / (1024**3)
    sdwl_total_size_gb = sdwl_data_size_gb + sdwl_labels_size_gb
    if VERBOSE:
        print(
            f"SegmentedDataWithLabels size: data={sdwl_data_size_gb:.2f} GB, labels={sdwl_labels_size_gb:.2f} GB, total={sdwl_total_size_gb:.2f} GB"
        )
    if sdwl_total_size_gb > critical_threshold_gb and VERBOSE:
        response = input(
            f"WARNING: SegmentedDataWithLabels is very large ({sdwl_total_size_gb:.2f} GB). Do you want to continue? (y/n): "
        )
        if response.lower() != "y":
            raise MemoryError(
                f"Aborting due to large SegmentedDataWithLabels size ({sdwl_total_size_gb:.2f} GB). Consider reducing window sizes or number of scales."
            )
    return sdwl_total_size_gb


def _check_gross_size(gross_size_gb):
    if VERBOSE:
        print(
            f"Gross size of all SegmentedDataWithLabels so far: {gross_size_gb:.2f} GB"
        )
    if gross_size_gb > CONFIG["MAX_ALLOWED_GB"]:
        raise MemoryError(
            f"Aborting due to large SegmentedDataWithLabels size ({gross_size_gb:.2f} GB). Consider reducing window sizes or number of scales."
        )


def _get_patch_tensors_at_target_size(
    window_size: int, data: torch.Tensor, stride: int
):
    assert (
        data.dim() == 3
    ), f"data must be a 3D tensor of shape (C,H,W), but got {data.shape}"
    patched_data = data.unfold(1, window_size, stride).unfold(
        2, window_size, stride
    )  # (C, nH, nW, window, window)
    # Rearrange to (nH, nW, C, window, window) so we can flatten nH and nW together
    patched_data = patched_data.permute(1, 2, 0, 3, 4).contiguous()
    # Flatten patch dimensions: (nH*nW, C, window, window)
    patched_data_collapsed_windows = patched_data.flatten(start_dim=0, end_dim=1)
    return patched_data_collapsed_windows


def _get_patch_tensors_dwls_at_target_sizes(
    window_sizes: List[int],
    dwl: DataWithLabelsGeoTethered,
    stride: int,
) -> List[SegmentedDataWithLabels]:
    gross_size_gb = 0
    sdwls_at_sizes = []
    for window_size in window_sizes:
        data_patches = _get_patch_tensors_at_target_size(window_size, dwl.data, stride)
        labels_patches = _get_patch_tensors_at_target_size(
            window_size, dwl.labels, stride
        )
        sdwl = SegmentedDataWithLabels(data_patches, labels_patches)
        accepting_indicies = _get_accepting_segment_indecies(
            sdwl, CONFIG["PERCENTAGE_EMPTY_TARGET"]
        )
        sdwl = _get_sdwl_patch_indicies(sdwl, accepting_indicies)
        sdwls_at_sizes.append(sdwl)
        _check_accepting_indecies(
            accepting_indicies, sdwl, CONFIG["TARGET_NUM_TO_TAKE"]
        )
        gross_size_gb += _check_dwl_size(sdwl)
        _check_gross_size(gross_size_gb)
    return sdwls_at_sizes


def _scale_multiplier_from_gammavariate(
    alpha: float = CONFIG["SIZE_CHANGE_ALPHA"], beta: float = CONFIG["SIZE_CHANGE_BETA"]
) -> float:
    scale_multiplier = random.gammavariate(alpha, beta)
    if scale_multiplier < 0.1 or scale_multiplier > 2.0:
        return _scale_multiplier_from_gammavariate(alpha, beta)
    return scale_multiplier


def _get_scale_change_multipliers(
    number_of_scales: int,
) -> List[float]:
    scale_change_multipliers = [
        _scale_multiplier_from_gammavariate() for _ in range(number_of_scales)
    ]
    print(f"Scale change multipliers: {scale_change_multipliers}")
    return scale_change_multipliers


def _get_window_sizes_from_scale_multipliers(
    base_window_size: int, scale_factors: List[float]
) -> List[int]:
    modified_window_sizes = [
        int(base_window_size * scale_factor) for scale_factor in scale_factors
    ]
    return modified_window_sizes


def _make_sdwls_at_varied_scales(
    dwl: DataWithLabelsGeoTethered,
    base_window_size: int,
    stride: int,
    number_of_scales: int = CONFIG["DEFAULT_NUMBER_OF_SCALES"],
) -> List[SegmentedDataWithLabels]:
    if VERBOSE:
        print(f"dwl shapes -- data: {dwl.data.shape}, labels: {dwl.labels.shape}")
    assert (
        dwl.data.dim() == 3
    ), f"Expected dwl.data to be a 3D tensor of shape (C,H,W), but got {dwl.data.shape}"
    scale_multipliers = _get_scale_change_multipliers(number_of_scales)
    modified_window_sizes = _get_window_sizes_from_scale_multipliers(
        base_window_size, scale_multipliers
    )
    dwls_at_targeted_sizes = _get_patch_tensors_dwls_at_target_sizes(
        modified_window_sizes, dwl, stride
    )
    return dwls_at_targeted_sizes


def _make_image_resizer(target_size: int) -> trch_trns.Resize:
    resizer = trch_trns.Resize(
        (target_size, target_size),
        interpolation=trch_trns.InterpolationMode.BILINEAR,
        antialias=True,
    )
    return resizer


def _resize_sdwl_patches(
    sdwl: SegmentedDataWithLabels, new_size: int
) -> SegmentedDataWithLabels:
    patch = sdwl.data
    label = sdwl.labels
    assert (
        patch.dim() == 4 and label.dim() == 4
    ), f"patch and label must be 4D tensors, but got {patch.shape} and {label.shape}"
    resizer = _make_image_resizer(new_size)
    patch_resized: Tuple = resizer(patch)
    label_resized: Tuple = resizer(label)
    assert (
        patch_resized.shape
        == label_resized.shape
        == (patch.shape[0], patch.shape[1], new_size, new_size)
    ), f"Resized patch and label must have shape ({patch.shape[0]}, {patch.shape[1]}, {new_size}, {new_size}), but got {patch_resized.shape} and {label_resized.shape}"
    return SegmentedDataWithLabels(patch_resized, label_resized)


def _get_sdwl_patch_indicies(
    sdwl: SegmentedDataWithLabels, indicies: torch.Tensor
) -> SegmentedDataWithLabels:
    sda_taken = SegmentedDataWithLabels(
        sdwl.data.index_select(0, indicies),
        sdwl.labels.index_select(0, indicies),
    )
    return sda_taken


def _get_percentage_of_un_reduced_tensor(
    num_positive: int, total: int, percentage_empty_target: float
) -> float:
    if num_positive == 0:
        return 0.0
    percentage_to_keep_of_total = percentage_empty_target / (total - num_positive)
    return percentage_to_keep_of_total


def _get_accepting_segment_indecies(
    sdwl: SegmentedDataWithLabels,
    percentage_empty_target: float,
) -> torch.Tensor:

    segmented_labels = sdwl.labels

    empty_mask = (segmented_labels == NEGATIVE_LABEL).all(dim=(1, 2, 3))

    total = segmented_labels.shape[0]
    num_empty = empty_mask.sum().item()
    num_positive = (~empty_mask).sum().item()

    empty_indices = torch.where(empty_mask)[0]
    non_empty_indices = torch.where(~empty_mask)[0]

    percentage_to_keep_of_total = _get_percentage_of_un_reduced_tensor(
        num_positive, total, percentage_empty_target
    )

    num_to_keep = int(percentage_to_keep_of_total * total)

    num_to_keep = min(num_to_keep, num_empty)
    perm = torch.randperm(num_empty, device=empty_indices.device)
    kept_empty_indices = empty_indices[perm[:num_to_keep]]

    accepting_indices = torch.cat([non_empty_indices, kept_empty_indices])

    accepting_indices = accepting_indices.sort().values

    return accepting_indices


def _get_num_to_take_per_scale(target: int, number_of_scales: int = None) -> int:
    num_to_take_per_scale = (
        target // number_of_scales if number_of_scales is not None else target
    )
    return num_to_take_per_scale


def _check_accepting_indecies(
    accepting_indicies: torch.Tensor,
    sdwl: SegmentedDataWithLabels,
    target_num_to_take: int,
):
    if len(accepting_indicies) / sdwl.data.shape[0] > 0.1:
        print(
            f"WARNING: accepting {len(accepting_indicies)} segments out of {sdwl.data.shape[0]} total ({len(accepting_indicies)/sdwl.data.shape[0]:.2%}). If you expect your dataset to be mostly empty, this may be fine."
        )
    if len(accepting_indicies) < target_num_to_take:
        print(
            f"WARNING: taking less than target number of segments for this scale: accepting {len(accepting_indicies)} segments, but we would like to take {target_num_to_take}. Consider increasing percentage_empty_target or adjusting your dataset if you want more segments at this scale."
        )
    assert (
        len(accepting_indicies) > 0
    ), "No segments accepted - try increasing percentage_empty_target!"


def _get_random_indicies(
    sdwl: SegmentedDataWithLabels, num_to_take: int
) -> torch.Tensor:
    assert (
        sdwl.data.dim() == 4
    ), f"Expected sdwl.data to be a 4D tensor of shape (N,C,H,W), but got {sdwl.data.shape}"
    result = torch.randperm(sdwl.data.shape[0])[:num_to_take]
    return result


def _stack_sdwls(sdwls: List[SegmentedDataWithLabels]) -> SegmentedDataWithLabels:
    data_from_sdwls = [sdwl.data for sdwl in sdwls]
    labels_from_sdwls = [sdwl.labels for sdwl in sdwls]
    spliced_data = torch.cat(data_from_sdwls, dim=0)
    spliced_labels = torch.cat(labels_from_sdwls, dim=0)
    return SegmentedDataWithLabels(spliced_data, spliced_labels)


def _randomise_order_of_sdwl_patches(
    sdwl: SegmentedDataWithLabels,
) -> SegmentedDataWithLabels:
    indices = torch.randperm(sdwl.data.shape[0])
    sdwl = _get_sdwl_patch_indicies(sdwl, indices)
    return sdwl


def apply_random_guassian_blur_to_sdwl(
    sdwl: SegmentedDataWithLabels, kernel_size: int = 3, sigma: float = 3.0
) -> SegmentedDataWithLabels:
    blurred_data_segments = []
    for idx in range(sdwl.data.shape[0]):
        patch = DataWithLabels(sdwl.data[idx], sdwl.labels[idx])
        blurred_patch = apply_random_guassian_blur_to_dwl(
            patch,
            kernel_size=kernel_size,
            sigma=sigma,
        )
        blurred_data_segments.append(blurred_patch.data)

    stacked_data = torch.stack(blurred_data_segments, dim=0)
    result_sdwl = SegmentedDataWithLabels(
        stacked_data,
        sdwl.labels,
    )

    return result_sdwl


def apply_random_guassian_blur_to_dwl(
    dwl_segment: DataWithLabels, kernel_size: int = 3, sigma: float = 3.0
) -> DataWithLabels:
    if random.random() < 0.33:
        return dwl_segment
    sigma = sigma * random.uniform(0.5, 1.5)
    kernel_size = int(kernel_size * random.uniform(1, 3))
    if kernel_size % 2 == 0:
        kernel_size += 1
    data = tv_tensors.Image(dwl_segment.data)
    blurred_data = trch_trns.functional.gaussian_blur(
        data,
        kernel_size=kernel_size,
        sigma=sigma,
    )
    return DataWithLabels(torch.Tensor(blurred_data), dwl_segment.labels)


def get_segments_with_sliding_window(
    dwl: DataWithLabelsGeoTethered,
    base_window_size: int = CONFIG["DEFAULT_BASE_WINDOW_SIZE"],
    stride: int = CONFIG["DEFAULT_STRIDE"],
    number_of_scales: int = CONFIG["DEFAULT_NUMBER_OF_SCALES"],
) -> SegmentedDataWithLabels:
    dwls_at_varried_sizes = _make_sdwls_at_varied_scales(
        dwl,
        base_window_size,
        stride,
        number_of_scales=number_of_scales,
    )

    target_num_patches_to_take = _get_num_to_take_per_scale(
        CONFIG["TARGET_NUM_TO_TAKE"],
        number_of_scales=number_of_scales,
    )

    sdwls = []
    for sdwl in dwls_at_varried_sizes:
        random_indices = _get_random_indicies(sdwl, target_num_patches_to_take)
        sdwl = _get_sdwl_patch_indicies(sdwl, random_indices)
        sdwl = _apply_rotation_to_sdwl(sdwl)
        sdwl = apply_random_guassian_blur_to_sdwl(sdwl)
        sdwl = _resize_sdwl_patches(sdwl, base_window_size)
        sdwls.append(sdwl)
    sdwl = _stack_sdwls(sdwls)
    sdwl = _randomise_order_of_sdwl_patches(sdwl)

    return sdwl


def remove_empty_segments(
    data_with_labels: SegmentedDataWithLabels,
) -> SegmentedDataWithLabels:
    not_empty = ~torch.isnan(data_with_labels.data)
    mean_occupied = not_empty.float().mean(dim=(1, 2))
    mask = mean_occupied > 0.5
    data_with_labels = SegmentedDataWithLabels(
        data_with_labels.data[mask], data_with_labels.labels[mask]
    )
    return data_with_labels


def remove_segments_missing_positive(
    data_with_labels: SegmentedDataWithLabels,
    keep_neg_prob: float = CONFIG["DEFAULT_KEEP_NEG_PROB"],
) -> SegmentedDataWithLabels:
    y = data_with_labels.labels  # (N,H,W) or (N,1,H,W) or (N,C,H,W)

    # Has at least one positive per segment
    has_pos = (y > 0).flatten(start_dim=1).any(dim=1)  # (N,)

    # Randomly keep some negative-only segments
    keep_neg = torch.rand(y.shape[0], device=y.device) < keep_neg_prob

    keep = has_pos | keep_neg  # elementwise OR

    return SegmentedDataWithLabels(
        data_with_labels.data[keep],
        y[keep],
    )


def infer_nans_segmented(sdwl: SegmentedDataWithLabels) -> SegmentedDataWithLabels:
    x = sdwl.data
    mean = torch.nanmean(x)
    out = torch.where(torch.isnan(x), mean, x)
    return SegmentedDataWithLabels(out, sdwl.labels)


def normalise_tensor_local(
    tensor: torch.Tensor,
    kernel_size: int = CONFIG["DEFAULT_KERNEL_SIZE"],
    sigma: float | None = CONFIG["DEFAULT_SIGMA"],
    eps: float = CONFIG["DEFAULT_EPS"],
) -> torch.Tensor:
    """
    Local background removal using a Gaussian KDE-like smoothing.
    Computes a Gaussian-smoothed field and subtracts it from the input.
    Supports (H,W), (C,H,W), (N,H,W), or (N,C,H,W) tensors.
    """
    x = tensor.float()

    if kernel_size % 2 == 0:
        kernel_size += 1

    added_batch = False
    added_channel = False

    if x.dim() == 2:
        x = x.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        added_batch = True
        added_channel = True
    elif x.dim() == 3:
        x = x.unsqueeze(1)  # (N,1,H,W)
        added_channel = True
    elif x.dim() != 4:
        raise ValueError("tensor must be 2D, 3D, or 4D")

    if sigma is None:
        sigma = kernel_size / 6.0
    if sigma <= 0:
        raise ValueError("sigma must be > 0")

    finite_mask = torch.isfinite(x)
    if finite_mask.any():
        min_val = x[finite_mask].min()
    else:
        min_val = x.new_tensor(0.0)

    padding = kernel_size // 2

    coords = (
        torch.arange(kernel_size, device=x.device, dtype=x.dtype)
        - (kernel_size - 1) / 2.0
    )
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    kernel_2d = g[:, None] * g[None, :]
    kernel_2d = kernel_2d / kernel_2d.sum()

    channels = x.shape[1]
    weight = kernel_2d.view(1, 1, kernel_size, kernel_size).repeat(channels, 1, 1, 1)

    # NaN-aware Gaussian smoothing
    valid = torch.isfinite(x)
    x_filled = torch.where(valid, x, torch.zeros_like(x))
    valid_f = valid.to(dtype=x.dtype)

    weighted_sum = F.conv2d(
        x_filled, weight, stride=1, padding=padding, groups=channels
    )
    weight_sum = F.conv2d(
        valid_f, weight, stride=1, padding=padding, groups=channels
    ).clamp_min(eps)
    local_mean = weighted_sum / weight_sum

    # KDE-like background subtraction (no local std normalization)
    out = x_filled - local_mean
    out = torch.where(valid, out, torch.zeros_like(out))
    out = out + min_val

    out_finite = torch.isfinite(out)
    if out_finite.any():
        out_min = out[out_finite].min()
        out = out - out_min

    if added_channel:
        out = out.squeeze(1)
    if added_batch:
        out = out.squeeze(0)

    return out


def normalise_dwl_local(
    dwl: DataWithLabels | DataWithLabelsGeoTethered,
    kernel_size: int = CONFIG["DEFAULT_KERNEL_SIZE"],
    sigma: float | None = None,
    eps: float = 1e-6,
) -> DataWithLabels | DataWithLabelsGeoTethered:
    """
    Apply local normalization to DataWithLabels.data while preserving labels and metadata.
    """

    def _print_stats(tag: str, t: torch.Tensor) -> None:
        t_f = t.float()
        valid = torch.isfinite(t_f)
        total = t_f.numel()
        valid_count = int(valid.sum().item())
        nan_count = total - valid_count
        nan_pct = (nan_count / total) * 100 if total > 0 else 0.0
        print(
            f"[{tag}] shape={tuple(t_f.shape)} dtype={t_f.dtype} total={total} valid={valid_count} nan%={nan_pct:.2f}"
        )

        if valid_count == 0:
            print(f"[{tag}] no valid values to summarize")
            return

        vals = t_f[valid]
        vmin = vals.min().item()
        vmax = vals.max().item()
        vmean = vals.mean().item()
        vstd = vals.std(unbiased=False).item()
        vmed = vals.median().item()
        q = torch.quantile(
            vals, torch.tensor([0.01, 0.05, 0.95, 0.99], device=vals.device)
        )
        print(
            f"[{tag}] min={vmin:.6f} max={vmax:.6f} mean={vmean:.6f} std={vstd:.6f} median={vmed:.6f} "
            f"p01={q[0].item():.6f} p05={q[1].item():.6f} p95={q[2].item():.6f} p99={q[3].item():.6f}"
        )

    _print_stats("local_normalise:input", dwl.data)

    out = normalise_tensor_local(
        dwl.data,
        kernel_size=kernel_size,
        sigma=sigma,
        eps=eps,
    )
    _print_stats("local_normalise:output", out)
    if isinstance(dwl, DataWithLabelsGeoTethered):
        return DataWithLabelsGeoTethered(out, dwl.labels, dwl.epsg, dwl.offset, dwl.res)
    else:
        return DataWithLabels(out, dwl.labels)


def determanistic_splice_tensors(
    tensors: List[torch.Tensor], seed: int = CONFIG["DEFAULT_SEED"]
) -> torch.Tensor:
    n_max = max(t.shape[0] for t in tensors)
    g = torch.Generator(device="cpu").manual_seed(seed)
    balanced = []
    for t in tensors:
        n = t.shape[0]
        if n == n_max:
            balanced.append(t)
        else:
            idx = torch.randint(0, n, (n_max,), generator=g, device="cpu").to(t.device)
            balanced.append(t.index_select(0, idx))

    x = torch.cat(balanced, dim=0)
    perm = torch.randperm(x.shape[0], generator=g, device="cpu").to(x.device)
    return x.index_select(0, perm)


def generate_train_test_sets(labeled_tensor: torch.Tensor):
    pass


def generate_folds(labeled_tensor: torch.Tensor) -> List[torch.Tensor]:
    pass


def get_geotiff_true_origin(geotiff_path: str) -> Tuple[int, int]:
    with rasterio.open(geotiff_path) as reader:
        transform: Affine = reader.transform

        x_origin: int
        y_origin: int
        x_origin, y_origin = transform * (0, 0)

    return int(x_origin), int(y_origin)


def tensor_and_offset_from_geotiff(
    geotiff_path: str,
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    with rasterio.open(geotiff_path) as reader:
        data = reader.read(masked=True)
    data = data[
        0
    ]  # we take only the first band -- WE ASSUME THIS IS ELEVATION! TO DO: include this important detail in read me!
    tensor = torch.from_numpy(data).float()
    true_origin = get_geotiff_true_origin(geotiff_path)
    return tensor, true_origin


def get_positive_geotiff_tensor_and_offset(
    folder: str,
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    path = os.path.join(CONFIG["DATA_LOCATION"], folder, CONFIG["POSITIVE_TIFF_NAME"])
    if not os.path.exists(path):
        print(
            f"Positive geotiff not found at: {path}. Using empty tensor and offset for pos class (0,0)."
        )
        empty_tensor = torch.zeros((1, 1))
        return empty_tensor, (0, 0)
    return tensor_and_offset_from_geotiff(path)


def get_combined_geotiff_tensor_and_offset(
    folder: str,
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    path = os.path.join(CONFIG["DATA_LOCATION"], folder, CONFIG["COMBINED_TIFF_NAME"])
    return tensor_and_offset_from_geotiff(path)


def retrieve_dataset_EPSG(folder: str) -> int:
    combined_path = os.path.join(
        CONFIG["DATA_LOCATION"], folder, CONFIG["COMBINED_TIFF_NAME"]
    )
    positive_path = os.path.join(
        CONFIG["DATA_LOCATION"], folder, CONFIG["POSITIVE_TIFF_NAME"]
    )

    with rasterio.open(combined_path) as reader:
        combined_epsg = reader.crs.to_epsg()

    if os.path.exists(positive_path):
        with rasterio.open(positive_path) as reader:
            positive_epsg = reader.crs.to_epsg()

        if combined_epsg != positive_epsg:
            raise ValueError(
                f"EPSG mismatch: {CONFIG['COMBINED_TIFF_NAME']} has EPSG:{combined_epsg}, {CONFIG['POSITIVE_TIFF_NAME']} has EPSG:{positive_epsg}."
            )
    return combined_epsg


def standardise_dwl(dwl: DataWithLabelsGeoTethered) -> DataWithLabelsGeoTethered:
    """
    Standardize a DataWithLabels by applying the same preprocessing as in training pipeline,
    excluding segmentation, rotation, and duplication.
    """
    dwl.data = put_nans_in_neggative_positions(dwl.data)
    dwl = normalise_dwl_local(dwl, kernel_size=CONFIG["DEFAULT_KERNEL_SIZE"])
    return dwl

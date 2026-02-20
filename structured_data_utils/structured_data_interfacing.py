import os
import torch
from typing import List, Tuple
from structured_data_utils.config.constants import (
    ESPSG,
    RES,
    EMPTY_VAL,
    STANDARDISATION_TARGET_TIFFS,
)
from common.data_managment import DataWithLabels, SegmentedDataWithLabels
from common.constants import (
    POSITIVE_LAS_DIR,
    NEGATIVE_LAS_DIR,
    POSITIVE_GEOTIFF_DIR,
    NEGATIVE_GEOTIFF_DIR,
    COMBINED_GEOTIFF_DIR,
    COMBINED_LAS_DIR,
    POSITIVE_TIFF_NAME,
    COMBINED_TIFF_NAME,
    DATA_LOCATION,
)
import subprocess
from torchvision import transforms as trch_trns
import torch.nn.functional as F
from typing import TYPE_CHECKING
import rasterio
from affine import Affine
import cv2

import random
import numpy as np

VERBOSE = True


def rotate_window(window: torch.Tensor, angle: float) -> torch.Tensor:
    """
    Rotate a 2D window by a specified angle.

    Args:
        window: 2D tensor of shape (height, width)
        angle: Rotation angle in degrees.

    Returns:
        Rotated window tensor of same shape
    """
    # Convert to numpy for rotation
    if isinstance(window, torch.Tensor):
        window_np = window.cpu().numpy() if window.is_cuda else window.numpy()
    else:
        window_np = np.asarray(window)

    # Get center of image
    h, w = window_np.shape
    center = (w / 2, h / 2)

    # Get rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Apply rotation
    rotated = cv2.warpAffine(
        window_np,
        rotation_matrix,
        (w, h),
        borderMode=cv2.BORDER_REFLECT,
        flags=cv2.INTER_LINEAR,
    )

    # Convert back to tensor
    if isinstance(window, torch.Tensor):
        return torch.from_numpy(rotated).to(window.dtype).to(window.device)
    else:
        return torch.from_numpy(rotated)


def standardise_geotiff(target_path: str, write_path: str, force: bool = False):
    if os.path.exists(write_path) and not force:
        print(
            f"Standardised file already exists at {write_path}. Skipping. Use force=True to override."
        )
        return

    p = subprocess.run(
        [
            "gdalwarp",
            "-t_srs",
            ESPSG,  # e.g. "EPSG:7856"
            "-tr",
            RES,
            RES,  # e.g. "1", "1"
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
        target_files if target_files is not None else STANDARDISATION_TARGET_TIFFS
    )

    for tiff in files_to_process:
        tiff_path = os.path.join(dir, tiff)
        write_path = tiff_path.replace(".tif", "_STANDARDISED.tif")
        standardise_geotiff(tiff_path, write_path, force=force)


def offset_meters_to_offset_pixels(offset):
    print(offset[0] / float(RES))
    offset_x_corrected = int(offset[0] / float(RES))
    offset_y_corrected = int(offset[1] / float(RES))
    return (offset_x_corrected, offset_y_corrected)


def pad_pos_mask_to_match(
    pos_tensor: torch.Tensor, other_tensor: torch.Tensor, offset: Tuple[int, int]
):
    pad_x = other_tensor.shape[-1] - pos_tensor.shape[-1]
    pad_y = other_tensor.shape[-2] - pos_tensor.shape[-2]

    print(f"pad_x: {pad_x}, pad_y: {pad_y}")

    padded = F.pad(pos_tensor, (0, pad_x, 0, pad_y), mode="constant", value=EMPTY_VAL)
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


def load_data_with_labels(folder_name: str) -> DataWithLabels:
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

    pos_mask = positive[0] != EMPTY_VAL

    labels[0, pos_mask] = 1

    epsg = retrieve_dataset_EPSG(folder_name)

    return DataWithLabels(combined, labels, epsg, offset_combined, RES)


def _apply_rotation_to_patches_and_patch_labels(
    patches: torch.Tensor,
    patch_labels: torch.Tensor,
    rotation_angles: List[float],
) -> SegmentedDataWithLabels:

    if rotation_angles is None:
        print(f"No rotation angles provided, using generator...")
        rotation_angles = [random.randint(0, 180) for _ in range(400)]
    print(f"Using rotation angles: {rotation_angles}")

    size_before_rotation = patches.shape[0]
    rotated_patches = []
    rotated_labels = []

    for patch, label in zip(patches, patch_labels):
        rotations_per_patch = 50
        for i in range(rotations_per_patch):
            angle = random.choice(rotation_angles)
            rotated_patch = rotate_window(patch, angle)
            rotated_label = rotate_window(label.float(), angle).long()
            rotated_patches.append(rotated_patch)
            rotated_labels.append(rotated_label)
        # print(f"Processed {len(rotated_patches)} patches")
    patches = torch.stack(rotated_patches)
    patch_labels = torch.stack(rotated_labels)

    size_after_rotation = patches.shape[0]

    assert len(patches) == len(patch_labels), "ok the rotation function is fucked..."
    print(f"len patches {len(patches)}")
    print(f"len patch labels {len(patch_labels)}")

    print(
        f"Dataset size after rotation: {size_after_rotation} patches (x{size_after_rotation/size_before_rotation:.1f} increase)"
    )
    return SegmentedDataWithLabels(patches, patch_labels)


def _get_patch_tensors_at_target_size(
    window_size: int, data: torch.Tensor, stride: int
):
    assert data.dim() == 3, "data must be a 3D tensor of shape (C,H,W)"
    data_pactched = data.unfold(1, window_size, stride).unfold(2, window_size, stride)
    return data_pactched


def _check_dwl_size(dwl: SegmentedDataWithLabels, critical_threshold_gb=1.0) -> float:
    dwl_data_size_bytes = dwl.data.element_size() * dwl.data.nelement()
    dwl_labels_size_bytes = dwl.labels.element_size() * dwl.labels.nelement()
    dwl_data_size_gb = (dwl_data_size_bytes) / (1024**3)
    dwl_labels_size_gb = (dwl_labels_size_bytes) / (1024**3)
    dwl_total_size_gb = dwl_data_size_gb + dwl_labels_size_gb
    if VERBOSE:
        print(
            f"SegmentedDataWithLabels size: data={dwl_data_size_gb:.2f} GB, labels={dwl_labels_size_gb:.2f} GB, total={dwl_total_size_gb:.2f} GB"
        )
    if dwl_total_size_gb > critical_threshold_gb:
        response = input(
            f"WARNING: SegmentedDataWithLabels is very large ({dwl_total_size_gb:.2f} GB). Do you want to continue? (y/n): "
        )
        if response.lower() != "y":
            raise MemoryError(
                f"Aborting due to large SegmentedDataWithLabels size ({dwl_total_size_gb:.2f} GB). Consider reducing window sizes or number of scales."
            )

    return dwl_total_size_gb


MAX_ALLOWED_GB = 1


def _get_patch_tensors_dwls_at_target_sizes(
    window_sizes: List[int],
    image_tensor_data: torch.Tensor,
    image_tensor_labels: torch.Tensor,
    stride: int,
) -> List[SegmentedDataWithLabels]:
    dwls_at_sizes = []
    for window_size in window_sizes:
        data_patches = _get_patch_tensors_at_target_size(
            window_size, image_tensor_data, stride
        )
        labels_patches = _get_patch_tensors_at_target_size(
            window_size, image_tensor_labels, stride
        )
        dwl = SegmentedDataWithLabels(data_patches, labels_patches)
        gross_size_gb = _check_dwl_size(dwl)
        if gross_size_gb > MAX_ALLOWED_GB:
            raise MemoryError(
                f"Aborting due to large SegmentedDataWithLabels size ({gross_size_gb:.2f} GB). Consider reducing window sizes or number of scales."
            )
        dwls_at_sizes.append(dwl)
    return dwls_at_sizes


def _size_change_percentage_from_gammavariate(alpha, beta) -> float:
    size_change_percentage = random.gammavariate(alpha, beta) - 30
    return size_change_percentage


def _get_size_change_percentages(
    number_of_scales: int, alpha: float, beta: float
) -> List[float]:
    size_change_percentages = [
        _size_change_percentage_from_gammavariate(alpha, beta)
        for _ in range(number_of_scales)
    ]
    return size_change_percentages


def _get_scale_factors_from_size_change_percentages(
    size_change_percentages: List[float],
) -> List[float]:
    scale_factors = [
        1 + (size_change_percentage / 100)
        for size_change_percentage in size_change_percentages
    ]
    return scale_factors


def _get_window_sizes_from_scale_factors(
    base_window_size: int, scale_factors: List[float]
) -> List[int]:
    modified_window_sizes = [
        int(base_window_size * scale_factor) for scale_factor in scale_factors
    ]
    return modified_window_sizes


def _make_dwls_at_varied_scales(
    data_with_labels: DataWithLabels,
    base_window_size: int,
    stride: int,
    number_of_scales: int,
) -> List[SegmentedDataWithLabels]:

    size_change_percentages = _get_size_change_percentages(
        number_of_scales, alpha=0.85, beta=117.6  # TODO: fix these magic numbers
    )
    scale_factors = _get_scale_factors_from_size_change_percentages(
        size_change_percentages
    )
    modified_window_sizes = _get_window_sizes_from_scale_factors(
        base_window_size, scale_factors
    )
    dwls_at_targeted_sizes = _get_patch_tensors_dwls_at_target_sizes(
        modified_window_sizes, data_with_labels.data, stride
    )
    return dwls_at_targeted_sizes


def _resize_patch(
    data_with_labels: SegmentedDataWithLabels, new_size: int
) -> SegmentedDataWithLabels:
    patch = data_with_labels.data
    label = data_with_labels.labels
    assert patch.dim() == 2 and label.dim() == 2, "patch and label must be 2D tensors"
    resizer = trch_trns.Resize(
        new_size, interpolation=trch_trns.InterpolationMode.BILINEAR, antialias=True
    )
    patch_resized = resizer.apply(patch)
    label_resized = resizer.apply(label)
    return SegmentedDataWithLabels(patch_resized, label_resized)


def _take_random_subset(
    data_with_labels: SegmentedDataWithLabels, random_indices: torch.Tensor
) -> SegmentedDataWithLabels:
    return SegmentedDataWithLabels(
        data_with_labels.data.index_select(0, random_indices),
        data_with_labels.labels.index_select(0, random_indices),
    )


def _take_indicies(
    data_with_labels: SegmentedDataWithLabels, indicies: torch.Tensor
) -> SegmentedDataWithLabels:
    return SegmentedDataWithLabels(
        data_with_labels.data.index_select(0, indicies),
        data_with_labels.labels.index_select(0, indicies),
    )


def _get_accepting_segment_indecies(
    label: torch.Tensor,
    varified_labels: list[torch.Tensor],
    percentage_empty_target: float,
) -> torch.Tensor:
    if label.numel() == 0:
        return []

    empty_mask = (label == 0).all(dim=(1, 2))

    total_count = len(varified_labels)
    if total_count == 0:
        accept_empty = True
    else:
        empty_count = sum((vl == 0).all().item() for vl in varified_labels)
        current_empty_percentage = empty_count / total_count
        accept_empty = current_empty_percentage < percentage_empty_target

    accepting = (~empty_mask) | accept_empty
    return torch.nonzero(accepting).squeeze(1).tolist()


def _get_num_to_take_per_scale(target: int, number_of_scales: int = None) -> int:
    num_to_take_per_scale = (
        target // number_of_scales if number_of_scales is not None else target
    )
    return num_to_take_per_scale


def _check_accepting_indecies(
    accepting_indicies, dwl: DataWithLabels, target_num_to_take: int
):
    if len(accepting_indicies) / dwl.data.shape[0] > 0.1:
        print(
            f"WARNING: accepting {len(accepting_indicies)} segments out of {dwl.data.shape[0]} total ({len(accepting_indicies)/dwl.data.shape[0]:.2%}). If you expect your dataset to be mostly empty, this may be fine."
        )
    if len(accepting_indicies) > target_num_to_take:
        print(
            f"WARNING: taking less than target number of segments for this scale: accepting {len(accepting_indicies)} segments, but target is {target_num_to_take}. Consider increasing percentage_empty_target or adjusting your dataset if you want more segments at this scale."
        )
    assert (
        len(accepting_indicies) > 0
    ), "No segments accepted - try increasing percentage_empty_target!"


def get_segments_with_sliding_window(
    data_with_labels: DataWithLabels,
    base_window_size=300,
    stride=300,
    rotation_angles=None,
    percentage_empty_target=0.30,
    number_of_scales=10,
) -> SegmentedDataWithLabels:

    dwls_at_varried_sizes = _make_dwls_at_varied_scales(
        data_with_labels,
        base_window_size,
        stride,
        number_of_scales=number_of_scales,
    )

    varified_patches = []
    varified_labels = []

    num_to_take = _get_num_to_take_per_scale(
        target=30000,  # TODO: fix this magic number
        number_of_scales=number_of_scales,
    )

    for dwl in dwls_at_varried_sizes:
        accepting_indicies = _get_accepting_segment_indecies(
            dwl.labels, varified_labels, percentage_empty_target
        )

        _check_accepting_indecies(accepting_indicies, dwl, num_to_take)

        data_with_labels_scaled = _take_indicies(
            data_with_labels_scaled, accepting_indicies
        )

        random_indices = torch.randperm(data_with_labels_scaled.data.shape[0])[
            :num_to_take
        ]
        data_with_labels_scaled = _take_random_subset(
            data_with_labels_scaled, random_indices
        )
        data_with_labels_scaled = _resize_patch(
            data_with_labels_scaled, base_window_size
        )

    patches = torch.stack(varified_patches)
    patch_labels = torch.stack(varified_labels)

    data_with_labels_out = _apply_rotation_to_patches_and_patch_labels(
        patches, patch_labels, rotation_angles
    )

    return data_with_labels_out


def remove_empty_segments(
    data_with_labels: SegmentedDataWithLabels,
) -> SegmentedDataWithLabels:
    print(data_with_labels.data.shape)
    not_empty = ~torch.isnan(data_with_labels.data)
    mean_occupied = not_empty.float().mean(dim=(1, 2))
    print(f"mean_occupied: {mean_occupied}")
    mask = mean_occupied > 0.5
    data_with_labels = SegmentedDataWithLabels(
        data_with_labels.data[mask], data_with_labels.labels[mask]
    )
    print(data_with_labels.data.shape)
    return data_with_labels


def remove_segments_missing_positive(
    data_with_labels: SegmentedDataWithLabels,
    keep_neg_prob: float = 0.5,
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


def infer_nans_segmented(dwl: SegmentedDataWithLabels) -> SegmentedDataWithLabels:
    x = dwl.data
    mean = torch.nanmean(x)
    out = torch.where(torch.isnan(x), mean, x)
    return SegmentedDataWithLabels(out, dwl.labels)


def normalise_tensor_local(
    tensor: torch.Tensor,
    kernel_size: int = 501,
    sigma: float | None = None,
    eps: float = 1e-6,
    target_mean: float = 0.0,
    target_std: float = 1.0,
) -> torch.Tensor:
    """
    Local background removal using a Gaussian KDE-like smoothing.
    Computes a Gaussian-smoothed field and subtracts it from the input.
    Supports (H,W), (C,H,W), (N,H,W), or (N,C,H,W) tensors.
    """
    x = tensor.float()

    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be an odd positive integer")

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

        out_vals = out[out_finite]
        out_mean = out_vals.mean()
        out_std = out_vals.std(unbiased=False).clamp_min(eps)
        out = (out - out_mean) / out_std
        out = out * target_std + target_mean

        shifted_min = out[out_finite].min()
        if shifted_min < 0:
            out = out - shifted_min

    if added_channel:
        out = out.squeeze(1)
    if added_batch:
        out = out.squeeze(0)

    return out


def normalise_data_with_labels_local(
    dwl: DataWithLabels,
    kernel_size: int,
    sigma: float | None = None,
    eps: float = 1e-6,
    target_mean: float = 0.0,
    target_std: float = 1.0,
) -> DataWithLabels:
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
        target_mean=target_mean,
        target_std=target_std,
    )
    _print_stats("local_normalise:output", out)
    return DataWithLabels(out, dwl.labels, dwl.epsg, dwl.offset, dwl.res)


def splice_tensors(tensors: List[torch.Tensor], seed: int = 0) -> torch.Tensor:
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
    path = os.path.join(DATA_LOCATION, folder, POSITIVE_TIFF_NAME)
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
    path = os.path.join(DATA_LOCATION, folder, COMBINED_TIFF_NAME)
    return tensor_and_offset_from_geotiff(path)


def retrieve_dataset_EPSG(folder: str) -> int:
    combined_path = os.path.join(DATA_LOCATION, folder, COMBINED_TIFF_NAME)
    positive_path = os.path.join(DATA_LOCATION, folder, POSITIVE_TIFF_NAME)

    with rasterio.open(combined_path) as reader:
        combined_epsg = reader.crs.to_epsg()

    if os.path.exists(positive_path):
        with rasterio.open(positive_path) as reader:
            positive_epsg = reader.crs.to_epsg()

        if combined_epsg != positive_epsg:
            raise ValueError(
                f"EPSG mismatch: {COMBINED_TIFF_NAME} has EPSG:{combined_epsg}, {POSITIVE_TIFF_NAME} has EPSG:{positive_epsg}."
            )
    return combined_epsg

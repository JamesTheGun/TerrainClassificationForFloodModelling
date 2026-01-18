
import os
import torch
from typing import List, Tuple
from structured_data_utils.config.constants import ESPSG, RES, EMPTY_VAL, STANDARDISATION_TARGET_TIFFS
from common.data_managment import DataWithLabels, SegmentedDataWithLabels
from common.constants import POSITIVE_LAS_DIR, NEGATIVE_LAS_DIR, POSITIVE_GEOTIFF_DIR, NEGATIVE_GEOTIFF_DIR, COMBINED_GEOTIFF_DIR, COMBINED_LAS_DIR, POSITIVE_TIFF_NAME, COMBINED_TIFF_NAME, DATA_LOCATION
import subprocess
import torch.nn.functional as F
from typing import TYPE_CHECKING
import rasterio
from affine import Affine

import random

def standardise_geotiff(target_path: str, write_path: str, force: bool = False):
    if os.path.exists(write_path) and not force:
        print(f"Standardised file already exists at {write_path}. Skipping. Use force=True to override.")
        return
    
    p = subprocess.run(
        [
            "gdalwarp",
            "-t_srs", ESPSG,        # e.g. "EPSG:7856"
            "-tr", RES, RES,        # e.g. "1", "1"
            "-tap",                # CRITICAL: align pixel grid
            "-r", "bilinear",      # use "near" for masks/classes
            "-of", "GTiff",
            "-overwrite",
            target_path,
            write_path,
        ],
        capture_output=True,
        text=True,
    )

    if p.returncode != 0:
        raise RuntimeError(p.stderr)

def standardise_dataset(dataset_name: str, force: bool = False, target_files: list = None):
    """Standardise dataset geotiffs.
    
    Args:
        dataset_name: Name of dataset folder in data/
        force: Force re-standardisation if file already exists
        target_files: Optional list of specific filenames to standardise.
                     If None, standardises all files in STANDARDISATION_TARGET_TIFFS
    """
    standardise_folder(os.path.join("data", dataset_name), force=force, target_files=target_files)

def standardise_folder(dir: str, force: bool = False, target_files: list = None):
    """Standardise all geotiffs in a folder.
    
    Args:
        dir: Directory containing geotiffs
        force: Force re-standardisation if file already exists
        target_files: Optional list of specific filenames to standardise.
                     If None, standardises all files in STANDARDISATION_TARGET_TIFFS
    """
    files_to_process = target_files if target_files is not None else STANDARDISATION_TARGET_TIFFS
    
    for tiff in files_to_process:
        tiff_path = os.path.join(dir, tiff)
        write_path = tiff_path.replace(".tif", "_STANDARDISED.tif")
        standardise_geotiff(tiff_path, write_path, force=force)

def offset_meters_to_offset_pixels(offset):
    print(offset[0]/float(RES))
    offset_x_corrected = int(offset[0]/float(RES))
    offset_y_corrected = int(offset[1]/float(RES))
    return (offset_x_corrected, offset_y_corrected)

def pad_pos_mask_to_match(pos_tensor: torch.Tensor, other_tensor: torch.Tensor, offset: Tuple[int, int]):
    pad_x = other_tensor.shape[-1] - pos_tensor.shape[-1]
    pad_y = other_tensor.shape[-2] - pos_tensor.shape[-2]

    print(f"pad_x: {pad_x}, pad_y: {pad_y}")

    padded = F.pad(pos_tensor, (0, pad_x, 0, pad_y), mode="constant", value=EMPTY_VAL)
    pixel_offset = offset_meters_to_offset_pixels(offset)
    #padded = torch.roll(padded, shifts = pixel_offset, dims=(-2,-1))
    assert (padded.shape == other_tensor.shape), f"padded shape {padded.shape} does not match other tensor shape {other_tensor.shape}"
    padded = torch.roll(padded, shifts = pixel_offset[0], dims=1)
    padded = torch.roll(padded, shifts = pixel_offset[1], dims=0)
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

def get_segments_with_sliding_window(data_with_labels: DataWithLabels, window_size=300, stride=300) -> SegmentedDataWithLabels:
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
    data_with_labels_out = SegmentedDataWithLabels(patches, patch_labels)
    return data_with_labels_out

def remove_empty_segments(data_with_labels: SegmentedDataWithLabels) -> SegmentedDataWithLabels:
    print(data_with_labels.data.shape)
    not_empty = ~torch.isnan(data_with_labels.data)
    mean_occupied = not_empty.float().mean(dim=(1,2))
    print(mean_occupied)
    mask = mean_occupied > 0.8
    data_with_labels = SegmentedDataWithLabels(data_with_labels.data[mask], data_with_labels.labels[mask])
    print(data_with_labels.data.shape)
    return data_with_labels

def remove_segments_missing_positive(
    data_with_labels: SegmentedDataWithLabels,
    keep_neg_prob: float = 0.1,
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

def bloat_positives(dwl: SegmentedDataWithLabels):
    data = dwl.data
    labels = dwl.labels

    rotations_data = [torch.rot90(data, k=k, dims=(1,2)) for k in range(4)]
    rotations_labels = [torch.rot90(labels, k=k, dims=(1,2)) for k in range(4)]

    bloated_data = torch.cat(rotations_data, dim = 0)
    bloated_labels = torch.cat(rotations_labels, dim = 0)

    return SegmentedDataWithLabels(bloated_data, bloated_labels)

def infer_nans_segmented(dwl: SegmentedDataWithLabels) -> SegmentedDataWithLabels:
    x = dwl.data
    mean = torch.nanmean(x)
    out = torch.where(torch.isnan(x), mean, x)
    return SegmentedDataWithLabels(out, dwl.labels)

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

def tensor_and_offset_from_geotiff(geotiff_path: str) -> Tuple[torch.Tensor, Tuple[int,int]]:
    with rasterio.open(geotiff_path) as reader:
        data = reader.read(masked=True)
    data = data[0] #we take only the first band -- WE ASSUME THIS IS ELEVATION! TO DO: include this important detail in read me!
    tensor = torch.from_numpy(data).float()
    true_origin = get_geotiff_true_origin(geotiff_path)
    return tensor, true_origin

def get_positive_geotiff_tensor_and_offset(folder: str) -> Tuple[torch.Tensor, Tuple[int,int]]:
    path = os.path.join(DATA_LOCATION, folder, POSITIVE_TIFF_NAME)
    if not os.path.exists(path):
        print(f"Positive geotiff not found at: {path}. Using empty tensor and offset for pos class (0,0).")
        empty_tensor = torch.zeros((1,1))
        return empty_tensor, (0,0)
    return tensor_and_offset_from_geotiff(path)

def get_combined_geotiff_tensor_and_offset(folder: str) -> Tuple[torch.Tensor, Tuple[int,int]]:
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
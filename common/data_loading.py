from dataclasses import dataclass
import pandas as pd
import numpy as np
from .constants import POSITIVE_LAS_DIR, NEGATIVE_LAS_DIR, KEEP_DIMS
from typing import List, Tuple
import torch
from sklearn.cluster import KMeans
import laspy

def load_combined_pos_neg_df() -> pd.DataFrame:
    pos_class = laspy.read(POSITIVE_LAS_DIR)
    neg_class = laspy.read(NEGATIVE_LAS_DIR)
    pos_class = pd.DataFrame({
        dim.name: pos_class[dim.name]
        for dim in pos_class.point_format.dimensions
        if dim.name in KEEP_DIMS
    })

    neg_class = pd.DataFrame({
        dim.name: neg_class[dim.name]
        for dim in neg_class.point_format.dimensions
        if dim.name in KEEP_DIMS
    })
    pos_class["label"] = 1
    neg_class["label"] = 0
    combined = pd.concat([pos_class, neg_class])
    combined = combined.infer_objects()
    combined = pd.concat([pos_class, neg_class])
    combined = combined.astype(float)
    #TO DO... unfuck this coercsion thing, like we need to make
    #sure we are actually doing this right... for now just
    #stuff everything into numeric...
    combined = combined.dropna(axis=1)
    return combined

def build_lidar_tensor(las_df: pd.DataFrame) -> torch.Tensor:
    #consider using mixed procision for some vars in future...
    tensor = torch.tensor(las_df.to_numpy(), dtype=torch.float32)
    return tensor

def las_split_kmeans(points: torch.Tensor,
                     split_count: int,
                     sample_size: int = 100_000) -> List[torch.Tensor]:

    coords = points[:, :2].cpu().numpy().astype(np.float32, copy=False)
    N = coords.shape[0]

    if N > sample_size:
        idx = np.random.choice(N, sample_size, replace=False)
        coords_sample = coords[idx]
    else:
        coords_sample = coords
        idx = np.arange(N)

    km = KMeans(
        n_clusters=split_count,
        init="k-means++",
        n_init=5,
        max_iter=50,
        tol=1e-3,
        algorithm="elkan"
    )

    km.fit(coords_sample)

    labels = km.predict(coords)

    labels = torch.from_numpy(labels).long()

    splits = []
    for f in range(split_count):
        idx_f = torch.nonzero(labels == f, as_tuple=False).squeeze(1)
        splits.append(idx_f)

    return splits

def quick_tensor():
    data = load_combined_pos_neg_df()
    return(build_lidar_tensor(data))
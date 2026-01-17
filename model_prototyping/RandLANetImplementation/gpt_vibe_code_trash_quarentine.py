import matplotlib.pyplot as plt
import numpy as np
import torch

def visualize_windows_from_tensor(
    points: torch.Tensor,
    folds: list[torch.Tensor],
    max_points: int = 200_000,
    fx: int = 1,   # x feature index
    fy: int = 2,   # y feature index
):
    """
    Fast visualization of spatial folds in XY space.

    - Downsamples the *entire* point cloud to at most `max_points`.
    - Background = grey points
    - Points belonging to a fold are coloured by fold index.

    points: (N, D)
    folds: list of 1D index tensors (point indices)
    """

    assert points.ndim == 2, "points must be (N, D)"
    N, D = points.shape
    assert D > max(fx, fy), "Feature indices out of range"

    # --- get XY as numpy ---
    xy = points[:, [fx, fy]].detach().cpu().numpy()   # (N, 2)

    # --- assign a fold id per point (-1 = no fold) ---
    fold_ids = np.full(N, -1, dtype=np.int32)

    for fold_id, idx in enumerate(folds):
        if isinstance(idx, torch.Tensor):
            idx_np = idx.detach().cpu().numpy()
        else:
            idx_np = np.asarray(idx, dtype=np.int64)

        if idx_np.size == 0:
            continue

        fold_ids[idx_np] = fold_id

    # --- global downsampling to max_points ---
    if N > max_points:
        sel = np.random.choice(N, size=max_points, replace=False)
        xy = xy[sel]
        fold_ids = fold_ids[sel]

    # --- build colours array (single scatter call) ---
    cmap = plt.get_cmap("tab20")
    colors = np.zeros((xy.shape[0], 4), dtype=float)

    # default: light grey background
    colors[:] = (0.8, 0.8, 0.8, 0.4)

    # colour points that belong to folds
    mask_fold = fold_ids >= 0
    unique_folds = np.unique(fold_ids[mask_fold])

    for fid in unique_folds:
        m = fold_ids == fid
        colors[m] = cmap(int(fid) % cmap.N)

    # --- plot ---
    plt.figure(figsize=(9, 9))
    plt.scatter(
        xy[:, 0],
        xy[:, 1],
        s=0.1,
        c=colors,
        rasterized=True,
    )

    plt.title("Fold visualization in XY\nGrey = all points, Colours = fold assignments")
    plt.xlabel(f"Feature {fx} (x)")
    plt.ylabel(f"Feature {fy} (y)")
    plt.tight_layout()
    plt.show()

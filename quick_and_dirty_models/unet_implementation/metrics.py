import matplotlib.pyplot as plt
import torch
import numpy as np


def show_matrices(matrices, titles=None, suptitle: str = None, cmap: str = "gray", vmin=None, vmax=None):
    """Display a list of 2D matrices/arrays as subplots in a row.
    
    Args:
        matrices: Iterable of 2D tensors/arrays
        titles: Optional list of titles for each subplot
        suptitle: Optional overall title
        cmap: Colormap to use (default: "gray")
        vmin: Minimum value for colormap
        vmax: Maximum value for colormap
    """
    # Convert all matrices to numpy
    mats = []
    for m in matrices:
        if isinstance(m, torch.Tensor):
            mats.append(m.detach().cpu().numpy())
        else:
            mats.append(np.asarray(m))
    
    n = len(mats)
    if n == 0:
        return
    
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]
    
    for i, (ax, mat) in enumerate(zip(axes, mats)):
        ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax)
        if titles is not None:
            try:
                ax.set_title(titles[i])
            except (IndexError, TypeError):
                ax.set_title(str(titles))
        ax.axis("off")
    
    if suptitle:
        plt.suptitle(suptitle)
    plt.tight_layout()
    plt.show()


def show_sample(x, y, logits, epoch: int, step: int = 0, threshold: float = 0.5):
    """Display a training sample with input, label, prediction probability, and binary prediction.
    
    Uses show_matrices for consistent visualization.
    
    Args:
        x: Input tensor (1,1,H,W)
        y: Ground truth labels (1,1,H,W) with values 0/1
        logits: Model logits (1,1,H,W)
        epoch: Epoch number for title
        step: Step number for title
        threshold: Threshold for binary prediction
    """
    # Extract single samples
    x0 = x[0, 0]
    y0 = y[0, 0]
    
    # Get probabilities and predictions
    p0 = torch.sigmoid(logits[0, 0])
    pred0 = (p0 > threshold).float()
    
    # Show input with natural range, others with [0,1]
    show_matrices([x0], titles=["Input"], suptitle=None)
    
    matrices = [y0, p0, pred0]
    titles = ["Label", "Pred Prob", f"Pred (>{threshold})"]
    suptitle = f"Epoch {epoch}  Step {step}"
    
    show_matrices(matrices, titles=titles, suptitle=suptitle, vmin=0, vmax=1)

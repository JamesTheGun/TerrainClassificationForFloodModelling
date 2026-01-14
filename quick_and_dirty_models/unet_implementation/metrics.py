
import matplotlib.pyplot as plt
import torch

def show_matrices(matrices, titles=None, suptitle: str = None, cmap: str = "gray", vmin=None, vmax=None):
    """Display a list of HxW matrices (torch tensors or numpy arrays) in a row.

    matrices: iterable of 2D tensors/arrays or 2D slices (e.g. x[0,0])
    titles:   optional iterable of titles for each subplot
    suptitle: optional overall title
    """
    mats = []
    for m in matrices:
        if isinstance(m, torch.Tensor):
            mats.append(m.detach().cpu().numpy())
        else:
            mats.append(m)

    n = len(mats)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for i, (ax, mat) in enumerate(zip(axes, mats)):
        ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax)
        if titles is not None:
            try:
                ax.set_title(titles[i])
            except Exception:
                ax.set_title(str(titles))
        ax.axis("off")

    if suptitle:
        plt.suptitle(suptitle)
    plt.tight_layout()
    plt.show()


def show_sample(x, y, logits, epoch: int, step: int = 0, threshold: float = 0.5):
    """Prepare standard sample tensors and display using `show_matrices`.

    x:      (1,1,H,W) float
    y:      (1,1,H,W) float (0/1)
    logits: (1,1,H,W) float
    """
    x0 = x[0, 0]
    y0 = y[0, 0]
    p0 = torch.sigmoid(logits[0, 0])
    pred0 = (p0 > threshold).float()

    titles = ["Input", "Label", "Pred prob (sigmoid)", f"Pred mask > {threshold}"]
    suptitle = f"Epoch {epoch}  Step {step}"
    show_matrices([x0, y0, p0, pred0], titles=titles, suptitle=suptitle, cmap="gray", vmin=0, vmax=1)

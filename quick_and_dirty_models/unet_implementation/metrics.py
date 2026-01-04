
import matplotlib.pyplot as plt
import torch

def show_sample(x, y, logits, epoch: int, step: int = 0, threshold: float = 0.5):
    """
    x:      (1,1,H,W) float
    y:      (1,1,H,W) float (0/1)
    logits: (1,1,H,W) float
    """
    x0 = x[0, 0].detach().cpu()
    y0 = y[0, 0].detach().cpu()
    p0 = torch.sigmoid(logits[0, 0]).detach().cpu()
    pred0 = (p0 > threshold).float()

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    axes[0].imshow(x0, cmap="gray")
    axes[0].set_title("Input")
    axes[1].imshow(y0, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Label")
    axes[2].imshow(p0, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("Pred prob (sigmoid)")
    axes[3].imshow(pred0, cmap="gray", vmin=0, vmax=1)
    axes[3].set_title(f"Pred mask > {threshold}")

    for ax in axes:
        ax.axis("off")
    plt.suptitle(f"Epoch {epoch}  Step {step}")
    plt.tight_layout()
    plt.show()

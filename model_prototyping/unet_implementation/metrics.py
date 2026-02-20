import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, MultipleLocator
from matplotlib.scale import FuncScale
import torch
import numpy as np


def plot_weighted_running_average_loss(losses: list, alpha: float = 0.1, learning_rates: list = None, num_epochs: int = None, fixed_sample_losses: list = None, show_raw_loss: bool = True, show_smooth: bool = False, show_responsive: bool = False, show_fixed_sample: bool = True, smooth_fixed_samples: bool = True, title: str = "Training Loss (Weighted Running Average)"):
    """Plot weighted running average loss with exponential smoothing and optional learning rate overlay.
    
    Args:
        losses: List of loss values from training
        alpha: Smoothing factor (0 to 1). Higher = more responsive to recent losses
        learning_rates: Optional list of learning rates per step to plot on secondary y-axis
        num_epochs: Number of training epochs (for x-axis labeling)
        fixed_sample_losses: Optional list of losses calculated on fixed 20-sample set per epoch
        show_raw_loss: Whether to plot raw training loss (default: False)
        show_smooth: Whether to plot smooth weighted average (default: False)
        show_responsive: Whether to plot responsive weighted average (default: False)
        show_fixed_sample: Whether to plot fixed sample loss (default: True)
        smooth_fixed_samples: Whether to apply running average to fixed samples (default: True)
        title: Title for the plot
    """
    if len(losses) == 0:
        print("No losses to plot")
        return
    
    # Calculate epoch positions if num_epochs is provided
    steps_per_epoch = len(losses) / num_epochs if num_epochs else 1
    epoch_steps = np.arange(len(losses)) / steps_per_epoch
    weighted_avg_smooth = []
    avg = losses[0]
    weighted_avg_smooth.append(avg)
    
    for loss in losses[1:]:
        avg = alpha * loss + (1 - alpha) * avg
        weighted_avg_smooth.append(avg)
    
    # Calculate more responsive weighted running average (local variation sensitive)
    alpha_responsive = min(alpha * 5, 0.5)  # 5x more responsive, capped at 0.5
    weighted_avg_responsive = []
    avg = losses[0]
    weighted_avg_responsive.append(avg)
    
    for loss in losses[1:]:
        avg = alpha_responsive * loss + (1 - alpha_responsive) * avg
        weighted_avg_responsive.append(avg)
    
    fig, ax1 = plt.subplots(figsize=(14, 6))
    
    # Set high-contrast dark theme (VS Code inspired)
    fig.patch.set_facecolor('#1e1e1e')
    ax1.set_facecolor('#0d1117')
    
    # Use epochs for x-axis if num_epochs provided, otherwise use steps
    x_axis = epoch_steps if num_epochs else np.arange(len(losses))
    x_label = "Epoch" if num_epochs else "Training Step"
    
    # Plot raw losses faintly on primary axis
    if show_raw_loss:
        ax1.plot(x_axis, losses, alpha=0.35, color='#6e7681', label='Raw Loss', linewidth=0.5)
    
    # Plot smooth weighted average on primary axis
    if show_smooth:
        ax1.plot(x_axis, weighted_avg_smooth, color='#58a6ff', label=f'Smooth (α={alpha})', linewidth=2.5)
    
    # Plot responsive weighted average on primary axis
    if show_responsive:
        ax1.plot(x_axis, weighted_avg_responsive, color='#3fb950', label=f'Responsive (α={alpha_responsive:.4f})', linewidth=1.8, linestyle='--')
    
    # Plot fixed sample loss if provided
    if show_fixed_sample and fixed_sample_losses is not None and len(fixed_sample_losses) > 0:
        # Fixed sample losses are per epoch, so create epoch x-axis for them
        fixed_x = np.linspace(0, num_epochs if num_epochs else 1, len(fixed_sample_losses))
        
        # Apply running average to fixed samples if requested
        if smooth_fixed_samples:
            fixed_avg = []
            avg = fixed_sample_losses[0]
            fixed_avg.append(avg)
            alpha_fixed = 0.2  # More responsive for fixed sample smoothing
            for loss in fixed_sample_losses[1:]:
                avg = alpha_fixed * loss + (1 - alpha_fixed) * avg
                fixed_avg.append(avg)
            ax1.plot(fixed_x, fixed_avg, color='#a371f7', label='Fixed 20-Sample (Smoothed)', linewidth=2.2, linestyle=':')
        else:
            ax1.plot(fixed_x, fixed_sample_losses, color='#a371f7', label='Fixed 20-Sample Loss', linewidth=2.2, linestyle=':')
    
    ax1.set_xlabel(x_label, color='#e6edf3', fontsize=11)
    ax1.set_ylabel("Loss", color='#58a6ff', fontsize=11)
    ax1.tick_params(axis='y', labelcolor='#58a6ff')
    ax1.tick_params(axis='x', labelcolor='#e6edf3')
    ax1.spines['left'].set_color('#58a6ff')
    
    # Use power scale (sqrt) which is more aggressive than log at exaggerating small differences
    ax1.set_yscale(FuncScale(ax1, (np.sqrt, lambda x: x**2)))
    ax1.grid(True, alpha=0.25, color='#30363d', linewidth=0.8)
    
    # Add x-axis grid with intelligent interval calculation
    if num_epochs:
        # Calculate a reasonable interval (5, 10, 25, 50, 100, etc.)
        intervals = [1, 2, 5, 10, 25, 50, 100, 250, 500, 1000]
        interval = 1
        for i in intervals:
            if num_epochs / i >= 3 and num_epochs / i <= 6:  # Aim for 3-6 major tick marks
                interval = i
                break
        ax1.xaxis.set_major_locator(MultipleLocator(interval))
        ax1.xaxis.set_minor_locator(MultipleLocator(interval // 2 if interval > 2 else 1))
    
    # Add finer granularity to loss axis
    ax1.yaxis.set_major_locator(LogLocator(base=10, numticks=10))
    ax1.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=10))
    ax1.grid(True, alpha=0.08, which='minor', color='#30363d', linewidth=0.3)
    
    # Add learning rate on secondary y-axis if provided
    if learning_rates is not None and len(learning_rates) > 0:
        ax2 = ax1.twinx()
        # Plot learning rates on same x-axis scale (epochs or steps)
        ax2.plot(x_axis, learning_rates, color='#fb8500', label='Learning Rate', linewidth=1.8, alpha=0.85)
        ax2.set_ylabel("Learning Rate", color='#fb8500', fontsize=11)
        ax2.tick_params(axis='y', labelcolor='#fb8500')
        ax2.spines['right'].set_color('#fb8500')
        ax2.set_yscale('log')
        
        # Add finer granularity to learning rate axis
        ax2.yaxis.set_major_locator(LogLocator(base=10, numticks=10))
        ax2.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=10))
        ax2.grid(True, alpha=0.1, which='minor', color='#30363d', linewidth=0.3)
        
        # Combine legends from both axes
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        legend = ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', 
                           facecolor='#0d1117', edgecolor='#30363d', framealpha=0.95)
        for text in legend.get_texts():
            text.set_color('#e6edf3')
    else:
        legend = ax1.legend(loc='upper left', facecolor='#0d1117', edgecolor='#30363d', framealpha=0.95)
        for text in legend.get_texts():
            text.set_color('#e6edf3')
    
    ax1.set_title(title, color='#e6edf3', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print(f"\nLoss Statistics:")
    print(f"  Final Fixed Sample Loss: {fixed_sample_losses[-1]:.6f}")
    print(f"  Initial loss: {losses[0]:.6f}")
    print(f"  Final loss: {losses[-1]:.6f}")
    print(f"  Min loss: {min(losses):.6f}")
    print(f"  Max loss: {max(losses):.6f}")
    print(f"  Final smooth average: {weighted_avg_smooth[-1]:.6f}")
    print(f"  Final responsive average: {weighted_avg_responsive[-1]:.6f}")
    print(f"  Total steps: {len(losses)}")


def show_matrices(matrices, titles=None, suptitle: str = None, cmap: str = "viridis", vmin=None, vmax=None, auto_scale: bool = True, scale_percentiles: tuple = (1, 99)):
    """Display a list of 2D matrices/arrays as subplots in a row.
    
    Args:
        matrices: Iterable of 2D tensors/arrays
        titles: Optional list of titles for each subplot
        suptitle: Optional overall title
        cmap: Colormap to use (default: "viridis")
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
        vmin_i = vmin
        vmax_i = vmax
        if auto_scale and (vmin is None or vmax is None):
            valid = np.isfinite(mat)
            if valid.any():
                lo, hi = np.percentile(mat[valid], scale_percentiles)
                if vmin is None:
                    vmin_i = lo
                if vmax is None:
                    vmax_i = hi
        ax.imshow(mat, cmap=cmap, vmin=vmin_i, vmax=vmax_i)
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

from typing import List

import os
import torch
from torch import nn
import torch.nn.functional as F
from model_prototyping.unet_implementation.metrics import show_sample, plot_weighted_running_average_loss
from model_prototyping.unet_implementation.constants import UNET_BLOCK_CONFIG
from structured_data_utils.data import ModelData
from structured_data_utils.structured_data_interfacing import tensor_and_offset_from_geotiff

class SimpleUnet(nn.Module):
    class Block(nn.Module):
        def __init__(self, in_ch: int, out_ch: int, groups: int = 8, block_config: dict = None):
            super().__init__()
            if block_config is None:
                block_config = UNET_BLOCK_CONFIG
            
            layers = []
            for i, conv_cfg in enumerate(block_config["conv_layers"]):
                # Alternate between in_ch and out_ch for first and second conv
                conv_in_ch = in_ch if i == 0 else out_ch
                
                # Build Conv2d kwargs dynamically
                conv_kwargs = {
                    "in_channels": conv_in_ch,
                    "out_channels": out_ch,
                    "kernel_size": conv_cfg["kernel_size"],
                    "padding": conv_cfg["padding"],
                    "bias": conv_cfg["bias"],
                }
                
                # Add optional dilation if specified
                if "dilation" in conv_cfg:
                    conv_kwargs["dilation"] = conv_cfg["dilation"]
                
                # Add Conv2d layer
                layers.append(nn.Conv2d(**conv_kwargs))
                
                # Add Normalization
                if block_config["norm_type"] == "GroupNorm":
                    num_groups = min(groups, out_ch)
                    layers.append(nn.GroupNorm(num_groups=num_groups, num_channels=out_ch))
                
                # Add Activation
                if block_config["activation"] == "ReLU":
                    layers.append(nn.ReLU(**block_config["activation_params"]))
            
            self.net = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    def __init__(self, in_channels=1, out_channels=1, features=(64, 128, 256), block_config: dict = None):
        super().__init__()
        if block_config is None:
            block_config = UNET_BLOCK_CONFIG
            
        self.input_norm = nn.InstanceNorm2d(in_channels)
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)

        ch = in_channels
        for f in features:
            self.downs.append(self.Block(ch, f, block_config=block_config))
            ch = f

        self.bottleneck = self.Block(features[-1], features[-1] * 2, block_config=block_config)

        rev = list(reversed(features))
        ch = features[-1] * 2
        for f in rev:
            # resize-conv (less checkerboard than ConvTranspose2d)
            self.ups.append(nn.Conv2d(ch, f, kernel_size=3, padding=0, bias=True))
            self.ups.append(self.Block(f * 2, f, block_config=block_config))
            ch = f

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(x)
        skips = []

        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skips = skips[::-1]

        for i in range(0, len(self.ups), 2):
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            x = self.ups[i](x)

            skip = skips[i // 2]
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)

            x = torch.cat([skip, x], dim=1)
            x = self.ups[i + 1](x)

        return self.final_conv(x)

def update_lr(current_lr: float, lr_decay: float, lr_decay_function: str, total_steps: int) -> float:
    # Apply learning rate decay
    if lr_decay_function == "exp":
        lr = current_lr * (lr_decay ** total_steps)
    elif lr_decay_function == "linear":
        lr = current_lr * max(0.0, 1.0 - lr_decay * total_steps)
    elif lr_decay_function == "polynomial":
        lr = current_lr * max(0.0, (1.0 - lr_decay * total_steps) ** 2)
    elif lr_decay_function == "inverse":
        # Inverse decay: fast early drop, then flattens
        lr = current_lr / (1.0 + lr_decay * total_steps)
    elif lr_decay_function == "sqrt":
        # Square root decay: slower flattening
        lr = current_lr / (1.0 + lr_decay * (total_steps ** 0.5)) + 0.00005
    elif lr_decay_function == "log":
        # Log decay: even slower flattening than sqrt
        import math
        lr = current_lr / (1.0 + lr_decay * math.log(1.0 + total_steps))
            
    return lr

def train_model(data: ModelData, num_epochs: int = 300, viz_every: int = 20, viz_steps: int = 1, pos_bias: float = 1.0, features: List[int] = [8, 16, 32], lr = 1e-3, lr_decay: float = 1.00003, lr_decay_function: str = "exp", block_config: dict = None) -> SimpleUnet:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleUnet(in_channels=1, out_channels=1, features=features, block_config=block_config).to(device)

    pos_weight = torch.tensor([pos_bias], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    all_losses = []  # Track all losses for visualization
    fixed_sample_losses = []  # Track loss on fixed 20 samples per epoch
    all_learning_rates = []  # Track learning rates
    
    # Sample and fix 20 random samples at the beginning
    all_data = data.segmented_data_with_labels.data
    all_labels = data.segmented_data_with_labels.labels
    num_samples = len(all_data)
    fixed_indices = torch.randperm(num_samples)[:min(20, num_samples)]
    fixed_data = all_data[fixed_indices].unsqueeze(1).to(device).float()  # (N, 1, H, W)
    fixed_labels = all_labels[fixed_indices].unsqueeze(1).to(device).float()  # (N, 1, H, W)
    
    # Initialize base learning rate for decay functions
    total_steps = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0

        for step, (this_data, labels) in enumerate(data.segmented_data_with_labels.get_hacky_fold_iterable()):
            # Make (B,C,H,W) with B=C=1
            this_data = this_data.unsqueeze(0).unsqueeze(0).to(device).float()
            labels = labels.unsqueeze(0).unsqueeze(0).to(device).float()

            logits = model(this_data)
            loss = criterion(logits, labels)

            lr = update_lr(lr, lr_decay, lr_decay_function, total_steps)

            total_steps += 1
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step_loss = loss.item()
            all_losses.append(step_loss)
            all_learning_rates.append(lr)
            epoch_loss += step_loss * this_data.size(0)

            if (epoch % viz_every == 0) and (step < viz_steps):
                print(f"Epoch {epoch}/{num_epochs} - loss: {epoch_loss:.4f}")
                print(f"lr: {lr}")
                model.eval()
                with torch.no_grad():
                    v_logits = model(this_data)
                show_sample(this_data, labels, v_logits, epoch=epoch, step=step)
                model.train()

        epoch_loss /= len(data.segmented_data_with_labels.data)
        
        # Calculate loss on fixed 20 samples
        model.eval()
        with torch.no_grad():
            fixed_logits = model(fixed_data)
            fixed_loss = criterion(fixed_logits, fixed_labels).item()
            fixed_sample_losses.append(fixed_loss)
        model.train()

    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    plot_weighted_running_average_loss(all_losses, alpha=0.01, learning_rates=all_learning_rates, num_epochs=num_epochs, fixed_sample_losses=fixed_sample_losses)

    return model

def test_model_on_model_data(model: SimpleUnet, test_data: ModelData):

    from model_prototyping.unet_implementation.metrics import show_matrices
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.eval()
    model.to(device)
    
    for data, empty_labels in test_data.segmented_data_with_labels.get_hacky_fold_iterable():
        data = data.unsqueeze(0).unsqueeze(0).to(device).float()
        with torch.no_grad():
            logits = model(data)

        probs = torch.sigmoid(logits)
        pred_binary = (probs > 0.5).float()

        logits_min = logits.min().item()
        logits_max = logits.max().item()
        logits_mean = logits.mean().item()

        print(f"Prediction probability range: [{probs.min():.4f}, {probs.max():.4f}]")
        print(f"Logits statistics: min={logits_min:.6f}, max={logits_max:.6f}, mean={logits_mean:.6f}")

        show_matrices(
            [data[0, 0], probs[0, 0], pred_binary[0, 0]],
            titles=["Input Data", "Prediction Probability", "Binary Prediction (>0.5)"],
            suptitle=f"Model Test Predictions",
            vmin=None,  # Input uses natural range, predictions use [0,1]
            vmax=None
        )


def test_model_visual(model: SimpleUnet, test_data_file: str = None):
    import os

    if test_data_file is None:
        test_data_file =  "COMBINED_STANDARDISED.tif"
    test_data_file = os.path.join("data", "TEST_SET", test_data_file)

    from structured_data_utils.data import ModelData

    test_data_model = ModelData()
    test_data_model.prepare_data("TEST_SET")
    print("If you got the warning about an empty tensor, that's expected for the test set.")

    test_model_on_model_data(model, test_data_model)
from typing import List

import os
import torch
from torch import nn
import torch.nn.functional as F
from quick_and_dirty_models.unet_implementation.metrics import show_sample
from structured_data_utils.data import ModelData
from structured_data_utils.structuring import tensor_and_offset_from_geotiff

class SimpleUnet(nn.Module):
    class Block(nn.Module):
        def __init__(self, in_ch: int, out_ch: int, groups: int = 8):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.GroupNorm(num_groups=min(groups, out_ch), num_channels=out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                nn.GroupNorm(num_groups=min(groups, out_ch), num_channels=out_ch),
                nn.ReLU(inplace=True),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    def __init__(self, in_channels=1, out_channels=1, features=(64, 128, 256)):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)

        ch = in_channels
        for f in features:
            self.downs.append(self.Block(ch, f))
            ch = f

        self.bottleneck = self.Block(features[-1], features[-1] * 2)

        rev = list(reversed(features))
        ch = features[-1] * 2
        for f in rev:
            # resize-conv (less checkerboard than ConvTranspose2d)
            self.ups.append(nn.Conv2d(ch, f, kernel_size=3, padding=1, bias=False))
            self.ups.append(self.Block(f * 2, f))
            ch = f

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

def train_model(data: ModelData, num_epochs: int = 300, viz_every: int = 20, viz_steps: int = 1, pos_bias: float = 1.0, features: List[int] = [8, 16, 32], lr = 1e-3, lr_decay: float = 1.00003) -> SimpleUnet:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleUnet(in_channels=1, out_channels=1, features = features).to(device)

    pos_weight = torch.tensor([pos_bias], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0

        for step, (this_data, labels) in enumerate(data.segmented_data_with_labels.get_hacky_fold_iterable()):
            # Make (B,C,H,W) with B=C=1
            this_data = this_data.unsqueeze(0).unsqueeze(0).to(device).float()
            labels = labels.unsqueeze(0).unsqueeze(0).to(device).float()

            if torch.isnan(this_data).any():
                #print("WARNING: somehow a nan is in your data set!")
                #print(torch.isnan(this_data))
                continue

            logits = model(this_data)
            loss = criterion(logits, labels)
            lr = lr ** lr_decay
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * this_data.size(0)

            # Visualize during training (cheap, but switch to eval/no_grad)
            if (epoch % viz_every == 0) and (step < viz_steps):
                print(f"Epoch {epoch}/{num_epochs} - loss: {epoch_loss:.4f}")
                print(f"lr: {lr}")
                model.eval()
                with torch.no_grad():
                    v_logits = model(this_data)
                show_sample(this_data, labels, v_logits, epoch=epoch, step=step)
                model.train()

        epoch_loss /= len(data.segmented_data_with_labels.data)

    return model

def test_model_on_model_data(model: SimpleUnet, test_data: ModelData):

    from quick_and_dirty_models.unet_implementation.metrics import show_matrices
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.eval()
    model.to(device)

    print("HERE:")

    print(test_data.segmented_data_with_labels.get_hacky_fold_iterable())
    
    for data, empty_labels in test_data.segmented_data_with_labels.get_hacky_fold_iterable():
        data = data.unsqueeze(0).unsqueeze(0).to(device).float()
        with torch.no_grad():
            logits = model(data)

        probs = torch.sigmoid(logits)
        pred_binary = (probs > 0.5).float()

        print(f"Prediction probability range: [{probs.min():.4f}, {probs.max():.4f}]")

        show_matrices(
            [data[0, 0], probs[0, 0], pred_binary[0, 0]],
            titles=["Input Data", "Prediction Probability", "Binary Prediction (>0.5)"],
            suptitle=f"Model Test Predictions",
            vmin=None,  # Input uses natural range, predictions use [0,1]
            vmax=None
        )


def test_model_visual(model: SimpleUnet, test_data_file: str = None):
    """Visualize model predictions on test data.
    
    Loads COMBINED_STANDARDISED geotiff from TEST_SET folder by default,
    or a specific file if provided. No ground truth available.
    The user can visually inspect the model's predictions.
    
    Args:
        model: Trained SimpleUnet model
        test_data_file: Optional specific geotiff file to test on.
                       Defaults to data/TEST_SET/COMBINED_STANDARDISED.tif
    """
    import os

    if test_data_file is None:
        test_data_file =  "COMBINED_STANDARDISED.tif"
    test_data_file = os.path.join("data", "TEST_SET", test_data_file)

    from structured_data_utils.data import ModelData

    test_data_model = ModelData()
    test_data_model.prepare_data("TEST_SET")
    print("If you got the warning about an empty tensor, that's expected for the test set.")

    test_model_on_model_data(model, test_data_model)

import torch
from torch import nn
import torch.nn.functional as F

class SimpleUnet(nn.Module):
    class EPBlock(nn.Module):
        def __init__(self, in_channels: int, out_channels: int):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.conv(x)

    def __init__(self, in_channels=1, out_channels=1, features=(64, 128, 256, 512)):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=5, stride=2)

        # Down path
        ch = in_channels
        for f in features:
            self.downs.append(self.EPBlock(ch, f))
            ch = f

        # Bottleneck (often doubled, but keeping simple)
        self.bottleneck = self.EPBlock(features[-1], features[-1])

        # Up path: (upconv) then (conv after concat)
        ch = features[-1]
        for f in reversed(features):
            self.ups.append(nn.ConvTranspose2d(ch, f, kernel_size=2, stride=2))
            self.ups.append(self.EPBlock(in_channels=f * 2, out_channels=f))
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
            x = self.ups[i](x)                 # upsample (B, f, H, W)
            skip = skips[i // 2]               # (B, f, H, W)

            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)

            x = torch.cat([skip, x], dim=1)    # (B, 2f, H, W)
            x = self.ups[i + 1](x)             # conv back to (B, f, H, W)

        return self.final_conv(x)

"""Segmentation modules
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vgg11 import VGG11


class DoubleConv(nn.Module):
    """Two consecutive Conv-BN-ReLU blocks."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class VGG11UNet(nn.Module):
    """U-Net style segmentation model with VGG11 encoder."""

    def __init__(self, in_channels: int = 3, num_classes: int = 3):
        """
        Args:
            in_channels: Number of input channels.
            num_classes: Number of segmentation classes.
        """
        super().__init__()

        self.encoder = VGG11(in_channels=in_channels)

        self.up5 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.dec5 = DoubleConv(512 + 512, 512)

        self.up4 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(512 + 512, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(256 + 256, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(128 + 128, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(64 + 64, 64)

        self.segmentation_head = nn.Conv2d(64, num_classes, kernel_size=1)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    @staticmethod
    def _match_size(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """Resize x to match spatial size of ref if needed."""
        if x.shape[-2:] != ref.shape[-2:]:
            x = F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, 3, H, W]

        Returns:
            Segmentation logits [B, num_classes, H, W]
        """
        bottleneck, features = self.encoder(x, return_features=True)

        s1 = features["stage1"]   # [B, 64, 224, 224]
        s2 = features["stage2"]   # [B, 128, 112, 112]
        s3 = features["stage3"]   # [B, 256, 56, 56]
        s4 = features["stage4"]   # [B, 512, 28, 28]
        s5 = features["stage5"]   # [B, 512, 14, 14]

        x = self.up5(bottleneck)  # 7 -> 14
        x = self._match_size(x, s5)
        x = torch.cat([x, s5], dim=1)
        x = self.dec5(x)

        x = self.up4(x)           # 14 -> 28
        x = self._match_size(x, s4)
        x = torch.cat([x, s4], dim=1)
        x = self.dec4(x)

        x = self.up3(x)           # 28 -> 56
        x = self._match_size(x, s3)
        x = torch.cat([x, s3], dim=1)
        x = self.dec3(x)

        x = self.up2(x)           # 56 -> 112
        x = self._match_size(x, s2)
        x = torch.cat([x, s2], dim=1)
        x = self.dec2(x)

        x = self.up1(x)           # 112 -> 224
        x = self._match_size(x, s1)
        x = torch.cat([x, s1], dim=1)
        x = self.dec1(x)

        logits = self.segmentation_head(x)
        return logits
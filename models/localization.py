"""Localization modules
"""

import torch
import torch.nn as nn

from .layers import CustomDropout
from .vgg11 import VGG11


class VGG11Localizer(nn.Module):
    """VGG11-based localizer."""

    def __init__(self, in_channels: int = 3, dropout_p: float = 0.5):
        """
        Initialize the VGG11Localizer model.

        Args:
            in_channels: Number of input channels.
            dropout_p: Dropout probability for the localization head.
        """
        super().__init__()

        self.encoder = VGG11(in_channels=in_channels)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        self.regressor = nn.Sequential(
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),

            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),

            nn.Linear(512, 4),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize linear layer weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for localization model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].

        Returns:
            Bounding box coordinates [B, 4] in (x_center, y_center, width, height)
            format in image pixel space (not normalized values).
        """
        _, _, h, w = x.shape

        x = self.encoder(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        box = self.regressor(x)
        box = torch.sigmoid(box)

        x_center = box[:, 0] * w
        y_center = box[:, 1] * h
        width = box[:, 2] * w
        height = box[:, 3] * h

        box = torch.stack([x_center, y_center, width, height], dim=1)
        return box
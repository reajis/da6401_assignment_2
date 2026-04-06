"""Unified multi-task model."""

import torch
import torch.nn as nn

from .classification import VGG11Classifier
from .localization import VGG11Localizer
from .segmentation import VGG11UNet



class MultiTaskPerceptionModel(nn.Module):
    """Multi-task model that loads trained task-specific checkpoints."""

    def __init__(
        self,
        num_breeds: int = 37,
        seg_classes: int = 3,
        in_channels: int = 3,
        classifier_path: str = "classifier.pth",
        localizer_path: str = "localizer.pth",
        unet_path: str = "unet.pth",
    ):
        super().__init__()

        import gdown
        gdown.download(
            id="1V3_uIyt7AGRH8QmTLXgoNV2_bVAX5O5d",
            output=classifier_path,
            quiet=False,
        )
        gdown.download(
            id="1F925PrKZdOhzCM23WvYJqGghcEPP7IvZ",
            output=localizer_path,
            quiet=False,
        )
        gdown.download(
            id="1UR0zrFBPbY1rHpKaMTTAtK6KDu3Au49C",
            output=unet_path,
            quiet=False,
        )

        self.classifier = VGG11Classifier(
            num_classes=num_breeds,
            in_channels=in_channels,
        )
        self.localizer = VGG11Localizer(
            in_channels=in_channels,
        )
        self.segmenter = VGG11UNet(
            in_channels=in_channels,
            num_classes=seg_classes,
        )

        self._load_checkpoint(self.classifier, classifier_path)
        self._load_checkpoint(self.localizer, localizer_path)
        self._load_checkpoint(self.segmenter, unet_path)

    @staticmethod
    def _extract_state_dict(checkpoint):
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            return checkpoint["state_dict"]
        return checkpoint

    def _load_checkpoint(self, model: nn.Module, checkpoint_path: str) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = self._extract_state_dict(checkpoint)
        model.load_state_dict(state_dict)

    def forward(self, x: torch.Tensor):
        classification_logits = self.classifier(x)
        localization_box = self.localizer(x)
        segmentation_logits = self.segmenter(x)

        return {
            "classification": classification_logits,
            "localization": localization_box,
            "segmentation": segmentation_logits,
        }
"""Unified multi-task model."""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import gdown

from .classification import VGG11Classifier
from .localization import VGG11Localizer
from .segmentation import VGG11UNet
from .vgg11 import VGG11


class MultiTaskPerceptionModel(nn.Module):
    """Shared-backbone multi-task model."""

    def __init__(
        self,
        num_breeds: int = 37,
        seg_classes: int = 3,
        in_channels: int = 3,
        classifier_path: str = "classifier.pth",
        localizer_path: str = "localizer.pth",
        unet_path: str = "unet.pth",
    ):
        """
        Initialize the shared backbone/heads using trained weights.

        Args:
            num_breeds: Number of output classes for classification head.
            seg_classes: Number of output classes for segmentation head.
            in_channels: Number of input channels.
            classifier_path: Path to trained classifier weights.
            localizer_path: Path to trained localizer weights.
            unet_path: Path to trained unet weights.
        """
        super().__init__()

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

        classifier_model = VGG11Classifier(
            num_classes=num_breeds,
            in_channels=in_channels,
        )
        localizer_model = VGG11Localizer(
            in_channels=in_channels,
        )
        segmentation_model = VGG11UNet(
            in_channels=in_channels,
            num_classes=seg_classes,
        )

        self._load_checkpoint(classifier_model, classifier_path)
        self._load_checkpoint(localizer_model, localizer_path)
        self._load_checkpoint(segmentation_model, unet_path)

        self.encoder = VGG11(in_channels=in_channels)
        self._initialize_shared_encoder(
            classifier_model.encoder.state_dict(),
            localizer_model.encoder.state_dict(),
            segmentation_model.encoder.state_dict(),
        )

        self.classification_avgpool = classifier_model.avgpool
        self.classifier_head = classifier_model.classifier

        self.localization_avgpool = localizer_model.avgpool
        self.localizer_head = localizer_model.regressor

        self.up5 = segmentation_model.up5
        self.dec5 = segmentation_model.dec5
        self.up4 = segmentation_model.up4
        self.dec4 = segmentation_model.dec4
        self.up3 = segmentation_model.up3
        self.dec3 = segmentation_model.dec3
        self.up2 = segmentation_model.up2
        self.dec2 = segmentation_model.dec2
        self.up1 = segmentation_model.up1
        self.dec1 = segmentation_model.dec1
        self.segmentation_head = segmentation_model.segmentation_head

    @staticmethod
    def _resolve_checkpoint_path(checkpoint_path: str) -> str:
        if os.path.isabs(checkpoint_path):
            return checkpoint_path

        if os.path.exists(checkpoint_path):
            return checkpoint_path

        model_dir = os.path.dirname(__file__)
        candidate = os.path.normpath(
            os.path.join(model_dir, "..", "checkpoints", os.path.basename(checkpoint_path))
        )
        return candidate

    @staticmethod
    def _extract_state_dict(checkpoint):
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            return checkpoint["state_dict"]
        return checkpoint

    def _load_checkpoint(self, model: nn.Module, checkpoint_path: str) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = self._extract_state_dict(checkpoint)
        model.load_state_dict(state_dict)

    def _initialize_shared_encoder(self, *encoder_state_dicts) -> None:
        shared_state = {}
        reference_state = encoder_state_dicts[0]

        for key in reference_state:
            values = [state_dict[key] for state_dict in encoder_state_dicts]

            if torch.is_floating_point(values[0]):
                stacked = torch.stack([value.detach().clone() for value in values], dim=0)
                shared_state[key] = stacked.mean(dim=0)
            else:
                shared_state[key] = values[0]

        self.encoder.load_state_dict(shared_state)

    @staticmethod
    def _match_size(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != ref.shape[-2:]:
            x = F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)
        return x

    def forward(self, x: torch.Tensor):
        """
        Forward pass.

        Args:
            x: Input tensor of shape [B, in_channels, H, W].

        Returns:
            A dict with keys:
            - 'classification': [B, num_breeds] logits tensor.
            - 'localization': [B, 4] bounding box tensor.
            - 'segmentation': [B, seg_classes, H, W] segmentation logits tensor.
        """
        bottleneck, features = self.encoder(x, return_features=True)

        cls_feat = self.classification_avgpool(bottleneck)
        cls_feat = torch.flatten(cls_feat, 1)
        classification_logits = self.classifier_head(cls_feat)

        loc_feat = self.localization_avgpool(bottleneck)
        loc_feat = torch.flatten(loc_feat, 1)
        localization_raw = torch.sigmoid(self.localizer_head(loc_feat))

        _, _, input_h, input_w = x.shape
        localization_box = torch.stack(
            [
                localization_raw[:, 0] * input_w,
                localization_raw[:, 1] * input_h,
                localization_raw[:, 2] * input_w,
                localization_raw[:, 3] * input_h,
            ],
            dim=1,
        )

        s1 = features["stage1"]
        s2 = features["stage2"]
        s3 = features["stage3"]
        s4 = features["stage4"]
        s5 = features["stage5"]

        seg = self.up5(bottleneck)
        seg = self._match_size(seg, s5)
        seg = torch.cat([seg, s5], dim=1)
        seg = self.dec5(seg)

        seg = self.up4(seg)
        seg = self._match_size(seg, s4)
        seg = torch.cat([seg, s4], dim=1)
        seg = self.dec4(seg)

        seg = self.up3(seg)
        seg = self._match_size(seg, s3)
        seg = torch.cat([seg, s3], dim=1)
        seg = self.dec3(seg)

        seg = self.up2(seg)
        seg = self._match_size(seg, s2)
        seg = torch.cat([seg, s2], dim=1)
        seg = self.dec2(seg)

        seg = self.up1(seg)
        seg = self._match_size(seg, s1)
        seg = torch.cat([seg, s1], dim=1)
        seg = self.dec1(seg)

        segmentation_logits = self.segmentation_head(seg)

        return {
            "classification": classification_logits,
            "localization": localization_box,
            "segmentation": segmentation_logits,
        }
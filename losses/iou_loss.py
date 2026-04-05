"""Custom IoU loss 
"""

import torch
import torch.nn as nn


class IoULoss(nn.Module):
    """IoU loss for bounding box regression.
    """

    def __init__(self, eps: float = 1e-6, reduction: str = "mean"):
        """
        Initialize the IoULoss module.
        Args:
            eps: Small value to avoid division by zero.
            reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
        """
        super().__init__()
        self.eps = eps
        self.reduction = reduction

        if self.reduction not in {"none", "mean", "sum"}:
            raise ValueError(
                f"reduction must be one of 'none', 'mean', or 'sum', got {self.reduction}"
            )

    @staticmethod
    def _xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
        """Convert boxes from (xc, yc, w, h) to (x1, y1, x2, y2)."""
        x_center = boxes[:, 0]
        y_center = boxes[:, 1]
        width = boxes[:, 2].clamp(min=0)
        height = boxes[:, 3].clamp(min=0)

        x1 = x_center - width / 2.0
        y1 = y_center - height / 2.0
        x2 = x_center + width / 2.0
        y2 = y_center + height / 2.0

        return torch.stack([x1, y1, x2, y2], dim=1)

    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """Compute IoU loss between predicted and target bounding boxes.
        Args:
            pred_boxes: [B, 4] predicted boxes in (x_center, y_center, width, height) format.
            target_boxes: [B, 4] target boxes in (x_center, y_center, width, height) format.
        """
        if pred_boxes.ndim != 2 or target_boxes.ndim != 2:
            raise ValueError("pred_boxes and target_boxes must have shape [B, 4]")
        if pred_boxes.shape != target_boxes.shape or pred_boxes.shape[1] != 4:
            raise ValueError(
                f"pred_boxes and target_boxes must both have shape [B, 4], "
                f"got {pred_boxes.shape} and {target_boxes.shape}"
            )

        pred_xyxy = self._xywh_to_xyxy(pred_boxes)
        target_xyxy = self._xywh_to_xyxy(target_boxes)

        x1 = torch.maximum(pred_xyxy[:, 0], target_xyxy[:, 0])
        y1 = torch.maximum(pred_xyxy[:, 1], target_xyxy[:, 1])
        x2 = torch.minimum(pred_xyxy[:, 2], target_xyxy[:, 2])
        y2 = torch.minimum(pred_xyxy[:, 3], target_xyxy[:, 3])

        inter_w = (x2 - x1).clamp(min=0)
        inter_h = (y2 - y1).clamp(min=0)
        inter_area = inter_w * inter_h

        pred_area = (
            (pred_xyxy[:, 2] - pred_xyxy[:, 0]).clamp(min=0) *
            (pred_xyxy[:, 3] - pred_xyxy[:, 1]).clamp(min=0)
        )
        target_area = (
            (target_xyxy[:, 2] - target_xyxy[:, 0]).clamp(min=0) *
            (target_xyxy[:, 3] - target_xyxy[:, 1]).clamp(min=0)
        )

        union_area = pred_area + target_area - inter_area
        iou = inter_area / (union_area + self.eps)

        loss = 1.0 - iou

        if self.reduction == "none":
            return loss
        if self.reduction == "sum":
            return loss.sum()
        return loss.mean()
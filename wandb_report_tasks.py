
"""W&B experiment runner for Assignment-2 tasks 2.1 to 2.8.

Assumptions:
1. Uses project modules:
   - data.pets_dataset.OxfordIIITPetDataset
   - models.classification.VGG11Classifier
   - models.localization.VGG11Localizer
   - models.segmentation.VGG11UNet
   - models.multitask.MultiTaskPerceptionModel
2. For Task 2.1, the main assignment VGG11 uses BatchNorm. This script defines a
   report-only "no BatchNorm" classifier for the required comparison.
3. For Task 2.5, the localization model predicts only [xc, yc, w, h] and has no
   explicit confidence head. This script logs a confidence proxy based on MC-dropout
   predictive stability.
4. For Task 2.7, you must pass three novel image paths via --novel_image_paths.
5. Task 2.8's written reflection cannot be authored automatically; this script logs
   the plots/tables needed for the reflection section.
"""

import argparse
import math
import os
import time
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader, Subset

from data.pets_dataset import OxfordIIITPetDataset
from losses.iou_loss import IoULoss
from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.multitask import MultiTaskPerceptionModel
from models.segmentation import VGG11UNet
from models.layers import CustomDropout


INPUT_IMAGE_SIZE = 224
NUM_CLASSES = 37
NUM_SEG_CLASSES = 3


# -----------------------------
# Simple tensor transforms
# -----------------------------
class ComposeTensor:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        for transform in self.transforms:
            image = transform(image)
        return image


class NormalizeTensor:
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(-1, 1, 1)

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        return (image - self.mean) / self.std


class RandomHorizontalFlipTensor:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() < self.p:
            return torch.flip(image, dims=[2])
        return image


def get_classification_transforms(train: bool = True):
    normalize = NormalizeTensor(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    if train:
        return ComposeTensor([RandomHorizontalFlipTensor(0.5), normalize])
    return ComposeTensor([normalize])


def get_eval_transform():
    return NormalizeTensor(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )


# -----------------------------
# Utility
# -----------------------------
def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def load_checkpoint_state(path: str):
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    return checkpoint


def save_checkpoint(model, path: str, epoch: int, metric: float):
    ensure_dir(str(Path(path).parent))
    torch.save(
        {
            "state_dict": model.state_dict(),
            "epoch": epoch,
            "best_metric": metric,
        },
        path,
    )


def xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    x_center = boxes[:, 0]
    y_center = boxes[:, 1]
    width = boxes[:, 2].clamp(min=0)
    height = boxes[:, 3].clamp(min=0)
    x1 = x_center - width / 2.0
    y1 = y_center - height / 2.0
    x2 = x_center + width / 2.0
    y2 = y_center + height / 2.0
    return torch.stack([x1, y1, x2, y2], dim=1)


def compute_batch_iou(pred_boxes: torch.Tensor, target_boxes: torch.Tensor, eps: float = 1e-6):
    pred_xyxy = xywh_to_xyxy(pred_boxes)
    target_xyxy = xywh_to_xyxy(target_boxes)

    x1 = torch.maximum(pred_xyxy[:, 0], target_xyxy[:, 0])
    y1 = torch.maximum(pred_xyxy[:, 1], target_xyxy[:, 1])
    x2 = torch.minimum(pred_xyxy[:, 2], target_xyxy[:, 2])
    y2 = torch.minimum(pred_xyxy[:, 3], target_xyxy[:, 3])

    inter_w = (x2 - x1).clamp(min=0)
    inter_h = (y2 - y1).clamp(min=0)
    inter_area = inter_w * inter_h

    pred_area = (
        (pred_xyxy[:, 2] - pred_xyxy[:, 0]).clamp(min=0)
        * (pred_xyxy[:, 3] - pred_xyxy[:, 1]).clamp(min=0)
    )
    target_area = (
        (target_xyxy[:, 2] - target_xyxy[:, 0]).clamp(min=0)
        * (target_xyxy[:, 3] - target_xyxy[:, 1]).clamp(min=0)
    )

    union_area = pred_area + target_area - inter_area
    return inter_area / (union_area + eps)


def compute_segmentation_metrics(logits: torch.Tensor, masks: torch.Tensor, num_classes: int = NUM_SEG_CLASSES):
    preds = torch.argmax(logits, dim=1)

    pixel_acc = (preds == masks).sum().item() / masks.numel()

    ious = []
    dices = []
    for cls in range(num_classes):
        pred_c = preds == cls
        mask_c = masks == cls

        intersection = (pred_c & mask_c).sum().item()
        union = (pred_c | mask_c).sum().item()
        denom_dice = pred_c.sum().item() + mask_c.sum().item()

        if union > 0:
            ious.append(intersection / union)
        if denom_dice > 0:
            dices.append((2.0 * intersection) / denom_dice)

    mean_iou = float(sum(ious) / len(ious)) if ious else 0.0
    mean_dice = float(sum(dices) / len(dices)) if dices else 0.0
    return pixel_acc, mean_iou, mean_dice


def tensor_to_numpy_image(image_tensor: torch.Tensor):
    image = image_tensor.detach().cpu().permute(1, 2, 0).numpy()
    image = np.clip(image, 0.0, 1.0)
    return image


def denormalize_image(image_tensor: torch.Tensor):
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=image_tensor.dtype, device=image_tensor.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=image_tensor.dtype, device=image_tensor.device).view(3, 1, 1)
    return torch.clamp(image_tensor * std + mean, 0.0, 1.0)


def colorize_mask(mask: np.ndarray):
    out = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    out[mask == 0] = np.array([0, 0, 0], dtype=np.uint8)
    out[mask == 1] = np.array([0, 255, 0], dtype=np.uint8)
    out[mask == 2] = np.array([255, 0, 0], dtype=np.uint8)
    return out


def make_box_overlay(image_np: np.ndarray, gt_xyxy=None, pred_xyxy=None, title=None):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(image_np)
    if gt_xyxy is not None:
        x1, y1, x2, y2 = gt_xyxy
        rect = patches.Rectangle((x1, y1), max(1.0, x2 - x1), max(1.0, y2 - y1),
                                 linewidth=2, edgecolor="green", facecolor="none")
        ax.add_patch(rect)
    if pred_xyxy is not None:
        x1, y1, x2, y2 = pred_xyxy
        rect = patches.Rectangle((x1, y1), max(1.0, x2 - x1), max(1.0, y2 - y1),
                                 linewidth=2, edgecolor="red", facecolor="none")
        ax.add_patch(rect)
    if title is not None:
        ax.set_title(title)
    ax.axis("off")
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)[..., :3]
    plt.close(fig)
    return image


def make_segmentation_triplet(image_np: np.ndarray, gt_mask: np.ndarray, pred_mask: np.ndarray, title=None):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image_np)
    axes[0].set_title("Original")
    axes[1].imshow(colorize_mask(gt_mask))
    axes[1].set_title("GT Trimap")
    axes[2].imshow(colorize_mask(pred_mask))
    axes[2].set_title("Pred Trimap")
    if title is not None:
        fig.suptitle(title)
    for ax in axes:
        ax.axis("off")
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)[..., :3]
    plt.close(fig)
    return image


def make_pipeline_showcase(image_np: np.ndarray, mask_np: np.ndarray, box_xyxy, title_text):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(image_np)
    x1, y1, x2, y2 = box_xyxy
    rect = patches.Rectangle((x1, y1), max(1.0, x2 - x1), max(1.0, y2 - y1),
                             linewidth=2, edgecolor="red", facecolor="none")
    axes[0].add_patch(rect)
    axes[0].set_title(title_text)
    axes[0].axis("off")

    axes[1].imshow(image_np)
    axes[1].imshow(colorize_mask(mask_np), alpha=0.45)
    axes[1].set_title("Segmentation Overlay")
    axes[1].axis("off")

    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)[..., :3]
    plt.close(fig)
    return image


# -----------------------------
# Dataset builders
# -----------------------------
def build_train_val_loaders(
    data_root,
    task,
    batch_size,
    num_workers,
    seed,
    val_split=0.1,
    max_train_samples=None,
    max_val_samples=None,
):
    train_transform = get_classification_transforms(train=True) if task == "classification" else ComposeTensor([get_eval_transform()])
    eval_transform = ComposeTensor([get_eval_transform()])

    full_train = OxfordIIITPetDataset(
        root=data_root,
        split="trainval",
        task=task,
        transform=train_transform,
        image_size=INPUT_IMAGE_SIZE,
    )
    full_val = OxfordIIITPetDataset(
        root=data_root,
        split="trainval",
        task=task,
        transform=eval_transform,
        image_size=INPUT_IMAGE_SIZE,
    )

    total = len(full_train)
    val_size = max(1, int(total * val_split))
    val_size = min(val_size, total - 1)

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(total, generator=generator).tolist()
    train_indices = indices[:-val_size]
    val_indices = indices[-val_size:]

    if max_train_samples is not None:
        train_indices = train_indices[:max_train_samples]
    if max_val_samples is not None:
        val_indices = val_indices[:max_val_samples]

    train_ds = Subset(full_train, train_indices)
    val_ds = Subset(full_val, val_indices)

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, val_loader


def build_test_loader(data_root, task, batch_size, num_workers, max_samples=None):
    dataset = OxfordIIITPetDataset(
        root=data_root,
        split="test",
        task=task,
        transform=ComposeTensor([get_eval_transform()]),
        image_size=INPUT_IMAGE_SIZE,
    )
    if max_samples is not None:
        dataset = Subset(dataset, list(range(min(len(dataset), max_samples))))
    pin_memory = torch.cuda.is_available()
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)


# -----------------------------
# Report-only no-BN models for task 2.1
# -----------------------------
class ReportVGG11NoBN(nn.Module):
    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool2d(2, 2)

        self.stage2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool2d(2, 2)

        self.stage3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool3 = nn.MaxPool2d(2, 2)

        self.stage4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool4 = nn.MaxPool2d(2, 2)

        self.stage5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool5 = nn.MaxPool2d(2, 2)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.pool1(self.stage1(x))
        x = self.pool2(self.stage2(x))
        x = self.pool3(self.stage3(x))
        x = self.pool4(self.stage4(x))
        x = self.pool5(self.stage5(x))
        return x


class ReportClassifierNoBN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, dropout_p=0.5):
        super().__init__()
        self.encoder = ReportVGG11NoBN()
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(4096, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.encoder(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


# -----------------------------
# Training / evaluation
# -----------------------------
def train_one_epoch_classification(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, total_correct, total_count = 0.0, 0, 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch = labels.size(0)
        total_loss += loss.item() * batch
        total_correct += (torch.argmax(logits, dim=1) == labels).sum().item()
        total_count += batch
    return total_loss / total_count, total_correct / total_count


@torch.no_grad()
def evaluate_classification(model, loader, criterion, device):
    model.eval()
    total_loss, total_correct, total_count = 0.0, 0, 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)

        batch = labels.size(0)
        total_loss += loss.item() * batch
        total_correct += (torch.argmax(logits, dim=1) == labels).sum().item()
        total_count += batch
    return total_loss / total_count, total_correct / total_count


def train_one_epoch_localization(model, loader, mse_criterion, iou_criterion, optimizer, device):
    model.train()
    total_loss, total_mse, total_iou_loss, total_iou, total_count = 0.0, 0.0, 0.0, 0.0, 0
    for images, boxes in loader:
        images = images.to(device)
        boxes = boxes.to(device)

        optimizer.zero_grad()
        pred_boxes = model(images)
        mse_loss = mse_criterion(pred_boxes, boxes)
        iou_loss = iou_criterion(pred_boxes, boxes)
        loss = mse_loss + iou_loss
        loss.backward()
        optimizer.step()

        batch = images.size(0)
        total_loss += loss.item() * batch
        total_mse += mse_loss.item() * batch
        total_iou_loss += iou_loss.item() * batch
        total_iou += compute_batch_iou(pred_boxes, boxes).sum().item()
        total_count += batch

    return total_loss / total_count, total_mse / total_count, total_iou_loss / total_count, total_iou / total_count


@torch.no_grad()
def evaluate_localization(model, loader, mse_criterion, iou_criterion, device):
    model.eval()
    total_loss, total_mse, total_iou_loss, total_iou, total_count = 0.0, 0.0, 0.0, 0.0, 0
    for images, boxes in loader:
        images = images.to(device)
        boxes = boxes.to(device)

        pred_boxes = model(images)
        mse_loss = mse_criterion(pred_boxes, boxes)
        iou_loss = iou_criterion(pred_boxes, boxes)
        loss = mse_loss + iou_loss

        batch = images.size(0)
        total_loss += loss.item() * batch
        total_mse += mse_loss.item() * batch
        total_iou_loss += iou_loss.item() * batch
        total_iou += compute_batch_iou(pred_boxes, boxes).sum().item()
        total_count += batch

    return total_loss / total_count, total_mse / total_count, total_iou_loss / total_count, total_iou / total_count


def train_one_epoch_segmentation(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, total_pixel_acc, total_miou, total_dice, total_count = 0.0, 0.0, 0.0, 0.0, 0
    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        batch = images.size(0)
        pixel_acc, mean_iou, mean_dice = compute_segmentation_metrics(logits.detach(), masks)
        total_loss += loss.item() * batch
        total_pixel_acc += pixel_acc * batch
        total_miou += mean_iou * batch
        total_dice += mean_dice * batch
        total_count += batch

    return (
        total_loss / total_count,
        total_pixel_acc / total_count,
        total_miou / total_count,
        total_dice / total_count,
    )


@torch.no_grad()
def evaluate_segmentation(model, loader, criterion, device):
    model.eval()
    total_loss, total_pixel_acc, total_miou, total_dice, total_count = 0.0, 0.0, 0.0, 0.0, 0
    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)
        loss = criterion(logits, masks)

        batch = images.size(0)
        pixel_acc, mean_iou, mean_dice = compute_segmentation_metrics(logits, masks)
        total_loss += loss.item() * batch
        total_pixel_acc += pixel_acc * batch
        total_miou += mean_iou * batch
        total_dice += mean_dice * batch
        total_count += batch

    return (
        total_loss / total_count,
        total_pixel_acc / total_count,
        total_miou / total_count,
        total_dice / total_count,
    )


def get_conv_layers(model):
    return [module for module in model.modules() if isinstance(module, nn.Conv2d)]


def capture_conv_activation(model, input_batch, conv_index=2):
    conv_layers = get_conv_layers(model)
    if conv_index >= len(conv_layers):
        raise ValueError(f"Model has only {len(conv_layers)} conv layers, cannot capture index {conv_index}")
    activation = {}

    def hook_fn(_, __, output):
        activation["tensor"] = output.detach().cpu()

    handle = conv_layers[conv_index].register_forward_hook(hook_fn)
    model.eval()
    with torch.no_grad():
        _ = model(input_batch)
    handle.remove()
    return activation["tensor"]


def plot_activation_histogram(act_a, act_b, label_a, label_b, title):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(act_a.flatten(), bins=100, alpha=0.6, label=label_a, density=True)
    ax.hist(act_b.flatten(), bins=100, alpha=0.6, label=label_b, density=True)
    ax.set_title(title)
    ax.set_xlabel("Activation value")
    ax.set_ylabel("Density")
    ax.legend()
    return fig


def init_run(args, name, job_type, group=None, config_extra=None):
    config = {
        "data_root": args.data_root,
        "input_image_size": INPUT_IMAGE_SIZE,
        "device": str(args.device),
    }
    if config_extra is not None:
        config.update(config_extra)
    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=name,
        job_type=job_type,
        group=group,
        config=config,
        reinit=True,
        mode=args.wandb_mode,
    )


def finish_run(run):
    if run is not None:
        wandb.finish()


def is_finite_number(x):
    return math.isfinite(float(x))


# -----------------------------
# Task 2.1
# -----------------------------
def run_task_21(args):
    summary = {}

    train_loader, val_loader = build_train_val_loaders(
        args.data_root,
        task="classification",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        val_split=args.val_split,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
    )

    models = {
        "with_bn": VGG11Classifier(num_classes=NUM_CLASSES, dropout_p=args.dropout_p_task21).to(args.device),
        "without_bn": ReportClassifierNoBN(num_classes=NUM_CLASSES, dropout_p=args.dropout_p_task21).to(args.device),
    }

    activation_batch = next(iter(val_loader))[0][: min(8, args.batch_size)].to(args.device)

    criterion = nn.CrossEntropyLoss()
    lr_candidates = [float(v) for v in args.task21_lr_candidates.split(",")]

    stable_lr_table = []

    for model_name, model in models.items():
        run = init_run(
            args,
            name=f"task2_1_{model_name}",
            job_type="task2_1",
            group="task2_1_bn_comparison",
            config_extra={
                "subtask": "2.1",
                "model_variant": model_name,
                "epochs": args.task21_epochs,
                "dropout_p": args.dropout_p_task21,
            },
        )
        try:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            best_val_acc = -1.0

            for epoch in range(args.task21_epochs):
                train_loss, train_acc = train_one_epoch_classification(model, train_loader, criterion, optimizer, args.device)
                val_loss, val_acc = evaluate_classification(model, val_loader, criterion, args.device)

                wandb.log(
                    {
                        "epoch": epoch + 1,
                        "train/loss": train_loss,
                        "train/accuracy": train_acc,
                        "val/loss": val_loss,
                        "val/accuracy": val_acc,
                        "learning_rate": optimizer.param_groups[0]["lr"],
                    }
                )

                if val_acc > best_val_acc:
                    best_val_acc = val_acc

            act = capture_conv_activation(model, activation_batch, conv_index=2)
            wandb.log(
                {
                    "activations/conv3_mean": float(act.mean().item()),
                    "activations/conv3_std": float(act.std().item()),
                }
            )
            summary[f"task21_{model_name}_best_val_acc"] = best_val_acc
            summary[f"task21_{model_name}_conv3_mean"] = float(act.mean().item())
            summary[f"task21_{model_name}_conv3_std"] = float(act.std().item())

            # quick stable LR sweep
            for lr in lr_candidates:
                test_model = VGG11Classifier(num_classes=NUM_CLASSES, dropout_p=args.dropout_p_task21).to(args.device) if model_name == "with_bn" else ReportClassifierNoBN(num_classes=NUM_CLASSES, dropout_p=args.dropout_p_task21).to(args.device)
                test_optimizer = torch.optim.Adam(test_model.parameters(), lr=lr, weight_decay=args.weight_decay)
                stable = True
                final_loss = None
                try:
                    for _ in range(args.task21_stability_epochs):
                        tr_loss, _ = train_one_epoch_classification(test_model, train_loader, criterion, test_optimizer, args.device)
                        final_loss = tr_loss
                        if (not is_finite_number(tr_loss)) or tr_loss > args.task21_divergence_threshold:
                            stable = False
                            break
                except Exception:
                    stable = False
                stable_lr_table.append([model_name, lr, int(stable), float(final_loss) if final_loss is not None else float("nan")])

            wandb.summary["best_val_accuracy"] = best_val_acc
        finally:
            finish_run(run)

    # joint activation histogram
    act_bn = capture_conv_activation(models["with_bn"], activation_batch, conv_index=2)
    act_no_bn = capture_conv_activation(models["without_bn"], activation_batch, conv_index=2)
    run = init_run(
        args,
        name="task2_1_activation_hist_and_lr_summary",
        job_type="task2_1",
        group="task2_1_bn_comparison",
        config_extra={"subtask": "2.1", "summary_run": True},
    )
    try:
        fig = plot_activation_histogram(
            act_bn.numpy(),
            act_no_bn.numpy(),
            "With BatchNorm",
            "Without BatchNorm",
            "Task 2.1: 3rd convolution activation distribution",
        )
        wandb.log({"task2_1/activation_histogram": wandb.Image(fig)})
        plt.close(fig)

        lr_table = wandb.Table(columns=["model_variant", "learning_rate", "stable", "final_train_loss"])
        for row in stable_lr_table:
            lr_table.add_data(*row)
        wandb.log({"task2_1/stable_lr_table": lr_table})

        max_stable = {}
        for model_name in ["with_bn", "without_bn"]:
            stable_lrs = [row[1] for row in stable_lr_table if row[0] == model_name and row[2] == 1]
            max_stable[model_name] = max(stable_lrs) if stable_lrs else None
            if max_stable[model_name] is not None:
                summary[f"task21_{model_name}_max_stable_lr"] = max_stable[model_name]

        wandb.summary.update(summary)
    finally:
        finish_run(run)

    return summary


# -----------------------------
# Task 2.2
# -----------------------------
def run_task_22(args):
    summary = {}
    train_loader, val_loader = build_train_val_loaders(
        args.data_root,
        task="classification",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        val_split=args.val_split,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
    )

    criterion = nn.CrossEntropyLoss()
    dropout_values = [0.0, 0.2, 0.5]
    final_table_rows = []

    for p in dropout_values:
        run = init_run(
            args,
            name=f"task2_2_dropout_{str(p).replace('.', '_')}",
            job_type="task2_2",
            group="task2_2_internal_dynamics",
            config_extra={
                "subtask": "2.2",
                "dropout_p": p,
                "epochs": args.task22_epochs,
            },
        )
        try:
            model = VGG11Classifier(num_classes=NUM_CLASSES, dropout_p=p).to(args.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            best_val_acc = -1.0
            best_val_loss = None

            for epoch in range(args.task22_epochs):
                train_loss, train_acc = train_one_epoch_classification(model, train_loader, criterion, optimizer, args.device)
                val_loss, val_acc = evaluate_classification(model, val_loader, criterion, args.device)

                wandb.log(
                    {
                        "epoch": epoch + 1,
                        "train/loss": train_loss,
                        "train/accuracy": train_acc,
                        "val/loss": val_loss,
                        "val/accuracy": val_acc,
                    }
                )

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_val_loss = val_loss

            final_table_rows.append([p, best_val_acc, best_val_loss])
            summary[f"task22_dropout_{p}_best_val_acc"] = best_val_acc
            summary[f"task22_dropout_{p}_best_val_loss"] = best_val_loss
        finally:
            finish_run(run)

    run = init_run(args, name="task2_2_summary", job_type="task2_2", group="task2_2_internal_dynamics", config_extra={"subtask": "2.2"})
    try:
        table = wandb.Table(columns=["dropout_p", "best_val_accuracy", "best_val_loss"])
        for row in final_table_rows:
            table.add_data(*row)
        wandb.log({"task2_2/summary_table": table})
        wandb.summary.update(summary)
    finally:
        finish_run(run)

    return summary


# -----------------------------
# Task 2.3
# -----------------------------
def set_encoder_trainability(unet_model: VGG11UNet, strategy: str):
    if strategy == "strict_feature_extractor":
        for param in unet_model.encoder.parameters():
            param.requires_grad = False

    elif strategy == "partial_fine_tuning":
        # freeze early blocks, fine-tune later blocks
        for name, module in unet_model.encoder.named_children():
            freeze = name in {"stage1", "pool1", "stage2", "pool2", "stage3", "pool3"}
            for param in module.parameters():
                param.requires_grad = not freeze

    elif strategy == "full_fine_tuning":
        for param in unet_model.encoder.parameters():
            param.requires_grad = True

    else:
        raise ValueError(f"Unknown transfer strategy: {strategy}")


def load_encoder_from_classifier(unet_model: VGG11UNet, classifier_ckpt_path: str):
    if not os.path.exists(classifier_ckpt_path):
        return False
    classifier_model = VGG11Classifier(num_classes=NUM_CLASSES)
    classifier_model.load_state_dict(load_checkpoint_state(classifier_ckpt_path))
    unet_model.encoder.load_state_dict(classifier_model.encoder.state_dict())
    return True


def run_task_23(args):
    summary = {}
    train_loader, val_loader = build_train_val_loaders(
        args.data_root,
        task="segmentation",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        val_split=args.val_split,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
    )

    criterion = nn.CrossEntropyLoss()
    strategies = [
        "strict_feature_extractor",
        "partial_fine_tuning",
        "full_fine_tuning",
    ]
    final_rows = []

    for strategy in strategies:
        run = init_run(
            args,
            name=f"task2_3_{strategy}",
            job_type="task2_3",
            group="task2_3_transfer_learning",
            config_extra={
                "subtask": "2.3",
                "transfer_strategy": strategy,
                "epochs": args.task23_epochs,
                "classifier_ckpt": args.classifier_ckpt,
            },
        )
        try:
            model = VGG11UNet(in_channels=3, num_classes=NUM_SEG_CLASSES).to(args.device)
            pretrained_loaded = load_encoder_from_classifier(model, args.classifier_ckpt)
            set_encoder_trainability(model, strategy)

            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=args.lr,
                weight_decay=args.weight_decay,
            )

            best_val_dice = -1.0
            best_val_pixel_acc = -1.0
            best_epoch_time = None

            for epoch in range(args.task23_epochs):
                start = time.time()
                train_loss, train_pixel_acc, train_miou, train_dice = train_one_epoch_segmentation(
                    model, train_loader, criterion, optimizer, args.device
                )
                val_loss, val_pixel_acc, val_miou, val_dice = evaluate_segmentation(
                    model, val_loader, criterion, args.device
                )
                epoch_time = time.time() - start
                if best_epoch_time is None or epoch_time < best_epoch_time:
                    best_epoch_time = epoch_time

                wandb.log(
                    {
                        "epoch": epoch + 1,
                        "train/loss": train_loss,
                        "train/pixel_accuracy": train_pixel_acc,
                        "train/miou": train_miou,
                        "train/dice": train_dice,
                        "val/loss": val_loss,
                        "val/pixel_accuracy": val_pixel_acc,
                        "val/miou": val_miou,
                        "val/dice": val_dice,
                        "epoch_time_seconds": epoch_time,
                        "pretrained_loaded": int(pretrained_loaded),
                    }
                )

                if val_dice > best_val_dice:
                    best_val_dice = val_dice
                if val_pixel_acc > best_val_pixel_acc:
                    best_val_pixel_acc = val_pixel_acc

            final_rows.append([strategy, best_val_pixel_acc, best_val_dice, best_epoch_time])
            summary[f"task23_{strategy}_best_val_pixel_acc"] = best_val_pixel_acc
            summary[f"task23_{strategy}_best_val_dice"] = best_val_dice
            summary[f"task23_{strategy}_best_epoch_time_seconds"] = best_epoch_time
        finally:
            finish_run(run)

    run = init_run(args, name="task2_3_summary", job_type="task2_3", group="task2_3_transfer_learning", config_extra={"subtask": "2.3"})
    try:
        table = wandb.Table(columns=["strategy", "best_val_pixel_acc", "best_val_dice", "best_epoch_time_seconds"])
        for row in final_rows:
            table.add_data(*row)
        wandb.log({"task2_3/summary_table": table})
        wandb.summary.update(summary)
    finally:
        finish_run(run)

    return summary


# -----------------------------
# Task 2.4
# -----------------------------
def load_classification_model(checkpoint_path: str, device: torch.device, dropout_p: float = 0.5):
    model = VGG11Classifier(num_classes=NUM_CLASSES, dropout_p=dropout_p).to(device)
    model.load_state_dict(load_checkpoint_state(checkpoint_path))
    model.eval()
    return model


def run_task_24(args):
    if args.feature_map_image is None:
        raise ValueError("Task 2.4 needs --feature_map_image pointing to a dog image.")

    run = init_run(
        args,
        name="task2_4_feature_maps",
        job_type="task2_4",
        group="task2_4_feature_maps",
        config_extra={
            "subtask": "2.4",
            "feature_map_image": args.feature_map_image,
            "classifier_ckpt": args.classifier_ckpt,
        },
    )
    summary = {}
    try:
        model = load_classification_model(args.classifier_ckpt, args.device, dropout_p=args.dropout_p_task21)

        image_np = mpimg.imread(args.feature_map_image).astype(np.float32)
        if image_np.ndim == 2:
            image_np = np.stack([image_np, image_np, image_np], axis=-1)
        if image_np.shape[-1] == 4:
            image_np = image_np[..., :3]
        if image_np.max() > 1.0:
            image_np = image_np / 255.0

        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float().unsqueeze(0)
        image_tensor = F.interpolate(image_tensor, size=(INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE), mode="bilinear", align_corners=False)
        image_tensor = get_eval_transform()(image_tensor.squeeze(0)).unsqueeze(0).to(args.device)

        conv_layers = get_conv_layers(model)
        first_conv = conv_layers[0]
        last_conv = conv_layers[-1]
        captured = {}

        def hook_first(_, __, output):
            captured["first"] = output.detach().cpu()

        def hook_last(_, __, output):
            captured["last"] = output.detach().cpu()

        h1 = first_conv.register_forward_hook(hook_first)
        h2 = last_conv.register_forward_hook(hook_last)
        with torch.no_grad():
            _ = model(image_tensor)
        h1.remove()
        h2.remove()

        rows = []
        for key, fmap in [("first_conv", captured["first"]), ("last_conv", captured["last"])]:
            fmap = fmap[0]  # [C,H,W]
            channel_count = min(args.feature_map_channels_to_log, fmap.shape[0])

            for channel_idx in range(channel_count):
                channel = fmap[channel_idx].numpy()
                channel = (channel - channel.min()) / (channel.max() - channel.min() + 1e-8)
                rows.append([key, channel_idx, wandb.Image(channel)])

            mean_map = fmap.mean(dim=0).numpy()
            mean_map = (mean_map - mean_map.min()) / (mean_map.max() - mean_map.min() + 1e-8)
            wandb.log({f"task2_4/{key}_mean_map": wandb.Image(mean_map)})

            summary[f"task24_{key}_channels_logged"] = channel_count

        table = wandb.Table(columns=["layer", "channel_index", "feature_map"])
        for row in rows:
            table.add_data(*row)
        wandb.log({"task2_4/feature_maps_table": table})
        wandb.summary.update(summary)
    finally:
        finish_run(run)
    return summary


# -----------------------------
# Task 2.5
# -----------------------------
def load_localization_model(checkpoint_path: str, device: torch.device, dropout_p: float = 0.5):
    model = VGG11Localizer(dropout_p=dropout_p).to(device)
    model.load_state_dict(load_checkpoint_state(checkpoint_path))
    return model


def mc_dropout_confidence(model, image_tensor, passes=8):
    # Assumption: model has no explicit objectness/confidence head.
    # We use predictive stability under MC dropout as a 0-1 confidence proxy.
    model.train()
    preds = []
    with torch.no_grad():
        for _ in range(passes):
            preds.append(model(image_tensor).detach().cpu())
    preds = torch.stack(preds, dim=0)  # [T, B, 4]
    mean_box = preds.mean(dim=0)
    std_box = preds.std(dim=0)

    norm = torch.tensor([INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE], dtype=std_box.dtype)
    uncertainty = (std_box / norm).mean(dim=1)
    confidence = torch.exp(-5.0 * uncertainty).clamp(0.0, 1.0)
    return mean_box, confidence


def run_task_25(args):
    run = init_run(
        args,
        name="task2_5_localization_table",
        job_type="task2_5",
        group="task2_5_detection",
        config_extra={
            "subtask": "2.5",
            "localizer_ckpt": args.localizer_ckpt,
            "mc_dropout_passes": args.task25_mc_passes,
        },
    )
    summary = {}
    try:
        loader = build_test_loader(
            args.data_root,
            task="localization",
            batch_size=1,
            num_workers=args.num_workers,
            max_samples=max(10, args.task25_num_images),
        )
        model = load_localization_model(args.localizer_ckpt, args.device, dropout_p=args.dropout_p_task21)
        mse_criterion = nn.MSELoss()
        iou_criterion = IoULoss(reduction="mean")

        table = wandb.Table(columns=["index", "overlay", "confidence_proxy", "iou", "mse", "failure_case"])

        iou_values = []
        confidence_values = []

        for idx, (images, boxes) in enumerate(loader):
            if idx >= args.task25_num_images:
                break

            images = images.to(args.device)
            boxes = boxes.to(args.device)

            mean_box, confidence = mc_dropout_confidence(model, images, passes=args.task25_mc_passes)
            pred_box = mean_box.to(args.device)
            iou = compute_batch_iou(pred_box, boxes)[0].item()
            mse = mse_criterion(pred_box, boxes).item()

            image_np = tensor_to_numpy_image(denormalize_image(images[0]).cpu())
            gt_xyxy = xywh_to_xyxy(boxes.detach().cpu())[0].numpy().tolist()
            pred_xyxy = xywh_to_xyxy(pred_box.detach().cpu())[0].numpy().tolist()

            if iou < 0.1:
                failure_case = "missed_object"
            elif confidence[0].item() < 0.4 and iou < 0.5:
                failure_case = "low_conf_low_iou"
            elif iou < 0.5:
                failure_case = "partial_localization"
            else:
                failure_case = "good"

            overlay = make_box_overlay(
                image_np,
                gt_xyxy=gt_xyxy,
                pred_xyxy=pred_xyxy,
                title=f"conf={confidence[0].item():.3f}, iou={iou:.3f}",
            )

            table.add_data(
                idx,
                wandb.Image(overlay),
                float(confidence[0].item()),
                float(iou),
                float(mse),
                failure_case,
            )

            iou_values.append(iou)
            confidence_values.append(float(confidence[0].item()))

        wandb.log({"task2_5/detection_table": table})
        summary["task25_mean_iou"] = float(np.mean(iou_values)) if iou_values else 0.0
        summary["task25_mean_confidence_proxy"] = float(np.mean(confidence_values)) if confidence_values else 0.0
        wandb.summary.update(summary)
    finally:
        finish_run(run)
    return summary


# -----------------------------
# Task 2.6
# -----------------------------
def load_segmentation_model(checkpoint_path: str, device: torch.device):
    model = VGG11UNet(in_channels=3, num_classes=NUM_SEG_CLASSES).to(device)
    model.load_state_dict(load_checkpoint_state(checkpoint_path))
    model.eval()
    return model


def run_task_26(args):
    run = init_run(
        args,
        name="task2_6_dice_vs_pixel_acc",
        job_type="task2_6",
        group="task2_6_segmentation_eval",
        config_extra={
            "subtask": "2.6",
            "unet_ckpt": args.unet_ckpt,
        },
    )
    summary = {}
    try:
        loader = build_test_loader(
            args.data_root,
            task="segmentation",
            batch_size=1,
            num_workers=args.num_workers,
            max_samples=max(5, args.task26_num_images),
        )
        model = load_segmentation_model(args.unet_ckpt, args.device)

        table = wandb.Table(columns=["index", "triplet", "pixel_accuracy", "dice"])
        pixel_accs = []
        dices = []

        count = 0
        for images, masks in loader:
            if count >= args.task26_num_images:
                break
            images = images.to(args.device)
            masks = masks.to(args.device)

            with torch.no_grad():
                logits = model(images)

            pixel_acc, _, dice = compute_segmentation_metrics(logits, masks)
            pred_mask = torch.argmax(logits, dim=1)[0].cpu().numpy()
            gt_mask = masks[0].cpu().numpy()
            image_np = tensor_to_numpy_image(denormalize_image(images[0]).cpu())

            triplet = make_segmentation_triplet(
                image_np=image_np,
                gt_mask=gt_mask,
                pred_mask=pred_mask,
                title=f"pixel_acc={pixel_acc:.3f}, dice={dice:.3f}",
            )

            table.add_data(count, wandb.Image(triplet), float(pixel_acc), float(dice))
            pixel_accs.append(pixel_acc)
            dices.append(dice)
            count += 1

        wandb.log({"task2_6/sample_triplets": table})

        # scatter plot
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(pixel_accs, dices)
        ax.set_xlabel("Pixel Accuracy")
        ax.set_ylabel("Dice Score")
        ax.set_title("Task 2.6: Dice vs Pixel Accuracy")
        wandb.log({"task2_6/dice_vs_pixel_acc_scatter": wandb.Image(fig)})
        plt.close(fig)

        summary["task26_mean_pixel_accuracy"] = float(np.mean(pixel_accs)) if pixel_accs else 0.0
        summary["task26_mean_dice"] = float(np.mean(dices)) if dices else 0.0
        wandb.summary.update(summary)
    finally:
        finish_run(run)
    return summary


# -----------------------------
# Task 2.7
# -----------------------------
def load_image_for_pipeline(image_path: str):
    image_np = mpimg.imread(image_path).astype(np.float32)
    if image_np.ndim == 2:
        image_np = np.stack([image_np, image_np, image_np], axis=-1)
    if image_np.shape[-1] == 4:
        image_np = image_np[..., :3]
    if image_np.max() > 1.0:
        image_np = image_np / 255.0

    orig_h, orig_w = image_np.shape[:2]
    tensor = torch.from_numpy(image_np).permute(2, 0, 1).float().unsqueeze(0)
    tensor = F.interpolate(tensor, size=(INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE), mode="bilinear", align_corners=False)
    tensor = get_eval_transform()(tensor.squeeze(0)).unsqueeze(0)
    return image_np, tensor, orig_h, orig_w


def scale_box_from_input_to_original(box_xywh, orig_h, orig_w):
    scale_x = orig_w / float(INPUT_IMAGE_SIZE)
    scale_y = orig_h / float(INPUT_IMAGE_SIZE)
    xc, yc, w, h = box_xywh
    return [xc * scale_x, yc * scale_y, w * scale_x, h * scale_y]


def xywh_list_to_xyxy_list(box):
    xc, yc, w, h = box
    return [xc - w / 2.0, yc - h / 2.0, xc + w / 2.0, yc + h / 2.0]


def run_task_27(args):
    if args.novel_image_paths is None or len(args.novel_image_paths) != 3:
        raise ValueError("Task 2.7 needs exactly three paths via --novel_image_paths.")

    run = init_run(
        args,
        name="task2_7_pipeline_showcase",
        job_type="task2_7",
        group="task2_7_pipeline",
        config_extra={
            "subtask": "2.7",
            "novel_images": args.novel_image_paths,
            "classifier_ckpt": args.classifier_ckpt,
            "localizer_ckpt": args.localizer_ckpt,
            "unet_ckpt": args.unet_ckpt,
        },
    )
    summary = {}
    try:
        multitask = MultiTaskPerceptionModel(
            classifier_path=args.classifier_ckpt,
            localizer_path=args.localizer_ckpt,
            unet_path=args.unet_ckpt,
        ).to(args.device)
        multitask.eval()

        class_dataset = OxfordIIITPetDataset(
            root=args.data_root,
            split="trainval",
            task="classification",
            transform=None,
            image_size=INPUT_IMAGE_SIZE,
        )
        class_names = [class_dataset.idx_to_class[i] for i in range(len(class_dataset.idx_to_class))]

        table = wandb.Table(columns=["image_path", "showcase", "top1_label", "bbox_xyxy"])

        for image_path in args.novel_image_paths:
            image_np, tensor, orig_h, orig_w = load_image_for_pipeline(image_path)
            tensor = tensor.to(args.device)

            with torch.no_grad():
                outputs = multitask(tensor)

            cls_logits = outputs["classification"]
            box_xywh = outputs["localization"][0].detach().cpu().tolist()
            seg_logits = outputs["segmentation"]

            top1_idx = int(torch.argmax(cls_logits, dim=1)[0].item())
            top1_label = class_names[top1_idx] if top1_idx < len(class_names) else f"class_{top1_idx}"

            pred_mask = torch.argmax(seg_logits, dim=1)[0].detach().cpu().numpy()
            pred_mask = np.array(ImageResizer.resize_mask_nearest(pred_mask, (orig_h, orig_w)))

            orig_box_xywh = scale_box_from_input_to_original(box_xywh, orig_h, orig_w)
            orig_box_xyxy = xywh_list_to_xyxy_list(orig_box_xywh)

            showcase = make_pipeline_showcase(
                image_np=image_np,
                mask_np=pred_mask,
                box_xyxy=orig_box_xyxy,
                title_text=f"Predicted breed: {top1_label}",
            )

            table.add_data(
                image_path,
                wandb.Image(showcase),
                top1_label,
                [round(v, 2) for v in orig_box_xyxy],
            )

        wandb.log({"task2_7/showcase_table": table})
        summary["task27_num_novel_images_logged"] = len(args.novel_image_paths)
        wandb.summary.update(summary)
    finally:
        finish_run(run)
    return summary


class ImageResizer:
    @staticmethod
    def resize_mask_nearest(mask_np, new_hw):
        tensor = torch.from_numpy(mask_np).float().unsqueeze(0).unsqueeze(0)
        tensor = F.interpolate(tensor, size=new_hw, mode="nearest")
        return tensor.squeeze(0).squeeze(0).long().numpy()


# -----------------------------
# Task 2.8
# -----------------------------
def run_task_28(args, previous_summaries):
    run = init_run(
        args,
        name="task2_8_meta_analysis",
        job_type="task2_8",
        group="task2_8_meta",
        config_extra={"subtask": "2.8"},
    )
    summary = {}
    try:
        table = wandb.Table(columns=["metric_name", "value"])
        for summary_dict in previous_summaries:
            for key, value in summary_dict.items():
                if value is None:
                    continue
                if isinstance(value, (int, float, np.floating)):
                    table.add_data(key, float(value))

        wandb.log({"task2_8/summary_table": table})

        # bar plot for key comparable metrics
        keys = []
        vals = []
        for summary_dict in previous_summaries:
            for key, value in summary_dict.items():
                if isinstance(value, (int, float, np.floating)):
                    if any(token in key for token in ["best_val_acc", "best_val_dice", "mean_iou", "mean_dice", "mean_pixel_accuracy"]):
                        keys.append(key)
                        vals.append(float(value))

        if keys:
            fig, ax = plt.subplots(figsize=(max(8, len(keys) * 0.5), 4))
            ax.bar(range(len(keys)), vals)
            ax.set_xticks(range(len(keys)))
            ax.set_xticklabels(keys, rotation=75, ha="right")
            ax.set_title("Task 2.8: Cross-task metric summary")
            ax.set_ylabel("Value")
            plt.tight_layout()
            wandb.log({"task2_8/cross_task_barplot": wandb.Image(fig)})
            plt.close(fig)

        summary["task28_num_metrics_logged"] = len(keys)
        summary["task28_note"] = 1.0  # placeholder numeric summary so it appears in W&B summary
        wandb.summary.update(summary)
    finally:
        finish_run(run)
    return summary


# -----------------------------
# CLI
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Run W&B experiments for assignment tasks 2.1 to 2.8")

    parser.add_argument("--data_root", type=str, required=True, help="Path to dataset/oxford_pet")
    parser.add_argument("--wandb_project", type=str, default="assignment2-wandb-report")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_val_samples", type=int, default=None)

    parser.add_argument("--classifier_ckpt", type=str, default="checkpoints/classifier.pth")
    parser.add_argument("--localizer_ckpt", type=str, default="checkpoints/localizer.pth")
    parser.add_argument("--unet_ckpt", type=str, default="checkpoints/unet.pth")

    parser.add_argument("--task21_epochs", type=int, default=5)
    parser.add_argument("--task22_epochs", type=int, default=5)
    parser.add_argument("--task23_epochs", type=int, default=5)
    parser.add_argument("--dropout_p_task21", type=float, default=0.5)
    parser.add_argument("--task21_lr_candidates", type=str, default="0.0001,0.0003,0.001,0.003,0.01")
    parser.add_argument("--task21_stability_epochs", type=int, default=2)
    parser.add_argument("--task21_divergence_threshold", type=float, default=100.0)

    parser.add_argument("--feature_map_image", type=str, default=None)
    parser.add_argument("--feature_map_channels_to_log", type=int, default=8)

    parser.add_argument("--task25_num_images", type=int, default=10)
    parser.add_argument("--task25_mc_passes", type=int, default=8)

    parser.add_argument("--task26_num_images", type=int, default=5)

    parser.add_argument("--novel_image_paths", nargs="*", default=None)

    parser.add_argument(
        "--which",
        type=str,
        default="all",
        choices=["all", "2.1", "2.2", "2.3", "2.4", "2.5", "2.6", "2.7", "2.8"],
        help="Which subtask to run",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {args.device}")

    all_summaries = []

    if args.which in {"all", "2.1"}:
        all_summaries.append(run_task_21(args))
    if args.which in {"all", "2.2"}:
        all_summaries.append(run_task_22(args))
    if args.which in {"all", "2.3"}:
        all_summaries.append(run_task_23(args))
    if args.which in {"all", "2.4"}:
        all_summaries.append(run_task_24(args))
    if args.which in {"all", "2.5"}:
        all_summaries.append(run_task_25(args))
    if args.which in {"all", "2.6"}:
        all_summaries.append(run_task_26(args))
    if args.which in {"all", "2.7"}:
        all_summaries.append(run_task_27(args))
    if args.which in {"all", "2.8"}:
        all_summaries.append(run_task_28(args, all_summaries))

    print("Finished requested W&B tasks.")


if __name__ == "__main__":
    main()

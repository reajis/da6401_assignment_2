"""training endpoint"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from data.pets_dataset import OxfordIIITPetDataset
from losses.iou_loss import IoULoss
from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet


INPUT_IMAGE_SIZE = 224


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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train VGG11 models on Oxford-IIIT Pet"
    )

    parser.add_argument(
        "--task",
        type=str,
        default="classification",
        choices=["classification", "localization", "segmentation"],
        help="Task to train",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Path to Oxford-IIIT Pet dataset root",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay",
    )
    parser.add_argument(
        "--dropout_p",
        type=float,
        default=0.5,
        help="Dropout probability for model head",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="Number of dataloader workers",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.1,
        help="Fraction of trainval split to use for validation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="Optional custom checkpoint path",
    )

    return parser.parse_args()


def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_transforms(task: str):
    normalize = NormalizeTensor(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    if task == "classification":
        train_transform = ComposeTensor([
            RandomHorizontalFlipTensor(p=0.5),
            normalize,
        ])
    else:
        train_transform = ComposeTensor([normalize])

    eval_transform = ComposeTensor([normalize])
    return train_transform, eval_transform


def build_dataloaders(args):
    train_transform, eval_transform = get_transforms(args.task)

    full_dataset_train = OxfordIIITPetDataset(
        root=args.data_root,
        split="trainval",
        task=args.task,
        transform=train_transform,
        image_size=INPUT_IMAGE_SIZE,
    )

    full_dataset_val = OxfordIIITPetDataset(
        root=args.data_root,
        split="trainval",
        task=args.task,
        transform=eval_transform,
        image_size=INPUT_IMAGE_SIZE,
    )

    total_samples = len(full_dataset_train)
    val_size = max(1, int(total_samples * args.val_split))
    val_size = min(val_size, total_samples - 1)
    train_size = total_samples - val_size

    generator = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(total_samples, generator=generator).tolist()

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_dataset = Subset(full_dataset_train, train_indices)
    val_dataset = Subset(full_dataset_val, val_indices)

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader


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


def compute_batch_iou(
    pred_boxes: torch.Tensor,
    target_boxes: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
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


def compute_segmentation_metrics(
    logits: torch.Tensor,
    masks: torch.Tensor,
    num_classes: int = 3,
):
    preds = torch.argmax(logits, dim=1)

    correct = (preds == masks).sum().item()
    total_pixels = masks.numel()
    pixel_acc = correct / total_pixels

    ious = []
    for cls in range(num_classes):
        pred_c = preds == cls
        mask_c = masks == cls

        intersection = (pred_c & mask_c).sum().item()
        union = (pred_c | mask_c).sum().item()

        if union > 0:
            ious.append(intersection / union)

    mean_iou = sum(ious) / len(ious) if len(ious) > 0 else 0.0
    return pixel_acc, mean_iou


def train_one_epoch_classification(model, loader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    running_correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(logits, dim=1)
        running_correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, running_correct / total


@torch.no_grad()
def evaluate_classification(model, loader, criterion, device):
    model.eval()

    running_loss = 0.0
    running_correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(logits, dim=1)
        running_correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, running_correct / total


def train_one_epoch_localization(
    model,
    loader,
    mse_criterion,
    iou_criterion,
    optimizer,
    device,
):
    model.train()

    running_loss = 0.0
    running_mse = 0.0
    running_iou_loss = 0.0
    running_iou = 0.0
    total = 0

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

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        running_mse += mse_loss.item() * batch_size
        running_iou_loss += iou_loss.item() * batch_size
        running_iou += compute_batch_iou(pred_boxes, boxes).sum().item()
        total += batch_size

    return (
        running_loss / total,
        running_mse / total,
        running_iou_loss / total,
        running_iou / total,
    )


@torch.no_grad()
def evaluate_localization(model, loader, mse_criterion, iou_criterion, device):
    model.eval()

    running_loss = 0.0
    running_mse = 0.0
    running_iou_loss = 0.0
    running_iou = 0.0
    total = 0

    for images, boxes in loader:
        images = images.to(device)
        boxes = boxes.to(device)

        pred_boxes = model(images)
        mse_loss = mse_criterion(pred_boxes, boxes)
        iou_loss = iou_criterion(pred_boxes, boxes)
        loss = mse_loss + iou_loss

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        running_mse += mse_loss.item() * batch_size
        running_iou_loss += iou_loss.item() * batch_size
        running_iou += compute_batch_iou(pred_boxes, boxes).sum().item()
        total += batch_size

    return (
        running_loss / total,
        running_mse / total,
        running_iou_loss / total,
        running_iou / total,
    )


def train_one_epoch_segmentation(model, loader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    running_pixel_acc = 0.0
    running_miou = 0.0
    total = 0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, masks)

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        pixel_acc, mean_iou = compute_segmentation_metrics(logits.detach(), masks)

        running_loss += loss.item() * batch_size
        running_pixel_acc += pixel_acc * batch_size
        running_miou += mean_iou * batch_size
        total += batch_size

    return (
        running_loss / total,
        running_pixel_acc / total,
        running_miou / total,
    )


@torch.no_grad()
def evaluate_segmentation(model, loader, criterion, device):
    model.eval()

    running_loss = 0.0
    running_pixel_acc = 0.0
    running_miou = 0.0
    total = 0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)
        loss = criterion(logits, masks)

        batch_size = images.size(0)
        pixel_acc, mean_iou = compute_segmentation_metrics(logits, masks)

        running_loss += loss.item() * batch_size
        running_pixel_acc += pixel_acc * batch_size
        running_miou += mean_iou * batch_size
        total += batch_size

    return (
        running_loss / total,
        running_pixel_acc / total,
        running_miou / total,
    )


def save_checkpoint(model, save_path: str, epoch: int, best_metric: float):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "state_dict": model.state_dict(),
        "epoch": epoch,
        "best_metric": best_metric,
    }

    torch.save(checkpoint, save_path)


def main():
    args = parse_args()
    set_seed(args.seed)

    if args.save_path is None:
        if args.task == "classification":
            args.save_path = "checkpoints/classifier.pth"
        elif args.task == "localization":
            args.save_path = "checkpoints/localizer.pth"
        else:
            args.save_path = "checkpoints/unet.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Training task: {args.task}")
    print(f"Fixed input image size: {INPUT_IMAGE_SIZE}x{INPUT_IMAGE_SIZE}")

    train_loader, val_loader = build_dataloaders(args)

    if args.task == "classification":
        model = VGG11Classifier(
            num_classes=37,
            in_channels=3,
            dropout_p=args.dropout_p,
        ).to(device)
        criterion = nn.CrossEntropyLoss()

    elif args.task == "localization":
        model = VGG11Localizer(
            in_channels=3,
            dropout_p=args.dropout_p,
        ).to(device)
        mse_criterion = nn.MSELoss()
        iou_criterion = IoULoss(reduction="mean")

    else:
        model = VGG11UNet(
            in_channels=3,
            num_classes=3,
        ).to(device)
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val_metric = float("-inf")

    for epoch in range(args.epochs):
        if args.task == "classification":
            train_loss, train_metric = train_one_epoch_classification(
                model, train_loader, criterion, optimizer, device
            )
            val_loss, val_metric = evaluate_classification(
                model, val_loader, criterion, device
            )

            print(
                f"Epoch [{epoch + 1}/{args.epochs}] "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_metric:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_metric:.4f}"
            )

        elif args.task == "localization":
            train_loss, train_mse, train_iou_loss, train_metric = train_one_epoch_localization(
                model, train_loader, mse_criterion, iou_criterion, optimizer, device
            )
            val_loss, val_mse, val_iou_loss, val_metric = evaluate_localization(
                model, val_loader, mse_criterion, iou_criterion, device
            )

            print(
                f"Epoch [{epoch + 1}/{args.epochs}] "
                f"Train Loss: {train_loss:.4f} | Train MSE: {train_mse:.4f} | "
                f"Train IoU Loss: {train_iou_loss:.4f} | Train IoU: {train_metric:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val MSE: {val_mse:.4f} | "
                f"Val IoU Loss: {val_iou_loss:.4f} | Val IoU: {val_metric:.4f}"
            )

        else:
            train_loss, train_pixel_acc, train_miou = train_one_epoch_segmentation(
                model, train_loader, criterion, optimizer, device
            )
            val_loss, val_pixel_acc, val_miou = evaluate_segmentation(
                model, val_loader, criterion, device
            )
            val_metric = val_miou

            print(
                f"Epoch [{epoch + 1}/{args.epochs}] "
                f"Train Loss: {train_loss:.4f} | Train Pixel Acc: {train_pixel_acc:.4f} | "
                f"Train mIoU: {train_miou:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Pixel Acc: {val_pixel_acc:.4f} | "
                f"Val mIoU: {val_miou:.4f}"
            )

        if val_metric > best_val_metric:
            best_val_metric = val_metric
            save_checkpoint(model, args.save_path, epoch + 1, best_val_metric)
            print(f"Saved best model to {args.save_path}")

    if args.task == "classification":
        print(f"Best Val Acc: {best_val_metric:.4f}")
    elif args.task == "localization":
        print(f"Best Val IoU: {best_val_metric:.4f}")
    else:
        print(f"Best Val mIoU: {best_val_metric:.4f}")


if __name__ == "__main__":
    main()
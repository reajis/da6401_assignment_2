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

    parser.add_argument("--task", type=str, default="classification",
                        choices=["classification", "localization", "segmentation"])
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout_p", type=float, default=0.2)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_path", type=str, default=None)

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
            RandomHorizontalFlipTensor(p=0.7),
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

    train_dataset = Subset(full_dataset_train, indices[:train_size])
    val_dataset = Subset(full_dataset_val, indices[train_size:])

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              pin_memory=pin_memory)

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers,
                            pin_memory=pin_memory)

    return train_loader, val_loader


# ---------------- LOCALIZATION UTILS ---------------- #

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


def compute_batch_iou(pred_boxes, target_boxes, eps=1e-6):
    pred_xyxy = xywh_to_xyxy(pred_boxes)
    target_xyxy = xywh_to_xyxy(target_boxes)

    x1 = torch.maximum(pred_xyxy[:, 0], target_xyxy[:, 0])
    y1 = torch.maximum(pred_xyxy[:, 1], target_xyxy[:, 1])
    x2 = torch.minimum(pred_xyxy[:, 2], target_xyxy[:, 2])
    y2 = torch.minimum(pred_xyxy[:, 3], target_xyxy[:, 3])

    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)

    area_p = (pred_xyxy[:, 2] - pred_xyxy[:, 0]).clamp(min=0) * \
             (pred_xyxy[:, 3] - pred_xyxy[:, 1]).clamp(min=0)
    area_t = (target_xyxy[:, 2] - target_xyxy[:, 0]).clamp(min=0) * \
             (target_xyxy[:, 3] - target_xyxy[:, 1]).clamp(min=0)

    union = area_p + area_t - inter
    return inter / (union + eps)


# ---------------- CLASSIFICATION ---------------- #

def train_one_epoch_classification(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate_classification(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


# ---------------- LOCALIZATION ---------------- #

def train_one_epoch_localization(model, loader, mse, iou, optimizer, device):
    model.train()
    total_loss = total_mse = total_iou_loss = total_iou = total = 0

    for images, boxes in loader:
        images, boxes = images.to(device), boxes.to(device)

        optimizer.zero_grad()
        pred_boxes = model(images)

        # (optional but safe)
        pred_boxes = pred_boxes.clamp(min=0, max=INPUT_IMAGE_SIZE)

        mse_loss = mse(pred_boxes, boxes)
        iou_loss = iou(pred_boxes, boxes)
        loss = mse_loss + iou_loss

        loss.backward()
        optimizer.step()

        bs = images.size(0)
        total_loss += loss.item() * bs
        total_mse += mse_loss.item() * bs
        total_iou_loss += iou_loss.item() * bs
        total_iou += compute_batch_iou(pred_boxes, boxes).sum().item()
        total += bs

    return total_loss/total, total_mse/total, total_iou_loss/total, total_iou/total


@torch.no_grad()
def evaluate_localization(model, loader, mse, iou, device):
    model.eval()
    total_loss = total_mse = total_iou_loss = total_iou = total = 0

    for images, boxes in loader:
        images, boxes = images.to(device), boxes.to(device)
        pred_boxes = model(images)


        pred_boxes = pred_boxes.clamp(min=0, max=INPUT_IMAGE_SIZE)

        mse_loss = mse(pred_boxes, boxes)
        iou_loss = iou(pred_boxes, boxes)
        loss = mse_loss + iou_loss

        bs = images.size(0)
        total_loss += loss.item() * bs
        total_mse += mse_loss.item() * bs
        total_iou_loss += iou_loss.item() * bs
        total_iou += compute_batch_iou(pred_boxes, boxes).sum().item()
        total += bs

    return total_loss/total, total_mse/total, total_iou_loss/total, total_iou/total


# ---------------- SEGMENTATION (UNCHANGED) ---------------- #
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


def compute_segmentation_metrics(logits, masks, num_classes=3):
    preds = torch.argmax(logits, dim=1)
    correct = (preds == masks).sum().item()
    pixel_acc = correct / masks.numel()

    ious = []
    for cls in range(num_classes):
        inter = ((preds == cls) & (masks == cls)).sum().item()
        union = ((preds == cls) | (masks == cls)).sum().item()
        if union > 0:
            ious.append(inter / union)

    return pixel_acc, sum(ious)/len(ious) if ious else 0


# ---------------- MAIN ---------------- #

def main():
    args = parse_args()
    set_seed(args.seed)

    if args.save_path is None:
        args.save_path = f"checkpoints/{args.task}.pth"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\nStarting training")
    print(f" Task: {args.task}")
    print(f" Device: {device}")
    print(f" Epochs: {args.epochs}, Batch size: {args.batch_size}\n")

    train_loader, val_loader = build_dataloaders(args)

    print(f" Train batches: {len(train_loader)}")
    print(f" Val batches: {len(val_loader)}\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    if args.task == "classification":
        model = VGG11Classifier(37, 3, args.dropout_p).to(device)
        class_counts = torch.zeros(37)
        for _, label in train_loader.dataset:
            class_counts[label] += 1

        weights = 1.0 / (class_counts + 1e-6)
        weights = weights / weights.sum() * len(weights)

        criterion = nn.CrossEntropyLoss(weight=weights.to(device))

    elif args.task == "localization":
        model = VGG11Localizer(3, args.dropout_p).to(device)
        mse = nn.MSELoss()
        iou = IoULoss()

    else:
        model = VGG11UNet(3, 3).to(device)
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_metric = -1

    for epoch in range(args.epochs):

        if args.task == "classification":
            train_loss, train_acc = train_one_epoch_classification(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = evaluate_classification(model, val_loader, criterion, device)
            val_metric = val_acc

        elif args.task == "localization":
            train_loss, _, _, train_iou = train_one_epoch_localization(model, train_loader, mse, iou, optimizer, device)
            val_loss, _, _, val_iou = evaluate_localization(model, val_loader, mse, iou, device)

            #  FIX: explicitly define val_metric
            val_metric = val_iou

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
            torch.save(model.state_dict(), args.save_path)

    print("Best metric:", best_val_metric)


if __name__ == "__main__":
    main()
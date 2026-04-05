"""Inference and evaluation."""

import argparse
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from data.pets_dataset import OxfordIIITPetDataset
from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet


INPUT_IMAGE_SIZE = 224


class NormalizeTensor:
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(-1, 1, 1)

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        return (image - self.mean) / self.std


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inference for Oxford-IIIT Pet tasks"
    )

    parser.add_argument(
        "--task",
        type=str,
        default="classification",
        choices=["classification", "localization", "segmentation"],
        help="Inference task",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Path to Oxford-IIIT Pet dataset root",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Path to input image",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--dropout_p",
        type=float,
        default=0.5,
        help="Dropout probability used during model construction",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=3,
        help="Number of top predictions to show for classification",
    )
    parser.add_argument(
        "--save_vis_path",
        type=str,
        default=None,
        help="Optional path to save visualization",
    )

    return parser.parse_args()


def get_eval_transform():
    return NormalizeTensor(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )


def load_rgb_image(image_path: str) -> np.ndarray:
    image = mpimg.imread(image_path)

    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)

    if image.shape[-1] == 4:
        image = image[..., :3]

    image = image.astype(np.float32)
    if image.max() > 1.0:
        image = image / 255.0

    return image


def image_to_tensor(image: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(image).permute(2, 0, 1).contiguous().float()
    tensor = tensor.unsqueeze(0)
    tensor = F.interpolate(
        tensor,
        size=(INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE),
        mode="bilinear",
        align_corners=False,
    )
    return tensor.squeeze(0)


def prepare_input(image: np.ndarray, transform) -> torch.Tensor:
    tensor = image_to_tensor(image)
    tensor = transform(tensor)
    return tensor.unsqueeze(0)


def load_classification_model(checkpoint_path: str, device: torch.device, dropout_p: float):
    model = VGG11Classifier(
        num_classes=37,
        in_channels=3,
        dropout_p=dropout_p,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model


def load_localization_model(checkpoint_path: str, device: torch.device, dropout_p: float):
    model = VGG11Localizer(
        in_channels=3,
        dropout_p=dropout_p,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model


def load_segmentation_model(checkpoint_path: str, device: torch.device):
    model = VGG11UNet(
        in_channels=3,
        num_classes=3,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model


def load_class_names(data_root: str):
    dataset = OxfordIIITPetDataset(
        root=data_root,
        split="trainval",
        task="classification",
        transform=None,
    )
    class_names = [dataset.idx_to_class[i] for i in range(len(dataset.idx_to_class))]
    return class_names


@torch.no_grad()
def predict_classification(model, input_tensor: torch.Tensor):
    logits = model(input_tensor)
    probs = torch.softmax(logits, dim=1)
    return probs.squeeze(0).cpu()


@torch.no_grad()
def predict_localization(model, input_tensor: torch.Tensor):
    pred_box = model(input_tensor)
    return pred_box.squeeze(0).cpu()


@torch.no_grad()
def predict_segmentation(model, input_tensor: torch.Tensor):
    logits = model(input_tensor)
    pred_mask = torch.argmax(logits, dim=1).squeeze(0).cpu()
    return pred_mask


def xywh_to_xyxy(box):
    x_center, y_center, width, height = box
    x1 = x_center - width / 2.0
    y1 = y_center - height / 2.0
    x2 = x_center + width / 2.0
    y2 = y_center + height / 2.0
    return [x1, y1, x2, y2]


def scale_box(box, scale_x: float, scale_y: float):
    x_center, y_center, width, height = box
    return [
        x_center * scale_x,
        y_center * scale_y,
        width * scale_x,
        height * scale_y,
    ]


def clamp_box_xyxy(box_xyxy, width: int, height: int):
    x1, y1, x2, y2 = box_xyxy
    x1 = max(0.0, min(float(x1), width - 1))
    y1 = max(0.0, min(float(y1), height - 1))
    x2 = max(0.0, min(float(x2), width - 1))
    y2 = max(0.0, min(float(y2), height - 1))
    return [x1, y1, x2, y2]


def draw_box_on_image(image: np.ndarray, box_xyxy, save_path: str):
    x1, y1, x2, y2 = box_xyxy

    figure, axis = plt.subplots(1)
    axis.imshow(image)
    rectangle = patches.Rectangle(
        (x1, y1),
        max(1.0, x2 - x1),
        max(1.0, y2 - y1),
        linewidth=2,
        edgecolor="red",
        facecolor="none",
    )
    axis.add_patch(rectangle)
    axis.axis("off")

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close(figure)


def mask_to_color_image(mask: torch.Tensor) -> np.ndarray:
    mask_np = mask.numpy()
    color_mask = np.zeros((mask_np.shape[0], mask_np.shape[1], 3), dtype=np.uint8)
    color_mask[mask_np == 0] = np.array([0, 0, 0], dtype=np.uint8)
    color_mask[mask_np == 1] = np.array([0, 255, 0], dtype=np.uint8)
    color_mask[mask_np == 2] = np.array([255, 0, 0], dtype=np.uint8)
    return color_mask


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    image_path = Path(args.image_path)
    checkpoint_path = Path(args.checkpoint)

    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    original_image = load_rgb_image(str(image_path))
    orig_h, orig_w = original_image.shape[:2]

    transform = get_eval_transform()
    input_tensor = prepare_input(original_image, transform).to(device)

    if args.task == "classification":
        model = load_classification_model(str(checkpoint_path), device, args.dropout_p)
        class_names = load_class_names(args.data_root)

        probs = predict_classification(model, input_tensor)

        top_k = min(args.top_k, probs.numel())
        top_probs, top_indices = torch.topk(probs, k=top_k)

        print(f"\nImage: {image_path}")
        print("Top predictions:")
        for rank, (idx, prob) in enumerate(
            zip(top_indices.tolist(), top_probs.tolist()), start=1
        ):
            class_name = class_names[idx] if idx < len(class_names) else f"class_{idx}"
            print(f"{rank}. Class {idx}: {class_name} | Probability: {prob:.4f}")

    elif args.task == "localization":
        model = load_localization_model(str(checkpoint_path), device, args.dropout_p)

        pred_box_resized = predict_localization(model, input_tensor).tolist()
        pred_box_resized_xyxy = xywh_to_xyxy(pred_box_resized)
        pred_box_resized_xyxy = clamp_box_xyxy(
            pred_box_resized_xyxy, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE
        )

        scale_x = orig_w / float(INPUT_IMAGE_SIZE)
        scale_y = orig_h / float(INPUT_IMAGE_SIZE)
        pred_box_original = scale_box(pred_box_resized, scale_x, scale_y)
        pred_box_original_xyxy = xywh_to_xyxy(pred_box_original)
        pred_box_original_xyxy = clamp_box_xyxy(pred_box_original_xyxy, orig_w, orig_h)

        print(f"\nImage: {image_path}")
        print("Predicted bounding box:")
        print(
            f"Resized-image space ({INPUT_IMAGE_SIZE}x{INPUT_IMAGE_SIZE}) "
            f"[xc, yc, w, h]: {[round(v, 2) for v in pred_box_resized]}"
        )
        print(
            f"Original-image space ({orig_w}x{orig_h}) "
            f"[xc, yc, w, h]: {[round(v, 2) for v in pred_box_original]}"
        )
        print(
            f"Original-image corners [x1, y1, x2, y2]: "
            f"{[round(v, 2) for v in pred_box_original_xyxy]}"
        )

        if args.save_vis_path is not None:
            draw_box_on_image(original_image, pred_box_original_xyxy, args.save_vis_path)
            print(f"Saved visualization to {args.save_vis_path}")

    else:
        model = load_segmentation_model(str(checkpoint_path), device)

        pred_mask = predict_segmentation(model, input_tensor)
        unique_labels = sorted(set(pred_mask.view(-1).tolist()))

        print(f"\nImage: {image_path}")
        print(f"Predicted segmentation labels present: {unique_labels}")

        if args.save_vis_path is not None:
            color_mask = mask_to_color_image(pred_mask)
            Path(args.save_vis_path).parent.mkdir(parents=True, exist_ok=True)
            plt.imsave(args.save_vis_path, color_mask)
            print(f"Saved segmentation mask to {args.save_vis_path}")


if __name__ == "__main__":
    main()
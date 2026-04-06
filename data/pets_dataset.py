"""Dataset loader for Oxford-IIIT Pet."""

from pathlib import Path
from typing import Callable, Optional, Tuple
import xml.etree.ElementTree as ET

import matplotlib.image as mpimg
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class OxfordIIITPetDataset(Dataset):
    """Oxford-IIIT Pet dataset loader for classification, localization, and segmentation."""

    def __init__(
        self,
        root: str,
        split: str = "trainval",
        task: str = "classification",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        image_size: int = 224,
    ):
        if split not in {"trainval", "test"}:
            raise ValueError(f"split must be 'trainval' or 'test', got {split}")
        if task not in {"classification", "localization", "segmentation"}:
            raise ValueError(
                f"task must be one of 'classification', 'localization', or 'segmentation', got {task}"
            )

        self.root = Path(root)
        self.split = split
        self.task = task
        self.transform = transform
        self.target_transform = target_transform
        self.image_size = image_size

        self.images_dir = self.root / "images"
        self.annotations_dir = self.root / "annotations"
        self.xmls_dir = self.annotations_dir / "xmls"
        self.trimaps_dir = self.annotations_dir / "trimaps"
        self.split_file = self.annotations_dir / f"{split}.txt"

        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        if not self.annotations_dir.exists():
            raise FileNotFoundError(f"Annotations directory not found: {self.annotations_dir}")
        if not self.split_file.exists():
            raise FileNotFoundError(f"Split file not found: {self.split_file}")

        self.samples = []
        self.class_to_idx = {}
        self.idx_to_class = {}

        with open(self.split_file, "r", encoding="utf-8") as file_obj:
            for line in file_obj:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                parts = line.split()
                if len(parts) < 4:
                    continue

                image_id, class_id, species, breed_id = parts[:4]
                image_path = self.images_dir / f"{image_id}.jpg"

                if not image_path.exists():
                    continue

                sample = {
                    "image_id": image_id,
                    "image_path": image_path,
                    "label": int(class_id) - 1,
                    "species": int(species) - 1,
                    "breed_id": int(breed_id) - 1,
                    "breed_name": self._breed_name_from_image_id(image_id),
                }

                if self.task == "localization":
                    bbox_path = self.xmls_dir / f"{image_id}.xml"
                    if not bbox_path.exists():
                        continue
                    sample["bbox_path"] = bbox_path

                if self.task == "segmentation":
                    mask_path = self.trimaps_dir / f"{image_id}.png"
                    if not mask_path.exists():
                        continue
                    sample["mask_path"] = mask_path

                self.samples.append(sample)
                self.class_to_idx[sample["breed_name"]] = sample["label"]
                self.idx_to_class[sample["label"]] = sample["breed_name"]

        if len(self.samples) == 0:
            raise RuntimeError(f"No valid samples found in split file: {self.split_file}")

    @staticmethod
    def _breed_name_from_image_id(image_id: str) -> str:
        parts = image_id.split("_")
        if len(parts) <= 1:
            return image_id
        return "_".join(parts[:-1])

    @staticmethod
    def _load_rgb_image(image_path: Path) -> np.ndarray:
        image = mpimg.imread(str(image_path))

        if image.ndim == 2:
            image = np.stack([image, image, image], axis=-1)

        if image.shape[-1] == 4:
            image = image[..., :3]

        image = image.astype(np.float32)
        if image.max() > 1.0:
            image = image / 255.0

        return image

    @staticmethod
    def _load_bbox_xyxy(xml_path: Path) -> Tuple[float, float, float, float]:
        tree = ET.parse(str(xml_path))
        root = tree.getroot()

        bndbox = root.find(".//object/bndbox")
        if bndbox is None:
            raise RuntimeError(f"Bounding box not found in annotation: {xml_path}")

        x1 = float(bndbox.find("xmin").text)
        y1 = float(bndbox.find("ymin").text)
        x2 = float(bndbox.find("xmax").text)
        y2 = float(bndbox.find("ymax").text)

        return x1, y1, x2, y2

    @staticmethod
    def _load_trimap(mask_path: Path) -> np.ndarray:
        mask = mpimg.imread(str(mask_path))

        if mask.ndim == 3:
            mask = mask[..., 0]

        mask = mask.astype(np.float32)
        if mask.max() <= 1.0:
            mask = np.rint(mask * 255.0)

        return mask.astype(np.int64)

    def _resize_image_tensor(self, image_tensor: torch.Tensor) -> torch.Tensor:
        image_tensor = image_tensor.unsqueeze(0)
        image_tensor = F.interpolate(
            image_tensor,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        return image_tensor.squeeze(0)

    def _resize_mask_tensor(self, mask_tensor: torch.Tensor) -> torch.Tensor:
        mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0).float()
        mask_tensor = F.interpolate(
            mask_tensor,
            size=(self.image_size, self.image_size),
            mode="nearest",
        )
        return mask_tensor.squeeze(0).squeeze(0).long()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]

        image = self._load_rgb_image(sample["image_path"])
        original_height, original_width = image.shape[:2]

        image_tensor = torch.from_numpy(image).permute(2, 0, 1).contiguous().float()
        image_tensor = self._resize_image_tensor(image_tensor)

        if self.transform is not None:
            image_tensor = self.transform(image_tensor)

        if self.task == "classification":
            target = sample["label"]
            if self.target_transform is not None:
                target = self.target_transform(target)
            return image_tensor,  torch.tensor(target, dtype=torch.long)

        if self.task == "localization":
            x1, y1, x2, y2 = self._load_bbox_xyxy(sample["bbox_path"])

            scale_x = self.image_size / float(original_width)
            scale_y = self.image_size / float(original_height)

            x1 *= scale_x
            x2 *= scale_x
            y1 *= scale_y
            y2 *= scale_y

            x_center = (x1 + x2) / 2.0
            y_center = (y1 + y2) / 2.0
            width = max(0.0, x2 - x1)
            height = max(0.0, y2 - y1)

            target = torch.tensor(
                [x_center, y_center, width, height],
                dtype=torch.float32,
            )

            if self.target_transform is not None:
                target = self.target_transform(target)
            return image_tensor, target

        mask = self._load_trimap(sample["mask_path"])
        mask_tensor = torch.from_numpy(mask).long()
        mask_tensor = self._resize_mask_tensor(mask_tensor)
        mask_tensor = (mask_tensor - 1).clamp(min=0, max=2)

        if self.target_transform is not None:
            mask_tensor = self.target_transform(mask_tensor)

        return image_tensor, mask_tensor
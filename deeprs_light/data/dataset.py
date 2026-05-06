"""
Core COCO-format Dataset class for remote sensing images.
"""

import os
from typing import Dict, List, Optional, Callable, Tuple

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

from deeprs_light.registry import DATASETS


class DeepRSCocoDataset(Dataset):
    """
    Standard COCO-format dataset for remote sensing.

    Responsibilities:
    - Image reading (via cv2)
    - COCO annotation parsing -> boxes, labels, masks, area, iscrowd
    - Transforms are injected via the `transforms` parameter (decoupled).

    The internal coordinate system for boxes is **xyxy** (absolute pixels).

    Returns per item: (image, target_dict)
    """

    def __init__(
        self,
        root: str,
        ann_file: str,
        transforms: Optional[Callable] = None,
        cache_images: bool = False,
    ):
        """
        Args:
            root: Image root directory.
            ann_file: Path to COCO JSON annotation file.
            transforms: Augmentation pipeline (e.g., Compose(...)).
            cache_images: If True, load all images into RAM at init.
        """
        self.root = root
        self.ann_file = ann_file
        self.transforms = transforms
        self.cache_images = cache_images

        # Lazy import to avoid hard dependency at module level
        from pycocotools.coco import COCO

        self.coco = COCO(ann_file)
        self.image_ids = sorted(self.coco.getImgIds())
        self._id_to_index = {img_id: i for i, img_id in enumerate(self.image_ids)}

        # Build image_id -> file_name mapping
        self._img_info: Dict[int, Dict] = {}
        for img_id in self.image_ids:
            info = self.coco.loadImgs([img_id])[0]
            self._img_info[img_id] = info

        # Build category_id -> contiguous index mapping
        cat_ids = sorted(self.coco.getCatIds())
        self.cat_id_to_label = {cat_id: i for i, cat_id in enumerate(cat_ids)}
        self.label_to_cat_id = {i: cat_id for cat_id, i in self.cat_id_to_label.items()}
        self.num_classes = len(cat_ids)

        # Optional image cache
        self._image_cache: Dict[int, np.ndarray] = {}
        if cache_images:
            for img_id in self.image_ids:
                self._image_cache[img_id] = self._load_image(img_id)

    def __len__(self) -> int:
        return len(self.image_ids)

    def _load_image(self, image_id: int) -> np.ndarray:
        """Load an image as a numpy array (H, W, C) in BGR format."""
        if image_id in self._image_cache:
            return self._image_cache[image_id]
        info = self._img_info[image_id]
        file_name = info["file_name"]
        path = os.path.join(self.root, file_name)
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Failed to load image: {path}")
        return image

    def _load_target(self, image_id: int) -> Dict:
        """
        Parse COCO annotations for a given image into a target dict.

        Returns a dict with keys:
            image_id, boxes, labels, masks, area, iscrowd
        Boxes are in **xyxy** format.
        If no annotations exist, boxes/labels/masks/area/iscrowd are empty tensors.
        """
        ann_ids = self.coco.getAnnIds(imgIds=[image_id])
        anns = self.coco.loadAnns(ann_ids)

        boxes = []
        labels = []
        masks = []
        areas = []
        iscrowds = []

        for ann in anns:
            x, y, w, h = ann["bbox"]
            # COCO bbox is xywh -> convert to xyxy
            x1, y1, x2, y2 = x, y, x + w, y + h
            boxes.append([x1, y1, x2, y2])
            labels.append(self.cat_id_to_label[ann["category_id"]])
            areas.append(ann["area"])
            iscrowds.append(ann.get("iscrowd", 0))

            # Instance segmentation mask
            if "segmentation" in ann and ann["segmentation"]:
                mask = self.coco.annToMask(ann)
                masks.append(mask)

        if boxes:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.long)
            areas = torch.tensor(areas, dtype=torch.float32)
            iscrowds = torch.tensor(iscrowds, dtype=torch.long)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.long)
            areas = torch.zeros((0,), dtype=torch.float32)
            iscrowds = torch.zeros((0,), dtype=torch.long)

        if masks:
            masks = torch.from_numpy(np.stack(masks, axis=0)).float()
        else:
            masks = torch.zeros((0, 1, 1), dtype=torch.float32)

        return {
            "image_id": image_id,
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "area": areas,
            "iscrowd": iscrowds,
        }

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        """
        Returns (image, target).
        """
        image_id = self.image_ids[idx]
        image = self._load_image(image_id)
        target = self._load_target(image_id)

        # Apply transforms if provided
        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    @staticmethod
    def collate_fn(batch: List[Tuple]) -> Tuple[torch.Tensor, List[Dict]]:
        """
        Collate function for DataLoader.
        Stacks images; targets are kept as a list of dicts.
        """
        images = torch.stack([item[0] for item in batch], dim=0)
        targets = [item[1] for item in batch]
        return images, targets


@DATASETS.register()
class DeepRSCocoDatasetReg(DeepRSCocoDataset):
    """Registered variant with default name 'deeprscocodatasetreg'."""

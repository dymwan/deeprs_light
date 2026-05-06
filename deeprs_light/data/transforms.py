"""
Data augmentation transforms decoupled from Dataset.
All transforms operate on (image: np.ndarray, target: Dict) pairs.
Image is always (H, W, C) in numpy until ToTensor converts it.
Target dict contains boxes (xyxy), labels, masks, area, iscrowd, image_id.
"""

from typing import Dict, List, Tuple, Union, Callable, Optional
import random

import numpy as np
import cv2
import torch

from deeprs_light.registry import TRANSFORMS
from deeprs_light.data.transforms_utils import (
    clamp_boxes,
    flip_boxes_horizontal,
    flip_boxes_vertical,
    scale_boxes,
    shift_boxes,
)


# ============================================================
# Compose
# ============================================================

@TRANSFORMS.register()
class Compose:
    """
    Apply a sequence of transforms in order to both image and target.

    Usage:
        transform = Compose([Resize((800, 800)), RandomHorizontalFlip(0.5), ...])
        image, target = transform(image, target)
    """

    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __call__(
        self, image: np.ndarray, target: Dict
    ) -> Tuple:
        for t in self.transforms:
            if image is None:
                return image, target
            image, target = t(image, target)
        return image, target


# ============================================================
# Conditional transforms
# ============================================================

@TRANSFORMS.register()
class RandomApply:
    """
    Apply a transform with probability p.

    Usage:
        RandomApply(ColorJitter(0.3, 0.3, 0.3, 0.1), p=0.5)
    """

    def __init__(self, transform: Callable, p: float = 0.5):
        self.transform = transform
        self.p = p

    def __call__(self, image, target):
        if random.random() < self.p:
            return self.transform(image, target)
        return image, target


@TRANSFORMS.register()
class OneOf:
    """
    Randomly choose one transform from a list to apply.
    Supports weighted selection.

    Usage:
        OneOf([HorizontalFlip(), VerticalFlip()], weights=[0.5, 0.5])
    """

    def __init__(
        self,
        transforms: List[Callable],
        weights: Optional[List[float]] = None,
    ):
        self.transforms = transforms
        if weights is None:
            weights = [1.0] * len(transforms)
        total = sum(weights)
        self.probs = [w / total for w in weights]

    def __call__(self, image, target):
        chosen = random.choices(self.transforms, weights=self.probs, k=1)[0]
        return chosen(image, target)


# ============================================================
# Geometric transforms — modify both image and target
# ============================================================

@TRANSFORMS.register()
class Resize:
    """
    Resize image and scale target boxes/masks accordingly.

    Args:
        size: Target (width, height) or single int for square.
        keep_ratio: If True, maintain aspect ratio and pad to size.
        pad_color: Padding color (B, G, R) for keep_ratio mode.
    """

    def __init__(
        self,
        size: Union[int, Tuple[int, int]],
        keep_ratio: bool = False,
        pad_color: Tuple[int, int, int] = (114, 114, 114),
    ):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
        self.keep_ratio = keep_ratio
        self.pad_color = pad_color

    def __call__(
        self, image: np.ndarray, target: Dict
    ) -> Tuple[np.ndarray, Dict]:
        h, w = image.shape[:2]
        target_w, target_h = self.size

        if self.keep_ratio:
            scale = min(target_w / w, target_h / h)
            new_w = int(round(w * scale))
            new_h = int(round(h * scale))
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            pad_w = target_w - new_w
            pad_h = target_h - new_h
            pad_left = pad_w // 2
            pad_top = pad_h // 2
            image = cv2.copyMakeBorder(
                image, pad_top, pad_h - pad_top,
                pad_left, pad_w - pad_left,
                cv2.BORDER_CONSTANT, value=self.pad_color,
            )

            if target["boxes"].shape[0] > 0:
                boxes = target["boxes"].numpy()
                boxes = scale_boxes(boxes, scale, scale)
                boxes = shift_boxes(boxes, pad_left, pad_top)
                target["boxes"] = torch.from_numpy(boxes)
            if target["masks"].shape[0] > 0 and target["masks"].ndim == 3:
                masks = target["masks"].numpy()
                masks_resized = []
                for m in masks:
                    m = cv2.resize(m, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                    m = cv2.copyMakeBorder(
                        m, pad_top, pad_h - pad_top,
                        pad_left, pad_w - pad_left,
                        cv2.BORDER_CONSTANT, value=0,
                    )
                    masks_resized.append(m)
                target["masks"] = torch.from_numpy(np.stack(masks_resized, axis=0))
            self._pad_offset = (pad_left, pad_top)
        else:
            scale_w = target_w / w
            scale_h = target_h / h
            image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

            if target["boxes"].shape[0] > 0:
                boxes = target["boxes"].numpy()
                boxes = scale_boxes(boxes, scale_w, scale_h)
                target["boxes"] = torch.from_numpy(boxes)
            if target["masks"].shape[0] > 0 and target["masks"].ndim == 3:
                masks = target["masks"].numpy()
                masks_resized = [
                    cv2.resize(m, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
                    for m in masks
                ]
                target["masks"] = torch.from_numpy(np.stack(masks_resized, axis=0))

        # Add size info to target for reference
        target["image_size"] = (target_w, target_h)

        return image, target


@TRANSFORMS.register()
class RandomHorizontalFlip:
    """Random horizontal flip with probability p."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(
        self, image: np.ndarray, target: Dict
    ) -> Tuple[np.ndarray, Dict]:
        if random.random() < self.p:
            image = cv2.flip(image, 1)  # 1 = horizontal
            w = image.shape[1]
            if target["boxes"].shape[0] > 0:
                target["boxes"] = torch.from_numpy(
                    flip_boxes_horizontal(target["boxes"].numpy(), w)
                )
            if target["masks"].shape[0] > 0 and target["masks"].ndim == 3:
                target["masks"] = torch.flip(target["masks"], dims=[2])
        return image, target


@TRANSFORMS.register()
class RandomVerticalFlip:
    """Random vertical flip with probability p."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(
        self, image: np.ndarray, target: Dict
    ) -> Tuple[np.ndarray, Dict]:
        if random.random() < self.p:
            image = cv2.flip(image, 0)
            h = image.shape[0]
            if target["boxes"].shape[0] > 0:
                target["boxes"] = torch.from_numpy(
                    flip_boxes_vertical(target["boxes"].numpy(), h)
                )
            if target["masks"].shape[0] > 0 and target["masks"].ndim == 3:
                target["masks"] = torch.flip(target["masks"], dims=[1])
        return image, target


@TRANSFORMS.register()
class RandomRotation90:
    """
    Random rotation by 0, 90, 180, or 270 degrees.
    Integer rotations avoid interpolation artifacts — important for
    remote sensing where pixel-level accuracy matters.
    """

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(
        self, image: np.ndarray, target: Dict
    ) -> Tuple[np.ndarray, Dict]:
        if random.random() < self.p:
            k = random.randint(0, 3)
            if k == 0:
                return image, target

            image = np.rot90(image, k)
            h, w = image.shape[:2]

            boxes = target["boxes"].numpy() if isinstance(target["boxes"], torch.Tensor) else target["boxes"]
            if boxes.shape[0] > 0:
                old_h, old_w = h, w
                for _ in range(k):
                    x1 = boxes[:, 0].copy()
                    y1 = boxes[:, 1].copy()
                    x2 = boxes[:, 2].copy()
                    y2 = boxes[:, 3].copy()
                    boxes[:, 0] = y1
                    boxes[:, 1] = old_w - x2
                    boxes[:, 2] = y2
                    boxes[:, 3] = old_w - x1
                    old_w, old_h = old_h, old_w
                target["boxes"] = torch.from_numpy(boxes.astype(np.float32))

            if target["masks"].shape[0] > 0 and target["masks"].ndim == 3:
                target["masks"] = torch.rot90(target["masks"], k, dims=[1, 2])

        return image, target


@TRANSFORMS.register()
class RandomResizedCrop:
    """
    Random crop + resize. Important for large remote sensing images
    to increase diversity during training.

    Args:
        crop_size: Output image size.
        scale: (min, max) ratio of crop area relative to original.
        p: Probability of applying.
    """

    def __init__(
        self,
        crop_size: Union[int, Tuple[int, int]],
        scale: Tuple[float, float] = (0.3, 1.0),
        p: float = 0.5,
    ):
        if isinstance(crop_size, int):
            self.crop_size = (crop_size, crop_size)
        else:
            self.crop_size = crop_size
        self.scale = scale
        self.p = p

    def __call__(
        self, image: np.ndarray, target: Dict
    ) -> Tuple[np.ndarray, Dict]:
        if random.random() >= self.p:
            return image, target

        h, w = image.shape[:2]
        crop_h, crop_w = self.crop_size

        scale_val = random.uniform(*self.scale)
        # Crop area should be at least as big as the output size
        crop_area_h = max(crop_h, int(h * scale_val))
        crop_area_w = max(crop_w, int(w * scale_val))

        x = random.randint(0, max(0, w - crop_area_w))
        y = random.randint(0, max(0, h - crop_area_h))

        # Crop image
        image = image[y:y + crop_area_h, x:x + crop_area_w]

        # Adjust boxes to cropped coordinate system
        boxes = target["boxes"].numpy() if isinstance(target["boxes"], torch.Tensor) else np.array(target["boxes"])
        if boxes.shape[0] > 0:
            boxes = shift_boxes(boxes, -x, -y)
            boxes, valid = clamp_boxes(boxes, crop_area_w, crop_area_h)
            labels = target["labels"].numpy() if isinstance(target["labels"], torch.Tensor) else np.array(target["labels"])
            labels = labels[valid]
            target["labels"] = torch.from_numpy(labels).long()
            target["boxes"] = torch.from_numpy(boxes)

            # Update area and iscrowd
            if "area" in target and target["area"].shape[0] > 0:
                target["area"] = target["area"][valid]
            if "iscrowd" in target and target["iscrowd"].shape[0] > 0:
                target["iscrowd"] = target["iscrowd"][valid]

        # Crop masks
        if target["masks"].shape[0] > 0 and target["masks"].ndim == 3:
            masks = target["masks"][:, y:y + crop_area_h, x:x + crop_area_w]
            target["masks"] = masks

        # Resize to target crop size
        if image.shape[0] != crop_h or image.shape[1] != crop_w:
            scale_w = crop_w / image.shape[1]
            scale_h = crop_h / image.shape[0]
            image = cv2.resize(image, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)
            if target["boxes"].shape[0] > 0:
                boxes = target["boxes"].numpy()
                boxes = scale_boxes(boxes, scale_w, scale_h)
                target["boxes"] = torch.from_numpy(boxes)

        return image, target


# ============================================================
# Pixel-level transforms — only modify image
# ============================================================

@TRANSFORMS.register()
class Normalize:
    """
    Normalize image: (image - mean) / std.
    Does NOT modify target boxes/masks coordinates.
    Image should be a torch.Tensor (after ToTensor) or float numpy.
    """

    def __init__(self, mean: Tuple[float, ...], std: Tuple[float, ...]):
        self.mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(std, dtype=np.float32).reshape(1, 1, 3)

    def __call__(
        self, image: np.ndarray, target: Dict
    ) -> Tuple[np.ndarray, Dict]:
        if isinstance(image, torch.Tensor):
            # Already converted to tensor — normalize inline
            mean_t = torch.from_numpy(self.mean.flatten()).to(image.device).view(3, 1, 1)
            std_t = torch.from_numpy(self.std.flatten()).to(image.device).view(3, 1, 1)
            image = (image - mean_t) / std_t
        else:
            image = (image.astype(np.float32) / 255.0 - self.mean) / self.std
        return image, target


@TRANSFORMS.register()
class ToTensor:
    """
    Convert numpy image (H,W,C) to torch.Tensor (C,H,W) in [0, 1].
    Convert target tensors: boxes, masks -> float32, labels -> long.
    """

    def __call__(
        self, image: np.ndarray, target: Dict
    ) -> Tuple[torch.Tensor, Dict]:
        # Image: HWC -> CHW, BGR -> RGB, uint8 -> float [0, 1]
        if isinstance(image, np.ndarray):
            image = image[..., [2, 1, 0]]  # BGR to RGB
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        # Convert target fields to tensors
        target = dict(target)
        for key in ("boxes", "area", "iscrowd"):
            if key in target and isinstance(target[key], np.ndarray):
                target[key] = torch.from_numpy(target[key])
        if "labels" in target and isinstance(target["labels"], np.ndarray):
            target["labels"] = torch.from_numpy(target["labels"]).long()

        return image, target


@TRANSFORMS.register()
class ColorJitter:
    """
    Brightness, contrast, saturation, and hue jitter.
    Only affects image, not target.
    """

    def __init__(
        self,
        brightness: float = 0.0,
        contrast: float = 0.0,
        saturation: float = 0.0,
        hue: float = 0.0,
        p: float = 0.5,
    ):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.p = p

    def __call__(
        self, image: np.ndarray, target: Dict
    ) -> Tuple[np.ndarray, Dict]:
        if random.random() >= self.p:
            return image, target

        # cv2 image is BGR -> convert to HSV for hue/sat adjustments
        if self.brightness > 0:
            delta = random.uniform(-self.brightness, self.brightness)
            image = np.clip(image.astype(np.float32) + delta * 255, 0, 255).astype(np.uint8)

        if self.contrast > 0:
            alpha = 1.0 + random.uniform(-self.contrast, self.contrast)
            mean = image.mean(axis=(0, 1), keepdims=True)
            image = np.clip((image.astype(np.float32) - mean) * alpha + mean, 0, 255).astype(np.uint8)

        if self.saturation > 0 or self.hue > 0:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
            if self.saturation > 0:
                alpha = 1.0 + random.uniform(-self.saturation, self.saturation)
                hsv[:, :, 1] = np.clip(hsv[:, :, 1] * alpha, 0, 255)
            if self.hue > 0:
                delta = random.uniform(-self.hue, self.hue) * 180
                hsv[:, :, 0] = (hsv[:, :, 0] + delta) % 180
            image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        return image, target


@TRANSFORMS.register()
class RandomBrightnessContrast:
    """Random brightness and contrast adjustment."""

    def __init__(
        self,
        brightness_limit: float = 0.2,
        contrast_limit: float = 0.2,
        p: float = 0.5,
    ):
        self.brightness_limit = brightness_limit
        self.contrast_limit = contrast_limit
        self.p = p

    def __call__(self, image, target):
        if random.random() >= self.p:
            return image, target
        alpha = 1.0 + random.uniform(-self.contrast_limit, self.contrast_limit)
        beta = random.uniform(-self.brightness_limit, self.brightness_limit) * 255
        image = np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
        return image, target


@TRANSFORMS.register()
class RandomGamma:
    """
    Random gamma correction. Simulates different sensor/illumination conditions.
    gamma_limit is in percent: (80, 120) means gamma in [0.8, 1.2].
    """

    def __init__(
        self,
        gamma_limit: Tuple[float, float] = (80, 120),
        p: float = 0.5,
    ):
        self.gamma_limit = gamma_limit
        self.p = p

    def __call__(self, image, target):
        if random.random() >= self.p:
            return image, target
        gamma = random.uniform(*self.gamma_limit) / 100.0
        table = (np.power(np.arange(256) / 255.0, gamma) * 255).astype(np.uint8)
        image = cv2.LUT(image, table)
        return image, target


@TRANSFORMS.register()
class GaussianNoise:
    """Add Gaussian noise to simulate remote sensing sensor noise."""

    def __init__(self, mean: float = 0.0, std: float = 0.05, p: float = 0.5):
        self.mean = mean
        self.std = std
        self.p = p

    def __call__(self, image, target):
        if random.random() >= self.p:
            return image, target
        noise = np.random.normal(self.mean, self.std * 255, image.shape).astype(np.float32)
        image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return image, target


# ============================================================
# Remote-sensing specific augmentations
# ============================================================

@TRANSFORMS.register()
class RandomCutMix:
    """
    CutMix augmentation (RS version).
    Randomly paste a patch from one image onto another within the same batch.
    This is a per-sample placeholder — the actual mixing happens at batch level
    via batch_transforms.BatchCutMix for better efficiency.

    For per-sample use, this acts as a no-op marker that signals the
    collate/batch_transform to perform CutMix.
    """

    def __init__(self, alpha: float = 1.0, p: float = 0.5):
        self.alpha = alpha
        self.p = p

    def __call__(self, image, target):
        # Per-sample CutMix requires access to a second image,
        # so it's marked here and executed at batch level.
        target["_cutmix_alpha"] = self.alpha
        target["_cutmix_p"] = self.p
        return image, target


@TRANSFORMS.register()
class Mosaic:
    """
    Mosaic augmentation: stitch 4 images into one.
    This is a per-sample placeholder — actual mosaic is performed at batch level
    via batch_transforms.BatchMosaic.
    """

    def __init__(self, output_size: Tuple[int, int] = (800, 800), p: float = 0.5):
        self.output_size = output_size
        self.p = p

    def __call__(self, image, target):
        target["_mosaic_size"] = self.output_size
        target["_mosaic_p"] = self.p
        return image, target

"""
Transform helper functions for box manipulation and coordinate conversions.
Also provides preset transform builders.
"""

from typing import Dict, List, Tuple, Union, Optional
import numpy as np


# ============================================================
# Box manipulation helpers
# ============================================================

def clamp_boxes(boxes: np.ndarray, width: int, height: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Clamp boxes to image boundaries [0, width] x [0, height].
    Removes boxes that become invalid (x2 <= x1 or y2 <= y1).

    Args:
        boxes: [N, 4] in xyxy format.
        width, height: Image dimensions.

    Returns:
        clamped_boxes: [M, 4] valid boxes (M <= N).
        valid_mask: [N] boolean mask, True for kept boxes.
    """
    if boxes.shape[0] == 0:
        return boxes, np.array([], dtype=bool)

    boxes = boxes.copy()
    boxes[:, 0] = np.clip(boxes[:, 0], 0, width)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, height)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, width)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, height)

    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    valid_mask = (w > 0) & (h > 0)

    return boxes[valid_mask], valid_mask


def filter_empty_boxes(
    boxes: np.ndarray, labels: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove boxes with zero or negative area (x2 <= x1 or y2 <= y1).
    Synchronously removes corresponding labels.

    Returns:
        (filtered_boxes, filtered_labels)
    """
    if boxes.shape[0] == 0:
        return boxes, labels

    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    valid = (w > 1) & (h > 1)
    return boxes[valid], labels[valid]


def flip_boxes_horizontal(boxes: np.ndarray, width: int) -> np.ndarray:
    """
    Flip boxes horizontally: [x1,y1,x2,y2] -> [W-x2, y1, W-x1, y2]

    Args:
        boxes: [N, 4] in xyxy format.

    Returns:
        Flipped boxes [N, 4].
    """
    if boxes.shape[0] == 0:
        return boxes
    boxes = boxes.copy()
    x1, x2 = boxes[:, 0].copy(), boxes[:, 2].copy()
    boxes[:, 0] = width - x2
    boxes[:, 2] = width - x1
    return boxes


def flip_boxes_vertical(boxes: np.ndarray, height: int) -> np.ndarray:
    """
    Flip boxes vertically: [x1,y1,x2,y2] -> [x1, H-y2, x2, H-y1]
    """
    if boxes.shape[0] == 0:
        return boxes
    boxes = boxes.copy()
    y1, y2 = boxes[:, 1].copy(), boxes[:, 3].copy()
    boxes[:, 1] = height - y2
    boxes[:, 3] = height - y1
    return boxes


def scale_boxes(
    boxes: np.ndarray, scale_w: float, scale_h: float
) -> np.ndarray:
    """
    Scale boxes by width and height factors.

    Args:
        boxes: [N, 4] xyxy.
        scale_w, scale_h: Scaling factors.

    Returns:
        Scaled boxes [N, 4].
    """
    if boxes.shape[0] == 0:
        return boxes
    boxes = boxes.copy()
    boxes[:, [0, 2]] *= scale_w
    boxes[:, [1, 3]] *= scale_h
    return boxes


def shift_boxes(boxes: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """
    Translate boxes by (dx, dy).

    Args:
        boxes: [N, 4] xyxy.
        dx, dy: Translation offsets.

    Returns:
        Shifted boxes [N, 4].
    """
    if boxes.shape[0] == 0:
        return boxes
    boxes = boxes.copy()
    boxes[:, [0, 2]] += dx
    boxes[:, [1, 3]] += dy
    return boxes


def boxes_xyxy_to_xywh(boxes: np.ndarray) -> np.ndarray:
    """[x1, y1, x2, y2] -> [x, y, w, h]"""
    if boxes.shape[0] == 0:
        return boxes
    boxes = np.array(boxes, dtype=np.float32)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    return np.stack([x1, y1, x2 - x1, y2 - y1], axis=1)


def boxes_xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """[x, y, w, h] -> [x1, y1, x2, y2]"""
    if boxes.shape[0] == 0:
        return boxes
    boxes = np.array(boxes, dtype=np.float32)
    x, y, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    return np.stack([x, y, x + w, y + h], axis=1)


def compute_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """
    Compute pairwise IoU matrix between two sets of boxes.

    Args:
        boxes_a: [N, 4] xyxy.
        boxes_b: [M, 4] xyxy.

    Returns:
        IoU matrix of shape [N, M].
    """
    if boxes_a.shape[0] == 0 or boxes_b.shape[0] == 0:
        return np.zeros((boxes_a.shape[0], boxes_b.shape[0]), dtype=np.float32)

    # Intersection
    x1 = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    y1 = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    x2 = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    y2 = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])

    inter_w = np.maximum(0, x2 - x1)
    inter_h = np.maximum(0, y2 - y1)
    inter_area = inter_w * inter_h

    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])

    union_area = area_a[:, None] + area_b[None, :] - inter_area
    iou = inter_area / np.maximum(union_area, 1e-6)

    return iou.astype(np.float32)


# ============================================================
# Transform presets
# ============================================================

def get_train_transforms(
    image_size: Union[int, Tuple[int, int]] = (800, 800),
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225),
    use_mosaic: bool = False,
    use_cutmix: bool = False,
):
    """
    Build a training transform pipeline.

    Default pipeline:
        Resize -> RandomHorizontalFlip -> RandomVerticalFlip ->
        RandomBrightnessContrast -> Normalize -> ToTensor

    Args:
        image_size: Target image size.
        mean, std: Normalization parameters (default: ImageNet).
        use_mosaic: If True, prepend Mosaic augmentation.
        use_cutmix: If True, append RandomCutMix.

    Returns:
        A Compose pipeline.
    """
    from deeprs_light.data.transforms import (
        Compose, Resize, RandomHorizontalFlip, RandomVerticalFlip,
        RandomBrightnessContrast, Normalize, ToTensor,
        Mosaic, RandomCutMix,
    )

    transforms = []
    if use_mosaic:
        transforms.append(Mosaic(image_size))

    transforms.extend([
        Resize(image_size),
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.3),
        RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        Normalize(mean=mean, std=std),
        ToTensor(),
    ])

    if use_cutmix:
        transforms.append(RandomCutMix())

    return Compose(transforms)


def get_val_transforms(
    image_size: Union[int, Tuple[int, int]] = (800, 800),
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225),
):
    """
    Build a validation/test transform pipeline.
    Only fixed Resize + Normalize + ToTensor, no random augmentations.
    """
    from deeprs_light.data.transforms import Compose, Resize, Normalize, ToTensor

    return Compose([
        Resize(image_size),
        Normalize(mean=mean, std=std),
        ToTensor(),
    ])

"""
Batch-level transforms applied after collate_fn and before model forward.
These operate on stacked images [B, C, H, W] and list of target dicts.
"""

from typing import Dict, List, Tuple, Callable, Optional
import random

import torch
import numpy as np


class BatchCompose:
    """
    Compose a sequence of batch-level transforms.

    Usage:
        batch_tf = BatchCompose([BatchMosaic((800, 800)), BatchMixUp(0.5)])
        images, targets = batch_tf(images, targets)
    """

    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __call__(
        self, images: torch.Tensor, targets: List[Dict]
    ) -> Tuple[torch.Tensor, List[Dict]]:
        for t in self.transforms:
            images, targets = t(images, targets)
        return images, targets


class BatchMosaic:
    """
    Mosaic 4 adjacent images in a batch into one.
    Each output image = a 2x2 grid of 4 input images.

    Args:
        output_size: (H, W) of the output mosaic image.
        p: Probability of applying.
    """

    def __init__(self, output_size: Tuple[int, int] = (800, 800), p: float = 0.5):
        self.output_size = output_size
        self.p = p

    def __call__(
        self, images: torch.Tensor, targets: List[Dict]
    ) -> Tuple[torch.Tensor, List[Dict]]:
        if random.random() >= self.p:
            return images, targets

        batch_size = images.shape[0]
        if batch_size < 4:
            return images, targets

        out_h, out_w = self.output_size
        new_images = []
        new_targets = []

        for i in range(0, batch_size - 3, 4):
            # Pick 4 images
            idxs = [i, i + 1, i + 2, i + 3]
            canvas = torch.zeros(3, out_h, out_w, dtype=images.dtype, device=images.device)

            # Center point (random offset)
            cx = int(random.uniform(out_w * 0.25, out_w * 0.75))
            cy = int(random.uniform(out_h * 0.25, out_h * 0.75))

            # Place 4 images at corners
            placements = [
                (0, 0, cx, cy),            # top-left
                (cx, 0, out_w, cy),        # top-right
                (0, cy, cx, out_h),        # bottom-left
                (cx, cy, out_w, out_h),     # bottom-right
            ]
            merged_boxes = []
            merged_labels = []
            merged_image_id = None

            for j, (x1, y1, x2, y2) in enumerate(placements):
                idx = idxs[j]
                img = images[idx]  # [3, H, W]
                _, h, w = img.shape

                # Resize image to fit placement area
                pw, ph = x2 - x1, y2 - y1
                scale = min(pw / w, ph / h)
                new_w, new_h = int(w * scale), int(h * scale)
                img_r = torch.nn.functional.interpolate(
                    img.unsqueeze(0), size=(new_h, new_w), mode="bilinear",
                ).squeeze(0)

                # Paste into canvas
                paste_x = x1 + (pw - new_w) // 2
                paste_y = y1 + (ph - new_h) // 2
                canvas[:, paste_y:paste_y + new_h, paste_x:paste_x + new_w] = img_r

                # Adjust boxes
                tgt = targets[idx]
                if tgt["boxes"].shape[0] > 0:
                    boxes = tgt["boxes"].clone()
                    boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale + paste_x
                    boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale + paste_y
                    # Clamp to placement area
                    boxes[:, [0, 2]] = torch.clamp(boxes[:, [0, 2]], x1, x2 - 1)
                    boxes[:, [1, 3]] = torch.clamp(boxes[:, [1, 3]], y1, y2 - 1)
                    valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
                    if valid.any():
                        merged_boxes.append(boxes[valid])
                        merged_labels.append(tgt["labels"][valid])

                if merged_image_id is None:
                    merged_image_id = tgt.get("image_id", 0)

            new_images.append(canvas)
            new_targets.append({
                "image_id": merged_image_id or 0,
                "boxes": torch.cat(merged_boxes, dim=0) if merged_boxes else torch.zeros((0, 4)),
                "labels": torch.cat(merged_labels, dim=0) if merged_labels else torch.zeros((0,), dtype=torch.long),
            })

        if new_images:
            return torch.stack(new_images, dim=0), new_targets
        return images, targets


class BatchMixUp:
    """
    MixUp: linearly interpolate two images and their labels.

    Args:
        alpha: Beta distribution parameter for mixing ratio.
        p: Probability of applying.
    """

    def __init__(self, alpha: float = 0.5, p: float = 0.5):
        self.alpha = alpha
        self.p = p

    def __call__(
        self, images: torch.Tensor, targets: List[Dict]
    ) -> Tuple[torch.Tensor, List[Dict]]:
        if random.random() >= self.p:
            return images, targets

        batch_size = images.shape[0]
        if batch_size < 2:
            return images, targets

        # Random pair indices
        idx = torch.randperm(batch_size, device=images.device)
        lam = np.random.beta(self.alpha, self.alpha)

        # Mix images
        images = lam * images + (1 - lam) * images[idx]

        # Merge targets (concatenate boxes with mix ratio)
        new_targets = []
        for i in range(batch_size):
            tgt_a = dict(targets[i])
            tgt_b = dict(targets[idx[i]])

            merged = {"image_id": tgt_a.get("image_id", 0)}
            # Concatenate boxes and labels from both images
            boxes = torch.cat([tgt_a.get("boxes", torch.zeros((0, 4))),
                             tgt_b.get("boxes", torch.zeros((0, 4)))], dim=0)
            labels = torch.cat([tgt_a.get("labels", torch.zeros((0,), dtype=torch.long)),
                              tgt_b.get("labels", torch.zeros((0,), dtype=torch.long))], dim=0)
            merged["boxes"] = boxes
            merged["labels"] = labels
            merged["_mix_ratio"] = lam
            new_targets.append(merged)

        return images, new_targets

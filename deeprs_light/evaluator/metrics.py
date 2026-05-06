"""
Confusion matrix and remote-sensing-specific evaluation metrics.

Metrics:
  - ConfusionMatrix: Per-class TP/FP/FN accumulation with IoU matching
  - Precision / Recall / F1
  - GTC: Ground Truth Completeness
  - GOC: Geometric Object Completeness
  - GUC: Geometric Usability Completeness
  - PoLis: Point on Line Statistics (localization accuracy)
"""

from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch


# ============================================================
# Confusion Matrix
# ============================================================

class ConfusionMatrix:
    """
    Multi-class confusion matrix with IoU-based TP/FP/FN matching.

    Usage:
        cm = ConfusionMatrix(num_classes=10, iou_threshold=0.5)
        cm.process_batch(gt_boxes_list, gt_labels_list, dt_boxes_list, dt_labels_list, dt_scores_list)
        metrics = cm.compute(["precision", "recall", "f1"])
    """

    def __init__(
        self,
        num_classes: int,
        iou_threshold: float = 0.5,
        class_names: Optional[List[str]] = None,
    ):
        """
        Args:
            num_classes: Number of classes (including background if applicable).
            iou_threshold: IoU threshold for considering a detection as TP.
            class_names: Optional list of class names for readability.
        """
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold
        self.class_names = class_names or [str(i) for i in range(num_classes)]
        self.reset()

    def reset(self):
        """Reset all counters."""
        self._counts: Dict[int, Dict[str, int]] = {
            c: {"tp": 0, "fp": 0, "fn": 0}
            for c in range(self.num_classes)
        }

    def process_batch(
        self,
        gt_boxes: List[torch.Tensor],
        gt_labels: List[torch.Tensor],
        dt_boxes: List[torch.Tensor],
        dt_labels: List[torch.Tensor],
        dt_scores: List[torch.Tensor],
    ):
        """
        Accumulate TP/FP/FN from a batch.

        Matching strategy (per image, per class):
        1. Sort detections by score descending
        2. For each detection, find the best-matching GT of the same class
           with IoU > iou_threshold that hasn't been matched yet
        3. Successful match -> TP, else -> FP
        4. Unmatched GTs -> FN

        Args:
            gt_boxes: List of GT box tensors [N_i, 4] xyxy.
            gt_labels: List of GT label tensors [N_i].
            dt_boxes: List of DT box tensors [M_i, 4] xyxy.
            dt_labels: List of DT label tensors [M_i].
            dt_scores: List of DT score tensors [M_i].
        """
        from deeprs_light.data.transforms_utils import compute_iou_matrix

        for gt_b, gt_l, dt_b, dt_l, dt_s in zip(
            gt_boxes, gt_labels, dt_boxes, dt_labels, dt_scores
        ):
            # Skip empty images
            if gt_b.numel() == 0 and dt_b.numel() == 0:
                continue

            gt_b = gt_b.numpy() if isinstance(gt_b, torch.Tensor) else np.array(gt_b)
            gt_l = gt_l.numpy() if isinstance(gt_l, torch.Tensor) else np.array(gt_l)
            dt_b = dt_b.numpy() if isinstance(dt_b, torch.Tensor) else np.array(dt_b)
            dt_l = dt_l.numpy() if isinstance(dt_l, torch.Tensor) else np.array(dt_l)
            dt_s = dt_s.numpy() if isinstance(dt_s, torch.Tensor) else np.array(dt_s)

            for c in range(self.num_classes):
                gt_mask = gt_l == c
                dt_mask = dt_l == c

                gt_c = gt_b[gt_mask]
                dt_c = dt_b[dt_mask]
                dt_s_c = dt_s[dt_mask]

                if len(gt_c) == 0 and len(dt_c) == 0:
                    continue

                if len(gt_c) == 0:
                    self._counts[c]["fp"] += len(dt_c)
                    continue

                if len(dt_c) == 0:
                    self._counts[c]["fn"] += len(gt_c)
                    continue

                # Compute IoU matrix
                ious = compute_iou_matrix(gt_c, dt_c)  # [N_gt, N_dt]

                # Match
                gt_matched = set()
                dt_order = np.argsort(-dt_s_c)

                for dt_idx in dt_order:
                    best_iou = 0
                    best_gt = -1
                    for gt_idx in range(len(gt_c)):
                        if gt_idx in gt_matched:
                            continue
                        if ious[gt_idx, dt_idx] > best_iou:
                            best_iou = ious[gt_idx, dt_idx]
                            best_gt = gt_idx

                    if best_iou >= self.iou_threshold and best_gt >= 0:
                        self._counts[c]["tp"] += 1
                        gt_matched.add(best_gt)
                    else:
                        self._counts[c]["fp"] += 1

                self._counts[c]["fn"] += len(gt_c) - len(gt_matched)

    def compute(
        self,
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Compute evaluation metrics from accumulated counts.

        Args:
            metrics: List of metric names to compute.
                     Default: ["precision", "recall", "f1"].

        Returns:
            Dict with per_class and macro/micro averages.
        """
        if metrics is None:
            metrics = ["precision", "recall", "f1"]

        per_class = {}
        total_tp, total_fp, total_fn = 0, 0, 0

        for c in range(self.num_classes):
            tp = self._counts[c]["tp"]
            fp = self._counts[c]["fp"]
            fn = self._counts[c]["fn"]
            p, r, f1 = compute_precision_recall_f1(tp, fp, fn)

            per_class[c] = {
                "class_name": self.class_names[c],
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }
            if "precision" in metrics:
                per_class[c]["precision"] = p
            if "recall" in metrics:
                per_class[c]["recall"] = r
            if "f1" in metrics:
                per_class[c]["f1"] = f1

            total_tp += tp
            total_fp += fp
            total_fn += fn

        # Macro average
        macro_avg = {}
        if "precision" in metrics:
            macro_avg["precision"] = np.mean([per_class[c]["precision"] for c in range(self.num_classes)])
        if "recall" in metrics:
            macro_avg["recall"] = np.mean([per_class[c]["recall"] for c in range(self.num_classes)])
        if "f1" in metrics:
            macro_avg["f1"] = np.mean([per_class[c]["f1"] for c in range(self.num_classes)])

        # Micro average
        mp, mr, mf1 = compute_precision_recall_f1(total_tp, total_fp, total_fn)
        micro_avg = {}
        if "precision" in metrics:
            micro_avg["precision"] = mp
        if "recall" in metrics:
            micro_avg["recall"] = mr
        if "f1" in metrics:
            micro_avg["f1"] = mf1

        return {
            "per_class": per_class,
            "macro_avg": macro_avg,
            "micro_avg": micro_avg,
        }


# ============================================================
# Base metrics
# ============================================================

def compute_precision_recall_f1(
    tp: int, fp: int, fn: int,
) -> Tuple[float, float, float]:
    """
    Compute precision, recall, and F1 from TP/FP/FN counts.

    Precision = TP / (TP + FP)
    Recall    = TP / (TP + FN)
    F1        = 2 * P * R / (P + R)

    Returns 0.0 for each metric when denominator is 0.
    """
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


# ============================================================
# Remote Sensing Quality Metrics
# ============================================================

def compute_gtc(tp: int, fn: int) -> float:
    """
    Ground Truth Completeness (GTC) — 地物真值完整性。

    Measures what proportion of ground truth objects are correctly detected.
    Equivalent to Recall, but contextualized for remote sensing:
    how completely does the algorithm capture real-world features?

    GTC = TP / (TP + FN)
    """
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def compute_goc(
    gt_areas: np.ndarray,
    dt_areas: np.ndarray,
    match_indices: List[Tuple[int, int]],
    iou_matrix: Optional[np.ndarray] = None,
) -> float:
    """
    Geometric Object Completeness (GOC) — 几何对象完整性。

    Measures how well detected objects cover ground truth in terms of
    geometric area. Uses intersection area of matched pairs relative
    to GT area.

    GOC = sum(intersection_area(gt_i, dt_j)) / sum(area(gt_i))

    Where (gt_i, dt_j) are matched pairs.

    Args:
        gt_areas: Area of each GT box [N_gt].
        dt_areas: Area of each DT box [N_dt].
        match_indices: List of (gt_idx, dt_idx) matched pairs.
        iou_matrix: Pre-computed IoU matrix [N_gt, N_dt] (optional).

    Returns:
        GOC score in [0, 1].
    """
    if len(gt_areas) == 0:
        return 0.0
    if len(match_indices) == 0:
        return 0.0

    total_gt_area = float(np.sum(gt_areas))
    if total_gt_area == 0:
        return 0.0

    inter_area_sum = 0.0
    for gt_idx, dt_idx in match_indices:
        if iou_matrix is not None:
            iou_val = iou_matrix[gt_idx, dt_idx]
            # Approximate intersection from IoU
            union = gt_areas[gt_idx] + dt_areas[dt_idx]
            inter_area_sum += iou_val * union / (1.0 + iou_val)
        else:
            # Conservative: use min area
            inter_area_sum += min(gt_areas[gt_idx], dt_areas[dt_idx])

    return inter_area_sum / total_gt_area


def compute_guc(
    gt_areas: np.ndarray,
    dt_areas: np.ndarray,
    match_indices: List[Tuple[int, int]],
    iou_matrix: Optional[np.ndarray] = None,
    usability_threshold: float = 0.5,
) -> float:
    """
    Geometric Usability Completeness (GUC) — 几何可用性完整性。

    Extends GOC by introducing a "usability" weight:
    only matched pairs with IoU >= usability_threshold contribute fully.
    Those below contribute proportionally.

    GUC = sum(w_k * intersection_area_k) / sum(area(gt_i))

    where w_k = 1 if IoU_k >= usability_threshold else IoU_k / usability_threshold

    Args:
        gt_areas: Area of each GT box [N_gt].
        dt_areas: Area of each DT box [N_dt].
        match_indices: List of (gt_idx, dt_idx) matched pairs.
        iou_matrix: Pre-computed IoU matrix [N_gt, N_dt].
        usability_threshold: IoU threshold above which detections count as "fully usable".

    Returns:
        GUC score in [0, 1].
    """
    if len(gt_areas) == 0:
        return 0.0
    if len(match_indices) == 0:
        return 0.0

    total_gt_area = float(np.sum(gt_areas))
    if total_gt_area == 0:
        return 0.0

    weighted_inter_sum = 0.0
    for gt_idx, dt_idx in match_indices:
        if iou_matrix is not None:
            iou_val = iou_matrix[gt_idx, dt_idx]
        else:
            iou_val = 0.5  # Fallback

        # Weight: fully usable if IoU >= threshold, else proportional
        if iou_val >= usability_threshold:
            w = 1.0
        else:
            w = iou_val / usability_threshold

        # Approximate intersection
        union = gt_areas[gt_idx] + dt_areas[dt_idx]
        inter_area = iou_val * union / (1.0 + iou_val)
        weighted_inter_sum += w * inter_area

    return weighted_inter_sum / total_gt_area


def compute_polis(
    gt_boxes: np.ndarray,
    dt_boxes: np.ndarray,
    match_indices: List[Tuple[int, int]],
    buffer_distance: float = 2.0,
) -> Dict[str, float]:
    """
    Point/Position on Line Statistics (PoLis) — 点/线位置统计。

    Measures localization accuracy by computing center-point distances
    between matched GT-DT pairs. Relevant for evaluating object positioning
    in remote sensing (buildings, vehicles, ships).

    Computes:
    - Center-point distances for each matched pair
    - Statistical summary: mean, std, median, max, RMSE
    - Rate of centers within buffer_distance of GT center

    Args:
        gt_boxes: GT boxes [N_gt, 4] xyxy.
        dt_boxes: DT boxes [N_dt, 4] xyxy.
        match_indices: Matched (gt_idx, dt_idx) pairs.
        buffer_distance: Buffer radius in pixels for buffer_rate.

    Returns:
        Dict with: mean_dist, std_dist, median_dist, max_dist, rmse, buffer_rate
    """
    if len(match_indices) == 0:
        return {
            "mean_dist": 0.0,
            "std_dist": 0.0,
            "median_dist": 0.0,
            "max_dist": 0.0,
            "rmse": 0.0,
            "buffer_rate": 0.0,
        }

    # Compute centers
    gt_centers = np.stack([
        (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2,
        (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2,
    ], axis=1)

    dt_centers = np.stack([
        (dt_boxes[:, 0] + dt_boxes[:, 2]) / 2,
        (dt_boxes[:, 1] + dt_boxes[:, 3]) / 2,
    ], axis=1)

    distances = []
    within_buffer = 0

    for gt_idx, dt_idx in match_indices:
        gt_c = gt_centers[gt_idx]
        dt_c = dt_centers[dt_idx]
        dist = float(np.linalg.norm(gt_c - dt_c))
        distances.append(dist)
        if dist <= buffer_distance:
            within_buffer += 1

    distances = np.array(distances)

    return {
        "mean_dist": float(np.mean(distances)),
        "std_dist": float(np.std(distances)),
        "median_dist": float(np.median(distances)),
        "max_dist": float(np.max(distances)),
        "rmse": float(np.sqrt(np.mean(distances ** 2))),
        "buffer_rate": within_buffer / len(distances),
    }


def compute_rs_quality_metrics(
    confusion_matrix: ConfusionMatrix,
    gt_areas_all: np.ndarray,
    dt_areas_all: np.ndarray,
    match_indices_all: List[Tuple[int, int]],
    gt_boxes_all: np.ndarray,
    dt_boxes_all: np.ndarray,
    iou_matrix: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    One-stop computation of all remote sensing quality metrics.

    Returns:
    {
        "GTC": float,
        "GOC": float,
        "GUC": float,
        "PoLis_mean_dist": float,
        "PoLis_std_dist": float,
        "PoLis_median_dist": float,
        "PoLis_rmse": float,
        "PoLis_buffer_rate": float,
    }
    """
    # Aggregate TP/FN across classes for GTC
    total_tp = sum(c["tp"] for c in confusion_matrix._counts.values())
    total_fn = sum(c["fn"] for c in confusion_matrix._counts.values())

    gtc = compute_gtc(total_tp, total_fn)
    goc = compute_goc(gt_areas_all, dt_areas_all, match_indices_all, iou_matrix)
    guc = compute_guc(gt_areas_all, dt_areas_all, match_indices_all, iou_matrix)
    polis = compute_polis(gt_boxes_all, dt_boxes_all, match_indices_all)

    return {
        "GTC": gtc,
        "GOC": goc,
        "GUC": guc,
        "PoLis_mean_dist": polis["mean_dist"],
        "PoLis_std_dist": polis["std_dist"],
        "PoLis_median_dist": polis["median_dist"],
        "PoLis_rmse": polis["rmse"],
        "PoLis_buffer_rate": polis["buffer_rate"],
    }

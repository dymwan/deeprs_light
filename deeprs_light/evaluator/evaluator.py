"""
DeepRSEvaluator: standardized evaluator combining COCO mAP,
confusion matrix, and remote sensing quality metrics.
"""

import json
import os
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import torch

from deeprs_light.evaluator.coco_eval import evaluate_coco, match_predictions_to_gt
from deeprs_light.evaluator.metrics import (
    ConfusionMatrix,
    compute_rs_quality_metrics,
)
from deeprs_light.data.transforms_utils import boxes_xyxy_to_xywh


class DeepRSEvaluator:
    """
    Standardized evaluator combining COCO metrics, confusion matrix,
    and remote-sensing-specific quality metrics.

    Interface:
    - process(inputs, outputs): Collect predictions after each inference batch.
    - evaluate(): Compute all metrics after full validation set traversal.
    - save_results(path): Persist results to JSON or TXT.

    Usage:
        evaluator = DeepRSEvaluator(ann_file="val.json", iou_type="bbox")

        for images, targets in val_dataloader:
            outputs = model(images)
            evaluator.process(targets, outputs)

        results = evaluator.evaluate()
        evaluator.save_results("results/exp1.json")
    """

    def __init__(
        self,
        ann_file: str,
        iou_type: str = "bbox",
        max_dets: Tuple[int, ...] = (100, 300, 1000),
        num_classes: Optional[int] = None,
        class_names: Optional[List[str]] = None,
        iou_threshold: float = 0.5,
    ):
        """
        Args:
            ann_file: Path to COCO annotation JSON (ground truth).
            iou_type: "bbox" for detection, "segm" for instance segmentation.
            max_dets: Max detections per image for AR.
            num_classes: Number of classes (inferred from COCO if None).
            class_names: Class name list.
            iou_threshold: IoU threshold for confusion matrix and RS metrics.
        """
        self.ann_file = ann_file
        self.iou_type = iou_type
        self.max_dets = max_dets
        self.iou_threshold = iou_threshold

        # Load COCO GT
        from pycocotools.coco import COCO
        self.coco_gt = COCO(ann_file)

        # Determine class info
        cat_ids = sorted(self.coco_gt.getCatIds())
        if num_classes is None:
            num_classes = len(cat_ids)
        if class_names is None:
            class_names = [c["name"] for c in self.coco_gt.loadCats(cat_ids)]
        self.num_classes = num_classes
        self.class_names = class_names
        self._cat_id_to_idx = {cat_id: i for i, cat_id in enumerate(cat_ids)}

        # Accumulators
        self.reset()

    def reset(self):
        """Reset all accumulators."""
        self._predictions: List[Dict] = []
        # For confusion matrix
        self._gt_boxes: List[torch.Tensor] = []
        self._gt_labels: List[torch.Tensor] = []
        self._dt_boxes: List[torch.Tensor] = []
        self._dt_labels: List[torch.Tensor] = []
        self._dt_scores: List[torch.Tensor] = []
        # For RS metrics
        self._gt_areas: List[float] = []
        self._dt_areas: List[float] = []
        self._gt_boxes_all: List[np.ndarray] = []
        self._dt_boxes_all: List[np.ndarray] = []

    def process(
        self,
        inputs: List[Dict],
        outputs: List[Dict],
    ):
        """
        Collect predictions from a batch.

        Converts model outputs to COCO results format and stores raw
        GT/DT data for confusion matrix and RS quality metrics.

        Args:
            inputs: Batch input metadata from DataLoader.
                    Each element should have at least "image_id".
                    Optionally contains GT boxes/labels for evaluation.
            outputs: Model outputs.
                     For detection: {"boxes": [N,4] xyxy, "scores": [N], "labels": [N]}
                     For segmentation: additionally {"masks": [N,H,W]}
        """
        for i, (inp, out) in enumerate(zip(inputs, outputs)):
            image_id = inp.get("image_id", 0)

            boxes = out.get("boxes")
            scores = out.get("scores")
            labels = out.get("labels")

            if boxes is None or scores is None or labels is None:
                continue

            if isinstance(boxes, torch.Tensor):
                boxes = boxes.detach().cpu()
            if isinstance(scores, torch.Tensor):
                scores = scores.detach().cpu()
            if isinstance(labels, torch.Tensor):
                labels = labels.detach().cpu()

            if boxes.shape[0] == 0:
                continue

            # Convert xyxy -> xywh for COCO results
            boxes_np = boxes.numpy()
            xywh = boxes_xyxy_to_xywh(boxes_np)
            w_arr = xywh[:, 2]
            h_arr = xywh[:, 3]

            for j in range(boxes.shape[0]):
                label_idx = int(labels[j].item())
                # Map back to COCO category_id
                cat_id = self._get_cat_id(label_idx)

                pred = {
                    "image_id": int(image_id),
                    "category_id": cat_id,
                    "bbox": [float(xywh[j, 0]), float(xywh[j, 1]),
                             float(w_arr[j]), float(h_arr[j])],
                    "score": float(scores[j].item()),
                }
                # Add segmentation if present
                if "masks" in out:
                    mask = out["masks"][j]
                    try:
                        from pycocotools import mask as mask_util
                        if isinstance(mask, torch.Tensor):
                            mask = mask.numpy()
                        rle = mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))
                        if isinstance(rle["counts"], bytes):
                            rle["counts"] = rle["counts"].decode("ascii")
                        pred["segmentation"] = rle
                    except Exception:
                        pass

                self._predictions.append(pred)

            # Store for confusion matrix
            self._dt_boxes.append(torch.from_numpy(boxes_np))
            self._dt_labels.append(torch.tensor([self._cat_id_to_idx.get(l.item(), l.item()) for l in labels], dtype=torch.long))
            self._dt_scores.append(scores.clone())

            # GT from input
            if "boxes" in inp and inp["boxes"].shape[0] > 0:
                gt_boxes = inp["boxes"]
                if isinstance(gt_boxes, torch.Tensor):
                    gt_boxes = gt_boxes.detach().cpu()
                self._gt_boxes.append(gt_boxes)

                gt_labels = inp.get("labels", torch.zeros(gt_boxes.shape[0], dtype=torch.long))
                if isinstance(gt_labels, torch.Tensor):
                    gt_labels = gt_labels.detach().cpu()
                self._gt_labels.append(gt_labels)

                if "area" in inp:
                    areas = inp["area"].numpy() if isinstance(inp["area"], torch.Tensor) else inp["area"]
                    self._gt_areas.extend(areas.tolist())
                    self._gt_boxes_all.append(gt_boxes.numpy())
            else:
                self._gt_boxes.append(torch.zeros((0, 4)))
                self._gt_labels.append(torch.zeros((0,), dtype=torch.long))

            # DT areas
            if boxes.shape[0] > 0:
                dt_areas = w_arr * h_arr
                self._dt_areas.extend(dt_areas.tolist())
                self._dt_boxes_all.append(boxes_np)

    def evaluate(self) -> Dict[str, Any]:
        """
        Compute all evaluation metrics.

        Returns:
            {
                "coco": {"AP": ..., "AP50": ..., ...},
                "classification": {"per_class": {...}, "macro_avg": {...}, ...},
                "rs_quality": {"GTC": ..., "GOC": ..., "GUC": ..., "PoLis_...": ...}
            }
        """
        results = {}

        # --- COCO metrics ---
        if len(self._predictions) > 0:
            coco_dt = self.coco_gt.loadRes(self._predictions)
            coco_results = evaluate_coco(
                self.coco_gt, coco_dt,
                iou_type=self.iou_type,
                max_dets=self.max_dets,
            )
        else:
            coco_results = {"AP": 0.0, "AP50": 0.0, "AP75": 0.0,
                           "AP_s": 0.0, "AP_m": 0.0, "AP_l": 0.0,
                           "AR1": 0.0, "AR10": 0.0, "AR100": 0.0,
                           "AR_s": 0.0, "AR_m": 0.0, "AR_l": 0.0}
        results["coco"] = coco_results

        # --- Confusion matrix ---
        cm = ConfusionMatrix(
            num_classes=self.num_classes,
            iou_threshold=self.iou_threshold,
            class_names=self.class_names,
        )
        if len(self._gt_boxes) > 0:
            cm.process_batch(
                self._gt_boxes, self._gt_labels,
                self._dt_boxes, self._dt_labels, self._dt_scores,
            )
        class_results = cm.compute()
        results["classification"] = class_results

        # --- RS quality metrics ---
        gt_areas_arr = np.array(self._gt_areas)
        dt_areas_arr = np.array(self._dt_areas)
        gt_boxes_arr = np.concatenate(self._gt_boxes_all, axis=0) if self._gt_boxes_all else np.zeros((0, 4))
        dt_boxes_arr = np.concatenate(self._dt_boxes_all, axis=0) if self._dt_boxes_all else np.zeros((0, 4))

        # Build match indices from confusion matrix
        # For RS metrics, we aggregate global matches
        match_indices = self._build_global_matches()

        rs_results = compute_rs_quality_metrics(
            confusion_matrix=cm,
            gt_areas_all=gt_areas_arr,
            dt_areas_all=dt_areas_arr,
            match_indices_all=match_indices,
            gt_boxes_all=gt_boxes_arr,
            dt_boxes_all=dt_boxes_arr,
        )
        results["rs_quality"] = rs_results

        return results

    def _build_global_matches(self) -> List[Tuple[int, int]]:
        """Build global (gt_idx, dt_idx) match pairs from accumulated data."""
        from deeprs_light.data.transforms_utils import compute_iou_matrix

        matches = []
        gt_offset = 0
        dt_offset = 0

        for gt_b, dt_b, gt_l, dt_l in zip(
            self._gt_boxes, self._dt_boxes,
            self._gt_labels, self._dt_labels,
        ):
            gt_np = gt_b.numpy() if isinstance(gt_b, torch.Tensor) else np.array(gt_b)
            dt_np = dt_b.numpy() if isinstance(dt_b, torch.Tensor) else np.array(dt_b)
            gt_lnp = gt_l.numpy() if isinstance(gt_l, torch.Tensor) else np.array(gt_l)
            dt_lnp = dt_l.numpy() if isinstance(dt_l, torch.Tensor) else np.array(dt_l)

            if len(gt_np) == 0 or len(dt_np) == 0:
                gt_offset += len(gt_np)
                dt_offset += len(dt_np)
                continue

            ious = compute_iou_matrix(gt_np, dt_np)
            gt_matched = set()

            for dt_local in range(len(dt_np)):
                best_iou = 0
                best_gt = -1
                for gt_local in range(len(gt_np)):
                    if gt_local in gt_matched:
                        continue
                    if gt_lnp[gt_local] != dt_lnp[dt_local]:
                        continue
                    if ious[gt_local, dt_local] > best_iou:
                        best_iou = ious[gt_local, dt_local]
                        best_gt = gt_local

                if best_iou >= self.iou_threshold and best_gt >= 0:
                    matches.append((gt_offset + best_gt, dt_offset + dt_local))
                    gt_matched.add(best_gt)

            gt_offset += len(gt_np)
            dt_offset += len(dt_np)

        return matches

    def _get_cat_id(self, label_idx: int) -> int:
        """Map label index back to COCO category_id."""
        cat_ids = sorted(self.coco_gt.getCatIds())
        if label_idx < len(cat_ids):
            return cat_ids[label_idx]
        return label_idx

    def save_results(
        self,
        output_path: str,
        format: str = "json",
    ):
        """
        Persist evaluation results.

        JSON format:
        {
            "meta": {"ann_file": ..., "iou_type": ..., "num_predictions": ...},
            "coco": {"AP": ..., ...},
            "classification": {...},
            "rs_quality": {...}
        }

        TXT format: grouped key-value pairs.
        """
        results = getattr(self, "_last_results", None)
        if results is None:
            results = self.evaluate()

        if format == "json":
            self._save_json(results, output_path)
        elif format == "txt":
            self._save_txt(results, output_path)
        else:
            raise ValueError(f"Unknown format: {format}. Use 'json' or 'txt'.")

    def _save_json(self, results: Dict, output_path: str):
        """Save results as JSON."""
        output = {
            "meta": {
                "ann_file": self.ann_file,
                "iou_type": self.iou_type,
                "num_predictions": len(self._predictions),
                "num_classes": self.num_classes,
                "iou_threshold": self.iou_threshold,
            },
            "coco": results.get("coco", {}),
            "classification": {
                "macro_avg": results.get("classification", {}).get("macro_avg", {}),
                "micro_avg": results.get("classification", {}).get("micro_avg", {}),
                "per_class": results.get("classification", {}).get("per_class", {}),
            },
            "rs_quality": results.get("rs_quality", {}),
        }

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"[DeepRSEvaluator] Results saved to '{output_path}'")

    def _save_txt(self, results: Dict, output_path: str):
        """Save results as a plain text report."""
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            # COCO metrics
            f.write("=== COCO Metrics ===\n")
            for k, v in results.get("coco", {}).items():
                f.write(f"{k}: {v:.4f}\n")

            # Classification
            f.write("\n=== Classification ===\n")
            clf = results.get("classification", {})
            if "macro_avg" in clf:
                f.write("-- Macro Average --\n")
                for k, v in clf["macro_avg"].items():
                    f.write(f"{k}: {v:.4f}\n")
            if "micro_avg" in clf:
                f.write("-- Micro Average --\n")
                for k, v in clf["micro_avg"].items():
                    f.write(f"{k}: {v:.4f}\n")

            # RS Quality
            f.write("\n=== RS Quality ===\n")
            for k, v in results.get("rs_quality", {}).items():
                f.write(f"{k}: {v:.4f}\n")

        print(f"[DeepRSEvaluator] Report saved to '{output_path}'")

"""
Thin wrapper around pycocotools for standardized COCO evaluation.
"""

from typing import Dict, List, Tuple


def evaluate_coco(
    coco_gt,
    coco_dt,
    iou_type: str = "bbox",
    max_dets: Tuple[int, ...] = (100, 300, 1000),
) -> Dict[str, float]:
    """
    Evaluate COCO-format detections against ground truth.

    Args:
        coco_gt: pycocotools COCO object (ground truth).
        coco_dt: pycocotools COCO object (detections, from COCO.loadRes()).
        iou_type: "bbox" for object detection, "segm" for instance segmentation.
        max_dets: Maximum detections per image for AR computation.

    Returns:
        Dict with keys:
            AP, AP50, AP75, AP_s, AP_m, AP_l,
            AR1, AR10, AR100, AR_s, AR_m, AR_l

    Process:
        1. Create COCOeval(coco_gt, coco_dt, iou_type)
        2. Set params.maxDets
        3. coco_eval.evaluate(), .accumulate(), .summarize()
        4. Extract stats from coco_eval.stats
    """
    from pycocotools.cocoeval import COCOeval

    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
    if max_dets:
        coco_eval.params.maxDets = list(max_dets)

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # coco_eval.stats order:
    # [AP, AP50, AP75, AP_s, AP_m, AP_l,
    #  AR1, AR10, AR100, AR_s, AR_m, AR_l]
    stats = coco_eval.stats

    return {
        "AP": float(stats[0]),
        "AP50": float(stats[1]),
        "AP75": float(stats[2]),
        "AP_s": float(stats[3]),
        "AP_m": float(stats[4]),
        "AP_l": float(stats[5]),
        "AR1": float(stats[6]),
        "AR10": float(stats[7]),
        "AR100": float(stats[8]),
        "AR_s": float(stats[9]),
        "AR_m": float(stats[10]),
        "AR_l": float(stats[11]),
    }


def match_predictions_to_gt(
    coco_gt,
    coco_dt,
    iou_type: str = "bbox",
    iou_threshold: float = 0.5,
    max_dets: int = 100,
) -> List[Dict]:
    """
    Match predictions to ground truth at a specific IoU threshold.
    Used as input for confusion matrix and RS quality metrics.

    Args:
        coco_gt: pycocotools COCO object (ground truth).
        coco_dt: pycocotools COCO object (detections).
        iou_type: "bbox" or "segm".
        iou_threshold: IoU threshold for a match.
        max_dets: Max detections per image.

    Returns:
        List of per-image match results:
        [
            {
                "image_id": int,
                "tp": [...],       # annotation_id for each TP
                "fp": [...],       # annotation_id for each FP
                "fn": [...],       # annotation_id for each FN
                "scores": [...],   # confidence score for each detection
                "gt_ids": [...],   # annotation_id for each GT
                "dt_ids": [...],   # annotation_id for each DT
            },
            ...
        ]
    """
    from pycocotools.cocoeval import COCOeval
    import numpy as np

    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
    coco_eval.params.maxDets = [max_dets]
    coco_eval.params.iouThrs = np.array([iou_threshold])

    coco_eval.evaluate()
    coco_eval.accumulate()

    results = []
    img_ids = list(coco_gt.getImgIds())

    for img_id in img_ids:
        # Get annotations for this image
        gt_ann_ids = coco_gt.getAnnIds(imgIds=[img_id])
        dt_ann_ids = coco_dt.getAnnIds(imgIds=[img_id])

        tp = []
        fp = []
        fn = []
        scores = []

        if len(gt_ann_ids) == 0 and len(dt_ann_ids) == 0:
            results.append({
                "image_id": img_id,
                "tp": [], "fp": [], "fn": [],
                "scores": [], "gt_ids": [], "dt_ids": [],
            })
            continue

        gts = coco_gt.loadAnns(gt_ann_ids)
        dts = coco_dt.loadAnns(dt_ann_ids)

        # Simple greedy matching
        gt_boxes = np.array([g["bbox"] for g in gts])  # xywh
        dt_boxes = np.array([d["bbox"] for d in dts])  # xywh
        dt_scores = np.array([d.get("score", 1.0) for d in dts])

        # Convert xywh to xyxy for IoU
        def xywh_to_xyxy(b):
            if len(b) == 0:
                return np.zeros((0, 4))
            return np.stack([b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]], axis=1)

        gt_xyxy = xywh_to_xyxy(gt_boxes)
        dt_xyxy = xywh_to_xyxy(dt_boxes)

        # Compute IoU matrix
        from deeprs_light.data.transforms_utils import compute_iou_matrix
        ious = compute_iou_matrix(gt_xyxy, dt_xyxy)

        gt_matched = set()
        dt_matched = set()

        # Sort detections by score descending
        dt_order = np.argsort(-dt_scores) if len(dt_scores) > 0 else []

        for dt_idx in dt_order:
            best_iou = 0
            best_gt = -1
            for gt_idx in range(len(gts)):
                if gt_idx in gt_matched:
                    continue
                if ious[gt_idx, dt_idx] > best_iou:
                    best_iou = ious[gt_idx, dt_idx]
                    best_gt = gt_idx

            if best_iou >= iou_threshold and best_gt >= 0:
                tp.append(gt_ann_ids[best_gt])
                gt_matched.add(best_gt)
            else:
                fp.append(dt_ann_ids[dt_idx])
            scores.append(dt_scores[dt_idx])

        # Unmatched GTs are FN
        for gt_idx in range(len(gts)):
            if gt_idx not in gt_matched:
                fn.append(gt_ann_ids[gt_idx])

        results.append({
            "image_id": img_id,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "scores": scores,
            "gt_ids": gt_ann_ids,
            "dt_ids": dt_ann_ids,
        })

    return results

#!/usr/bin/env python
"""
acc_ass.py — Remote sensing accuracy assessment tool.

Supports vector (shapefile, geopackage) and raster (GeoTIFF) inputs
with multiple precision types: confusion matrix (cm), average precision (ap),
geometric quality (go), and point/line statistics (pl).

Usage — single pair:
    python Scripts/acc_ass.py \
        --pred pred.shp --gt gt.tif --mode 10 \
        --precision cm,ap --output results.csv

Usage — batch (multiple pairs + overall):
    python Scripts/acc_ass.py \
        --pairs pairs.csv \
        --precision all --output results.csv

See Scripts/readme.md for full documentation.
"""

import argparse
import csv as csv_module
import os
import sys
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

# Add project root to path for deeprs_light imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Remote sensing accuracy assessment (vector & raster)."
    )
    # Mutually exclusive: single pair vs batch CSV
    single_group = parser.add_argument_group("Single pair mode")
    single_group.add_argument("--pred", default=None, help="Prediction file path (shp/gpkg/tif).")
    single_group.add_argument("--gt", default=None, help="Ground truth file path (shp/gpkg/tif).")
    single_group.add_argument("--mode", default=None,
                              help="Mode string 'xy': 0=vector, 1=raster, 2=continuous.")

    batch_group = parser.add_argument_group("Batch mode")
    batch_group.add_argument("--pairs", default=None,
                             help="CSV file with columns: name,pred,gt,mode. "
                                  "Each row is one prediction-ground truth pair. "
                                  "Overall metrics are automatically computed.")

    parser.add_argument("--precision", required=True,
                        help="Precision types: cm, ap, go, pl. Comma-separated or 'all'.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    parser.add_argument("--field", default=None,
                        help="Class field name(s). Single string for both, "
                             "or 'pred_field,gt_field' for separate.")
    parser.add_argument("--band", type=int, default=None, help="Raster band index (default: 1).")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold (default: 0.5).")
    return parser.parse_args()


def validate_mode(mode: str):
    """Validate mode string. Raises NotImplementedError for unsupported modes."""
    if len(mode) != 2:
        raise ValueError(f"Mode must be a 2-digit string, got '{mode}'")
    pred_type = int(mode[0])
    gt_type = int(mode[1])
    if pred_type == 2 or gt_type == 2:
        raise NotImplementedError(
            f"Mode '{mode}': continuous probability raster (type 2) not yet supported."
        )
    if pred_type not in (0, 1) or gt_type not in (0, 1):
        raise ValueError(f"Invalid mode '{mode}': each digit must be 0, 1, or 2.")
    return mode


def parse_field(field: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    if field is None:
        return None, None
    parts = [f.strip() for f in field.split(",")]
    if len(parts) == 1:
        return parts[0], parts[0]
    if len(parts) == 2:
        return parts[0], parts[1]
    raise ValueError(
        f"Field must be a single name or 'pred_field,gt_field', got '{field}'"
    )


def parse_precisions(precision_str: str) -> List[str]:
    if precision_str.strip().lower() == "all":
        return ["cm", "ap", "go", "pl"]
    parts = [p.strip().lower() for p in precision_str.split(",")]
    valid = {"cm", "ap", "go", "pl"}
    for p in parts:
        if p not in valid:
            raise ValueError(f"Unknown precision type '{p}'. Valid: {valid}")
    return parts


# ============================================================
# File type detection
# ============================================================

def _is_raster(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    return ext in (".tif", ".tiff", ".vrt")


def _is_vector(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    return ext in (".shp", ".gpkg", ".geojson", ".gml", ".kml")


# ============================================================
# Data Loading
# ============================================================

def load_vector(
    path: str, field: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str], int, object]:
    try:
        import geopandas as gpd
    except ImportError:
        raise ImportError("geopandas is required for vector loading. pip install geopandas")

    gdf = gpd.read_file(path)
    crs = gdf.crs

    boxes = []
    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            boxes.append([0, 0, 0, 0])
        else:
            minx, miny, maxx, maxy = geom.bounds
            boxes.append([minx, miny, maxx, maxy])
    boxes = np.array(boxes, dtype=np.float32)

    if field is not None and field in gdf.columns:
        labels = gdf[field].values
        unique_vals = sorted(set(labels))
        val_to_id = {v: i for i, v in enumerate(unique_vals)}
        labels = np.array([val_to_id[v] for v in labels], dtype=np.int64)
        class_names = [str(v) for v in unique_vals]
    else:
        labels = np.zeros(len(boxes), dtype=np.int64)
        class_names = ["object"]
    num_classes = len(class_names)

    return boxes, labels, class_names, num_classes, crs


def load_raster(
    path: str, band: Optional[int] = None,
) -> Tuple[np.ndarray, object, object]:
    try:
        import rasterio
    except ImportError:
        raise ImportError("rasterio is required for raster loading. pip install rasterio")

    if band is None:
        band = 1

    with rasterio.open(path) as src:
        arr = src.read(band)
        transform = src.transform
        crs = src.crs

    return arr, transform, crs


def load_data(
    path: str, data_type: str, band: Optional[int], field: Optional[str],
) -> Dict:
    data_type = int(data_type)
    if data_type == 0:
        if not _is_vector(path):
            raise ValueError(
                f"Mode expects vector input (type 0), but '{path}' does not look like a vector file."
            )
        boxes, labels, class_names, num_classes, crs = load_vector(path, field)
        return {
            "type": "vector", "path": path,
            "boxes": boxes, "labels": labels,
            "class_names": class_names, "num_classes": num_classes, "crs": crs,
        }
    elif data_type == 1:
        if not _is_raster(path):
            print(f"[WARN] Mode expects raster (type 1), but '{path}' does not look like a tif.")
        arr, transform, crs = load_raster(path, band)
        return {
            "type": "raster", "path": path,
            "array": arr, "transform": transform, "crs": crs,
        }
    else:
        raise NotImplementedError(f"Data type {data_type} not supported.")


# ============================================================
# Raster -> Vector conversion
# ============================================================

def raster_to_vector(
    raster: np.ndarray, transform=None, crs=None,
) -> Tuple[np.ndarray, np.ndarray, List[str], int]:
    try:
        import cv2
    except ImportError:
        raise ImportError("opencv-python is required for raster->vector conversion.")

    unique_classes = np.unique(raster)
    class_ids = sorted([c for c in unique_classes if c > 0])

    if len(class_ids) == 0:
        return (np.zeros((0, 4), dtype=np.float32),
                np.zeros((0,), dtype=np.int64), ["background"], 1)

    all_boxes, all_labels = [], []
    for new_label, class_val in enumerate(class_ids):
        binary = (raster == class_val).astype(np.uint8) * 255
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        for comp_id in range(1, num_labels):
            x, y, w, h, area = stats[comp_id]
            if area < 4:
                continue
            all_boxes.append([float(x), float(y), float(x + w), float(y + h)])
            all_labels.append(new_label)

    if len(all_boxes) == 0:
        return (np.zeros((0, 4), dtype=np.float32),
                np.zeros((0,), dtype=np.int64),
                [str(c) for c in class_ids], len(class_ids))

    boxes = np.array(all_boxes, dtype=np.float32)
    labels = np.array(all_labels, dtype=np.int64)
    class_names = [str(c) for c in class_ids]
    return boxes, labels, class_names, len(class_names)


# ============================================================
# Pixel-level Metrics (fast path for mode 11)
# ============================================================

def pixel_confusion_matrix(
    pred: np.ndarray, gt: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> Dict:
    """Pixel-level confusion matrix. Returns intermediate + final."""
    all_classes = sorted(set(np.unique(pred)) | set(np.unique(gt)))
    if class_names is None:
        class_names = [str(c) for c in all_classes]

    per_class = {}
    raw = {}  # intermediate: per-class tp/fp/fn
    total_tp, total_fp, total_fn = 0, 0, 0

    for idx, c in enumerate(all_classes):
        tp = int(np.sum((pred == c) & (gt == c)))
        fp = int(np.sum((pred == c) & (gt != c)))
        fn = int(np.sum((pred != c) & (gt == c)))

        raw[int(c)] = {"tp": tp, "fp": fp, "fn": fn}

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * p * r / (p + r) if (p + r) > 0 else 0.0)
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

        per_class[int(c)] = {
            "class_name": class_names[idx],
            "tp": tp, "fp": fp, "fn": fn,
            "precision": p, "recall": r, "f1": f1, "iou": iou,
        }
        total_tp += tp
        total_fp += fp
        total_fn += fn

    n = len(all_classes) or 1
    macro_avg = {
        "precision": sum(v["precision"] for v in per_class.values()) / n,
        "recall": sum(v["recall"] for v in per_class.values()) / n,
        "f1": sum(v["f1"] for v in per_class.values()) / n,
        "iou": sum(v["iou"] for v in per_class.values()) / n,
    }
    mp = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    mr = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_avg = {
        "precision": mp, "recall": mr,
        "f1": (2 * mp * mr / (mp + mr) if (mp + mr) > 0 else 0.0),
        "iou": total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else 0.0,
    }

    return {
        "_intermediate": {"type": "cm", "raw": raw, "all_classes": all_classes,
                          "class_names": class_names, "total_tp": total_tp,
                          "total_fp": total_fp, "total_fn": total_fn},
        "per_class": per_class, "macro_avg": macro_avg, "micro_avg": micro_avg,
    }


def pixel_geometric_quality(pred: np.ndarray, gt: np.ndarray) -> Dict:
    """Pixel-level geometric quality. Returns intermediate + final."""
    total_gt = int(np.sum(gt > 0))
    if total_gt == 0:
        return {"_intermediate": {"type": "go", "tp": 0, "fn": 0, "total_gt_area": 0,
                                  "inter_area_sum": 0.0, "weighted_sum": 0.0},
                "GTC": 0.0, "GOC": 0.0, "GUC": 0.0}

    tp = int(np.sum((pred > 0) & (gt > 0)))
    fn = int(np.sum((pred == 0) & (gt > 0)))
    gtc = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    all_classes = sorted(set(np.unique(pred)) | set(np.unique(gt)))
    ious = []
    per_class_data = {}
    for c in all_classes:
        if c == 0:
            continue
        tp_c = int(np.sum((pred == c) & (gt == c)))
        fp_c = int(np.sum((pred == c) & (gt != c)))
        fn_c = int(np.sum((pred != c) & (gt == c)))
        iou_c = tp_c / (tp_c + fp_c + fn_c) if (tp_c + fp_c + fn_c) > 0 else 0.0
        ious.append(iou_c)
        per_class_data[c] = {"tp": tp_c, "fp": fp_c, "fn": fn_c, "iou": iou_c}
    goc = np.mean(ious) if ious else 0.0
    guc = goc  # pixel level: GUC ~= GOC

    inter_area_sum = float(sum(p["iou"] * (p["tp"] + p["fp"] + p["fn"]) for p in per_class_data.values()))
    total_gt_area = float(np.sum(gt > 0))

    return {
        "_intermediate": {"type": "go", "tp": tp, "fn": fn,
                          "total_gt_area": total_gt_area,
                          "inter_area_sum": inter_area_sum,
                          "weighted_sum": inter_area_sum,  # pixel-level GUC = GOC
                          },
        "GTC": float(gtc), "GOC": float(goc), "GUC": float(guc),
    }


def pixel_polis(
    pred: np.ndarray, gt: np.ndarray, buffer_distance: float = 2.0,
) -> Dict:
    """Pixel-level PoLis. Returns intermediate + final."""
    try:
        import cv2
    except ImportError:
        raise ImportError("opencv-python is required for PoLis computation.")

    # Extract GT centroids
    gt_centroids = []
    for c in np.unique(gt):
        if c == 0: continue
        binary = (gt == c).astype(np.uint8) * 255
        num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        for comp_id in range(1, num_labels):
            if stats[comp_id, cv2.CC_STAT_AREA] < 4: continue
            cx, cy = centroids[comp_id]
            gt_centroids.append({"class": int(c), "cx": cx, "cy": cy})

    # Extract pred centroids
    pred_centroids = []
    for c in np.unique(pred):
        if c == 0: continue
        binary = (pred == c).astype(np.uint8) * 255
        num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        for comp_id in range(1, num_labels):
            if stats[comp_id, cv2.CC_STAT_AREA] < 4: continue
            cx, cy = centroids[comp_id]
            pred_centroids.append({"class": int(c), "cx": cx, "cy": cy})

    if len(gt_centroids) == 0 or len(pred_centroids) == 0:
        return {"_intermediate": {"type": "pl", "distances": []},
                "PoLis_mean_dist": 0.0, "PoLis_std_dist": 0.0,
                "PoLis_median_dist": 0.0, "PoLis_max_dist": 0.0,
                "PoLis_rmse": 0.0, "PoLis_buffer_rate": 0.0}

    pred_by_class = {}
    for i, p in enumerate(pred_centroids):
        pred_by_class.setdefault(p["class"], []).append((i, p))

    distances = []
    for gt_c in gt_centroids:
        cls = gt_c["class"]
        if cls not in pred_by_class: continue
        best_dist = float("inf")
        for _, pred_c in pred_by_class[cls]:
            dist = np.sqrt((gt_c["cx"] - pred_c["cx"]) ** 2 + (gt_c["cy"] - pred_c["cy"]) ** 2)
            if dist < best_dist: best_dist = dist
        distances.append(best_dist)

    distances = np.array(distances) if distances else np.array([])

    if len(distances) == 0:
        return {"_intermediate": {"type": "pl", "distances": []},
                "PoLis_mean_dist": 0.0, "PoLis_std_dist": 0.0,
                "PoLis_median_dist": 0.0, "PoLis_max_dist": 0.0,
                "PoLis_rmse": 0.0, "PoLis_buffer_rate": 0.0}

    buffer_rate = float(np.sum(distances <= buffer_distance)) / len(distances)
    return {
        "_intermediate": {"type": "pl", "distances": distances.tolist()},
        "PoLis_mean_dist": float(np.mean(distances)),
        "PoLis_std_dist": float(np.std(distances)),
        "PoLis_median_dist": float(np.median(distances)),
        "PoLis_max_dist": float(np.max(distances)),
        "PoLis_rmse": float(np.sqrt(np.mean(distances ** 2))),
        "PoLis_buffer_rate": buffer_rate,
    }


# ============================================================
# Vector-level Metrics
# ============================================================

def vector_cm(
    pred_boxes, pred_labels, gt_boxes, gt_labels,
    num_classes, class_names, iou_threshold=0.5,
) -> Dict:
    """Returns intermediate + final."""
    import torch
    from deeprs_light.evaluator.metrics import ConfusionMatrix

    if len(gt_boxes) == 0:
        return {"_intermediate": {"type": "cm", "raw": {}, "all_classes": [],
                                  "class_names": class_names, "total_tp": 0,
                                  "total_fp": 0, "total_fn": 0}}

    cm = ConfusionMatrix(num_classes=num_classes, iou_threshold=iou_threshold,
                         class_names=class_names)
    cm.process_batch(
        gt_boxes=[torch.from_numpy(gt_boxes)],
        gt_labels=[torch.from_numpy(gt_labels).long()],
        dt_boxes=[torch.from_numpy(pred_boxes)],
        dt_labels=[torch.from_numpy(pred_labels).long()],
        dt_scores=[torch.ones(len(pred_labels))],
    )

    raw = {}
    total_tp, total_fp, total_fn = 0, 0, 0
    all_classes = sorted(cm._counts.keys())

    for c in all_classes:
        cnt = cm._counts[c]
        raw[c] = {"tp": cnt["tp"], "fp": cnt["fp"], "fn": cnt["fn"]}
        total_tp += cnt["tp"]
        total_fp += cnt["fp"]
        total_fn += cnt["fn"]

    result = cm.compute(metrics=["precision", "recall", "f1"])
    result["_intermediate"] = {
        "type": "cm", "raw": raw, "all_classes": all_classes,
        "class_names": class_names, "total_tp": total_tp,
        "total_fp": total_fp, "total_fn": total_fn,
    }
    return result


def vector_ap(
    pred_boxes, pred_labels, gt_boxes, gt_labels,
    class_names, iou_type="bbox",
) -> Dict:
    """Returns intermediate (as COCO result list) + final dict."""
    import json, tempfile
    from pycocotools.coco import COCO
    from deeprs_light.evaluator.coco_eval import evaluate_coco

    gt_xywh = np.stack([
        gt_boxes[:, 0], gt_boxes[:, 1],
        gt_boxes[:, 2] - gt_boxes[:, 0], gt_boxes[:, 3] - gt_boxes[:, 1],
    ], axis=1)
    pred_xywh = np.stack([
        pred_boxes[:, 0], pred_boxes[:, 1],
        pred_boxes[:, 2] - pred_boxes[:, 0], pred_boxes[:, 3] - pred_boxes[:, 1],
    ], axis=1)

    categories = [{"id": i, "name": name} for i, name in enumerate(class_names)]

    # GT annotations (accumulable: image_id offset handled by accumulator)
    gt_annotations = []
    for i, (box, label) in enumerate(zip(gt_xywh, gt_labels)):
        gt_annotations.append({
            "id": i, "image_id": 1, "category_id": int(label),
            "bbox": [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
            "area": float(box[2] * box[3]), "iscrowd": 0,
        })
    # Pred results
    pred_results = []
    for i, (box, label) in enumerate(zip(pred_xywh, pred_labels)):
        pred_results.append({
            "image_id": 1, "category_id": int(label),
            "bbox": [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
            "score": 1.0,
        })

    if len(gt_boxes) == 0:
        return {"_intermediate": {"type": "ap", "gt_anns": [], "pred_results": [],
                                  "categories": categories},
                "AP": 0.0, "AP50": 0.0, "AP75": 0.0}

    # Compute per-pair AP
    gt_coco = {"images": [{"id": 1, "file_name": "img", "width": 100000, "height": 100000}],
               "annotations": gt_annotations, "categories": categories}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(gt_coco, f)
        tmp_path = f.name
    try:
        coco_gt = COCO(tmp_path)
        coco_dt = coco_gt.loadRes(pred_results)
        final = evaluate_coco(coco_gt, coco_dt, iou_type=iou_type)
    finally:
        os.unlink(tmp_path)

    final["_intermediate"] = {
        "type": "ap", "gt_anns": gt_annotations, "pred_results": pred_results,
        "categories": categories,
    }
    return final


def vector_go(
    pred_boxes, pred_labels, gt_boxes, gt_labels, iou_threshold=0.5,
) -> Dict:
    """Returns intermediate + final for geometric quality."""
    from deeprs_light.data.transforms_utils import compute_iou_matrix

    gt_areas = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
    pred_areas = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])

    if len(gt_boxes) == 0:
        return {"_intermediate": {"type": "go", "tp": 0, "fn": 0,
                                  "total_gt_area": 0.0, "inter_area_sum": 0.0,
                                  "weighted_sum": 0.0},
                "GTC": 0.0, "GOC": 0.0, "GUC": 0.0}

    ious = compute_iou_matrix(gt_boxes, pred_boxes)
    gt_matched = set()
    total_tp = 0
    inter_area_sum = 0.0
    weighted_sum = 0.0
    total_gt_area = float(np.sum(gt_areas))

    pred_order = np.argsort(-pred_areas)
    for pred_idx in pred_order:
        best_iou, best_gt = 0.0, -1
        for gt_idx in range(len(gt_boxes)):
            if gt_idx in gt_matched: continue
            if gt_labels[gt_idx] != pred_labels[pred_idx]: continue
            if ious[gt_idx, pred_idx] > best_iou:
                best_iou, best_gt = ious[gt_idx, pred_idx], gt_idx
        if best_iou >= iou_threshold and best_gt >= 0:
            gt_matched.add(best_gt)
            total_tp += 1
            iou_val = best_iou
            union = gt_areas[best_gt] + pred_areas[pred_idx]
            inter = iou_val * union / (1.0 + iou_val) if iou_val < 1.0 else min(gt_areas[best_gt], pred_areas[pred_idx])
            inter_area_sum += inter
            w = 1.0 if iou_val >= iou_threshold else iou_val / iou_threshold
            weighted_sum += w * inter

    total_fn = len(gt_boxes) - len(gt_matched)
    gtc = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    goc = inter_area_sum / total_gt_area if total_gt_area > 0 else 0.0
    guc = weighted_sum / total_gt_area if total_gt_area > 0 else 0.0

    return {
        "_intermediate": {"type": "go", "tp": total_tp, "fn": total_fn,
                          "total_gt_area": total_gt_area,
                          "inter_area_sum": inter_area_sum,
                          "weighted_sum": weighted_sum},
        "GTC": float(gtc), "GOC": float(goc), "GUC": float(guc),
    }


def vector_pl(
    pred_boxes, pred_labels, gt_boxes, gt_labels,
    iou_threshold=0.5, buffer_distance=2.0,
) -> Dict:
    """Returns intermediate + final for PoLis."""
    from deeprs_light.data.transforms_utils import compute_iou_matrix

    if len(gt_boxes) == 0:
        return {"_intermediate": {"type": "pl", "distances": []},
                "PoLis_mean_dist": 0.0, "PoLis_std_dist": 0.0,
                "PoLis_median_dist": 0.0, "PoLis_max_dist": 0.0,
                "PoLis_rmse": 0.0, "PoLis_buffer_rate": 0.0}

    gt_cx = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2
    gt_cy = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2
    pred_cx = (pred_boxes[:, 0] + pred_boxes[:, 2]) / 2
    pred_cy = (pred_boxes[:, 1] + pred_boxes[:, 3]) / 2

    ious = compute_iou_matrix(gt_boxes, pred_boxes)
    distances = []
    gt_matched = set()

    pred_areas = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
    pred_order = np.argsort(-pred_areas)
    for pred_idx in pred_order:
        best_iou, best_gt = 0.0, -1
        for gt_idx in range(len(gt_boxes)):
            if gt_idx in gt_matched: continue
            if gt_labels[gt_idx] != pred_labels[pred_idx]: continue
            if ious[gt_idx, pred_idx] > best_iou:
                best_iou, best_gt = ious[gt_idx, pred_idx], gt_idx
        if best_iou >= iou_threshold and best_gt >= 0:
            dist = np.sqrt((gt_cx[best_gt] - pred_cx[pred_idx]) ** 2 +
                          (gt_cy[best_gt] - pred_cy[pred_idx]) ** 2)
            distances.append(float(dist))
            gt_matched.add(best_gt)

    dist_arr = np.array(distances) if distances else np.array([])
    if len(dist_arr) == 0:
        return {"_intermediate": {"type": "pl", "distances": []},
                "PoLis_mean_dist": 0.0, "PoLis_std_dist": 0.0,
                "PoLis_median_dist": 0.0, "PoLis_max_dist": 0.0,
                "PoLis_rmse": 0.0, "PoLis_buffer_rate": 0.0}

    return {
        "_intermediate": {"type": "pl", "distances": distances},
        "PoLis_mean_dist": float(np.mean(dist_arr)),
        "PoLis_std_dist": float(np.std(dist_arr)),
        "PoLis_median_dist": float(np.median(dist_arr)),
        "PoLis_max_dist": float(np.max(dist_arr)),
        "PoLis_rmse": float(np.sqrt(np.mean(dist_arr ** 2))),
        "PoLis_buffer_rate": float(np.sum(dist_arr <= buffer_distance)) / len(dist_arr),
    }


# ============================================================
# Dispatcher
# ============================================================

def dispatch(
    mode: str, precision: str,
    pred_data: Dict, gt_data: Dict,
    iou_threshold: float = 0.5,
) -> Dict:
    """
    Route to the appropriate computation.

    Returns:
        A dict with both intermediate data ("_intermediate") and final metrics.
    """
    pred_type = pred_data["type"]
    gt_type = gt_data["type"]

    if pred_type == "vector" and gt_type == "vector":
        return _dispatch_vec_vec(precision, pred_data, gt_data, iou_threshold)

    if pred_type == "raster" and gt_type == "vector":
        pred_boxes, pred_labels, pred_class_names, pred_num_classes = \
            raster_to_vector(pred_data["array"], pred_data.get("transform"))
        pred_vec = {"type": "vector", "boxes": pred_boxes, "labels": pred_labels,
                    "class_names": pred_class_names, "num_classes": pred_num_classes}
        return _dispatch_vec_vec(precision, pred_vec, gt_data, iou_threshold)

    if pred_type == "raster" and gt_type == "raster":
        return _dispatch_ras_ras(precision, pred_data["array"], gt_data["array"],
                                 iou_threshold)

    raise ValueError(f"Unsupported mode '{mode}' (pred={pred_type}, gt={gt_type})")


def _dispatch_vec_vec(precision, pred_data, gt_data, iou_threshold) -> Dict:
    pred_boxes = pred_data["boxes"]
    pred_labels = pred_data["labels"]
    gt_boxes = gt_data["boxes"]
    gt_labels = gt_data["labels"]

    all_labels = sorted(set(pred_labels) | set(gt_labels))
    num_classes = max(len(all_labels), max(all_labels) + 1 if len(all_labels) > 0 else 1)
    class_names = [str(c) for c in range(num_classes)]

    if len(gt_boxes) == 0:
        return _empty_result(precision)

    if precision == "cm":
        return vector_cm(pred_boxes, pred_labels, gt_boxes, gt_labels,
                         num_classes, class_names, iou_threshold)
    elif precision == "ap":
        return vector_ap(pred_boxes, pred_labels, gt_boxes, gt_labels, class_names)
    elif precision == "go":
        return vector_go(pred_boxes, pred_labels, gt_boxes, gt_labels, iou_threshold)
    elif precision == "pl":
        return vector_pl(pred_boxes, pred_labels, gt_boxes, gt_labels, iou_threshold)
    return {}


def _dispatch_ras_ras(precision, pred_arr, gt_arr, iou_threshold) -> Dict:
    if pred_arr.shape != gt_arr.shape:
        raise ValueError(f"Raster shape mismatch: pred {pred_arr.shape} vs gt {gt_arr.shape}.")

    if precision == "cm":
        return pixel_confusion_matrix(pred_arr, gt_arr)
    elif precision == "ap":
        pred_boxes, pred_labels, pred_class_names, _ = raster_to_vector(pred_arr)
        gt_boxes, gt_labels, gt_class_names, _ = raster_to_vector(gt_arr)
        class_names = list(set(pred_class_names) | set(gt_class_names)) or ["0"]
        if len(gt_boxes) == 0:
            return _empty_result("ap")
        return vector_ap(pred_boxes, pred_labels, gt_boxes, gt_labels, class_names)
    elif precision == "go":
        return pixel_geometric_quality(pred_arr, gt_arr)
    elif precision == "pl":
        return pixel_polis(pred_arr, gt_arr)
    return {}


def _empty_result(precision: str) -> Dict:
    if precision == "ap":
        return {"_intermediate": {"type": "ap", "gt_anns": [], "pred_results": [],
                                  "categories": [], "num_pairs": 0},
                "AP": 0.0, "AP50": 0.0, "AP75": 0.0}
    if precision == "cm":
        return {"_intermediate": {"type": "cm", "raw": {}, "all_classes": [],
                                  "class_names": [], "total_tp": 0,
                                  "total_fp": 0, "total_fn": 0}}
    if precision == "go":
        return {"_intermediate": {"type": "go", "tp": 0, "fn": 0,
                                  "total_gt_area": 0.0, "inter_area_sum": 0.0,
                                  "weighted_sum": 0.0},
                "GTC": 0.0, "GOC": 0.0, "GUC": 0.0}
    if precision == "pl":
        return {"_intermediate": {"type": "pl", "distances": []},
                "PoLis_mean_dist": 0.0, "PoLis_std_dist": 0.0,
                "PoLis_median_dist": 0.0, "PoLis_rmse": 0.0}
    return {}


# ============================================================
# Batch Accumulator: aggregates intermediate results across pairs
# ============================================================

class BatchAccumulator:
    """
    Accumulates intermediate per-pair results and computes overall metrics,
    grouped by mode. Different modes (e.g., 00 vs 11) have different
    counting semantics (object-level vs pixel-level), so overall metrics
    are computed independently per mode.

    Usage:
        acc = BatchAccumulator(precisions=["cm", "go"], iou_threshold=0.5)
        for each pair:
            result = dispatch(...)
            acc.update("pair_name", "11", result)
        overall = acc.finalize()
        # overall = {"11": {"GTC": 0.75, ...}}
    """

    def __init__(self, precisions: List[str], iou_threshold: float = 0.5):
        self.precisions = precisions
        self.iou_threshold = iou_threshold
        self._modes: Dict[str, Dict] = {}  # mode -> per-precision accumulator data

    def _ensure_mode(self, mode: str):
        """Initialize accumulator storage for a mode if not already present."""
        if mode not in self._modes:
            self._modes[mode] = {
                "num_pairs": 0,
                # cm
                "cm_raw": {},
                "cm_classes": set(),
                "cm_class_names": [],
                "cm_total_tp": 0,
                "cm_total_fp": 0,
                "cm_total_fn": 0,
                # ap
                "ap_gt_anns": [],
                "ap_pred_results": [],
                "ap_categories": [],
                "ap_image_counter": 0,
                # go
                "go_tp": 0,
                "go_fn": 0,
                "go_total_gt_area": 0.0,
                "go_inter_area_sum": 0.0,
                "go_weighted_sum": 0.0,
                # pl
                "pl_distances": [],
            }

    def update(self, name: str, mode: str, result: Dict):
        """
        Accumulate intermediate results from one pair.

        Args:
            name: Pair name (for logging).
            mode: Mode string (e.g., "00", "11").
            result: The full result dict returned by dispatch().
        """
        inter = result.get("_intermediate", {})
        if not inter:
            return
        self._ensure_mode(mode)
        store = self._modes[mode]
        prec_type = inter["type"]
        store["num_pairs"] += 1

        if prec_type == "cm":
            self._accumulate_cm(store, inter)
        elif prec_type == "ap":
            self._accumulate_ap(store, inter)
        elif prec_type == "go":
            self._accumulate_go(store, inter)
        elif prec_type == "pl":
            self._accumulate_pl(store, inter)

    def _accumulate_cm(self, store: Dict, inter: Dict):
        raw = inter.get("raw", {})
        store["cm_classes"].update(inter.get("all_classes", []))
        store["cm_class_names"] = inter.get("class_names", [])
        store["cm_total_tp"] += inter.get("total_tp", 0)
        store["cm_total_fp"] += inter.get("total_fp", 0)
        store["cm_total_fn"] += inter.get("total_fn", 0)

        for cls_id, cnt in raw.items():
            if cls_id not in store["cm_raw"]:
                store["cm_raw"][cls_id] = {"tp": 0, "fp": 0, "fn": 0}
            store["cm_raw"][cls_id]["tp"] += cnt["tp"]
            store["cm_raw"][cls_id]["fp"] += cnt["fp"]
            store["cm_raw"][cls_id]["fn"] += cnt["fn"]

    def _accumulate_ap(self, store: Dict, inter: Dict):
        gt_anns = inter.get("gt_anns", [])
        pred_results = inter.get("pred_results", [])
        categories = inter.get("categories", [])

        img_id = store["ap_image_counter"] + 1
        ann_offset = len(store["ap_gt_anns"])

        for ann in gt_anns:
            ann = dict(ann)
            ann["id"] += ann_offset
            ann["image_id"] = img_id
            store["ap_gt_anns"].append(ann)

        for pred in pred_results:
            pred = dict(pred)
            pred["image_id"] = img_id
            store["ap_pred_results"].append(pred)

        if categories:
            store["ap_categories"] = categories
        store["ap_image_counter"] += 1

    def _accumulate_go(self, store: Dict, inter: Dict):
        store["go_tp"] += inter.get("tp", 0)
        store["go_fn"] += inter.get("fn", 0)
        store["go_total_gt_area"] += inter.get("total_gt_area", 0.0)
        store["go_inter_area_sum"] += inter.get("inter_area_sum", 0.0)
        store["go_weighted_sum"] += inter.get("weighted_sum", 0.0)

    def _accumulate_pl(self, store: Dict, inter: Dict):
        store["pl_distances"].extend(inter.get("distances", []))

    def finalize(self) -> Dict[str, Dict]:
        """
        Compute overall metrics for each mode independently.

        Returns:
            Dict[mode_str -> flat_dict of metric_name -> value]
            e.g. {"00": {"macro_precision": ..., "GTC": ...},
                   "11": {"macro_precision": ..., "GTC": ...}}
        """
        overall = {}
        for mode in sorted(self._modes.keys()):
            store = self._modes[mode]
            mode_result = {}

            if "cm" in self.precisions:
                mode_result.update(self._finalize_cm(store))
            if "ap" in self.precisions:
                mode_result.update(self._finalize_ap(store))
            if "go" in self.precisions:
                mode_result.update(self._finalize_go(store))
            if "pl" in self.precisions:
                mode_result.update(self._finalize_pl(store))

            overall[mode] = mode_result
        return overall

    def _finalize_cm(self, store: Dict) -> Dict:
        all_classes = sorted(store["cm_classes"])
        n = len(all_classes) or 1

        per_class = {}
        for c in all_classes:
            cnt = store["cm_raw"].get(c, {"tp": 0, "fp": 0, "fn": 0})
            tp, fp, fn = cnt["tp"], cnt["fp"], cnt["fn"]
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * p * r / (p + r) if (p + r) > 0 else 0.0)
            iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
            per_class[int(c)] = {"precision": p, "recall": r, "f1": f1, "iou": iou}

        flat = {}
        for cls_id, metrics in per_class.items():
            name = str(cls_id)
            for k in ("precision", "recall", "f1", "iou"):
                flat[f"{k}_class_{name}"] = metrics[k]

        macro_p = sum(v["precision"] for v in per_class.values()) / n if n > 0 else 0.0
        macro_r = sum(v["recall"] for v in per_class.values()) / n if n > 0 else 0.0
        macro_f1 = sum(v["f1"] for v in per_class.values()) / n if n > 0 else 0.0
        macro_iou = sum(v["iou"] for v in per_class.values()) / n if n > 0 else 0.0
        flat["macro_precision"] = macro_p
        flat["macro_recall"] = macro_r
        flat["macro_f1"] = macro_f1
        flat["macro_iou"] = macro_iou

        tp, fp, fn = store["cm_total_tp"], store["cm_total_fp"], store["cm_total_fn"]
        mp = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        mr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        flat["micro_precision"] = mp
        flat["micro_recall"] = mr
        flat["micro_f1"] = (2 * mp * mr / (mp + mr) if (mp + mr) > 0 else 0.0)
        flat["micro_iou"] = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        return flat

    def _finalize_ap(self, store: Dict) -> Dict:
        if len(store["ap_gt_anns"]) == 0:
            return {"AP": 0.0, "AP50": 0.0, "AP75": 0.0,
                    "AP_s": 0.0, "AP_m": 0.0, "AP_l": 0.0,
                    "AR1": 0.0, "AR10": 0.0, "AR100": 0.0,
                    "AR_s": 0.0, "AR_m": 0.0, "AR_l": 0.0}

        import json, tempfile
        from pycocotools.coco import COCO
        from deeprs_light.evaluator.coco_eval import evaluate_coco

        images = [{"id": i + 1, "file_name": f"img_{i+1}", "width": 100000, "height": 100000}
                  for i in range(store["ap_image_counter"])]
        gt_coco = {"images": images, "annotations": store["ap_gt_anns"],
                   "categories": store["ap_categories"]}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(gt_coco, f)
            tmp_path = f.name
        try:
            coco_gt = COCO(tmp_path)
            coco_dt = coco_gt.loadRes(store["ap_pred_results"])
            return evaluate_coco(coco_gt, coco_dt, iou_type="bbox")
        finally:
            os.unlink(tmp_path)

    def _finalize_go(self, store: Dict) -> Dict:
        gtc = store["go_tp"] / (store["go_tp"] + store["go_fn"]) if (store["go_tp"] + store["go_fn"]) > 0 else 0.0
        goc = store["go_inter_area_sum"] / store["go_total_gt_area"] if store["go_total_gt_area"] > 0 else 0.0
        guc = store["go_weighted_sum"] / store["go_total_gt_area"] if store["go_total_gt_area"] > 0 else 0.0
        return {"GTC": gtc, "GOC": goc, "GUC": guc}

    def _finalize_pl(self, store: Dict) -> Dict:
        if not store["pl_distances"]:
            return {"PoLis_mean_dist": 0.0, "PoLis_std_dist": 0.0,
                    "PoLis_median_dist": 0.0, "PoLis_max_dist": 0.0,
                    "PoLis_rmse": 0.0, "PoLis_buffer_rate": 0.0}
        d = np.array(store["pl_distances"])
        buf_dist = 2.0
        return {
            "PoLis_mean_dist": float(np.mean(d)),
            "PoLis_std_dist": float(np.std(d)),
            "PoLis_median_dist": float(np.median(d)),
            "PoLis_max_dist": float(np.max(d)),
            "PoLis_rmse": float(np.sqrt(np.mean(d ** 2))),
            "PoLis_buffer_rate": float(np.sum(d <= buf_dist)) / len(d),
        }


# ============================================================
# Output Writer (with pair column)
# ============================================================

def write_csv_rows(rows: List[Dict], output_path: str):
    """
    Write rows to CSV with columns: precision_type, metric, class, pair, value

    Each row dict: {"precision_type": str, "metric": str, "class": str,
                    "pair": str, "value": float|int|str}
    """
    if not output_path.endswith(".csv"):
        raise NotImplementedError(f"Only CSV output is supported. Got '{output_path}'.")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv_module.writer(f)
        writer.writerow(["precision_type", "metric", "class", "pair", "value"])
        for row in rows:
            val = row["value"]
            if isinstance(val, float):
                val = f"{val:.6f}"
            writer.writerow([row["precision_type"], row["metric"],
                             row["class"], row["pair"], val])


def result_to_rows(precision_type: str, pair_name: str, result: Dict) -> List[Dict]:
    """
    Convert a flat result dict into CSV row dicts.

    Args:
        precision_type: e.g. "cm", "ap", "go", "pl"
        pair_name: e.g. "region1" or "overall"
        result: flat dict of metric_name -> value (excluding "_intermediate")
    """
    rows = []
    for metric_name, value in result.items():
        if metric_name.startswith("_"):  # skip intermediate data
            continue
        if isinstance(value, dict):
            continue  # skip nested (should already be flattened)

        # Determine class identifier
        if "_class_" in metric_name:
            parts = metric_name.rsplit("_class_", 1)
            base_metric = parts[0]
            class_id = parts[1]
        else:
            base_metric = metric_name
            class_id = "all"

        rows.append({
            "precision_type": precision_type,
            "metric": base_metric,
            "class": class_id,
            "pair": pair_name,
            "value": value,
        })
    return rows


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    if not args.output.endswith(".csv"):
        raise NotImplementedError(f"Only CSV output is supported. Got '{args.output}'.")

    precisions = parse_precisions(args.precision)
    pred_field, gt_field = parse_field(args.field)
    band = args.band if args.band is not None else 1

    # --- Batch mode ---
    if args.pairs:
        run_batch(args, precisions, pred_field, gt_field, band)
    else:
        # --- Single mode ---
        run_single(args, precisions, pred_field, gt_field, band)


def run_single(args, precisions, pred_field, gt_field, band):
    """Run single pred-gt pair mode."""
    if not args.pred or not args.gt or not args.mode:
        raise ValueError("--pred, --gt, and --mode are required in single mode. "
                         "Use --pairs for batch mode.")

    validate_mode(args.mode)
    print(f"[acc_ass] mode={args.mode}, precision={precisions}")
    print(f"[acc_ass] pred='{args.pred}', gt='{args.gt}'")

    pred_data = load_data(args.pred, args.mode[0], band, pred_field)
    gt_data = load_data(args.gt, args.mode[1], band, gt_field)
    pair_name = os.path.splitext(os.path.basename(args.pred))[0]

    all_rows = []
    for prec in precisions:
        print(f"[acc_ass] Computing {prec}...")
        result = dispatch(args.mode, prec, pred_data, gt_data, args.iou)
        # Flatten and extract non-intermediate fields
        flat = _extract_final(prec, result)
        rows = result_to_rows(prec, pair_name, flat)
        all_rows.extend(rows)
        for k, v in flat.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")

    write_csv_rows(all_rows, args.output)
    print(f"[acc_ass] Results saved to '{args.output}'")


def run_batch(args, precisions, pred_field, gt_field, band):
    """Run batch mode with --pairs CSV."""
    if not os.path.exists(args.pairs):
        raise FileNotFoundError(f"Pairs CSV not found: '{args.pairs}'")

    # Read pairs
    pairs = []
    with open(args.pairs, "r") as f:
        reader = csv_module.DictReader(f)
        for row in reader:
            name = row.get("name", "").strip()
            pred = row.get("pred", "").strip()
            gt = row.get("gt", "").strip()
            mode = row.get("mode", "").strip()
            if not pred or not gt or not mode:
                print(f"[WARN] Skipping incomplete row: {row}")
                continue
            if not name:
                name = os.path.splitext(os.path.basename(pred))[0]
            validate_mode(mode)
            pairs.append({"name": name, "pred": pred, "gt": gt, "mode": mode})

    if not pairs:
        raise ValueError("No valid pairs found in CSV.")

    print(f"[acc_ass] Batch mode: {len(pairs)} pairs, precision={precisions}")

    # Initialize accumulator per precision (internally manages per-mode storage)
    accumulators = {prec: BatchAccumulator([prec], args.iou) for prec in precisions}

    all_rows = []

    # Process each pair
    for pair in pairs:
        name = pair["name"]
        mode = pair["mode"]
        print(f"\n[acc_ass] Pair '{name}' | pred='{pair['pred']}' | gt='{pair['gt']}' | mode={mode}")

        pred_data = load_data(pair["pred"], mode[0], band, pred_field)
        gt_data   = load_data(pair["gt"],   mode[1], band, gt_field)

        for prec in precisions:
            result = dispatch(mode, prec, pred_data, gt_data, args.iou)
            inter = result.get("_intermediate", {})

            if inter:
                accumulators[prec].update(name, mode, result)

            # Per-pair metrics
            flat = _extract_final(prec, result)
            rows = result_to_rows(prec, name, flat)
            all_rows.extend(rows)

            # Print summary
            print(f"  [{prec}]", end="")
            for k, v in flat.items():
                if isinstance(v, float):
                    print(f" {k}={v:.4f}", end="")
            print()

    # Compute and write overall metrics (per mode)
    print("\n[acc_ass] Computing overall metrics (per mode)...")
    for prec in precisions:
        overall_by_mode = accumulators[prec].finalize()
        for mode_str, overall in overall_by_mode.items():
            pair_label = f"overall_mode_{mode_str}"
            rows = result_to_rows(prec, pair_label, overall)
            all_rows.extend(rows)
            print(f"  [{prec} {pair_label}]", end="")
            for k, v in overall.items():
                if isinstance(v, float):
                    print(f" {k}={v:.4f}", end="")
            print()

    write_csv_rows(all_rows, args.output)
    print(f"\n[acc_ass] Results saved to '{args.output}'")


def _extract_final(precision_type: str, result: Dict) -> Dict:
    """Extract the final (non-intermediate) metrics from a result dict."""
    if precision_type == "cm":
        return _flatten_cm(result)
    # For ap/go/pl, just strip _intermediate
    return {k: v for k, v in result.items() if not k.startswith("_")}


def _flatten_cm(result: Dict) -> Dict:
    """Flatten nested CM result into flat key-value dict."""
    flat = {}
    for cls_id, metrics in result.get("per_class", {}).items():
        name = metrics.get("class_name", str(cls_id))
        for k in ("precision", "recall", "f1", "iou"):
            if k in metrics:
                flat[f"{k}_class_{name}"] = metrics[k]
    for k, v in result.get("macro_avg", {}).items():
        flat[f"macro_{k}"] = v
    for k, v in result.get("micro_avg", {}).items():
        flat[f"micro_{k}"] = v
    return flat


if __name__ == "__main__":
    main()

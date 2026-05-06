#!/usr/bin/env python
"""
acc_ass.py — Remote sensing accuracy assessment tool.

Supports vector (shapefile, geopackage) and raster (GeoTIFF) inputs
with multiple precision types: confusion matrix (cm), average precision (ap),
geometric quality (go), and point/line statistics (pl).

Usage:
    python Scripts/acc_ass.py \
        --pred pred.shp \
        --gt gt.tif \
        --mode 10 \
        --precision cm,ap \
        --output results.csv

See Scripts/readme.md for full documentation.
"""

import argparse
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
    parser.add_argument("--pred", required=True, help="Prediction file path (shp/gpkg/tif).")
    parser.add_argument("--gt", required=True, help="Ground truth file path (shp/gpkg/tif).")
    parser.add_argument("--mode", required=True,
                       help="Mode string 'xy': x=pred type, y=gt type. "
                            "0=vector, 1=binary/multi-value raster, 2=continuous raster.")
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
    """
    Parse field argument into (pred_field, gt_field).

    - None -> (None, None)
    - "class_id" -> ("class_id", "class_id")
    - "f1,f2" -> ("f1", "f2")  (length must be 2)
    """
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
    """Parse precision types. 'all' -> ['cm', 'ap', 'go', 'pl']."""
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
    """Check if a file is a raster (GeoTIFF) by extension."""
    ext = os.path.splitext(path)[1].lower()
    return ext in (".tif", ".tiff", ".vrt")


def _is_vector(path: str) -> bool:
    """Check if a file is a vector by extension."""
    ext = os.path.splitext(path)[1].lower()
    return ext in (".shp", ".gpkg", ".geojson", ".gml", ".kml")


# ============================================================
# Data Loading
# ============================================================

def load_vector(
    path: str, field: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str], int, object]:
    """
    Load a vector file and extract boxes + labels.

    Args:
        path: Path to shp/gpkg/geojson.
        field: Attribute field for class labels. None = all class 0 (binary).

    Returns:
        (boxes_xyxy[N,4], labels[N], class_names, num_classes, crs)
    """
    try:
        import geopandas as gpd
    except ImportError:
        raise ImportError("geopandas is required for vector loading. pip install geopandas")

    gdf = gpd.read_file(path)
    if gdf.crs is None:
        print(f"[WARN] No CRS found in '{path}'. Assuming same CRS as counterpart.")
    crs = gdf.crs

    # Extract bounding boxes from geometry
    boxes = []
    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            boxes.append([0, 0, 0, 0])
        else:
            minx, miny, maxx, maxy = geom.bounds
            boxes.append([minx, miny, maxx, maxy])
    boxes = np.array(boxes, dtype=np.float32)

    # Extract labels
    if field is not None and field in gdf.columns:
        labels = gdf[field].values
        # Convert to integer class IDs
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
    """
    Load a raster file.

    Args:
        path: Path to GeoTIFF.
        band: Band index (1-based). None defaults to 1.

    Returns:
        (array[H,W], geotransform, crs)
    """
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
    """
    Unified data loader. Returns a dict describing the loaded data.

    Returns:
    {
        "type": "vector" | "raster",
        "path": str,
        # Vector fields:
        "boxes": np.ndarray,    # [N,4] xyxy
        "labels": np.ndarray,   # [N]
        "class_names": List[str],
        "num_classes": int,
        "crs": ...,
        # Raster fields:
        "array": np.ndarray,    # [H,W]
        "transform": ...,
    }
    """
    data_type = int(data_type)

    if data_type == 0:
        if not _is_vector(path):
            raise ValueError(
                f"Mode expects vector input (type 0), but '{path}' does not look like a vector file."
            )
        boxes, labels, class_names, num_classes, crs = load_vector(path, field)
        return {
            "type": "vector",
            "path": path,
            "boxes": boxes,
            "labels": labels,
            "class_names": class_names,
            "num_classes": num_classes,
            "crs": crs,
        }
    elif data_type == 1:
        if not _is_raster(path):
            print(f"[WARN] Mode expects raster (type 1), but '{path}' does not look like a tif.")
        arr, transform, crs = load_raster(path, band)
        return {
            "type": "raster",
            "path": path,
            "array": arr,
            "transform": transform,
            "crs": crs,
        }
    else:
        raise NotImplementedError(f"Data type {data_type} not supported.")


# ============================================================
# Raster -> Vector conversion
# ============================================================

def raster_to_vector(
    raster: np.ndarray,
    transform=None,
    crs=None,
) -> Tuple[np.ndarray, np.ndarray, List[str], int]:
    """
    Convert a classification raster to vector boxes + labels.

    Uses connected component analysis (cv2) to extract per-class regions
    and their bounding boxes. Much faster than full polygonization.

    Args:
        raster: [H, W] integer array (class per pixel).

    Returns:
        (boxes_xyxy[N,4], labels[N], class_names, num_classes)
    """
    try:
        import cv2
    except ImportError:
        raise ImportError("opencv-python is required for raster->vector conversion.")

    unique_classes = np.unique(raster)
    # Filter out background (0)
    class_ids = sorted([c for c in unique_classes if c > 0])

    if len(class_ids) == 0:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            ["background"],
            1,
        )

    all_boxes = []
    all_labels = []

    for new_label, class_val in enumerate(class_ids):
        binary = (raster == class_val).astype(np.uint8) * 255
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
        # stats[0] is the background component, skip it
        for comp_id in range(1, num_labels):
            x, y, w, h, area = stats[comp_id]
            if area < 4:  # Ignore tiny noise components
                continue
            all_boxes.append([float(x), float(y), float(x + w), float(y + h)])
            all_labels.append(new_label)

    if len(all_boxes) == 0:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            [str(c) for c in class_ids],
            len(class_ids),
        )

    boxes = np.array(all_boxes, dtype=np.float32)
    labels = np.array(all_labels, dtype=np.int64)
    class_names = [str(c) for c in class_ids]

    return boxes, labels, class_names, len(class_names)


# ============================================================
# Pixel-level Metrics (fast path for mode 11)
# ============================================================

def pixel_confusion_matrix(
    pred: np.ndarray,
    gt: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> Dict:
    """
    Pixel-level confusion matrix computation. O(H*W), very fast.

    Args:
        pred: [H, W] int, predicted class per pixel.
        gt: [H, W] int, ground truth class per pixel.

    Returns:
        Dict with per_class, macro_avg, micro_avg precision/recall/f1/iou.
    """
    # Find all unique classes in pred and gt
    all_classes = sorted(set(np.unique(pred)) | set(np.unique(gt)))
    num_classes = len(all_classes)

    if class_names is None:
        class_names = [str(c) for c in all_classes]

    per_class = {}
    total_tp, total_fp, total_fn = 0, 0, 0

    for idx, c in enumerate(all_classes):
        tp = int(np.sum((pred == c) & (gt == c)))
        fp = int(np.sum((pred == c) & (gt != c)))
        fn = int(np.sum((pred != c) & (gt == c)))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

        per_class[int(c)] = {
            "class_name": class_names[idx],
            "tp": tp, "fp": fp, "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "iou": iou,
        }
        total_tp += tp
        total_fp += fp
        total_fn += fn

    # Macro average
    n = len(all_classes) or 1
    macro_avg = {
        "precision": sum(v["precision"] for v in per_class.values()) / n,
        "recall": sum(v["recall"] for v in per_class.values()) / n,
        "f1": sum(v["f1"] for v in per_class.values()) / n,
        "iou": sum(v["iou"] for v in per_class.values()) / n,
    }

    # Micro average
    mp = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    mr = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_avg = {
        "precision": mp,
        "recall": mr,
        "f1": (2 * mp * mr / (mp + mr) if (mp + mr) > 0 else 0.0),
        "iou": total_tp / (total_tp + total_fp + total_fn)
               if (total_tp + total_fp + total_fn) > 0 else 0.0,
    }

    return {
        "per_class": per_class,
        "macro_avg": macro_avg,
        "micro_avg": micro_avg,
    }


def pixel_geometric_quality(
    pred: np.ndarray,
    gt: np.ndarray,
) -> Dict:
    """
    Pixel-level geometric quality: GTC, GOC, GUC.

    Args:
        pred: [H, W] int, predicted class per pixel.
        gt: [H, W] int, ground truth class per pixel.

    Returns:
        {"GTC": float, "GOC": float, "GUC": float}
    """
    total_gt = int(np.sum(gt > 0))
    if total_gt == 0:
        return {"GTC": 0.0, "GOC": 0.0, "GUC": 0.0}

    # GTC: proportion of GT pixels correctly predicted (pixel-level recall)
    tp = int(np.sum((pred > 0) & (gt > 0)))
    fn = int(np.sum((pred == 0) & (gt > 0)))
    gtc = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # GOC: per-class IoU averaged (geometric completeness)
    all_classes = sorted(set(np.unique(pred)) | set(np.unique(gt)))
    ious = []
    for c in all_classes:
        if c == 0:
            continue
        tp_c = int(np.sum((pred == c) & (gt == c)))
        fp_c = int(np.sum((pred == c) & (gt != c)))
        fn_c = int(np.sum((pred != c) & (gt == c)))
        iou_c = tp_c / (tp_c + fp_c + fn_c) if (tp_c + fp_c + fn_c) > 0 else 0.0
        ious.append(iou_c)
    goc = np.mean(ious) if ious else 0.0

    # GUC: weighted GOC — pixel-level usability = whether pixel was correctly classified
    guc = goc  # At pixel level, GUC ~= GOC (all pixels weighted equally)

    return {"GTC": float(gtc), "GOC": float(goc), "GUC": float(guc)}


def pixel_polis(
    pred: np.ndarray,
    gt: np.ndarray,
    buffer_distance: float = 2.0,
) -> Dict:
    """
    Pixel-level PoLis: extract connected component centroids and compute distances.

    Args:
        pred: [H, W] int.
        gt: [H, W] int.
        buffer_distance: Buffer radius in pixels.

    Returns:
        PoLis metrics dict.
    """
    try:
        import cv2
    except ImportError:
        raise ImportError("opencv-python is required for PoLis computation.")

    # Extract connected components from GT and compute centroids
    gt_centroids = []
    for c in np.unique(gt):
        if c == 0:
            continue
        binary = (gt == c).astype(np.uint8) * 255
        num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
        for comp_id in range(1, num_labels):
            area = stats[comp_id, cv2.CC_STAT_AREA]
            if area < 4:
                continue
            cx, cy = centroids[comp_id]
            gt_centroids.append({"class": int(c), "cx": cx, "cy": cy, "area": area})

    # Extract centroids from pred
    pred_centroids = []
    for c in np.unique(pred):
        if c == 0:
            continue
        binary = (pred == c).astype(np.uint8) * 255
        num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
        for comp_id in range(1, num_labels):
            area = stats[comp_id, cv2.CC_STAT_AREA]
            if area < 4:
                continue
            cx, cy = centroids[comp_id]
            pred_centroids.append({"class": int(c), "cx": cx, "cy": cy, "area": area})

    if len(gt_centroids) == 0 or len(pred_centroids) == 0:
        return {
            "PoLis_mean_dist": 0.0,
            "PoLis_std_dist": 0.0,
            "PoLis_median_dist": 0.0,
            "PoLis_max_dist": 0.0,
            "PoLis_rmse": 0.0,
            "PoLis_buffer_rate": 0.0,
        }

    # Match pred centroids to nearest GT centroids of the same class
    distances = []
    gt_matched = set()
    pred_by_class = {}
    for i, p in enumerate(pred_centroids):
        pred_by_class.setdefault(p["class"], []).append((i, p))

    for gt_idx, gt_c in enumerate(gt_centroids):
        cls = gt_c["class"]
        if cls not in pred_by_class:
            continue
        best_dist = float("inf")
        best_pred_idx = -1
        for pred_idx, pred_c in pred_by_class[cls]:
            if pred_idx in set():  # already matched — not applicable for centroid matching
                continue
            dist = np.sqrt((gt_c["cx"] - pred_c["cx"]) ** 2 +
                          (gt_c["cy"] - pred_c["cy"]) ** 2)
            if dist < best_dist:
                best_dist = dist
                best_pred_idx = pred_idx
        if best_pred_idx >= 0:
            distances.append(best_dist)
            gt_matched.add(gt_idx)

    if len(distances) == 0:
        return {
            "PoLis_mean_dist": 0.0,
            "PoLis_std_dist": 0.0,
            "PoLis_median_dist": 0.0,
            "PoLis_max_dist": 0.0,
            "PoLis_rmse": 0.0,
            "PoLis_buffer_rate": 0.0,
        }

    distances = np.array(distances)
    buffer_rate = float(np.sum(distances <= buffer_distance)) / len(distances)

    return {
        "PoLis_mean_dist": float(np.mean(distances)),
        "PoLis_std_dist": float(np.std(distances)),
        "PoLis_median_dist": float(np.median(distances)),
        "PoLis_max_dist": float(np.max(distances)),
        "PoLis_rmse": float(np.sqrt(np.mean(distances ** 2))),
        "PoLis_buffer_rate": buffer_rate,
    }


# ============================================================
# Vector-level Metrics (reuse existing evaluator code)
# ============================================================

def vector_cm(
    pred_boxes: np.ndarray,
    pred_labels: np.ndarray,
    gt_boxes: np.ndarray,
    gt_labels: np.ndarray,
    num_classes: int,
    class_names: List[str],
    iou_threshold: float = 0.5,
) -> Dict:
    """
    Vector-level confusion matrix using existing ConfusionMatrix class.
    """
    import torch
    from deeprs_light.evaluator.metrics import ConfusionMatrix

    cm = ConfusionMatrix(
        num_classes=num_classes,
        iou_threshold=iou_threshold,
        class_names=class_names,
    )

    cm.process_batch(
        gt_boxes=[torch.from_numpy(gt_boxes)],
        gt_labels=[torch.from_numpy(gt_labels).long()],
        dt_boxes=[torch.from_numpy(pred_boxes)],
        dt_labels=[torch.from_numpy(pred_labels).long()],
        dt_scores=[torch.ones(len(pred_labels))],  # All scores = 1.0
    )

    return cm.compute(metrics=["precision", "recall", "f1"])


def vector_ap(
    pred_boxes: np.ndarray,
    pred_labels: np.ndarray,
    gt_boxes: np.ndarray,
    gt_labels: np.ndarray,
    class_names: List[str],
    iou_type: str = "bbox",
) -> Dict:
    """
    Vector-level AP computation via COCO format.
    Creates temporary COCO JSON structures and calls evaluate_coco().
    """
    import json
    import tempfile
    from pycocotools.coco import COCO
    from deeprs_light.evaluator.coco_eval import evaluate_coco

    # Helper: xyxy -> xywh
    xyxy = gt_boxes
    gt_xywh = np.stack([
        xyxy[:, 0],
        xyxy[:, 1],
        xyxy[:, 2] - xyxy[:, 0],
        xyxy[:, 3] - xyxy[:, 1],
    ], axis=1)

    pred_xywh = np.stack([
        pred_boxes[:, 0],
        pred_boxes[:, 1],
        pred_boxes[:, 2] - pred_boxes[:, 0],
        pred_boxes[:, 3] - pred_boxes[:, 1],
    ], axis=1)

    # Build categories
    categories = [{"id": i, "name": name} for i, name in enumerate(class_names)]

    # Build ground truth COCO dict
    gt_coco = {
        "images": [{"id": 1, "file_name": "img", "width": 100000, "height": 100000}],
        "annotations": [],
        "categories": categories,
    }
    for i, (box, label) in enumerate(zip(gt_xywh, gt_labels)):
        gt_coco["annotations"].append({
            "id": i,
            "image_id": 1,
            "category_id": int(label),
            "bbox": [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
            "area": float(box[2] * box[3]),
            "iscrowd": 0,
        })

    # Build prediction results list
    pred_results = []
    for i, (box, label) in enumerate(zip(pred_xywh, pred_labels)):
        pred_results.append({
            "image_id": 1,
            "category_id": int(label),
            "bbox": [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
            "score": 1.0,
        })

    # Write GT to temp file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(gt_coco, f)
        tmp_path = f.name

    try:
        coco_gt = COCO(tmp_path)
        coco_dt = coco_gt.loadRes(pred_results)
        results = evaluate_coco(coco_gt, coco_dt, iou_type=iou_type)
    finally:
        os.unlink(tmp_path)

    return results


def vector_go(
    pred_boxes: np.ndarray,
    pred_labels: np.ndarray,
    gt_boxes: np.ndarray,
    gt_labels: np.ndarray,
    iou_threshold: float = 0.5,
) -> Dict:
    """
    Vector-level geometric quality (GTC, GOC, GUC).

    Matches pred-gt boxes by class and IoU, then computes area-based metrics.
    """
    from deeprs_light.data.transforms_utils import compute_iou_matrix

    # Compute areas
    gt_areas = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
    pred_areas = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])

    # Global matching across all classes
    ious = compute_iou_matrix(gt_boxes, pred_boxes)
    matches = []
    gt_matched = set()
    total_tp, total_fn = 0, 0

    # Sort predictions by area descending for matching priority
    pred_order = np.argsort(-pred_areas)
    for pred_idx in pred_order:
        best_iou = 0.0
        best_gt = -1
        for gt_idx in range(len(gt_boxes)):
            if gt_idx in gt_matched:
                continue
            if gt_labels[gt_idx] != pred_labels[pred_idx]:
                continue
            if ious[gt_idx, pred_idx] > best_iou:
                best_iou = ious[gt_idx, pred_idx]
                best_gt = gt_idx
        if best_iou >= iou_threshold and best_gt >= 0:
            matches.append((best_gt, pred_idx))
            gt_matched.add(best_gt)
            total_tp += 1

    total_fn = len(gt_boxes) - len(gt_matched)

    # GTC
    gtc = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0

    # GOC: sum of intersection areas / sum of GT areas
    inter_area_sum = 0.0
    total_gt_area = float(np.sum(gt_areas))
    for gt_idx, pred_idx in matches:
        # Approximate intersection from IoU * union
        iou_val = ious[gt_idx, pred_idx]
        union = gt_areas[gt_idx] + pred_areas[pred_idx]
        inter = iou_val * union / (1.0 + iou_val) if iou_val < 1.0 else min(gt_areas[gt_idx], pred_areas[pred_idx])
        inter_area_sum += inter
    goc = inter_area_sum / total_gt_area if total_gt_area > 0 else 0.0

    # GUC: weighted by IoU threshold
    weighted_sum = 0.0
    for gt_idx, pred_idx in matches:
        iou_val = ious[gt_idx, pred_idx]
        w = 1.0 if iou_val >= iou_threshold else iou_val / iou_threshold
        union = gt_areas[gt_idx] + pred_areas[pred_idx]
        inter = iou_val * union / (1.0 + iou_val) if iou_val < 1.0 else min(gt_areas[gt_idx], pred_areas[pred_idx])
        weighted_sum += w * inter
    guc = weighted_sum / total_gt_area if total_gt_area > 0 else 0.0

    return {"GTC": float(gtc), "GOC": float(goc), "GUC": float(guc)}


def vector_pl(
    pred_boxes: np.ndarray,
    pred_labels: np.ndarray,
    gt_boxes: np.ndarray,
    gt_labels: np.ndarray,
    iou_threshold: float = 0.5,
    buffer_distance: float = 2.0,
) -> Dict:
    """
    Vector-level PoLis: center point distance statistics.
    """
    from deeprs_light.data.transforms_utils import compute_iou_matrix

    # Compute centers
    gt_cx = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2
    gt_cy = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2
    pred_cx = (pred_boxes[:, 0] + pred_boxes[:, 2]) / 2
    pred_cy = (pred_boxes[:, 1] + pred_boxes[:, 3]) / 2

    # Match by IoU (same class)
    ious = compute_iou_matrix(gt_boxes, pred_boxes)
    distances = []
    gt_matched = set()

    pred_order = np.argsort(-(pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1]))
    for pred_idx in pred_order:
        best_iou = 0.0
        best_gt = -1
        for gt_idx in range(len(gt_boxes)):
            if gt_idx in gt_matched:
                continue
            if gt_labels[gt_idx] != pred_labels[pred_idx]:
                continue
            if ious[gt_idx, pred_idx] > best_iou:
                best_iou = ious[gt_idx, pred_idx]
                best_gt = gt_idx
        if best_iou >= iou_threshold and best_gt >= 0:
            dist = np.sqrt((gt_cx[best_gt] - pred_cx[pred_idx]) ** 2 +
                          (gt_cy[best_gt] - pred_cy[pred_idx]) ** 2)
            distances.append(dist)
            gt_matched.add(best_gt)

    if len(distances) == 0:
        return {
            "PoLis_mean_dist": 0.0, "PoLis_std_dist": 0.0,
            "PoLis_median_dist": 0.0, "PoLis_max_dist": 0.0,
            "PoLis_rmse": 0.0, "PoLis_buffer_rate": 0.0,
        }

    distances = np.array(distances)
    return {
        "PoLis_mean_dist": float(np.mean(distances)),
        "PoLis_std_dist": float(np.std(distances)),
        "PoLis_median_dist": float(np.median(distances)),
        "PoLis_max_dist": float(np.max(distances)),
        "PoLis_rmse": float(np.sqrt(np.mean(distances ** 2))),
        "PoLis_buffer_rate": float(np.sum(distances <= buffer_distance)) / len(distances),
    }


# ============================================================
# Dispatcher
# ============================================================

def dispatch(
    mode: str,
    precision: str,
    pred_data: Dict,
    gt_data: Dict,
    iou_threshold: float = 0.5,
) -> Dict:
    """
    Route to the appropriate computation based on mode and precision type.

    Args:
        mode: 2-digit mode string.
        precision: One of 'cm', 'ap', 'go', 'pl'.
        pred_data: Loaded prediction data dict.
        gt_data: Loaded ground truth data dict.
        iou_threshold: IoU threshold for vector matching.

    Returns:
        Flat dict of metric name -> value.
    """
    pred_type = pred_data["type"]
    gt_type = gt_data["type"]

    # ---- Mode 00: vector - vector ----
    if pred_type == "vector" and gt_type == "vector":
        return dispatch_vec_vec(precision, pred_data, gt_data, iou_threshold)

    # ---- Mode 10: raster pred, vector gt ----
    if pred_type == "raster" and gt_type == "vector":
        # Convert pred raster -> vector, then dispatch as vec-vec
        pred_boxes, pred_labels, pred_class_names, pred_num_classes = \
            raster_to_vector(pred_data["array"], pred_data.get("transform"))
        pred_vec = {
            "type": "vector",
            "boxes": pred_boxes,
            "labels": pred_labels,
            "class_names": pred_class_names,
            "num_classes": pred_num_classes,
        }
        return dispatch_vec_vec(precision, pred_vec, gt_data, iou_threshold)

    # ---- Mode 11: raster - raster ----
    if pred_type == "raster" and gt_type == "raster":
        return dispatch_ras_ras(precision, pred_data["array"], gt_data["array"],
                                iou_threshold)

    raise ValueError(f"Unsupported mode '{mode}' (pred={pred_type}, gt={gt_type})")


def dispatch_vec_vec(
    precision: str,
    pred_data: Dict,
    gt_data: Dict,
    iou_threshold: float,
) -> Dict:
    """Vector-vector dispatch."""
    pred_boxes = pred_data["boxes"]
    pred_labels = pred_data["labels"]
    gt_boxes = gt_data["boxes"]
    gt_labels = gt_data["labels"]

    # Use union of class sets
    all_labels = sorted(set(pred_labels) | set(gt_labels))
    num_classes = max(len(all_labels), max(all_labels) + 1 if len(all_labels) > 0 else 1)
    class_names = [str(c) for c in range(num_classes)]

    if len(gt_boxes) == 0:
        return _empty_results(precision)

    if precision == "cm":
        result = vector_cm(pred_boxes, pred_labels, gt_boxes, gt_labels,
                          num_classes, class_names, iou_threshold)
        return _flatten_cm(result)

    elif precision == "ap":
        return vector_ap(pred_boxes, pred_labels, gt_boxes, gt_labels, class_names)

    elif precision == "go":
        return vector_go(pred_boxes, pred_labels, gt_boxes, gt_labels, iou_threshold)

    elif precision == "pl":
        return vector_pl(pred_boxes, pred_labels, gt_boxes, gt_labels, iou_threshold)

    return {}


def dispatch_ras_ras(
    precision: str,
    pred_arr: np.ndarray,
    gt_arr: np.ndarray,
    iou_threshold: float,
) -> Dict:
    """Raster-raster dispatch. Uses fast pixel-level path when possible."""
    # Ensure same shape
    if pred_arr.shape != gt_arr.shape:
        raise ValueError(
            f"Raster shape mismatch: pred {pred_arr.shape} vs gt {gt_arr.shape}."
        )

    if precision == "cm":
        result = pixel_confusion_matrix(pred_arr, gt_arr)
        return _flatten_cm(result)

    elif precision == "ap":
        # Must convert both to vector for COCO evaluation
        pred_boxes, pred_labels, pred_class_names, _ = raster_to_vector(pred_arr)
        gt_boxes, gt_labels, gt_class_names, _ = raster_to_vector(gt_arr)
        class_names = list(set(pred_class_names) | set(gt_class_names)) or ["0"]
        if len(gt_boxes) == 0:
            return _empty_results("ap")
        return vector_ap(pred_boxes, pred_labels, gt_boxes, gt_labels, class_names)

    elif precision == "go":
        # Pixel-level fast path for geometric quality
        return pixel_geometric_quality(pred_arr, gt_arr)

    elif precision == "pl":
        return pixel_polis(pred_arr, gt_arr)

    return {}


def _empty_results(precision: str) -> Dict:
    """Return zero-filled results for empty ground truth."""
    if precision == "ap":
        return {"AP": 0.0, "AP50": 0.0, "AP75": 0.0}
    if precision == "cm":
        return {"macro_precision": 0.0, "macro_recall": 0.0, "macro_f1": 0.0,
                "micro_precision": 0.0, "micro_recall": 0.0, "micro_f1": 0.0}
    if precision == "go":
        return {"GTC": 0.0, "GOC": 0.0, "GUC": 0.0}
    if precision == "pl":
        return {"PoLis_mean_dist": 0.0, "PoLis_std_dist": 0.0,
                "PoLis_median_dist": 0.0, "PoLis_rmse": 0.0}
    return {}


def _flatten_cm(result: Dict) -> Dict:
    """Flatten nested CM result into flat key-value dict for CSV output."""
    flat = {}
    # Per-class
    for cls_id, metrics in result.get("per_class", {}).items():
        name = metrics.get("class_name", str(cls_id))
        for k in ("precision", "recall", "f1", "iou"):
            if k in metrics:
                flat[f"{k}_class_{name}"] = metrics[k]
    # Macro
    for k, v in result.get("macro_avg", {}).items():
        flat[f"macro_{k}"] = v
    # Micro
    for k, v in result.get("micro_avg", {}).items():
        flat[f"micro_{k}"] = v
    return flat


# ============================================================
# Output Writer
# ============================================================

def write_csv(all_results: Dict[str, Dict], output_path: str):
    """
    Write results to CSV.

    Format:
        metric,class,value
        AP,all,0.45
        AP50,all,0.72
        macro_precision,all,0.785
        GTC,all,0.75
        PoLis_mean_dist,all,2.3
        precision_class_0,0,0.85
        precision_class_1,1,0.72
        ...
    """
    if not output_path.endswith(".csv"):
        raise NotImplementedError(
            f"Only CSV output is supported. Got '{output_path}'."
        )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "w") as f:
        f.write("precision_type,metric,class,value\n")
        for prec_type, metrics in all_results.items():
            for metric_name, value in metrics.items():
                # Determine class identifier
                # Flattened CM keys like "precision_class_ship" -> class="ship"
                # Other keys like "AP50" -> class="all"
                if "_class_" in metric_name:
                    parts = metric_name.rsplit("_class_", 1)
                    base_metric = parts[0]
                    class_id = parts[1]
                else:
                    base_metric = metric_name
                    class_id = "all"

                if isinstance(value, float):
                    f.write(f"{prec_type},{base_metric},{class_id},{value:.6f}\n")
                elif isinstance(value, int):
                    f.write(f"{prec_type},{base_metric},{class_id},{value}\n")
                else:
                    f.write(f"{prec_type},{base_metric},{class_id},{value}\n")


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    # Validate
    validate_mode(args.mode)
    if not args.output.endswith(".csv"):
        raise NotImplementedError(
            f"Only CSV output is supported. Got '{args.output}'."
        )

    pred_field, gt_field = parse_field(args.field)
    precisions = parse_precisions(args.precision)
    band = args.band if args.band is not None else 1

    print(f"[acc_ass] mode={args.mode}, precision={precisions}")
    print(f"[acc_ass] pred='{args.pred}', gt='{args.gt}'")

    # Load data
    pred_data = load_data(args.pred, args.mode[0], band, pred_field)
    gt_data = load_data(args.gt, args.mode[1], band, gt_field)

    print(f"[acc_ass] pred: {pred_data['type']} "
          f"({'boxes=' + str(pred_data.get('boxes', pred_data.get('array', '')).shape)})")
    print(f"[acc_ass] gt:   {gt_data['type']} "
          f"({'boxes=' + str(gt_data.get('boxes', gt_data.get('array', '')).shape)})")

    # Compute
    all_results = {}
    for prec in precisions:
        print(f"[acc_ass] Computing {prec}...")
        result = dispatch(args.mode, prec, pred_data, gt_data, args.iou)
        all_results[prec] = result
        # Print summary
        for k, v in result.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")

    # Output
    write_csv(all_results, args.output)
    print(f"[acc_ass] Results saved to '{args.output}'")


if __name__ == "__main__":
    main()

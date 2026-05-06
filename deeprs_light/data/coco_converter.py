"""
Offline conversion utilities: transform custom detection/segmentation
formats into standard COCO JSON.
"""

import os
import json
import glob
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


def convert_detection_to_coco(
    data_dir: str,
    output_path: str,
    categories: List[Dict],
    label_map: Optional[Dict[str, int]] = None,
    image_dir: str = "images",
    label_dir: str = "labels",
    image_ext: str = ".png",
) -> str:
    """
    Convert custom detection data to standard COCO JSON format.

    Assumes input structure:
        data_dir/
            images/xxx.png
            labels/xxx.txt   # Each line: class_name x_center y_center w h (normalized YOLO format)

    Args:
        data_dir: Root data directory.
        output_path: Output COCO JSON path.
        categories: List of category dicts [{"id": 1, "name": "ship", "supercategory": "vehicle"}, ...].
        label_map: Optional dict mapping label-file class names to category_id.
                   If None, auto-builds from category names.
        image_dir: Subdirectory for images.
        label_dir: Subdirectory for labels.
        image_ext: Image file extension.

    Returns:
        output_path
    """
    # Build label map if not provided
    if label_map is None:
        label_map = {cat["name"]: cat["id"] for cat in categories}

    # Category metadata for COCO
    coco_categories = [
        {"id": cat["id"], "name": cat["name"], "supercategory": cat.get("supercategory", "")}
        for cat in categories
    ]

    images = []
    annotations = []
    ann_id = 0

    img_dir = os.path.join(data_dir, image_dir)
    lbl_dir = os.path.join(data_dir, label_dir)

    image_files = sorted(glob.glob(os.path.join(img_dir, f"*{image_ext}")))
    if not image_files:
        raise FileNotFoundError(f"No images found in '{img_dir}' with ext '{image_ext}'")

    for img_id, img_path in enumerate(image_files, start=1):
        file_name = os.path.basename(img_path)
        image = cv2.imread(img_path)
        if image is None:
            print(f"[WARN] Failed to load '{img_path}', skipping.")
            continue
        h, w = image.shape[:2]

        images.append({
            "id": img_id,
            "file_name": os.path.join(image_dir, file_name),
            "width": w,
            "height": h,
        })

        # Parse label file
        base_name = os.path.splitext(file_name)[0]
        label_path = os.path.join(lbl_dir, f"{base_name}.txt")

        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    class_name = parts[0]
                    try:
                        cx, cy, bw, bh = map(float, parts[1:5])
                    except ValueError:
                        continue

                    cat_id = label_map.get(class_name)
                    if cat_id is None:
                        print(f"[WARN] Unknown class '{class_name}' in '{label_path}', skipping.")
                        continue

                    # YOLO normalized -> pixel absolute xywh
                    abs_x = (cx - bw / 2) * w
                    abs_y = (cy - bh / 2) * h
                    abs_w = bw * w
                    abs_h = bh * h

                    area = abs_w * abs_h
                    ann_id += 1
                    annotations.append({
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": cat_id,
                        "bbox": [round(abs_x, 2), round(abs_y, 2),
                                 round(abs_w, 2), round(abs_h, 2)],
                        "area": round(area, 2),
                        "iscrowd": 0,
                    })

    coco_json = {
        "images": images,
        "annotations": annotations,
        "categories": coco_categories,
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(coco_json, f, indent=2)

    print(f"[convert_detection_to_coco] Written {len(images)} images,"
          f" {len(annotations)} annotations to '{output_path}'")
    return output_path


def convert_segmentation_to_coco(
    data_dir: str,
    output_path: str,
    categories: List[Dict],
    mask_dir: str = "masks",
    image_dir: str = "images",
    image_ext: str = ".png",
) -> str:
    """
    Convert semantic segmentation masks to COCO instance segmentation format.

    Each PNG mask has pixel values representing category IDs.
    Connected components are extracted as individual instances.

    Args:
        data_dir: Root data directory.
        output_path: Output COCO JSON path.
        categories: Category list.
        mask_dir: Subdirectory for mask PNGs.
        image_dir: Subdirectory for original images.
        image_ext: Image file extension.

    Returns:
        output_path
    """
    coco_categories = [
        {"id": cat["id"], "name": cat["name"], "supercategory": cat.get("supercategory", "")}
        for cat in categories
    ]

    images = []
    annotations = []
    ann_id = 0

    img_dir = os.path.join(data_dir, image_dir)
    msk_dir = os.path.join(data_dir, mask_dir)

    image_files = sorted(glob.glob(os.path.join(img_dir, f"*{image_ext}")))
    if not image_files:
        raise FileNotFoundError(f"No images found in '{img_dir}'")

    for img_id, img_path in enumerate(image_files, start=1):
        file_name = os.path.basename(img_path)
        image = cv2.imread(img_path)
        if image is None:
            continue
        h, w = image.shape[:2]

        images.append({
            "id": img_id,
            "file_name": os.path.join(image_dir, file_name),
            "width": w,
            "height": h,
        })

        # Parse mask
        base_name = os.path.splitext(file_name)[0]
        mask_path = os.path.join(msk_dir, f"{base_name}{image_ext}")

        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue

            unique_cats = np.unique(mask)
            for cat in unique_cats:
                if cat == 0:
                    continue  # Background
                cat_id = int(cat)
                if cat_id not in [c["id"] for c in categories]:
                    continue

                # Find connected components for this category
                binary = (mask == cat).astype(np.uint8) * 255
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                    binary, connectivity=8
                )

                for label_id in range(1, num_labels):
                    # Extract instance mask
                    inst_mask = (labels == label_id).astype(np.uint8)

                    # Encode as RLE using pycocotools
                    try:
                        from pycocotools import mask as mask_util
                        rle = mask_util.encode(
                            np.asfortranarray(inst_mask.astype(np.uint8))
                        )
                        # Convert bytes to string for JSON
                        if isinstance(rle["counts"], bytes):
                            rle["counts"] = rle["counts"].decode("ascii")
                    except ImportError:
                        # Fallback: store as polygon
                        contours, _ = cv2.findContours(
                            inst_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                        )
                        rle = None
                        seg = []
                        for cnt in contours:
                            if cnt.shape[0] >= 3:
                                seg.append(cnt.flatten().tolist())
                        # Use the contour as segmentation
                        rle = {"segmentation": seg}

                    ann_id += 1
                    ann = {
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": cat_id,
                        "bbox": list(cv2.boundingRect(inst_mask)),  # [x, y, w, h]
                        "area": float(inst_mask.sum()),
                        "iscrowd": 0,
                    }
                    if isinstance(rle, dict) and "counts" in rle:
                        ann["segmentation"] = rle
                    elif isinstance(rle, dict) and "segmentation" in rle:
                        ann["segmentation"] = rle["segmentation"]

                    annotations.append(ann)

    coco_json = {
        "images": images,
        "annotations": annotations,
        "categories": coco_categories,
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(coco_json, f, indent=2)

    print(f"[convert_segmentation_to_coco] Written {len(images)} images,"
          f" {len(annotations)} annotations to '{output_path}'")
    return output_path

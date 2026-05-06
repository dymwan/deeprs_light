#!/usr/bin/env python
"""
CLI tool: Convert custom detection/segmentation formats to standard COCO JSON.

Usage:
    # Detection (YOLO format to COCO)
    python tools/convert_to_coco.py detection \\
        --data_dir /data/my_dataset \\
        --output /data/coco/train.json \\
        --categories '[{"id":1,"name":"ship"},{"id":2,"name":"vehicle"}]'

    # Segmentation (mask PNGs to COCO)
    python tools/convert_to_coco.py segmentation \\
        --data_dir /data/my_dataset \\
        --output /data/coco/train.json \\
        --categories '[{"id":1,"name":"building"},{"id":2,"name":"road"}]'
"""

import argparse
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from deeprs_light.data.coco_converter import (
    convert_detection_to_coco,
    convert_segmentation_to_coco,
)


def main():
    parser = argparse.ArgumentParser(
        description="Convert custom formats to COCO JSON (detection & segmentation)."
    )
    subparsers = parser.add_subparsers(dest="task", required=True)

    # --- Detection ---
    det_parser = subparsers.add_parser(
        "detection", help="Convert detection data to COCO."
    )
    det_parser.add_argument("--data_dir", required=True, help="Root data directory.")
    det_parser.add_argument("--output", required=True, help="Output COCO JSON path.")
    det_parser.add_argument(
        "--categories", required=True,
        help='JSON string of category list, e.g., \'[{"id":1,"name":"ship"}]\'.'
    )
    det_parser.add_argument("--image_dir", default="images", help="Image subdirectory.")
    det_parser.add_argument("--label_dir", default="labels", help="Label subdirectory.")
    det_parser.add_argument("--image_ext", default=".png", help="Image file extension.")
    det_parser.add_argument("--label_map", default=None,
                           help='JSON string of class_name -> category_id mapping.')

    # --- Segmentation ---
    seg_parser = subparsers.add_parser(
        "segmentation", help="Convert segmentation mask data to COCO."
    )
    seg_parser.add_argument("--data_dir", required=True, help="Root data directory.")
    seg_parser.add_argument("--output", required=True, help="Output COCO JSON path.")
    seg_parser.add_argument(
        "--categories", required=True,
        help='JSON string of category list.'
    )
    seg_parser.add_argument("--mask_dir", default="masks", help="Mask subdirectory.")
    seg_parser.add_argument("--image_dir", default="images", help="Image subdirectory.")
    seg_parser.add_argument("--image_ext", default=".png", help="Image file extension.")

    args = parser.parse_args()

    # Parse categories
    categories = json.loads(args.categories)

    if args.task == "detection":
        label_map = None
        if args.label_map:
            label_map = json.loads(args.label_map)
        convert_detection_to_coco(
            data_dir=args.data_dir,
            output_path=args.output,
            categories=categories,
            label_map=label_map,
            image_dir=args.image_dir,
            label_dir=args.label_dir,
            image_ext=args.image_ext,
        )
    elif args.task == "segmentation":
        convert_segmentation_to_coco(
            data_dir=args.data_dir,
            output_path=args.output,
            categories=categories,
            mask_dir=args.mask_dir,
            image_dir=args.image_dir,
            image_ext=args.image_ext,
        )


if __name__ == "__main__":
    main()

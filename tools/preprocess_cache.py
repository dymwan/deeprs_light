#!/usr/bin/env python
"""
CLI tool: Build a preprocessed cache from a COCO dataset.

Usage:
    # Basic
    python tools/preprocess_cache.py \\
        --root /data/images \\
        --ann_file /data/train.json \\
        --output cache/train_lmdb \\
        --backend lmdb \\
        --size 800 \\
        --num_workers 8

    # With custom transforms
    python tools/preprocess_cache.py \\
        --root /data/images \\
        --ann_file /data/train.json \\
        --output cache/train_lmdb \\
        --size 800 \\
        --num_workers 8 \\
        --custom_transforms "my_module.EdgeMapTarget,my_module.GeoMetaTarget"

    # Check cache status
    python tools/preprocess_cache.py \\
        --output cache/train_lmdb \\
        --check

    # Check if cache has required keys
    python tools/preprocess_cache.py \\
        --output cache/train_lmdb \\
        --check_keys boxes,labels,edge_map
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main():
    parser = argparse.ArgumentParser(
        description="Build/manage preprocessed cache for deeprs_light."
    )
    parser.add_argument("--root", default=None, help="Image root directory.")
    parser.add_argument("--ann_file", default=None, help="COCO annotation JSON path.")
    parser.add_argument("--output", required=True, help="Cache output path.")
    parser.add_argument("--backend", default="lmdb", choices=["lmdb", "pt"],
                       help="Cache backend type.")
    parser.add_argument("--size", type=int, default=800, help="Resize target size.")
    parser.add_argument("--mean", nargs=3, type=float, default=[0.485, 0.456, 0.406],
                       help="Normalize mean (3 values).")
    parser.add_argument("--std", nargs=3, type=float, default=[0.229, 0.224, 0.225],
                       help="Normalize std (3 values).")
    parser.add_argument("--num_workers", type=int, default=0,
                       help="Number of parallel workers (0 = single-process).")
    parser.add_argument("--custom_transforms", default=None,
                       help="Comma-separated import paths for custom PreprocessTransform classes.")
    parser.add_argument("--resume", action="store_true",
                       help="Resume from interrupted run.")
    parser.add_argument("--check", action="store_true",
                       help="Check cache status (read-only).")
    parser.add_argument("--check_keys", default=None,
                       help="Comma-separated list of required target keys to check.")

    args = parser.parse_args()

    if args.check:
        return _check_cache(args.output)
    if args.check_keys:
        return _check_required_keys(args.output, args.check_keys)

    # --- Build cache ---
    if args.root is None or args.ann_file is None:
        parser.error("--root and --ann_file are required for cache building.")

    _build_cache(
        root=args.root,
        ann_file=args.ann_file,
        output=args.output,
        backend=args.backend,
        size=args.size,
        mean=tuple(args.mean),
        std=tuple(args.std),
        num_workers=args.num_workers,
        custom_transforms=args.custom_transforms,
        resume=args.resume,
    )


def _build_cache(
    root, ann_file, output, backend, size, mean, std,
    num_workers, custom_transforms, resume,
):
    """Run cache preprocessing."""
    from deeprs_light.data.dataset import DeepRSCocoDataset
    from deeprs_light.data.cache_backend import LMDBBackend, PTBackend
    from deeprs_light.data.preprocess import (
        PreprocessPipeline, FixedResize, FixedNormalize,
    )

    # Create dataset
    print(f"Loading dataset from '{root}':{ann_file}...")
    dataset = DeepRSCocoDataset(root=root, ann_file=ann_file)

    # Create backend
    if backend == "lmdb":
        cache_backend = LMDBBackend(output)
    else:
        cache_backend = PTBackend(output, single_file=False)

    # Build transform list
    transforms = [
        FixedResize(size),
        FixedNormalize(mean=mean, std=std),
    ]

    # Load custom transforms if specified
    if custom_transforms:
        for path in custom_transforms.split(","):
            path = path.strip()
            if not path:
                continue
            try:
                module_name, class_name = path.rsplit(".", 1)
                import importlib
                mod = importlib.import_module(module_name)
                cls = getattr(mod, class_name)
                transforms.append(cls())
                print(f"  Added custom transform: {class_name}")
            except Exception as e:
                print(f"  [WARN] Failed to load '{path}': {e}")

    pipeline = PreprocessPipeline(
        dataset=dataset,
        backend=cache_backend,
        transforms=transforms,
    )

    if resume:
        num_done = pipeline.resume()
    else:
        print(f"Starting preprocessing with {num_workers} workers...")
        num_done, manifest = pipeline.run(num_workers=num_workers)
        print(f"\nDone: {num_done} images processed.")
        print(f"Manifest: {manifest.target_keys} target keys.")
        print(f"Config hash: {manifest.config_hash}")


def _check_cache(output):
    """Check cache status."""
    from deeprs_light.data.cache_manifest import CacheManifest

    manifest = CacheManifest.load(output)
    if manifest is None:
        print(f"No cache found at '{output}'.")
        return 1

    print(f"Cache: {output}")
    print(f"  Version:      {manifest.version}")
    print(f"  Images:       {manifest.num_samples}")
    print(f"  Image size:   {manifest.image_size}")
    print(f"  Target keys:  {manifest.target_keys}")
    print(f"  Created:      {manifest.created_at}")
    print(f"  Config hash:  {manifest.config_hash}")
    print(f"  Source:       {manifest.source}")
    return 0


def _check_required_keys(output, keys_str):
    """Check if cache has required target keys."""
    from deeprs_light.data.cache_manifest import CacheManifest, check_target_keys

    required = [k.strip() for k in keys_str.split(",")]
    manifest = CacheManifest.load(output)
    if manifest is None:
        print(f"No cache found at '{output}'.")
        return 1

    ok, missing = check_target_keys(required, manifest)
    if ok:
        print(f"OK: All {len(required)} required keys present in cache.")
        print(f"  Required: {required}")
        print(f"  Available: {manifest.target_keys}")
        return 0
    else:
        print(f"MISSING KEYS: {missing}")
        print(f"  Required:  {required}")
        print(f"  Available: {manifest.target_keys}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

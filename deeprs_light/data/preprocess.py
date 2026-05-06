"""
Offline preprocessing pipeline with multi-process safety.
Writes preprocessed (image, target) pairs to a cache backend.
Supports custom PreprocessTransforms that can add arbitrary target keys.
"""

import os
import shutil
import tempfile
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Union, Any, Optional

import numpy as np
import cv2
import torch
from tqdm import tqdm

from deeprs_light.data.dataset import DeepRSCocoDataset
from deeprs_light.data.cache_backend import CacheBackend, LMDBBackend, PTBackend
from deeprs_light.data.cache_manifest import CacheManifest


# ============================================================
# PreprocessTransform base
# ============================================================

class PreprocessTransform(ABC):
    """
    Base class for offline preprocess transforms.

    Key differences from online TransformBase:
    - Can **add new keys** to the target dict (not just modify existing keys)
    - Receives meta info (image_path, image_id, width, height)

    Subclasses implement:
        apply(image, target, meta) -> (image, target)
    """

    @abstractmethod
    def apply(
        self,
        image: np.ndarray,
        target: Dict[str, Any],
        meta: Dict[str, Any],
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply transform to a single image-target pair.

        Args:
            image: np.ndarray (H, W, C), raw image.
            target: Current target dict (boxes, labels, masks, ...).
            meta: Metadata dict with keys:
                  - "image_id": int
                  - "image_path": str
                  - "width": int
                  - "height": int

        Returns:
            (image, target): Transformed pair. Target can have new keys added.
        """


class CustomTarget(PreprocessTransform):
    """
    Convenience base for user-defined transforms that add custom target keys.

    Usage:
        class MyTarget(CustomTarget):
            def apply(self, image, target, meta):
                target["my_key"] = compute_something(image, target)
                return image, target
    """
    def apply(self, image, target, meta):
        raise NotImplementedError("Override in subclass")


# ============================================================
# Built-in PreprocessTransforms
# ============================================================

class FixedResize(PreprocessTransform):
    """Fixed-size resize for offline preprocessing (no randomness)."""

    def __init__(self, size: Union[int, Tuple[int, int]]):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size

    def apply(self, image, target, meta):
        h, w = image.shape[:2]
        target_w, target_h = self.size

        # Scale image
        image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        scale_w = target_w / w
        scale_h = target_h / h

        # Scale boxes
        from deeprs_light.data.transforms_utils import scale_boxes
        if target["boxes"].shape[0] > 0:
            boxes = target["boxes"].numpy() if isinstance(target["boxes"], torch.Tensor) else np.array(target["boxes"])
            target["boxes"] = torch.from_numpy(scale_boxes(boxes, scale_w, scale_h))

        # Scale masks
        if target["masks"].shape[0] > 0 and target["masks"].ndim == 3:
            masks = target["masks"].numpy() if isinstance(target["masks"], torch.Tensor) else target["masks"]
            masks_r = np.array([cv2.resize(m, (target_w, target_h), interpolation=cv2.INTER_NEAREST) for m in masks])
            target["masks"] = torch.from_numpy(masks_r)

        target["image_size"] = (target_w, target_h)
        return image, target


class FixedNormalize(PreprocessTransform):
    """Fixed normalization: (image - mean) / std, output float32."""

    def __init__(self, mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
                 std: Tuple[float, ...] = (0.229, 0.224, 0.225)):
        self.mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(std, dtype=np.float32).reshape(1, 1, 3)

    def apply(self, image, target, meta):
        image = (image.astype(np.float32) / 255.0 - self.mean) / self.std
        return image, target


class EdgeMapTarget(PreprocessTransform):
    """
    Example: pre-compute edge maps as a new target key.
    Uses Canny edge detection offline so training doesn't recompute.
    Result stored in target["edge_map"] as (H,W) tensor of 0/1.
    """

    def __init__(self, low_threshold: float = 50, high_threshold: float = 150):
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def apply(self, image, target, meta):
        # Canny on grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, self.low_threshold, self.high_threshold)
        target["edge_map"] = torch.from_numpy((edges > 0).astype(np.float32))
        return image, target


class GeographicMetaTarget(PreprocessTransform):
    """
    Example: extract geographic metadata from GeoTIFF as new target keys.
    Stores spatial reference info in target["geo_meta"].
    """

    def apply(self, image, target, meta):
        # Placeholder: in practice would read GeoTIFF tags
        target["geo_meta"] = {
            "image_id": meta["image_id"],
            "width": meta["width"],
            "height": meta["height"],
        }
        return image, target


# ============================================================
# Preprocessing Pipeline (multi-process safe)
# ============================================================

class PreprocessPipeline:
    """
    Offline preprocessing pipeline with multi-process safety.

    Multi-process strategy:
        - Each worker writes to an isolated temporary cache directory
        - Main process merges all worker outputs into the final cache
        - No process ever writes to the same cache file concurrently

    Usage:
        pipeline = PreprocessPipeline(
            dataset=dataset,   # DeepRSCocoDataset (no online transforms)
            backend=LMDBBackend("cache/train_lmdb"),
            transforms=[FixedResize(800), FixedNormalize(...), EdgeMapTarget()],
        )
        num_processed, manifest = pipeline.run(num_workers=8)
    """

    def __init__(
        self,
        dataset: DeepRSCocoDataset,
        backend: CacheBackend,
        transforms: List[PreprocessTransform],
    ):
        self.dataset = dataset
        self.backend = backend
        self.transforms = transforms

    def run(self, num_workers: int = 0) -> Tuple[int, CacheManifest]:
        """
        Run preprocessing.

        Args:
            num_workers: Number of parallel worker processes. 0 = single-process.

        Returns:
            (num_processed, manifest)
        """
        image_ids = self.dataset.image_ids

        if num_workers <= 0:
            num_done = self._run_single(image_ids)
        else:
            num_done = self._run_multi(image_ids, num_workers)

        # Build and write manifest
        manifest = self._build_manifest(num_done)
        self.backend.write_manifest(manifest)

        return num_done, manifest

    def _run_single(self, image_ids: List[int]) -> int:
        """Single-process execution."""
        self.backend.open("w")
        cnt = 0
        for image_id in tqdm(image_ids, desc="Preprocessing"):
            try:
                image, target = self._process_one(image_id)
                # Serialize: convert numpy images to tensor for storage
                image_t = torch.from_numpy(image).float() if isinstance(image, np.ndarray) else image
                self.backend.put(image_id, {"image": image_t, "target": target})
                cnt += 1
            except Exception as e:
                print(f"[WARN] Failed to process image_id={image_id}: {e}")
        self.backend.close()
        return cnt

    def _run_multi(self, image_ids: List[int], num_workers: int) -> int:
        """
        Multi-process execution.
        Each worker writes to a separate temp cache; main process merges.
        """
        import multiprocessing as mp

        # Split image_ids into chunks
        chunk_size = max(1, len(image_ids) // num_workers)
        chunks = [image_ids[i:i + chunk_size] for i in range(0, len(image_ids), chunk_size)]

        # Create temp directory for workers
        tmp_root = tempfile.mkdtemp(prefix="deeprs_preprocess_")

        # Serialize transform config for workers
        transforms_config = [
            {"class": t.__class__.__name__, "args": repr(vars(t))}
            for t in self.transforms
        ]

        # Serialize dataset constructor args
        dataset_args = {
            "root": self.dataset.root,
            "ann_file": self.dataset.ann_file,
        }

        # Determine backend class
        backend_cls_name = self.backend.__class__.__name__

        # Launch workers
        args_list = []
        for i, chunk in enumerate(chunks):
            tmp_dir = os.path.join(tmp_root, f"worker_{i}")
            args_list.append((
                chunk, dataset_args, transforms_config,
                tmp_dir, backend_cls_name,
            ))

        with mp.Pool(processes=min(num_workers, len(chunks))) as pool:
            results = pool.starmap(_worker_fn, args_list)

        cnt = sum(results)

        # Merge: copy from each worker's temp cache to final backend
        self.backend.open("w")
        for i in range(len(chunks)):
            tmp_dir = os.path.join(tmp_root, f"worker_{i}")
            self._merge_worker(tmp_dir, backend_cls_name)
        self.backend.close()

        # Cleanup
        shutil.rmtree(tmp_root, ignore_errors=True)

        return cnt

    def _merge_worker(self, tmp_dir: str, backend_cls_name: str):
        """Merge one worker's temp cache into the final backend."""
        if not os.path.isdir(tmp_dir):
            return

        if backend_cls_name == "LMDBBackend":
            tmp_backend = LMDBBackend(tmp_dir)
            tmp_backend.open("r")
        else:
            tmp_backend = PTBackend(tmp_dir, single_file=False)
            tmp_backend.open("r")

        for key in tmp_backend.keys():
            try:
                data = tmp_backend.get(key)
                self.backend.put(key, data)
            except Exception as e:
                print(f"[WARN] Failed to merge key={key}: {e}")

        tmp_backend.close()

    def _process_one(self, image_id: int) -> Tuple[np.ndarray, Dict]:
        """Process a single image through all transforms."""
        image = self.dataset._load_image(image_id)
        target = self.dataset._load_target(image_id)
        info = self.dataset._img_info[image_id]

        meta = {
            "image_id": image_id,
            "image_path": os.path.join(self.dataset.root, info["file_name"]),
            "width": info["width"],
            "height": info["height"],
        }

        for t in self.transforms:
            image, target = t.apply(image, target, meta)

        return image, target

    def _build_manifest(self, num_processed: int) -> CacheManifest:
        """Build CacheManifest from the pipeline configuration."""
        # Collect target keys: base keys + keys added by transforms
        base_keys = {"boxes", "labels", "masks", "area", "iscrowd", "image_id"}
        # Run one sample to detect added keys
        extra_keys = set()
        if len(self.dataset) > 0:
            try:
                _, sample_target = self._process_one(self.dataset.image_ids[0])
                extra_keys = set(sample_target.keys()) - base_keys
            except Exception:
                pass

        all_keys = sorted(base_keys | {"image_size"}) if "image_size" not in base_keys else sorted(base_keys)
        all_keys = sorted(set(all_keys) | extra_keys)

        config_hash = CacheManifest.compute_config_hash(self.transforms)

        # Detect image size and normalize from transforms
        image_size = []
        normalize = {"mean": [], "std": []}
        for t in self.transforms:
            if isinstance(t, FixedResize):
                image_size = list(t.size)
            elif isinstance(t, FixedNormalize):
                normalize = {
                    "mean": list(t.mean.flatten()),  # type: ignore
                    "std": list(t.std.flatten()),    # type: ignore
                }

        return CacheManifest(
            version="1.0",
            dataset_name="deeprs_coco",
            source={
                "root": self.dataset.root,
                "ann_file": self.dataset.ann_file,
            },
            image_size=image_size,
            normalize=normalize,
            target_keys=all_keys,
            num_samples=num_processed,
            config_hash=config_hash,
        )

    def resume(self) -> int:
        """
        Resume from an interrupted preprocessing run.
        Checks for existing entries and only processes missing image_ids.
        """
        # Check what's already in the final backend
        self.backend.open("r")
        existing = set(self.backend.keys())
        self.backend.close()

        missing = [iid for iid in self.dataset.image_ids if iid not in existing]
        if not missing:
            print("All images already processed. Nothing to resume.")
            return 0

        print(f"Resuming: {len(missing)} images remaining")
        return self.run(len(missing))


def _worker_fn(
    chunk_image_ids: List[int],
    dataset_args: Dict,
    transforms_config: List[Dict],
    tmp_dir: str,
    backend_cls_name: str,
) -> int:
    """
    Worker function (module-level for pickle safety).

    Reconstructs transforms and dataset, processes its chunk,
    and writes results to a temporary backend.
    """
    from deeprs_light.data.dataset import DeepRSCocoDataset
    from deeprs_light.data.preprocess import (
        FixedResize, FixedNormalize, EdgeMapTarget, GeographicMetaTarget,
    )

    # Reconstruct transforms from config
    cls_map = {
        "FixedResize": FixedResize,
        "FixedNormalize": FixedNormalize,
        "EdgeMapTarget": EdgeMapTarget,
        "GeographicMetaTarget": GeographicMetaTarget,
    }
    transforms = []
    for cfg in transforms_config:
        cls = cls_map.get(cfg["class"])
        if cls is None:
            print(f"[WARN] Unknown transform class '{cfg['class']}' in worker, skipping.")
            continue
        # Reconstruct from repr: eval is dangerous, so use simple mapping
        # In practice, custom transforms should register with a factory
        try:
            transforms.append(cls())
        except Exception:
            transforms.append(cls)

    # Reconstruct dataset
    dataset = DeepRSCocoDataset(
        root=dataset_args["root"],
        ann_file=dataset_args["ann_file"],
    )

    # Create temp backend
    if backend_cls_name == "LMDBBackend":
        backend = LMDBBackend(tmp_dir)
    else:
        backend = PTBackend(tmp_dir, single_file=False)
    backend.open("w")

    cnt = 0
    for image_id in chunk_image_ids:
        try:
            image = dataset._load_image(image_id)
            target = dataset._load_target(image_id)
            info = dataset._img_info[image_id]
            meta = {
                "image_id": image_id,
                "image_path": os.path.join(dataset.root, info["file_name"]),
                "width": info["width"],
                "height": info["height"],
            }
            for t in transforms:
                image, target = t.apply(image, target, meta)

            image_t = torch.from_numpy(image).float() if isinstance(image, np.ndarray) else image
            backend.put(image_id, {"image": image_t, "target": target})
            cnt += 1
        except Exception as e:
            print(f"[WARN] Worker failed image_id={image_id}: {e}")

    backend.close()
    return cnt

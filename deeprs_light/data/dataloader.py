"""
Unified DataLoader builder with online/cache mode switching and
target key validation.
"""

from typing import Dict, List, Optional, Callable

from torch.utils.data import DataLoader

from deeprs_light.registry import DATASETS, CACHE_BACKENDS
from deeprs_light.data.transforms import Compose
from deeprs_light.data.transforms_utils import get_train_transforms, get_val_transforms


def build_dataloader(
    name: str,
    root: str = None,
    ann_file: str = None,
    cache_dir: str = None,
    cache_backend: str = "lmdb",
    required_keys: Optional[List[str]] = None,
    auto_rebuild: bool = False,
    split: str = "train",
    batch_size: int = 16,
    num_workers: int = 4,
    transforms: Optional[List[Callable]] = None,
    shuffle: bool = True,
    pin_memory: bool = True,
    drop_last: bool = False,
    **kwargs,
) -> DataLoader:
    """
    Unified DataLoader builder. Switches between online and cache mode.

    Mode A — Online (cache_dir=None):
        Loads raw COCO JSON + images, applies full transform pipeline.

    Mode B — Cache (cache_dir is not None):
        Reads from preprocessed cache, applies light online augmentations.
        Validates required target keys against cache manifest.

    Args:
        name: Dataset registry name.
        root: Image root directory (online mode).
        ann_file: COCO annotation JSON path (online mode).
        cache_dir: Cache directory path (cache mode).
        cache_backend: "lmdb" or "pt" (cache mode).
        required_keys: Target keys the model needs. Checked against cache manifest.
        auto_rebuild: If True, rebuild cache when keys are missing.
        split: "train" or "val".
        batch_size: Batch size.
        num_workers: DataLoader workers.
        transforms: Transform list (None = use preset).
        shuffle: Whether to shuffle.
        pin_memory: Pin memory for faster GPU transfer.
        drop_last: Drop last incomplete batch.
        **kwargs: Extra args passed to Dataset constructor.

    Returns:
        A torch DataLoader.

    Usage:
        # Online mode
        loader = build_dataloader("deeprs_coco", root="/data/imgs",
                                  ann_file="/data/train.json")

        # Cache mode with key validation
        loader = build_dataloader(
            "deeprs_coco", cache_dir="cache/train_lmdb",
            required_keys=["boxes", "labels", "edge_map"],
        )
    """
    # --- Cache mode ---
    if cache_dir is not None:
        return _build_cache_loader(
            name=name,
            cache_dir=cache_dir,
            cache_backend=cache_backend,
            required_keys=required_keys,
            auto_rebuild=auto_rebuild,
            split=split,
            batch_size=batch_size,
            num_workers=num_workers,
            transforms=transforms,
            shuffle=shuffle,
            pin_memory=pin_memory,
            drop_last=drop_last,
            **kwargs,
        )

    # --- Online mode ---
    dataset_cls = DATASETS.get(name)
    if transforms is None:
        transforms = get_train_transforms() if split == "train" else get_val_transforms()
    else:
        transforms = Compose(transforms)

    dataset = dataset_cls(
        root=root,
        ann_file=ann_file,
        transforms=transforms,
        **kwargs,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=dataset_cls.collate_fn,
    )


def _build_cache_loader(
    name: str,
    cache_dir: str,
    cache_backend: str,
    required_keys: Optional[List[str]],
    auto_rebuild: bool,
    split: str,
    batch_size: int,
    num_workers: int,
    transforms: Optional[List[Callable]],
    shuffle: bool,
    pin_memory: bool,
    drop_last: bool,
    **kwargs,
) -> DataLoader:
    """Internal: build a DataLoader from cache."""
    from deeprs_light.data.cache_manifest import CacheManifest, check_target_keys
    from deeprs_light.data.cache_dataset import CocoCacheDataset

    # Read manifest
    manifest = CacheManifest.load(cache_dir)
    if manifest is None:
        if auto_rebuild:
            _auto_rebuild_cache(name, cache_dir, cache_backend, **kwargs)
            manifest = CacheManifest.load(cache_dir)
        else:
            raise RuntimeError(
                f"No manifest found in '{cache_dir}'. "
                f"Run preprocess_cache.py first, or set auto_rebuild=True."
            )

    # Validate required keys
    if required_keys is not None and manifest is not None:
        ok, missing = check_target_keys(required_keys, manifest)
        if not ok:
            if auto_rebuild:
                _auto_rebuild_cache(name, cache_dir, cache_backend, **kwargs)
                manifest = CacheManifest.load(cache_dir)
            else:
                raise RuntimeError(
                    f"Cache at '{cache_dir}' is missing target keys: {missing}. "
                    f"Available keys: {manifest.target_keys}. "
                    f"Re-run preprocess_cache.py with the required transforms, "
                    f"or set auto_rebuild=True."
                )

    # Build transforms
    if transforms is None:
        # Cache mode: light augment only (no Resize/Normalize)
        from deeprs_light.data.transforms import (
            Compose, RandomHorizontalFlip, RandomVerticalFlip,
            ColorJitter, RandomBrightnessContrast,
        )
        if split == "train":
            transforms = Compose([
                RandomHorizontalFlip(p=0.5),
                RandomVerticalFlip(p=0.3),
                ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.3),
            ])
        else:
            transforms = None
    else:
        transforms = Compose(transforms)

    backend_cls = CACHE_BACKENDS.get(cache_backend)
    backend = backend_cls(cache_dir)
    backend.open("r")

    dataset = CocoCacheDataset(backend=backend, transforms=transforms, **kwargs)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=CocoCacheDataset.collate_fn,
    )


def _auto_rebuild_cache(
    dataset_name: str,
    cache_dir: str,
    cache_backend: str,
    **kwargs,
):
    """Trigger cache rebuild automatically."""
    from deeprs_light.data.cache_manifest import CacheManifest
    print(f"[deeprs_light] Auto-rebuilding cache at '{cache_dir}'...")
    manifest = CacheManifest.load(cache_dir)
    if manifest is None:
        raise RuntimeError(
            f"Cannot auto-rebuild: no source info available. "
            f"Run preprocess_cache.py manually first time."
        )
    # Use stored source info from manifest
    # In practice this would call PreprocessPipeline with the recorded config
    raise NotImplementedError("Auto-rebuild not yet implemented. Run preprocess_cache.py manually.")


def get_cached_target_keys(cache_dir: str, cache_backend: str = "lmdb") -> List[str]:
    """
    Quickly check available target keys in a cache (reads manifest only).

    Usage:
        keys = get_cached_target_keys("cache/train_lmdb")
    """
    from deeprs_light.data.cache_manifest import CacheManifest
    manifest = CacheManifest.load(cache_dir)
    if manifest is None:
        raise RuntimeError(f"No manifest found in '{cache_dir}'")
    return manifest.target_keys

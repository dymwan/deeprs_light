"""
Dataset that reads from offline preprocessed cache (LMDB or PT backend).
Provides the same interface as DeepRSCocoDataset but skips heavy transforms.
"""

from typing import Dict, List, Optional, Callable, Tuple

import torch
from torch.utils.data import Dataset

from deeprs_light.registry import DATASETS
from deeprs_light.data.cache_backend import CacheBackend


@DATASETS.register("coco_cache")
class CocoCacheDataset(Dataset):
    """
    Dataset reading from preprocessed cache.

    Data is already resized and normalized; online transforms should only
    include light augmentations (Flip, ColorJitter) that don't change image size.

    Usage:
        backend = LMDBBackend("cache/train_lmdb")
        backend.open("r")

        dataset = CocoCacheDataset(
            backend=backend,
            transforms=Compose([RandomHorizontalFlip(0.5)]),
        )
        loader = DataLoader(dataset, batch_size=16, collate_fn=dataset.collate_fn)
    """

    def __init__(
        self,
        backend: CacheBackend,
        transforms: Optional[Callable] = None,
    ):
        """
        Args:
            backend: Opened cache backend.
            transforms: Light online augmentations (e.g., RandomFlip, ColorJitter).
                        Should NOT include Resize or Normalize.
        """
        self.backend = backend
        self.transforms = transforms
        self._keys = backend.keys()

    def __len__(self) -> int:
        return len(self._keys)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        image_id = self._keys[idx]

        # Read from cache
        cached = self.backend.get(image_id)
        image = cached["image"]
        target = cached["target"]

        # Ensure image_id is present
        if "image_id" not in target:
            target["image_id"] = image_id

        # Apply online transforms
        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    @staticmethod
    def collate_fn(batch: List[Tuple]) -> Tuple[torch.Tensor, List[Dict]]:
        """
        Collate: stack images, keep targets as list of dicts.
        Custom keys in targets are preserved.
        """
        images = torch.stack([item[0] for item in batch], dim=0)
        targets = [item[1] for item in batch]
        return images, targets

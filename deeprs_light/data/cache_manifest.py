"""
Cache manifest: records what target keys and config a cache contains.
Enables fast checking of cache compatibility without opening the full dataset.
"""

import os
import json
import hashlib
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime


@dataclass
class CacheManifest:
    """
    Manifest file (manifest.json) stored at cache root.

    Records the complete configuration used to build the cache,
    enabling CocoCacheDataset to quickly verify target key availability
    without reading individual cache entries.

    Storage format (JSON):
    {
        "version": "1.0",
        "dataset_name": "my_coco_dataset",
        "source": {"root": "/data/images", "ann_file": "/data/train.json"},
        "image_size": [800, 800],
        "normalize": {"mean": [0.485, ...], "std": [0.229, ...]},
        "target_keys": ["boxes", "labels", "masks", "edge_map", ...],
        "num_samples": 50000,
        "config_hash": "a1b2c3d4...",
        "created_at": "2026-05-06T10:30:00"
    }
    """

    version: str = "1.0"
    dataset_name: str = ""
    source: Dict[str, str] = field(default_factory=dict)
    image_size: List[int] = field(default_factory=list)
    normalize: Dict[str, List[float]] = field(default_factory=dict)
    target_keys: List[str] = field(default_factory=list)
    num_samples: int = 0
    config_hash: str = ""
    created_at: str = ""

    def save(self, path: str):
        """
        Write manifest to path/manifest.json.

        Args:
            path: Directory path (e.g., "cache/train_lmdb/") or file path.
        """
        if path.endswith(".json"):
            file_path = path
        else:
            os.makedirs(path, exist_ok=True)
            file_path = os.path.join(path, "manifest.json")

        self.created_at = datetime.now().isoformat()
        with open(file_path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @staticmethod
    def load(path: str) -> Optional["CacheManifest"]:
        """
        Load manifest from path/manifest.json.

        Args:
            path: Directory path (e.g., "cache/train_lmdb/") or direct file path.

        Returns:
            CacheManifest or None if not found.
        """
        if path.endswith(".json"):
            file_path = path
        else:
            file_path = os.path.join(path, "manifest.json")

        if not os.path.exists(file_path):
            return None

        with open(file_path, "r") as f:
            data = json.load(f)

        return CacheManifest(
            version=data.get("version", "1.0"),
            dataset_name=data.get("dataset_name", ""),
            source=data.get("source", {}),
            image_size=data.get("image_size", []),
            normalize=data.get("normalize", {}),
            target_keys=data.get("target_keys", []),
            num_samples=data.get("num_samples", 0),
            config_hash=data.get("config_hash", ""),
            created_at=data.get("created_at", ""),
        )

    @staticmethod
    def compute_config_hash(transforms: List[Any]) -> str:
        """
        Compute a deterministic SHA256 hash from a transform pipeline.

        Iterates over transforms, recording class name and init args (via repr),
        concatenates them, and computes SHA256.

        This allows detecting when a cache needs to be rebuilt because the
        transform configuration has changed.

        Args:
            transforms: List of PreprocessTransform instances.

        Returns:
            Hex digest string.
        """
        parts = []
        for t in transforms:
            cls_name = t.__class__.__name__
            try:
                args_repr = repr(vars(t))
            except Exception:
                args_repr = ""
            parts.append(f"{cls_name}:{args_repr}")
        config_str = "|".join(parts)
        return hashlib.sha256(config_str.encode()).hexdigest()


def check_target_keys(
    required_keys: List[str],
    manifest: CacheManifest,
) -> Tuple[bool, List[str]]:
    """
    Check if cached target_keys satisfy the model's requirements.

    Args:
        required_keys: Target keys the model needs (e.g., ["boxes", "labels", "edge_map"]).
        manifest: The cache manifest.

    Returns:
        (ok, missing): ok=True means all required keys present;
                       missing is the list of keys not found (empty if ok).
    """
    available = set(manifest.target_keys)
    missing = [k for k in required_keys if k not in available]
    return len(missing) == 0, missing

"""
Cache backend abstraction with LMDB and PT implementations.
Provides unified get/put/keys/len interface for cached datasets.
"""

import os
import glob
import json
import struct
from abc import ABC, abstractmethod
from typing import Dict, List, Union, Any, Optional

import numpy as np
import torch

from deeprs_light.registry import CACHE_BACKENDS
from deeprs_light.data.cache_manifest import CacheManifest


# ============================================================
# Serialization helpers
# ============================================================

def _serialize_tensor(t: torch.Tensor) -> Dict:
    """Serialize a torch.Tensor to a msgpack-friendly dict."""
    t_c = t.contiguous().cpu()
    return {
        "_t": True,
        "dtype": str(t_c.dtype).split(".")[-1],
        "shape": list(t_c.shape),
        "data": t_c.numpy().tobytes(),
    }


def _deserialize_tensor(d: Dict) -> torch.Tensor:
    """Deserialize a tensor dict back to torch.Tensor."""
    dtype = getattr(torch, d["dtype"])
    arr = np.frombuffer(d["data"], dtype=np.dtype(d["dtype"].replace("float32", "float32")))
    # Map torch dtype string to numpy dtype
    dtype_map = {
        "float32": np.float32,
        "float64": np.float64,
        "int64": np.int64,
        "int32": np.int32,
        "uint8": np.uint8,
        "bool": np.bool_,
    }
    np_dtype = dtype_map.get(d["dtype"], np.float32)
    arr = np.frombuffer(d["data"], dtype=np_dtype)
    return torch.from_numpy(arr.reshape(d["shape"]))


def _encode(data: Dict[str, Any]) -> bytes:
    """
    Encode a data dict (image + target) to bytes via msgpack.
    Tensors are serialized inline.
    """
    import msgpack
    def _encode_value(v):
        if isinstance(v, torch.Tensor):
            return _serialize_tensor(v)
        if isinstance(v, np.ndarray):
            return _serialize_tensor(torch.from_numpy(v))
        if isinstance(v, dict):
            return {kk: _encode_value(vv) for kk, vv in v.items()}
        if isinstance(v, list):
            return [_encode_value(vv) for vv in v]
        return v

    encoded = _encode_value(data)
    return msgpack.packb(encoded, use_bin_type=True)


def _decode(raw: bytes) -> Dict[str, Any]:
    """
    Decode msgpack bytes back to a data dict.
    Tensor dicts are deserialized back to torch.Tensor.
    """
    import msgpack

    def _decode_value(v):
        if isinstance(v, dict):
            if v.get("_t"):
                return _deserialize_tensor(v)
            return {kk: _decode_value(vv) for kk, vv in v.items()}
        if isinstance(v, list):
            return [_decode_value(vv) for vv in v]
        return v

    data = msgpack.unpackb(raw, raw=False)
    return _decode_value(data)


# ============================================================
# Abstract backend
# ============================================================

class CacheBackend(ABC):
    """
    Abstract cache backend providing unified read/write interface.

    Subclasses: LMDBBackend, PTBackend.
    """

    def __init__(self, path: str):
        self.path = path
        self._open = False

    @abstractmethod
    def open(self, mode: str = "r"):
        """Open or create the cache. mode: 'r' (read) or 'w' (write)."""

    @abstractmethod
    def put(self, key: int, data: Dict[str, Any]):
        """Write one record. key = image_id, data = {"image": Tensor, "target": Dict}."""

    @abstractmethod
    def get(self, key: int) -> Dict[str, Any]:
        """Read one record. Returns {"image": Tensor, "target": Dict}."""

    @abstractmethod
    def keys(self) -> List[int]:
        """Get all cached image_ids."""

    @abstractmethod
    def close(self):
        """Close the cache."""

    @abstractmethod
    def __len__(self) -> int:
        """Total number of cached entries."""

    def write_manifest(self, manifest: CacheManifest):
        """Write manifest to the cache directory."""
        manifest.save(self.path)

    def read_manifest(self) -> Optional[CacheManifest]:
        """Read manifest.json from the cache directory."""
        return CacheManifest.load(self.path)

    def __enter__(self):
        self.open("r")
        return self

    def __exit__(self, *args):
        self.close()


# ============================================================
# LMDB Backend
# ============================================================

@CACHE_BACKENDS.register("lmdb")
class LMDBBackend(CacheBackend):
    """
    LMDB cache backend.

    Features:
    - Memory-mapped (mmap) for zero-copy reads
    - Lock-free concurrent reads (perfect for num_workers > 0)
    - msgpack serialization for structured data

    Storage structure:
        <path>/
            data.mdb
            lock.mdb
            manifest.json

    Usage:
        backend = LMDBBackend("cache/train_lmdb")
        backend.open("r")
        data = backend.get(image_id)
        backend.close()
    """

    def __init__(
        self,
        path: str,
        map_size: int = 1024 * 1024 * 1024 * 100,  # 100 GB default
        readonly: bool = True,
    ):
        super().__init__(path)
        self.map_size = map_size
        self.readonly = readonly
        self._env = None
        self._txn = None

    def open(self, mode: str = "r"):
        try:
            import lmdb
        except ImportError:
            raise ImportError(
                "lmdb is required for LMDBBackend. Install with: pip install lmdb"
            )

        os.makedirs(self.path, exist_ok=True)
        self._env = lmdb.open(
            self.path,
            map_size=self.map_size,
            readonly=(mode == "r"),
            max_dbs=1,
            lock=True,
            readahead=False,
            meminit=False,
        )
        self._open = True

    def put(self, key: int, data: Dict[str, Any]):
        raw = _encode(data)
        k = struct.pack(">I", key)
        with self._env.begin(write=True) as txn:
            txn.put(k, raw)

    def get(self, key: int) -> Dict[str, Any]:
        k = struct.pack(">I", key)
        with self._env.begin() as txn:
            raw = txn.get(k)
        if raw is None:
            raise KeyError(f"Key {key} not found in LMDB cache '{self.path}'")
        return _decode(raw)

    def keys(self) -> List[int]:
        keys = []
        with self._env.begin() as txn:
            cursor = txn.cursor()
            for k, _ in cursor:
                keys.append(struct.unpack(">I", k)[0])
        return sorted(keys)

    def close(self):
        if self._env is not None:
            self._env.close()
            self._env = None
            self._open = False

    def __len__(self) -> int:
        with self._env.begin() as txn:
            return txn.stat()["entries"]


# ============================================================
# PT (torch.save) Backend
# ============================================================

@CACHE_BACKENDS.register("pt")
class PTBackend(CacheBackend):
    """
    .pt file cache backend.

    Two modes:
    A. Single-file (default): all entries in one .pt file.
       Fast reads, limited by RAM. Best for debugging/small datasets.
    B. Multi-file: one .pt per image. Scales to larger datasets.

    Storage structure:
    Single-file: <path>.pt + <path>.manifest.json
    Multi-file:  <path>/{0000001.pt, 0000002.pt, ..., index.json, manifest.json}
    """

    def __init__(
        self,
        path: str,
        single_file: bool = True,
    ):
        super().__init__(path)
        self.single_file = single_file
        self._data: Dict[int, Dict] = {}
        self._index: Dict[int, str] = {}

    def open(self, mode: str = "r"):
        if self.single_file:
            if mode == "r" and os.path.exists(self.path):
                self._data = torch.load(self.path, map_location="cpu", weights_only=False)
            else:
                self._data = {}
        else:
            # Multi-file mode
            os.makedirs(self.path, exist_ok=True)
            index_path = os.path.join(self.path, "index.json")
            if mode == "r" and os.path.exists(index_path):
                with open(index_path, "r") as f:
                    raw_index = json.load(f)
                    self._index = {int(k): v for k, v in raw_index.items()}
            else:
                self._index = {}
        self._open = True

    def put(self, key: int, data: Dict[str, Any]):
        if self.single_file:
            # Move tensors to CPU before saving
            self._data[key] = _to_cpu(data)
        else:
            fname = f"{key:07d}.pt"
            fpath = os.path.join(self.path, fname)
            torch.save(_to_cpu(data), fpath)
            self._index[key] = fname

    def get(self, key: int) -> Dict[str, Any]:
        if self.single_file:
            if key not in self._data:
                raise KeyError(f"Key {key} not found in PT cache '{self.path}'")
            return self._data[key]
        else:
            if key not in self._index:
                raise KeyError(f"Key {key} not found in PT cache index '{self.path}'")
            fpath = os.path.join(self.path, self._index[key])
            return torch.load(fpath, map_location="cpu", weights_only=False)

    def keys(self) -> List[int]:
        if self.single_file:
            return sorted(self._data.keys())
        return sorted(self._index.keys())

    def flush(self):
        """Persist in-memory data to disk."""
        if self.single_file:
            torch.save(self._data, self.path)
        else:
            index_path = os.path.join(self.path, "index.json")
            with open(index_path, "w") as f:
                json.dump(self._index, f, indent=2)

    def close(self):
        self.flush()
        if self.single_file:
            self._data.clear()
        self._index.clear()
        self._open = False

    def __len__(self) -> int:
        if self.single_file:
            return len(self._data)
        return len(self._index)


def _to_cpu(data: Dict[str, Any]) -> Dict[str, Any]:
    """Move all tensors in a nested dict to CPU."""
    if isinstance(data, dict):
        return {k: _to_cpu(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_to_cpu(v) for v in data]
    if isinstance(data, torch.Tensor):
        return data.cpu()
    return data

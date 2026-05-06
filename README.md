# deeprs_light

A lightweight, standardized testing scaffold for remote sensing computer vision.
Maximum architectural flexibility (CNN, Transformer, Mamba) with engineering
rigor for the "dirty work" — a foundation for deeprs_v2.

## Install

```bash
pip install -e .
```

## Quick Overview

### 1. Data Pipeline

```python
from deeprs_light.data import build_dataloader

# Online mode: load from COCO JSON
loader = build_dataloader(
    "deeprs_coco",
    root="/data/images",
    ann_file="/data/train.json",
    batch_size=16,
)

# Cache mode: load from preprocessed cache
loader = build_dataloader(
    "deeprs_coco",
    cache_dir="cache/train_lmdb",
    required_keys=["boxes", "labels", "edge_map"],
)
```

### 2. Tracker & Logger

```python
from deeprs_light.tracker import HookManager, Logger, ProgressTracker, LRTracker

logger = Logger("runs/exp1")
manager = HookManager()
manager.register(LRTracker(optimizer)).register(ProgressTracker(logger, total_steps))

for epoch in range(epochs):
    for step, batch in enumerate(loader):
        # ... training logic ...
        manager.call("on_step_end", step, {"loss": loss.item(), "lr": get_lr(optimizer)})
```

### 3. Evaluator

```python
from deeprs_light.evaluator import DeepRSEvaluator

evaluator = DeepRSEvaluator(ann_file="val.json")
for images, targets in val_loader:
    outputs = model(images)
    evaluator.process(targets, outputs)

results = evaluator.evaluate()
# results = {"coco": {...}, "classification": {...}, "rs_quality": {"GTC": ..., "GOC": ..., ...}}
evaluator.save_results("results/exp1.json")
```

### 4. CLI Tools

```bash
# Convert custom format to COCO
python tools/convert_to_coco.py detection --data_dir /data --output train.json --categories '[{"id":1,"name":"ship"}]'

# Build preprocessed cache
python tools/preprocess_cache.py --root /data/images --ann_file train.json --output cache/train --num_workers 8

# Check cache status
python tools/preprocess_cache.py --output cache/train --check
```

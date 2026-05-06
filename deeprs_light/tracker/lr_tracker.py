"""
Learning rate tracking utilities.
Non-invasively reads learning rates from torch.optim.Optimizer.
"""

from typing import Dict, List, Optional

import torch

from deeprs_light.tracker.hooks import HookBase


def get_lr(optimizer: torch.optim.Optimizer) -> List[float]:
    """
    Read current learning rates from an optimizer's param_groups.

    Non-invasive: directly accesses optimizer.param_groups[i]["lr"].

    Args:
        optimizer: Any PyTorch optimizer.

    Returns:
        List of learning rates, one per param_group.

    Usage:
        lrs = get_lr(optimizer)
        print(lrs)  # [0.001, 0.0001]
    """
    return [pg["lr"] for pg in optimizer.param_groups]


def format_lr(lr_list: List[float]) -> str:
    """
    Format learning rate list as a human-readable string.

    Args:
        lr_list: List of learning rates.

    Returns:
        Formatted string like "1e-3, 1e-4".
    """
    if len(lr_list) == 1:
        return f"{lr_list[0]:.2e}"
    return ", ".join(f"{lr:.2e}" for lr in lr_list)


class LRTracker(HookBase):
    """
    Hook that injects the current learning rate into logs at each step.

    Usage:
        tracker = LRTracker(optimizer)
        manager.register(tracker)

        # Now logs["lr"] is populated automatically at each on_step_end.
    """

    def __init__(self, optimizer: torch.optim.Optimizer):
        self.optimizer = optimizer

    def on_step_end(self, step: int, logs: Optional[Dict] = None):
        if logs is not None:
            lrs = get_lr(self.optimizer)
            logs["lr"] = lrs[0] if len(lrs) == 1 else lrs

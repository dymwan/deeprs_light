"""
Lightweight Hook/Callback system for non-invasive training monitoring.
"""

from typing import Dict, List, Optional


class HookBase:
    """
    Base class for training hooks.

    All methods are no-ops by default; subclasses override the ones they need.
    Third-party training loops just call HookManager.call() at the right times.

    Lifecycle:
        on_train_begin -> [on_epoch_begin -> on_step_end * N -> on_epoch_end] * E -> on_train_end
    """

    def on_train_begin(self, logs: Optional[Dict] = None):
        """Called once before training starts."""

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict] = None):
        """Called at the start of each epoch."""

    def on_step_end(self, step: int, logs: Optional[Dict] = None):
        """Called after each training step. logs should contain 'loss', 'lr', etc."""

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        """Called at the end of each epoch."""

    def on_train_end(self, logs: Optional[Dict] = None):
        """Called once after training finishes."""


class HookManager:
    """
    Manages a collection of hooks, calling them in registration order.

    Usage:
        manager = HookManager()
        manager.register(LRTracker(optimizer))
        manager.register(ProgressTracker(logger, total_steps))

        manager.call("on_train_begin")
        for epoch in range(max_epochs):
            manager.call("on_epoch_begin", epoch)
            for step, batch in enumerate(dataloader):
                # ... training logic ...
                manager.call("on_step_end", step, {"loss": loss_val, "lr": lr_val})
            manager.call("on_epoch_end", epoch, {"epoch_loss": avg_loss})
        manager.call("on_train_end")
    """

    def __init__(self):
        self._hooks: List[HookBase] = []

    def register(self, hook: HookBase) -> "HookManager":
        """
        Register a hook. Returns self for chaining.

        Usage:
            manager.register(hook1).register(hook2)
        """
        if not isinstance(hook, HookBase):
            raise TypeError(f"Expected HookBase, got {type(hook)}")
        self._hooks.append(hook)
        return self

    def call(self, method: str, *args, **kwargs):
        """
        Call a method on all registered hooks in order.

        Args:
            method: Method name (e.g., "on_step_end").
            *args, **kwargs: Passed to each hook's method.
        """
        for hook in self._hooks:
            fn = getattr(hook, method, None)
            if fn is not None:
                fn(*args, **kwargs)

    def __len__(self) -> int:
        return len(self._hooks)

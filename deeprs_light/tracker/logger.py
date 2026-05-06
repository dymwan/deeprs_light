"""
Unified logger: rich terminal progress + TensorBoard + CSV.
"""

import os
import csv
from typing import Dict, Optional

from deeprs_light.tracker.hooks import HookBase
from deeprs_light.tracker.meters import SmoothedValue


class Logger:
    """
    Unified logging to terminal (rich), TensorBoard, and CSV.

    Usage:
        logger = Logger(log_dir="runs/exp1", use_tensorboard=True, use_csv=True)
        logger.log_metrics({"loss": 0.5, "lr": 1e-3}, step=100)
        logger.close()
    """

    def __init__(
        self,
        log_dir: str,
        use_tensorboard: bool = True,
        use_csv: bool = True,
        use_rich: bool = True,
        window_size: int = 20,
    ):
        """
        Args:
            log_dir: Output directory for logs.
            use_tensorboard: Enable TensorBoard writer.
            use_csv: Enable CSV output.
            use_rich: Enable rich terminal progress display.
            window_size: Smoothing window for terminal display.
        """
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        self.use_tensorboard = use_tensorboard
        self.use_csv = use_csv
        self.use_rich = use_rich
        self.window_size = window_size

        # TensorBoard
        self._writer = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self._writer = SummaryWriter(log_dir=log_dir)
            except ImportError:
                print("[WARN] tensorboard not installed. Disabling TensorBoard output.")
                self.use_tensorboard = False

        # CSV
        self._csv_file = None
        self._csv_writer = None
        self._csv_header_written = False
        if use_csv:
            csv_path = os.path.join(log_dir, "metrics.csv")
            self._csv_file = open(csv_path, "w", newline="")
            self._csv_writer = csv.writer(self._csv_file)

        # Rich
        self._console = None
        self._progress = None
        self._task_id = None
        self._smoothed = {}
        if use_rich:
            try:
                from rich.console import Console
                from rich.progress import Progress, TextColumn, BarColumn
                self._console = Console()
                self._progress = Progress(
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TextColumn("{task.fields[metrics]}"),
                    console=self._console,
                )
                self._progress.start()
            except ImportError:
                print("[WARN] rich not installed. Disabling rich output.")
                self.use_rich = False

    def log_metrics(self, metrics: Dict[str, float], step: int):
        """
        Log scalar metrics to all enabled outputs.

        Args:
            metrics: Dict of name -> value (e.g., {"loss": 0.5, "lr": 0.001}).
            step: Global step number.
        """
        # TensorBoard
        if self.use_tensorboard and self._writer is not None:
            for k, v in metrics.items():
                self._writer.add_scalar(k, v, global_step=step)

        # CSV
        if self.use_csv and self._csv_writer is not None:
            if not self._csv_header_written:
                self._csv_writer.writerow(["step"] + list(metrics.keys()))
                self._csv_header_written = True
            row = [step] + list(metrics.values())
            self._csv_writer.writerow(row)
            self._csv_file.flush()

        # Rich terminal
        if self.use_rich and self._progress is not None:
            if self._task_id is None:
                self._task_id = self._progress.add_task(
                    "[cyan]Training",
                    total=None,
                    metrics="",
                )
            # Build metrics display string
            parts = []
            for k, v in metrics.items():
                if k not in self._smoothed:
                    self._smoothed[k] = SmoothedValue(window_size=self.window_size)
                sv = self._smoothed[k]
                sv.update(v)
                parts.append(f"{k}: {sv.smooth:.4f}")
            self._progress.update(self._task_id, metrics="  ".join(parts))

    def log_text(self, tag: str, text: str, step: int = 0):
        """Log text to TensorBoard."""
        if self.use_tensorboard and self._writer is not None:
            self._writer.add_text(tag, text, global_step=step)

    def add_figure(self, tag: str, figure, step: int = 0):
        """Log a matplotlib figure to TensorBoard."""
        if self.use_tensorboard and self._writer is not None:
            self._writer.add_figure(tag, figure, global_step=step)

    def log_scalar(self, tag: str, value: float, step: int):
        """Log a single scalar (convenience wrapper)."""
        self.log_metrics({tag: value}, step)

    def close(self):
        """Close all outputs."""
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
        if self._progress is not None:
            self._progress.stop()

    def __del__(self):
        self.close()


class ProgressTracker(HookBase):
    """
    Hook that logs training progress (loss, lr) at each step.

    Integrates with Logger to update terminal, TensorBoard, and CSV.

    Usage:
        logger = Logger("runs/exp1")
        tracker = ProgressTracker(logger, total_steps)
        manager.register(tracker)
    """

    def __init__(self, logger: Logger, total_steps: int):
        self.logger = logger
        self.total_steps = total_steps
        self.loss_meter = SmoothedValue(window_size=20)

    def on_step_end(self, step: int, logs: Optional[Dict] = None):
        if logs is None:
            return
        loss = logs.get("loss", 0)
        self.loss_meter.update(loss)
        metrics = {"loss": self.loss_meter.smooth}
        if "lr" in logs:
            lr_val = logs["lr"]
            if isinstance(lr_val, list):
                metrics["lr"] = lr_val[0]
            else:
                metrics["lr"] = lr_val
        self.logger.log_metrics(metrics, step)

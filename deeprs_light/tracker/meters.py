"""
Metric tracking utilities: AverageMeter and SmoothedValue.
"""

import collections
from typing import Optional


class AverageMeter:
    """
    Compute the global weighted average of a stream of values.

    Usage:
        meter = AverageMeter()
        meter.update(loss_val, n=batch_size)
        print(meter.avg)  # Global average
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all statistics."""
        self.val = 0.0       # Most recent value
        self.sum = 0.0       # Weighted sum
        self.count = 0       # Total sample count

    def update(self, val: float, n: int = 1):
        """
        Record a new value.

        Args:
            val: The value to record.
            n: Sample count (e.g., batch size) for weighted averaging.
        """
        self.val = val
        self.sum += val * n
        self.count += n

    @property
    def avg(self) -> float:
        """Global weighted average = sum / count. Returns 0 if empty."""
        if self.count == 0:
            return 0.0
        return self.sum / self.count

    def __repr__(self):
        return f"AverageMeter(val={self.val:.4f}, avg={self.avg:.4f})"


class SmoothedValue:
    """
    Sliding-window average + global average.

    Usage:
        sv = SmoothedValue(window_size=20)
        sv.update(loss_val)
        print(sv.smooth)       # Average over last window_size values
        print(sv.global_avg)   # Average over all values
    """

    def __init__(self, window_size: int = 20, fmt: Optional[str] = None):
        """
        Args:
            window_size: Number of recent values for the sliding average.
            fmt: Format string for __repr__ (default: "{smooth:.4f}").
        """
        self.deque = collections.deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt or "{smooth:.4f}"

    def reset(self):
        """Reset all statistics."""
        self.deque.clear()
        self.total = 0.0
        self.count = 0

    def update(self, val: float):
        """Record a new value."""
        self.deque.append(val)
        self.total += val
        self.count += 1

    @property
    def smooth(self) -> float:
        """Sliding window average. Returns 0 if empty."""
        if len(self.deque) == 0:
            return 0.0
        return sum(self.deque) / len(self.deque)

    @property
    def global_avg(self) -> float:
        """Global average. Returns 0 if empty."""
        if self.count == 0:
            return 0.0
        return self.total / self.count

    @property
    def latest(self) -> float:
        """Most recent value. Returns 0 if empty."""
        if len(self.deque) == 0:
            return 0.0
        return self.deque[-1]

    def __repr__(self):
        return self.fmt.format(smooth=self.smooth)

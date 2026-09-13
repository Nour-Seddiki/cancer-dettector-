"""Shared training plumbing: device setup, checkpointing, LR schedule, metrics.

Several of these exist because of specific things that bit the EN->FR project on this
machine - see the comments on `save_checkpoint`, `cap_gpu_memory` and `keep_awake`.
"""

import ctypes
import math
import time
from pathlib import Path

import torch


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cap_gpu_memory(fraction=0.8, device=None):
    """Cap the CUDA caching allocator to a fraction of VRAM.

    On Windows/WDDM the driver silently spills past physical VRAM into shared system RAM
    instead of raising OOM, which turns a training run into a PCIe-bound crawl with no
    error to point at. Capping the allocator makes the allocator fail fast (so you lower
    the batch size) rather than quietly running an order of magnitude slower.
    """
    device = device or get_device()
    if device.type != "cuda":
        return
    try:
        torch.cuda.set_per_process_memory_fraction(fraction, device.index or 0)
        total = torch.cuda.get_device_properties(device).total_memory / 1e9
        print(f"  CUDA allocator capped at {fraction:.0%} of {total:.1f} GB")
    except (AttributeError, RuntimeError) as e:
        print(f"  could not cap CUDA memory: {e}")


_ES_CONTINUOUS = 0x80000000
_ES_SYSTEM_REQUIRED = 0x00000001
_ES_AWAYMODE_REQUIRED = 0x00000040


def keep_awake(enable=True):
    """Stop Modern Standby from suspending a long run on an idle laptop.

    Windows will drop into Modern Standby after ~30 idle minutes even with a training loop
    saturating the GPU; the run is suspended and often killed on wake. No-op off Windows.
    """
    try:
        flags = _ES_CONTINUOUS
        if enable:
            flags |= _ES_SYSTEM_REQUIRED | _ES_AWAYMODE_REQUIRED
        ctypes.windll.kernel32.SetThreadExecutionState(ctypes.c_uint(flags))
    except (AttributeError, OSError):
        pass  # not Windows, or the call is unavailable


def save_checkpoint(state, path, retries=5, delay=5.0):
    """Save with retries - OneDrive transiently locks large files while syncing."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        try:
            torch.save(state, path)
            return True
        except (RuntimeError, OSError, PermissionError) as e:
            if attempt == retries - 1:
                print(f"  WARNING: failed to save {path} after {retries} attempts: {e}")
                return False
            print(f"  save to {path.name} failed ({e}); retrying in {delay}s...")
            time.sleep(delay)
    return False


def cosine_lr(it, base_lr, warmup_iters, max_iters, min_lr_ratio=0.1):
    """Linear warmup then cosine decay - same schedule as the translation project."""
    min_lr = base_lr * min_lr_ratio
    if it < warmup_iters:
        return base_lr * (it + 1) / (warmup_iters + 1)
    if it >= max_iters:
        return min_lr
    ratio = (it - warmup_iters) / max(1, max_iters - warmup_iters)
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * ratio)) * (base_lr - min_lr)


class AverageMeter:
    def __init__(self):
        self.total = 0.0
        self.count = 0

    def update(self, value, n=1):
        self.total += float(value) * n
        self.count += n

    @property
    def avg(self):
        return self.total / self.count if self.count else 0.0


def format_seconds(s):
    m, s = divmod(int(s), 60)
    h, m = divmod(m, 60)
    return f"{h:d}h{m:02d}m" if h else f"{m:d}m{s:02d}s"


class EarlyStopping:
    """Stop when the monitored metric has not improved for `patience` evaluations."""

    def __init__(self, patience=5, mode="min", min_delta=0.0):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best = None
        self.bad_rounds = 0

    def is_better(self, value):
        if self.best is None:
            return True
        if self.mode == "min":
            return value < self.best - self.min_delta
        return value > self.best + self.min_delta

    def step(self, value):
        """Returns (improved, should_stop)."""
        if self.is_better(value):
            self.best = value
            self.bad_rounds = 0
            return True, False
        self.bad_rounds += 1
        return False, self.bad_rounds >= self.patience

from __future__ import annotations

import os
from pathlib import Path

import torch


def configure_torch_runtime(device: torch.device) -> None:
    """Enable safe runtime settings for ROCm/CUDA-backed training."""
    if device.type != "cuda":
        return

    if getattr(torch.version, "hip", None):
        miopen_cache_dir = Path.home() / ".cache" / "miopen"
        miopen_cache_dir.mkdir(parents=True, exist_ok=True)
        (miopen_cache_dir / "miopen-lockfiles").mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MIOPEN_USER_DB_PATH", str(miopen_cache_dir))

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = not bool(getattr(torch.version, "hip", None))

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

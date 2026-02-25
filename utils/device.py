"""Device and CUDA utility helpers."""

from __future__ import annotations

import torch


def resolve_device(prefer_cuda: bool = True) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def cuda_summary() -> dict[str, str | int | bool]:
    available = torch.cuda.is_available()
    return {
        "cuda_available": available,
        "cuda_device_count": torch.cuda.device_count() if available else 0,
        "cuda_device_name": torch.cuda.get_device_name(0) if available else "",
    }

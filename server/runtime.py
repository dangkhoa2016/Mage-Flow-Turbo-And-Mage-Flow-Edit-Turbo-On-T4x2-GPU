from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class DeviceInfo:
    index: int
    name: str

    @property
    def cuda_name(self) -> str:
        return f"cuda:{self.index}"


@dataclass(frozen=True)
class RuntimeConfig:
    t2i_device: str = "cuda:0"
    edit_device: str = "cuda:1"
    allow_cpu_fallback: bool = False


def detect_cuda_devices() -> list[DeviceInfo]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required on the Kaggle runtime to inspect CUDA devices") from exc

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; this demo requires Kaggle NVIDIA T4 x2")

    return [DeviceInfo(i, torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]


def validate_required_gpu_layout(
    devices: Sequence[DeviceInfo],
    config: RuntimeConfig = RuntimeConfig(),
) -> dict[str, str]:
    if config.allow_cpu_fallback:
        raise RuntimeError("CPU fallback must remain disabled for the intended T4 x2 demo")
    if len(devices) < 2:
        raise RuntimeError(f"Expected at least 2 CUDA devices, found {len(devices)}")
    if config.t2i_device != "cuda:0" or config.edit_device != "cuda:1":
        raise RuntimeError("Public routing contract requires T2I=cuda:0 and Edit=cuda:1")
    return {
        "t2i_device": config.t2i_device,
        "edit_device": config.edit_device,
        "cuda_0_name": devices[0].name,
        "cuda_1_name": devices[1].name,
    }

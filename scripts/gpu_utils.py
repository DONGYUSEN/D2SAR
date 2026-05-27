"""gpu_utils — Shared GPU detection and initialization utilities.

Centralizes all GPU/CUDA fallback logic so tops_insar.py, tops_geometry.py,
strip_insar.py, and tops_data_utils.py all use the same code path.
"""

from __future__ import annotations

__all__ = [
    "init_cuda_device",
    "check_cuda_available",
    "get_gpu_count",
    "GpuInfo",
]

import logging
from dataclasses import dataclass, field

LOG = logging.getLogger(__name__)


@dataclass
class GpuInfo:
    """Result of GPU detection and initialization."""
    available: bool = False
    device_id: int = -1
    backend: str = "cpu"          # "cpu" | "isce3" | "cupy" | "pytorch"
    device_name: str = "cpu"
    error: str | None = None


def get_gpu_count() -> int:
    """Return number of available CUDA devices across all backends."""
    # Try ISCE3 first (most relevant for the pipeline)
    try:
        import isce3.cuda.core as cuda_core
        if hasattr(cuda_core.Device, "numDevices"):
            return int(cuda_core.Device.numDevices())
    except Exception:
        pass

    # Try CuPy
    try:
        import cupy as cp
        return int(cp.cuda.runtime.getDeviceCount())
    except Exception:
        pass

    # Try PyTorch
    try:
        import torch
        if torch.cuda.is_available():
            return int(torch.cuda.device_count())
    except Exception:
        pass

    return 0


def _init_isce3_cuda(device_id: int) -> GpuInfo:
    """Initialize CUDA device via ISCE3 bindings."""
    try:
        import isce3.cuda.core as cuda_core

        ngpus = 0
        if hasattr(cuda_core.Device, "numDevices"):
            ngpus = int(cuda_core.Device.numDevices())
        elif hasattr(cuda_core.Device, "get_count"):
            ngpus = int(cuda_core.Device.get_count())
        else:
            for i in range(16):
                try:
                    _ = cuda_core.Device(i)
                    ngpus = i + 1
                except Exception:
                    break

        if device_id >= ngpus:
            return GpuInfo(
                available=False, device_id=device_id, backend="isce3",
                error=f"device {device_id} >= available {ngpus}",
            )

        dev = cuda_core.Device(device_id)
        cuda_core.set_device(dev)
        name = str(dev.name) if hasattr(dev, "name") else f"cuda:{device_id}"
        return GpuInfo(available=True, device_id=device_id, backend="isce3", device_name=name)

    except Exception as exc:
        return GpuInfo(
            available=False, device_id=device_id, backend="isce3",
            error=str(exc),
        )


def _init_cupy_cuda(device_id: int) -> GpuInfo:
    """Initialize CUDA device via CuPy."""
    try:
        import cupy as cp
        count = int(cp.cuda.runtime.getDeviceCount())
        if device_id >= count:
            return GpuInfo(
                available=False, device_id=device_id, backend="cupy",
                error=f"device {device_id} >= available {count}",
            )
        cp.cuda.Device(device_id).use()
        props = cp.cuda.runtime.getDeviceProperties(device_id)
        name = props["name"].decode() if isinstance(props["name"], bytes) else str(props["name"])
        return GpuInfo(available=True, device_id=device_id, backend="cupy", device_name=name)

    except Exception as exc:
        return GpuInfo(
            available=False, device_id=device_id, backend="cupy",
            error=str(exc),
        )


def _init_pytorch_cuda(device_id: int) -> GpuInfo:
    """Initialize CUDA device via PyTorch."""
    try:
        import torch
        if not torch.cuda.is_available():
            return GpuInfo(
                available=False, device_id=device_id, backend="pytorch",
                error="CUDA not available",
            )
        count = int(torch.cuda.device_count())
        if device_id >= count:
            return GpuInfo(
                available=False, device_id=device_id, backend="pytorch",
                error=f"device {device_id} >= available {count}",
            )
        name = str(torch.cuda.get_device_name(device_id))
        torch.cuda.set_device(device_id)
        return GpuInfo(available=True, device_id=device_id, backend="pytorch", device_name=name)

    except Exception as exc:
        return GpuInfo(
            available=False, device_id=device_id, backend="pytorch",
            error=str(exc),
        )


# Priority order for backends
_BACKEND_INIT_ORDER = [
    ("isce3", _init_isce3_cuda),
    ("cupy", _init_cupy_cuda),
    ("pytorch", _init_pytorch_cuda),
]


def init_cuda_device(
    device_id: int = 0,
    *,
    gpu_mode: str = "auto",
    prefer_backend: str | None = None,
    log: logging.Logger = LOG,
) -> GpuInfo:
    """Initialize a CUDA device with automatic backend fallback.

    Parameters
    ----------
    device_id : int
        Preferred GPU device index.
    gpu_mode : str
        ``"cpu"`` → skip entirely.  ``"gpu"`` → fail if no GPU available.
        ``"auto"`` → try GPU, fall back to CPU gracefully.
    prefer_backend : str or None
        If set (``"isce3"`` | ``"cupy"`` | ``"pytorch"``), try that backend
        first.  Default ``None`` tries backends in priority order.
    log : logging.Logger
        Logger for status messages.

    Returns
    -------
    GpuInfo
        Result with ``.available``, ``.backend``, ``.device_name``.
    """
    if gpu_mode == "cpu":
        log.info("GPU mode=cpu, skipping GPU init")
        return GpuInfo(available=False, device_id=-1, backend="cpu", device_name="cpu")

    info = GpuInfo(available=False, device_id=device_id, device_name="cpu")

    backends = list(_BACKEND_INIT_ORDER)

    if prefer_backend is not None:
        idx = next((i for i, (name, _) in enumerate(backends) if name == prefer_backend), -1)
        if idx >= 0:
            entry = backends.pop(idx)
            backends.insert(0, entry)

    for name, init_fn in backends:
        info = init_fn(device_id)
        if info.available:
            log.info(
                "CUDA device %d ready: backend=%s name=%s",
                device_id, info.backend, info.device_name,
            )
            return info
        log.debug("CUDA backend %s failed for device %d: %s", name, device_id, info.error)

    info = GpuInfo(
        available=False,
        device_id=device_id,
        device_name="cpu",
        error="no CUDA backend succeeded",
    )

    if gpu_mode == "gpu":
        log.error("GPU mode=gpu but no CUDA device available: %s", info.error)
    else:
        log.info("GPU auto mode: no CUDA device available, using CPU")

    return info


def check_cuda_available(
    device_id: int = 0,
    *,
    log: logging.Logger = LOG,
) -> bool:
    """Quick check whether a specific CUDA device is usable.

    Returns True if any backend can reach the device.
    Does NOT set the device; see ``init_cuda_device`` for persistent init.
    """
    for name, init_fn in _BACKEND_INIT_ORDER:
        info = init_fn(device_id)
        if info.available:
            return True
    return False

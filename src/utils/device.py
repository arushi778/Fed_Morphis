"""Hardware device resolution and GPU inspection utilities for FedMorphis."""

from __future__ import annotations

import importlib
from typing import Any

SUPPORTED_DEVICES: tuple[str, ...] = ("auto", "cuda", "cpu")


class FallbackDevice(str):
    """Lightweight device representation used when PyTorch is not installed.

    Subclasses ``str`` so equality checks like ``device == "cpu"`` and
    ``str(device) == "cpu"`` hold while also exposing a ``.type`` attribute
    compatible with ``torch.device``.
    """

    @property
    def type(self) -> str:
        """Return the device type string (e.g., ``'cpu'``)."""
        return str(self)


def _get_torch() -> Any | None:
    """Import and return the ``torch`` package if installed, else ``None``."""
    try:
        return importlib.import_module("torch")
    except (ImportError, ModuleNotFoundError):
        return None


def is_pytorch_available() -> bool:
    """Return ``True`` if PyTorch is importable in the current environment."""
    return _get_torch() is not None


def is_cuda_available() -> bool:
    """Return ``True`` if PyTorch is installed and CUDA execution is available."""
    torch = _get_torch()
    if torch is None:
        return False
    cuda_mod = getattr(torch, "cuda", None)
    if cuda_mod is None or not hasattr(cuda_mod, "is_available"):
        return False
    return bool(cuda_mod.is_available())


def resolve_device(device: str = "auto") -> Any:
    """Resolve a target compute device from a configuration string.

    Supported values are ``"auto"``, ``"cuda"``, and ``"cpu"`` (case-insensitive).

    Behavior:
        - ``"auto"``: Returns CUDA if PyTorch is installed and CUDA is available,
          otherwise returns CPU.
        - ``"cuda"``: Returns CUDA only if PyTorch is installed and CUDA is
          available; otherwise raises a ``RuntimeError`` (never falls back
          silently to CPU).
        - ``"cpu"``: Returns CPU.

    When PyTorch is installed, returns a ``torch.device`` instance. When PyTorch
    is unavailable, ``"auto"`` and ``"cpu"`` return a ``FallbackDevice("cpu")``
    (a ``str`` subclass with ``.type == "cpu"``), while ``"cuda"`` raises a
    ``RuntimeError``.

    Args:
        device: Requested device identifier (``"auto"``, ``"cuda"``, or ``"cpu"``).

    Returns:
        A ``torch.device`` object when PyTorch is installed, or
        ``FallbackDevice("cpu")`` when PyTorch is not installed and CPU/auto is used.

    Raises:
        ValueError: If ``device`` is not a string or not in ``("auto", "cuda", "cpu")``.
        RuntimeError: If ``device="cuda"`` is explicitly requested when PyTorch
            is not installed or CUDA is unavailable.
    """
    if not isinstance(device, str):
        raise ValueError(
            f"Invalid device specification {device!r}: expected a string in {SUPPORTED_DEVICES}."
        )

    normalized = device.strip().lower()
    if normalized not in SUPPORTED_DEVICES:
        raise ValueError(
            f"Unsupported device {device!r}. Supported values are: {', '.join(SUPPORTED_DEVICES)}."
        )

    torch = _get_torch()

    if normalized == "cuda":
        if torch is None:
            raise RuntimeError(
                "CUDA device was explicitly requested (device='cuda'), "
                "but PyTorch is not installed in the current environment."
            )
        cuda_mod = getattr(torch, "cuda", None)
        if cuda_mod is None or not cuda_mod.is_available():
            raise RuntimeError(
                "CUDA device was explicitly requested (device='cuda'), "
                "but CUDA is not available in the current PyTorch installation."
            )
        return torch.device("cuda")

    if normalized == "auto":
        if torch is not None:
            cuda_mod = getattr(torch, "cuda", None)
            if cuda_mod is not None and cuda_mod.is_available():
                return torch.device("cuda")
            return torch.device("cpu")
        return FallbackDevice("cpu")

    # normalized == "cpu"
    if torch is not None:
        return torch.device("cpu")
    return FallbackDevice("cpu")


def get_device_info() -> dict[str, Any]:
    """Return a structured dictionary describing PyTorch and CUDA hardware status.

    Note:
        ``cuda_runtime_version`` reports ``torch.version.cuda`` (the CUDA runtime
        bundled with PyTorch), not the host NVIDIA display driver's maximum CUDA
        version from ``nvidia-smi`` and without requiring ``nvcc``.

    Returns:
        Dictionary containing:
            - ``pytorch_available`` (bool)
            - ``pytorch_version`` (str | None)
            - ``cuda_available`` (bool)
            - ``cuda_runtime_version`` (str | None)
            - ``device_count`` (int)
            - ``gpu_names`` (list[str])
            - ``gpu_memory`` (list[dict[str, Any]])
            - ``default_device`` (str)
    """
    torch = _get_torch()
    if torch is None:
        return {
            "pytorch_available": False,
            "pytorch_version": None,
            "cuda_available": False,
            "cuda_runtime_version": None,
            "device_count": 0,
            "gpu_names": [],
            "gpu_memory": [],
            "default_device": "cpu",
        }

    pytorch_version = getattr(torch, "__version__", None)
    version_mod = getattr(torch, "version", None)
    cuda_runtime_version = getattr(version_mod, "cuda", None) if version_mod is not None else None

    cuda_mod = getattr(torch, "cuda", None)
    cuda_avail = bool(
        cuda_mod is not None
        and hasattr(cuda_mod, "is_available")
        and cuda_mod.is_available()
    )

    device_count = 0
    gpu_names: list[str] = []
    gpu_memory: list[dict[str, Any]] = []

    if cuda_avail and cuda_mod is not None:
        if hasattr(cuda_mod, "device_count"):
            device_count = int(cuda_mod.device_count())

        for idx in range(device_count):
            name = (
                str(cuda_mod.get_device_name(idx))
                if hasattr(cuda_mod, "get_device_name")
                else f"CUDA Device {idx}"
            )
            gpu_names.append(name)

            mem_entry: dict[str, Any] = {
                "device_index": idx,
                "device_name": name,
                "total_memory_bytes": None,
                "total_memory_mb": None,
                "allocated_memory_bytes": None,
                "reserved_memory_bytes": None,
            }

            if hasattr(cuda_mod, "get_device_properties"):
                try:
                    props = cuda_mod.get_device_properties(idx)
                    total_bytes = getattr(props, "total_memory", None)
                    if total_bytes is not None:
                        mem_entry["total_memory_bytes"] = int(total_bytes)
                        mem_entry["total_memory_mb"] = round(
                            int(total_bytes) / (1024 * 1024), 2
                        )
                except Exception:
                    pass

            if hasattr(cuda_mod, "memory_allocated"):
                try:
                    mem_entry["allocated_memory_bytes"] = int(
                        cuda_mod.memory_allocated(idx)
                    )
                except Exception:
                    pass

            if hasattr(cuda_mod, "memory_reserved"):
                try:
                    mem_entry["reserved_memory_bytes"] = int(
                        cuda_mod.memory_reserved(idx)
                    )
                except Exception:
                    pass

            gpu_memory.append(mem_entry)

    default_device = str(resolve_device("auto"))

    return {
        "pytorch_available": True,
        "pytorch_version": str(pytorch_version) if pytorch_version is not None else None,
        "cuda_available": cuda_avail,
        "cuda_runtime_version": (
            str(cuda_runtime_version) if cuda_runtime_version is not None else None
        ),
        "device_count": device_count,
        "gpu_names": gpu_names,
        "gpu_memory": gpu_memory,
        "default_device": default_device,
    }


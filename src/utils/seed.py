"""Reproducibility and random seed management utilities for FedMorphis.

Limitations of Deterministic Execution:
    1. Enabling deterministic algorithms (`deterministic=True`) may reduce training and
       inference performance, particularly for convolutional networks where cuDNN
       autotuning (`torch.backends.cudnn.benchmark`) is disabled.
    2. Complete bit-level reproducibility is not guaranteed across different CPU/GPU
       architectures, operating systems, NVIDIA driver versions, CUDA runtimes, or
       PyTorch/NumPy releases.
    3. Certain GPU operations (e.g., atomic scatter/gather operations or specific
       interpolation backward passes) may not have deterministic CUDA implementations
       or may emit warnings when `warn_only=True` is active.
    4. Setting `PYTHONHASHSEED` inside a running Python process records the configured
       seed in `os.environ` (and applies to child processes), but Python's internal
       hash randomization for the active interpreter is fixed at interpreter startup.
"""

from __future__ import annotations

import importlib
import os
import random
from typing import Any

import numpy as np

MAX_UINT32_SEED: int = 2**32 - 1


def _get_torch() -> Any | None:
    """Import and return the `torch` module if installed, otherwise return `None`."""
    try:
        return importlib.import_module("torch")
    except (ImportError, ModuleNotFoundError):
        return None


def _validate_seed(seed: int) -> int:
    """Validate that `seed` is an integer within the valid 32-bit unsigned range."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError(
            f"Random seed must be an integer, got {type(seed).__name__} ({seed!r})."
        )
    if not (0 <= seed <= MAX_UINT32_SEED):
        raise ValueError(
            f"Random seed must be between 0 and {MAX_UINT32_SEED} inclusive, got {seed}."
        )
    return seed


def set_seed(seed: int = 42, deterministic: bool = True) -> int:
    """Initialize global random seeds across Python, NumPy, and PyTorch (if installed).

    Args:
        seed: Non-negative integer seed in ``[0, 2**32 - 1]``. Defaults to ``42``.
        deterministic: Whether to configure deterministic execution flags in PyTorch
            and cuDNN when PyTorch is available. Defaults to ``True``.

    Returns:
        The validated integer seed that was applied.

    Raises:
        TypeError: If ``seed`` is not an ``int`` (or is a ``bool``), or if
            ``deterministic`` is not a ``bool``.
        ValueError: If ``seed`` is outside ``[0, 2**32 - 1]``.
    """
    validated_seed = _validate_seed(seed)
    if not isinstance(deterministic, bool):
        raise TypeError(
            f"'deterministic' must be a boolean, got {type(deterministic).__name__}."
        )

    os.environ["PYTHONHASHSEED"] = str(validated_seed)
    random.seed(validated_seed)
    np.random.seed(validated_seed)

    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    torch = _get_torch()
    if torch is not None:
        if hasattr(torch, "manual_seed"):
            torch.manual_seed(validated_seed)

        cuda_module = getattr(torch, "cuda", None)
        if cuda_module is not None and hasattr(cuda_module, "is_available"):
            if cuda_module.is_available():
                if hasattr(cuda_module, "manual_seed"):
                    cuda_module.manual_seed(validated_seed)
                if hasattr(cuda_module, "manual_seed_all"):
                    cuda_module.manual_seed_all(validated_seed)

        if hasattr(torch, "use_deterministic_algorithms"):
            try:
                torch.use_deterministic_algorithms(deterministic, warn_only=True)
            except TypeError:
                torch.use_deterministic_algorithms(deterministic)

        backends = getattr(torch, "backends", None)
        cudnn = getattr(backends, "cudnn", None) if backends is not None else None
        if cudnn is not None:
            cudnn.deterministic = deterministic
            cudnn.benchmark = not deterministic

    return validated_seed


def seed_worker(worker_id: int) -> None:
    """Initialize random seeds for a DataLoader worker process.

    When PyTorch is available, derives a 32-bit worker seed from
    ``torch.initial_seed() % 2**32``. When PyTorch is not installed, derives a
    deterministic fallback seed using ``worker_id`` and Python's RNG state.

    Args:
        worker_id: Non-negative integer identifier of the DataLoader worker.

    Raises:
        TypeError: If ``worker_id`` is not an integer.
        ValueError: If ``worker_id`` is negative.
    """
    if isinstance(worker_id, bool) or not isinstance(worker_id, int):
        raise TypeError(
            f"'worker_id' must be an integer, got {type(worker_id).__name__}."
        )
    if worker_id < 0:
        raise ValueError(f"'worker_id' must be non-negative, got {worker_id}.")

    torch = _get_torch()
    if torch is not None and hasattr(torch, "initial_seed"):
        worker_seed = (int(torch.initial_seed()) + worker_id) % (2**32)
    else:
        base_seed = int(os.environ.get("PYTHONHASHSEED", "42"))
        worker_seed = (base_seed + worker_id) % (2**32)

    np.random.seed(worker_seed)
    random.seed(worker_seed)


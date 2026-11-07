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
from collections.abc import Mapping
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


def capture_rng_state(dataloader_generator: Any | None = None) -> dict[str, Any]:
    """Capture a snapshot of Python, NumPy, PyTorch, CUDA, and DataLoader generator RNG states.

    Args:
        dataloader_generator: Optional :class:`torch.Generator` attached to a DataLoader.

    Returns:
        Dictionary of serializable RNG states suitable for checkpoint persistence.
    """
    state: dict[str, Any] = {
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_rng_state": None,
        "cuda_rng_state_all": None,
        "dataloader_generator_state": None,
    }
    torch = _get_torch()
    if torch is not None:
        if hasattr(torch, "get_rng_state"):
            state["torch_rng_state"] = torch.get_rng_state().clone()
        cuda_mod = getattr(torch, "cuda", None)
        if (
            cuda_mod is not None
            and hasattr(cuda_mod, "is_available")
            and cuda_mod.is_available()
            and hasattr(cuda_mod, "get_rng_state_all")
        ):
            try:
                state["cuda_rng_state_all"] = [
                    s.clone() for s in cuda_mod.get_rng_state_all()
                ]
            except Exception:
                state["cuda_rng_state_all"] = None
        if dataloader_generator is not None and hasattr(
            dataloader_generator, "get_state"
        ):
            state["dataloader_generator_state"] = (
                dataloader_generator.get_state().clone()
            )
    return state


def restore_rng_state(
    rng_state: Mapping[str, Any] | None,
    dataloader_generator: Any | None = None,
) -> bool:
    """Restore Python, NumPy, PyTorch, CUDA, and DataLoader generator RNG states from a checkpoint.

    Safely returns ``False`` without raising if ``rng_state`` is ``None`` or absent
    (preserving backward compatibility with legacy checkpoints).
    """
    if not isinstance(rng_state, Mapping) or len(rng_state) == 0:
        return False

    restored_any = False
    py_state = rng_state.get("python_random_state")
    if py_state is not None:
        random.setstate(py_state)
        restored_any = True

    np_state = rng_state.get("numpy_random_state")
    if np_state is not None:
        np.random.set_state(np_state)
        restored_any = True

    torch = _get_torch()
    if torch is not None:
        torch_state = rng_state.get("torch_rng_state")
        if torch_state is not None and hasattr(torch, "set_rng_state"):
            if hasattr(torch_state, "to"):
                torch_state = torch_state.to(device="cpu", dtype=torch.uint8)
            torch.set_rng_state(torch_state)
            restored_any = True

        cuda_states = rng_state.get("cuda_rng_state_all")
        cuda_mod = getattr(torch, "cuda", None)
        if (
            cuda_states is not None
            and cuda_mod is not None
            and hasattr(cuda_mod, "is_available")
            and cuda_mod.is_available()
            and hasattr(cuda_mod, "set_rng_state_all")
        ):
            try:
                cpu_cuda_states = [
                    s.to(device="cpu", dtype=torch.uint8) if hasattr(s, "to") else s
                    for s in cuda_states
                ]
                cuda_mod.set_rng_state_all(cpu_cuda_states)
                restored_any = True
            except Exception:
                pass

        dl_gen_state = rng_state.get("dataloader_generator_state")
        if (
            dataloader_generator is not None
            and dl_gen_state is not None
            and hasattr(dataloader_generator, "set_state")
        ):
            if hasattr(dl_gen_state, "to"):
                dl_gen_state = dl_gen_state.to(device="cpu", dtype=torch.uint8)
            dataloader_generator.set_state(dl_gen_state)
            restored_any = True

    return restored_any



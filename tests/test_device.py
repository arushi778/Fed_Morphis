"""Unit tests for device resolution and hardware inspection (src/utils/device.py)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.utils.config import load_yaml_config
from src.utils.device import (
    FallbackDevice,
    get_device_info,
    is_cuda_available,
    is_pytorch_available,
    resolve_device,
)


def test_resolve_device_accepts_auto() -> None:
    """Verify resolve_device('auto') succeeds and returns either cpu or cuda."""
    device = resolve_device("auto")
    assert str(device) in {"cpu", "cuda"}
    assert getattr(device, "type", str(device)) in {"cpu", "cuda"}


def test_resolve_device_accepts_cpu() -> None:
    """Verify resolve_device('cpu') succeeds and resolves to CPU."""
    device = resolve_device("cpu")
    assert str(device) == "cpu"
    assert getattr(device, "type", str(device)) == "cpu"

    # Case-insensitive and whitespace-tolerant
    device_upper = resolve_device("  CPU ")
    assert str(device_upper) == "cpu"


@pytest.mark.parametrize("invalid_device", ["tpu", "mps", "gpu", "cuda:99", "", "invalid", 123, None])
def test_resolve_device_rejects_invalid_values(invalid_device: object) -> None:
    """Verify unsupported or non-string device values raise an informative ValueError."""
    with pytest.raises(ValueError, match="device"):
        resolve_device(invalid_device)  # type: ignore[arg-type]


def test_explicit_cuda_request_fails_when_pytorch_unavailable() -> None:
    """Verify resolve_device('cuda') raises RuntimeError when PyTorch is not installed."""
    with patch("src.utils.device._get_torch", return_value=None):
        with pytest.raises(RuntimeError, match="CUDA device was explicitly requested"):
            resolve_device("cuda")


def test_explicit_cuda_request_fails_when_cuda_unavailable_in_pytorch() -> None:
    """Verify resolve_device('cuda') raises RuntimeError when PyTorch has no CUDA."""
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False

    with patch("src.utils.device._get_torch", return_value=mock_torch):
        with pytest.raises(RuntimeError, match="CUDA is not available"):
            resolve_device("cuda")


def test_resolve_device_with_mock_pytorch_cuda_available() -> None:
    """Verify resolve_device returns torch.device('cuda') when CUDA is available."""
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True
    mock_torch.device.side_effect = lambda name: SimpleNamespace(type=name, __str__=lambda self: name)

    with patch("src.utils.device._get_torch", return_value=mock_torch):
        auto_dev = resolve_device("auto")
        cuda_dev = resolve_device("cuda")
        cpu_dev = resolve_device("cpu")

        assert auto_dev.type == "cuda"
        assert cuda_dev.type == "cuda"
        assert cpu_dev.type == "cpu"


def test_resolve_device_without_pytorch_returns_fallback_cpu_for_auto_and_cpu() -> None:
    """Verify auto and cpu resolve to FallbackDevice('cpu') when PyTorch is absent."""
    with patch("src.utils.device._get_torch", return_value=None):
        auto_dev = resolve_device("auto")
        cpu_dev = resolve_device("cpu")

        assert isinstance(auto_dev, FallbackDevice)
        assert auto_dev == "cpu"
        assert auto_dev.type == "cpu"
        assert cpu_dev == "cpu"
        assert cpu_dev.type == "cpu"


def test_get_device_info_structure_current_environment() -> None:
    """Verify get_device_info returns all expected keys and valid types."""
    info = get_device_info()
    expected_keys = {
        "pytorch_available",
        "pytorch_version",
        "cuda_available",
        "cuda_runtime_version",
        "device_count",
        "gpu_names",
        "gpu_memory",
        "default_device",
    }
    assert set(info.keys()) == expected_keys
    assert isinstance(info["pytorch_available"], bool)
    assert isinstance(info["cuda_available"], bool)
    assert isinstance(info["device_count"], int)
    assert isinstance(info["gpu_names"], list)
    assert isinstance(info["gpu_memory"], list)
    assert info["default_device"] in {"cpu", "cuda"}


def test_get_device_info_when_pytorch_unavailable() -> None:
    """Verify get_device_info reports clean defaults when PyTorch is not installed."""
    with patch("src.utils.device._get_torch", return_value=None):
        info = get_device_info()
        assert info == {
            "pytorch_available": False,
            "pytorch_version": None,
            "cuda_available": False,
            "cuda_runtime_version": None,
            "device_count": 0,
            "gpu_names": [],
            "gpu_memory": [],
            "default_device": "cpu",
        }
        assert is_pytorch_available() is False
        assert is_cuda_available() is False


def test_get_device_info_with_mock_cuda_gpu() -> None:
    """Verify get_device_info populates GPU names, memory, and PyTorch CUDA runtime."""
    mock_torch = MagicMock()
    mock_torch.__version__ = "2.6.0+cu128"
    mock_torch.version = SimpleNamespace(cuda="12.8")
    mock_torch.cuda.is_available.return_value = True
    mock_torch.cuda.device_count.return_value = 1
    mock_torch.cuda.get_device_name.return_value = "Simulated Biomedical GPU"
    mock_torch.cuda.get_device_properties.return_value = SimpleNamespace(
        total_memory=8 * 1024 * 1024 * 1024
    )
    mock_torch.cuda.memory_allocated.return_value = 0
    mock_torch.cuda.memory_reserved.return_value = 0
    mock_torch.device.side_effect = lambda name: name

    with patch("src.utils.device._get_torch", return_value=mock_torch):
        info = get_device_info()
        assert info["pytorch_available"] is True
        assert info["pytorch_version"] == "2.6.0+cu128"
        assert info["cuda_available"] is True
        assert info["cuda_runtime_version"] == "12.8"
        assert info["device_count"] == 1
        assert info["gpu_names"] == ["Simulated Biomedical GPU"]
        assert len(info["gpu_memory"]) == 1
        assert info["gpu_memory"][0]["total_memory_mb"] == 8192.0
        assert info["default_device"] == "cuda"


def test_project_yaml_device_config_integration() -> None:
    """Verify configs/project.yaml device setting resolves cleanly."""
    project_cfg = load_yaml_config("project.yaml")
    assert project_cfg["device"] == "auto"
    resolved = resolve_device(project_cfg["device"])
    assert str(resolved) in {"cpu", "cuda"}


@pytest.mark.skipif(not is_pytorch_available(), reason="PyTorch is not installed")
def test_resolve_device_returns_torch_device_when_installed() -> None:
    """Verify resolve_device returns a real torch.device instance when PyTorch is installed."""
    import torch

    dev = resolve_device("cpu")
    assert isinstance(dev, torch.device)
    assert dev.type == "cpu"


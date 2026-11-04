"""Environment verification tests for the PyTorch, CUDA, MONAI, Albumentations, and TorchMetrics stack."""

from __future__ import annotations

import importlib
import os

import numpy as np
import pytest

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

from src.utils.device import (
    get_device_info,
    is_cuda_available,
    is_pytorch_available,
    resolve_device,
)


def _has_package(pkg_name: str) -> bool:
    """Return True if the specified package can be imported."""
    try:
        importlib.import_module(pkg_name)
        return True
    except (ImportError, ModuleNotFoundError):
        return False


@pytest.mark.skipif(not is_pytorch_available(), reason="PyTorch is not installed")
def test_pytorch_cpu_tensor_and_conv2d() -> None:
    """Verify CPU tensor matmul and 2D convolution execute cleanly when PyTorch is installed."""
    import torch

    cpu_device = resolve_device("cpu")
    assert cpu_device.type == "cpu"

    x = torch.randn((2, 3, 16, 16), device=cpu_device)
    conv = torch.nn.Conv2d(3, 4, kernel_size=3, padding=1).to(cpu_device)
    out = conv(x)
    assert out.shape == (2, 4, 16, 16)
    assert bool(torch.isfinite(out).all().item())


@pytest.mark.skipif(not is_cuda_available(), reason="CUDA is not available")
def test_live_cuda_tensor_matmul_and_conv2d() -> None:
    """Verify live unmocked CUDA tensor matrix multiplication and Conv2d on the GPU."""
    import torch

    auto_device = resolve_device("auto")
    cuda_device = resolve_device("cuda")
    assert auto_device.type == "cuda"
    assert cuda_device.type == "cuda"

    info = get_device_info()
    assert info["pytorch_available"] is True
    assert info["cuda_available"] is True
    assert info["device_count"] >= 1
    assert len(info["gpu_names"]) >= 1
    assert info["default_device"] == "cuda"

    # Real CUDA matrix multiplication
    a = torch.randn((128, 128), device=cuda_device, dtype=torch.float32)
    b = torch.randn((128, 128), device=cuda_device, dtype=torch.float32)
    c = torch.matmul(a, b)
    torch.cuda.synchronize(cuda_device)

    assert c.is_cuda
    assert c.shape == (128, 128)
    assert bool(torch.isfinite(c).all().item())

    # Real CUDA 2D convolution
    conv = torch.nn.Conv2d(3, 8, kernel_size=3, padding=1).to(cuda_device)
    img = torch.randn((2, 3, 32, 32), device=cuda_device)
    conv_out = conv(img)
    torch.cuda.synchronize(cuda_device)

    assert conv_out.is_cuda
    assert conv_out.shape == (2, 8, 32, 32)
    assert bool(torch.isfinite(conv_out).all().item())


@pytest.mark.skipif(
    not (is_cuda_available() and _has_package("monai")),
    reason="CUDA or MONAI is not available",
)
def test_live_monai_dice_loss_on_cuda() -> None:
    """Verify MONAI DiceLoss forward and backward passes execute on CUDA."""
    import torch
    from monai.losses import DiceLoss

    cuda_device = resolve_device("cuda")
    loss_fn = DiceLoss(to_onehot_y=True, softmax=True)
    logits = torch.randn((2, 5, 32, 32), device=cuda_device, requires_grad=True)
    targets = torch.randint(0, 5, (2, 1, 32, 32), device=cuda_device)

    loss = loss_fn(logits, targets)
    loss.backward()
    torch.cuda.synchronize(cuda_device)

    assert loss.is_cuda
    assert bool(torch.isfinite(loss).all().item())
    assert logits.grad is not None
    assert logits.grad.is_cuda
    assert bool(torch.isfinite(logits.grad).all().item())


@pytest.mark.skipif(
    not (is_pytorch_available() and _has_package("torchvision")),
    reason="PyTorch or torchvision is not available",
)
def test_torchvision_basic_operation() -> None:
    """Verify torchvision imports and executes a basic bounding-box IoU operation."""
    import torch
    from torchvision.ops import box_iou

    device = resolve_device("auto")
    boxes1 = torch.tensor([[0.0, 0.0, 10.0, 10.0]], device=device)
    boxes2 = torch.tensor([[0.0, 0.0, 5.0, 5.0]], device=device)
    iou = box_iou(boxes1, boxes2)
    assert iou.shape == (1, 1)
    assert pytest.approx(float(iou.item()), rel=1e-4) == 0.25


@pytest.mark.skipif(
    not _has_package("albumentations"),
    reason="Albumentations is not installed",
)
def test_albumentations_image_and_mask_transform() -> None:
    """Verify Albumentations transforms a synthetic retinal image and lesion mask."""
    import albumentations as A

    pipeline = A.Compose(
        [
            A.HorizontalFlip(p=1.0),
            A.VerticalFlip(p=1.0),
        ]
    )
    img = np.arange(32 * 32 * 3, dtype=np.uint8).reshape(32, 32, 3)
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[0, 0] = 4

    result = pipeline(image=img, mask=mask)
    assert result["image"].shape == (32, 32, 3)
    assert result["mask"].shape == (32, 32)
    assert result["mask"][-1, -1] == 4


@pytest.mark.skipif(
    not (is_pytorch_available() and _has_package("torchmetrics")),
    reason="PyTorch or torchmetrics is not available",
)
def test_torchmetrics_jaccard_index_calculation() -> None:
    """Verify TorchMetrics computes MulticlassJaccardIndex on sample segmentation predictions."""
    import torch
    from torchmetrics.classification import MulticlassJaccardIndex

    device = resolve_device("auto")
    metric = MulticlassJaccardIndex(num_classes=5).to(device)
    preds = torch.zeros((2, 16, 16), dtype=torch.long, device=device)
    targets = torch.zeros((2, 16, 16), dtype=torch.long, device=device)
    preds[:, :8, :8] = 1
    targets[:, :8, :8] = 1

    score = metric(preds, targets)
    assert bool(torch.isfinite(score).all().item())
    assert float(score.item()) > 0.0


"""Live verification script for PyTorch, CUDA, MONAI, torchvision, Albumentations, and TorchMetrics.

Run with:
    .\\.venv311\\Scripts\\python.exe scripts\\verify_torch_cuda.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.utils.device import get_device_info, resolve_device
from src.utils.seed import set_seed


def main() -> int:
    """Execute end-to-end hardware and deep-learning stack verification."""
    print("=" * 72)
    print("FedMorphis Phase 0, Task 4: Deep Learning & CUDA Stack Verification")
    print("=" * 72)

    # 1. Print Python executable and version
    print(f"[1] Python Executable : {sys.executable}")
    print(f"    Python Version    : {sys.version.splitlines()[0]}")

    try:
        import torch
    except ImportError as exc:
        print(f"[FAIL] PyTorch is not installed: {exc}")
        return 1

    # 2-4. Print PyTorch version, CUDA runtime version, and CUDA availability
    print(f"[2] torch.__version__ : {torch.__version__}")
    print(f"[3] torch.version.cuda: {torch.version.cuda}")
    cuda_avail = torch.cuda.is_available()
    print(f"[4] CUDA Available    : {cuda_avail}")

    if not cuda_avail:
        print("[FAIL] CUDA is not available in PyTorch. GPU verification cannot pass.")
        return 1

    # 5-6. Print GPU device count, name, and memory information
    device_count = torch.cuda.device_count()
    device_name = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    total_vram_mb = round(props.total_memory / (1024 * 1024), 2)
    print(f"[5] GPU Device Count  : {device_count}")
    print(f"    GPU 0 Name        : {device_name} (Compute Capability {props.major}.{props.minor})")
    print(f"[6] GPU 0 Total VRAM  : {total_vram_mb} MiB ({props.total_memory} bytes)")

    # 7-8. Verify get_device_info() and resolve_device("auto")
    set_seed(42, deterministic=True)
    device_info = get_device_info()
    resolved = resolve_device("auto")
    print(f"[7] get_device_info() : {device_info}")
    print(f"[8] resolve_device('auto'): {resolved} (type={resolved.type})")

    if resolved.type != "cuda":
        print(f"[FAIL] Expected resolve_device('auto') to select 'cuda', got {resolved!r}")
        return 1

    # 9-12. Create real tensor on CUDA, perform matmul, synchronize, verify finite & on CUDA
    mat_a = torch.randn((256, 256), device=resolved, dtype=torch.float32)
    mat_b = torch.randn((256, 256), device=resolved, dtype=torch.float32)
    mat_c = torch.matmul(mat_a, mat_b)
    torch.cuda.synchronize(resolved)

    is_on_cuda = mat_c.is_cuda
    is_finite = bool(torch.isfinite(mat_c).all().item())
    c_norm = float(torch.linalg.norm(mat_c).item())
    print(
        f"[9-12] CUDA MatMul (256x256) -> shape={tuple(mat_c.shape)}, "
        f"device={mat_c.device}, finite={is_finite}, L2_norm={c_norm:.4f}"
    )
    if not (is_on_cuda and is_finite):
        print("[FAIL] CUDA matrix multiplication check failed.")
        return 1

    # Verify small PyTorch 2D convolution on CUDA
    conv = torch.nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1).to(resolved)
    sample_img = torch.randn((2, 3, 64, 64), device=resolved)
    conv_out = conv(sample_img)
    torch.cuda.synchronize(resolved)
    conv_ok = conv_out.is_cuda and bool(torch.isfinite(conv_out).all().item())
    print(
        f"[Conv2d] CUDA Conv2d -> input={tuple(sample_img.shape)}, "
        f"output={tuple(conv_out.shape)}, device={conv_out.device}, ok={conv_ok}"
    )
    if not conv_ok:
        print("[FAIL] CUDA Conv2d execution failed.")
        return 1

    # Verify MONAI segmentation loss (DiceLoss) on CUDA
    import monai
    from monai.losses import DiceLoss

    dice_loss_fn = DiceLoss(to_onehot_y=True, softmax=True)
    logits = torch.randn((2, 5, 32, 32), device=resolved, requires_grad=True)
    targets = torch.randint(0, 5, (2, 1, 32, 32), device=resolved)
    loss_val = dice_loss_fn(logits, targets)
    loss_val.backward()
    torch.cuda.synchronize(resolved)
    monai_ok = (
        loss_val.is_cuda
        and bool(torch.isfinite(loss_val).all().item())
        and logits.grad is not None
        and bool(torch.isfinite(logits.grad).all().item())
    )
    print(
        f"[MONAI] version={monai.__version__} | CUDA DiceLoss={float(loss_val.item()):.4f}, "
        f"grad_finite={monai_ok}"
    )
    if not monai_ok:
        print("[FAIL] MONAI CUDA DiceLoss verification failed.")
        return 1

    # Verify torchvision import and basic transform operation
    import torchvision
    from torchvision.ops import box_iou

    boxes_a = torch.tensor([[0.0, 0.0, 10.0, 10.0]], device=resolved)
    boxes_b = torch.tensor([[5.0, 5.0, 15.0, 15.0]], device=resolved)
    iou_tensor = box_iou(boxes_a, boxes_b)
    torch.cuda.synchronize(resolved)
    tv_ok = iou_tensor.is_cuda and bool(torch.isfinite(iou_tensor).all().item())
    print(
        f"[torchvision] version={torchvision.__version__} | CUDA box_iou={float(iou_tensor.item()):.4f}, "
        f"ok={tv_ok}"
    )
    if not tv_ok:
        print("[FAIL] torchvision verification failed.")
        return 1

    # Verify Albumentations construction and execution on synthetic image + mask
    import albumentations as A

    transform = A.Compose(
        [
            A.HorizontalFlip(p=1.0),
            A.RandomBrightnessContrast(p=1.0),
        ]
    )
    synth_img = np.random.randint(0, 256, size=(64, 64, 3), dtype=np.uint8)
    synth_mask = np.random.randint(0, 5, size=(64, 64), dtype=np.uint8)
    augmented = transform(image=synth_img, mask=synth_mask)
    alb_ok = (
        augmented["image"].shape == (64, 64, 3)
        and augmented["mask"].shape == (64, 64)
    )
    print(
        f"[Albumentations] version={A.__version__} | output_shape={augmented['image'].shape}, "
        f"mask_shape={augmented['mask'].shape}, ok={alb_ok}"
    )
    if not alb_ok:
        print("[FAIL] Albumentations verification failed.")
        return 1

    # Verify TorchMetrics calculation with sample tensors
    import torchmetrics
    from torchmetrics.classification import MulticlassJaccardIndex

    metric = MulticlassJaccardIndex(num_classes=5).to(resolved)
    pred_classes = torch.randint(0, 5, (2, 32, 32), device=resolved)
    true_classes = torch.randint(0, 5, (2, 32, 32), device=resolved)
    jaccard_score = metric(pred_classes, true_classes)
    torch.cuda.synchronize(resolved)
    tm_ok = jaccard_score.is_cuda and bool(torch.isfinite(jaccard_score).all().item())
    print(
        f"[TorchMetrics] version={torchmetrics.__version__} | CUDA MulticlassJaccardIndex="
        f"{float(jaccard_score.item()):.4f}, ok={tm_ok}"
    )
    if not tm_ok:
        print("[FAIL] TorchMetrics verification failed.")
        return 1

    print("=" * 72)
    print("[PASS] All PyTorch, CUDA, MONAI, torchvision, Albumentations, and TorchMetrics checks passed!")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())


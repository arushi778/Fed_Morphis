# FedMorphis: Self-Supervised and Federated Learning for Biomedical Applications

**FedMorphis** is a research-grade software prototype investigating the synergy of **Self-Supervised Learning (SSL)** and **Federated Learning (FL)** for multi-institutional biomedical image segmentation, focusing on **diabetic retinopathy (DR) lesion segmentation** across the **DDR** and **IDRiD** fundus photography datasets.

---

## 1. Project Overview

Annotated pixel-level biomedical datasets are scarce, expensive to curate, and strictly governed by patient privacy regulations that prevent centralizing clinical data across hospitals. At the same time, clinical institutions often hold abundant *unlabeled* retinal images alongside smaller labeled cohorts suffering from cross-scanner domain shift.

FedMorphis addresses both challenges simultaneously:
1. **Label Scarcity**: Uses **SimCLR** contrastive representation learning to pretrain a **ResNet-50** encoder on unlabeled fundus images before transferring weights to a **U-Net** segmentation network.
2. **Data Privacy & Institutional Heterogeneity**: Uses the **Flower** federated framework (**FedAvg** and **FedProx**) to train both SSL encoders and downstream lesion segmentation models across decentralized client nodes without sharing raw patient imagery.

---

## 2. Research Objectives

- **Multi-Lesion Segmentation**: Segment four primary diabetic retinopathy lesions—Microaneurysms (`MA`), Hemorrhages (`HE`), Hard Exudates (`EX`), and Soft Exudates (`SE`)—from color fundus photographs using a U-Net with a ResNet-50 encoder.
- **Label-Efficient Representation Learning**: Quantify the downstream segmentation gains of SimCLR self-supervised pretraining compared to random initialization and ImageNet initialization under varying label budgets.
- **Decentralized Privacy-Preserving Training**: Compare **FedAvg** and **FedProx** across IID, non-IID (Dirichlet), and cross-dataset (DDR vs. IDRiD) client splits.
- **Federated Self-Supervised Pretraining (FedSSL)**: Evaluate whether federated contrastive pretraining across distributed unlabeled retinal repositories produces representations competitive with centralized pretraining.
- **Reproducible Benchmarking & Clinical Inspection**: Track all experiments systematically via MLflow and provide an interactive Streamlit dashboard + FastAPI inference service for visual inspection of predicted lesion masks.

---

## 3. Technology Stack

| Layer | Technologies |
| :--- | :--- |
| **Core Language** | Python ($\ge 3.11$; verified on Python 3.11.9 & Python 3.14.4) |
| **Deep Learning & Vision** | PyTorch, torchvision, MONAI, TorchMetrics |
| **Data Processing & Augmentation** | Albumentations, OpenCV (`opencv-python-headless`), NumPy, Pandas, scikit-learn |
| **Federated Learning** | Flower (`flwr`) supporting FedAvg and FedProx |
| **Experiment Tracking & Config** | MLflow, PyYAML, `python-dotenv` |
| **Visualization, API & UI** | Streamlit, FastAPI, Uvicorn, Matplotlib, Plotly |
| **Testing** | `pytest` |

---

## 4. Project Structure

```text
FedMorphis/
├── configs/
│   ├── project.yaml          # Global seed, device, and relative path settings
│   ├── dataset.yaml          # DDR & IDRiD paths, lesion classes, splits, normalization
│   ├── model.yaml            # U-Net + ResNet-50 encoder architecture settings
│   ├── ssl.yaml              # SimCLR projection head, NT-Xent temperature, optimizer
│   ├── federated.yaml        # Flower client/server settings, FedAvg & FedProx parameters
│   └── experiment.yaml       # MLflow tracking, checkpoints, and evaluation metrics
├── data/
│   ├── raw/
│   │   ├── DDR/              # Raw DDR dataset files (images & lesion masks)
│   │   └── IDRiD/            # Raw IDRiD dataset files (images & lesion masks)
│   ├── processed/            # Standardized, FOV-cropped, and resized tensors/patches
│   └── partitions/           # Client partition manifests (IID, Dirichlet, cross-center)
├── checkpoints/
│   ├── ssl/                  # Pretrained SimCLR encoder weights
│   ├── segmentation/         # Centralized U-Net segmentation checkpoints
│   └── federated/            # Global & client-aggregated federated checkpoints
├── experiments/              # MLflow tracking artifacts and comparative CSV summaries
├── logs/                     # Structured runtime logs
├── docs/
│   └── architecture.md       # Detailed architectural specification and training paradigms
├── scripts/                  # CLI entrypoints for preprocessing, training, and evaluation
├── src/
│   ├── data/                 # Dataset classes, transforms, and federated partitioners
│   ├── models/               # ResNet-50 backbone, SimCLR head, and U-Net architecture
│   ├── ssl/                  # Contrastive loss (NT-Xent) and SSL training engine
│   ├── segmentation/         # Dice/Focal losses and supervised segmentation trainer
│   ├── federated/            # Flower Client, Server, FedAvg, and FedProx strategies
│   ├── evaluation/           # Dice, IoU, AUPRC, Sensitivity, and Specificity metrics
│   ├── experiments/          # Experiment runner and MLflow logging hooks
│   └── utils/                # Pathlib resolution, YAML config loader, seeding, logging
├── app/
│   └── pages/                # Streamlit multi-page interactive dashboard
├── tests/                    # Automated pytest verification suite
├── .env.example              # Environment variable template
├── .gitignore                # Git exclusion rules
├── pyproject.toml            # Python package metadata and pytest configuration
├── requirements.txt          # Dependency specification
└── README.md                 # Project documentation
```

---

## 5. Environment Setup

### Hardware Verification Notes
During Phase 0 inspection on the development host:
- **GPU Detected**: NVIDIA GeForce RTX 5050 Laptop GPU (8 GB VRAM, Driver `592.19`, CUDA Runtime `13.1`, Blackwell architecture).
- **CUDA Compiler (`nvcc`)**: Full CUDA Toolkit (`nvcc`) is not on the system `PATH`, though the NVIDIA driver and WDDM runtime are active.
- **Python Installations**: Python `3.14.4` (`C:\Python314\python.exe`) and Python `3.11.9` (`C:\Users\Arushi\AppData\Local\Programs\Python\Python311\python.exe`).

### Creating and Activating the Virtual Environment (PowerShell)

The primary deep-learning and CUDA environment uses **Python 3.11.9** in `.venv311` (while preserving the original Python 3.14 `.venv`):

```powershell
# Activate the verified Python 3.11 + CUDA 12.8 virtual environment
.\.venv311\Scripts\Activate.ps1
```

If creating `.venv311` from scratch:

```powershell
& "C:\Users\Arushi\AppData\Local\Programs\Python\Python311\python.exe" -m venv .venv311 --system-site-packages
.\.venv311\Scripts\Activate.ps1
```

---

## 6. How to Install Dependencies & Verify the Deep Learning Stack

### Verified Deep Learning Stack in `.venv311`
For the **NVIDIA GeForce RTX 5050 Laptop GPU** (`Compute Capability 12.0`, Blackwell architecture), install the `cu128` PyTorch wheels and biomedical segmentation packages:

```powershell
.\.venv311\Scripts\python.exe -m pip install pytest monai albumentations torchmetrics
.\.venv311\Scripts\python.exe -m pip install --force-reinstall --no-deps torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

### Live CUDA & Stack Verification (`scripts/verify_torch_cuda.py`)
Run the end-to-end verification script to confirm live CUDA tensor execution, 2D convolution, MONAI `DiceLoss`, `torchvision`, `Albumentations`, and `TorchMetrics`:

```powershell
.\.venv311\Scripts\python.exe scripts\verify_torch_cuda.py
```

**Verified Hardware & Stack Output**:
- **Python**: `3.11.9` (`.venv311\Scripts\python.exe`)
- **PyTorch**: `2.11.0+cu128` (`torch.version.cuda == "12.8"`, `torch.cuda.is_available() == True`)
- **GPU**: `NVIDIA GeForce RTX 5050 Laptop GPU` (`8150.56 MiB` total VRAM, Compute Capability `12.0`)
- **Packages Verified on CUDA**: `torchvision 0.26.0+cu128`, `monai 1.6.1`, `albumentations 2.0.8`, `torchmetrics 1.9.0`

---

## 7. Reproducibility & Hardware Device Management

FedMorphis provides centralized utilities in `src/utils/seed.py` and `src/utils/device.py` to enforce reproducible experiments and explicit compute device resolution across centralized and federated training runs.

### Random Seeds, Worker Seeding & Deterministic Execution
- **`set_seed(seed=42, deterministic=True)`** initializes global random seeds across Python's `random`, `numpy.random`, `os.environ["PYTHONHASHSEED"]`, and PyTorch CPU/CUDA generators (when PyTorch is installed).
- **`seed_worker(worker_id)`** provides a deterministic `worker_init_fn` for multi-process data loading, deriving worker-specific 32-bit seeds from `torch.initial_seed() % 2**32` (when PyTorch is available) so each worker produces distinct yet reproducible augmentation streams.
- **Deterministic Execution (`deterministic=True`)**: Configures `CUBLAS_WORKSPACE_CONFIG=":4096:8"`, enables `torch.use_deterministic_algorithms(True, warn_only=True)`, sets `torch.backends.cudnn.deterministic = True`, and disables `torch.backends.cudnn.benchmark = False`.
- **Important Limitations**:
  - Enabling deterministic execution disables cuDNN kernel autotuning (`cudnn.benchmark = False`), which can **reduce training and inference throughput**.
  - **Complete bit-level reproducibility is not guaranteed** across different CPU/GPU hardware architectures, operating systems, NVIDIA driver versions, CUDA runtimes, or library versions.
  - Certain GPU operations (such as atomic scatter/gather kernels or specific interpolation backward passes) may lack deterministic CUDA implementations.

### Device Selection, CUDA Availability & Hardware Inspection
- **`resolve_device(device="auto")`** supports `"auto"`, `"cuda"`, and `"cpu"`:
  - `"auto"`: Resolves to `"cuda"` if PyTorch is installed and CUDA is available; otherwise falls back to `"cpu"`.
  - `"cuda"`: Resolves to `"cuda"` only when PyTorch is installed with active CUDA support; **raises a `RuntimeError`** if PyTorch is missing or CUDA is unavailable (never silently falls back to CPU).
  - `"cpu"`: Explicitly selects CPU execution.
  - Returns a `torch.device` instance when PyTorch is installed, or a lightweight `FallbackDevice("cpu")` when PyTorch is not yet installed.
- **`get_device_info()`** inspects runtime hardware availability without requiring `nvcc` or a standalone CUDA Toolkit installation. It reports `pytorch_available`, `pytorch_version`, `cuda_available`, `cuda_runtime_version` (the CUDA runtime bundled with PyTorch via `torch.version.cuda`, distinct from the host driver version reported by `nvidia-smi`), `device_count`, `gpu_names`, `gpu_memory`, and `default_device`.

### Usage Example

```python
from src.utils.seed import set_seed, seed_worker
from src.utils.device import resolve_device, get_device_info

set_seed(42)

device = resolve_device("auto")
info = get_device_info()
print(f"Selected device: {device} | Hardware info: {info}")
```

---

## 8. Structured Logging & Configuration Validation

### Centralized Logging (`src/utils/logger.py`)
- **Obtaining a Logger**: Application modules obtain a named logger via `get_logger(name, level="INFO", log_file="fedmorphis.log")`.
- **Log Storage & Rotation**: Log files are written beneath the project's `logs/` directory (resolved dynamically via `get_logs_dir()` in `src/utils/paths.py`, independent of the current working directory). File logging uses Python's `RotatingFileHandler` (default `5 MB` max file size with `3` backup files) to prevent unbounded disk growth.
- **Handler Idempotency & Isolation**: Repeated calls to `get_logger()` with the same logger name update the log level without adding duplicate handlers, and `logger.propagate = False` prevents duplicate root output or interference with third-party libraries.

```python
from src.utils.logger import get_logger

logger = get_logger("fedmorphis.training", level="INFO")
logger.info("Starting configuration check.")
```

### Configuration Loading, Merging & Validation (`src/utils/config.py`)
- **Loading & Validating**: `load_yaml_config(config_path, validate=True)` parses YAML files under `configs/`, rejecting malformed YAML or non-mapping documents with a `ConfigValidationError`. `validate_config(cfg, section_name)` and `validate_all_configs()` check required fields, types, numeric bounds, split sums, and supported choices across all 6 configuration files (`project`, `dataset`, `model`, `ssl`, `federated`, `experiment`).
- **Deep Merging Overrides**: `deep_merge_configs(base, overrides)` recursively merges nested configuration dictionaries without mutating either input dictionary, preserving sibling keys while applying overrides.

```python
from src.utils.config import load_yaml_config, deep_merge_configs, validate_config

base_cfg = load_yaml_config("dataset.yaml", validate=True)
overrides = {"batch_size": 16, "normalization": {"fov_crop": False}}
merged_cfg = deep_merge_configs(base_cfg, overrides)
validate_config(merged_cfg, "dataset")
```

### Running the Logging and Configuration Tests

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_logger.py tests/test_config.py -v
```

---

## 9. Dataset Acquisition Guidance, Placement & Inventory Audit

### 9.1 Official Dataset Acquisition Sources
Datasets are **not** downloaded automatically and must be obtained through their authorized channels:

1. **IDRiD (Indian Diabetic Retinopathy Image Dataset)**:
   - **Official Source**: IEEE DataPort (`https://dx.doi.org/10.21227/H25W98` / `https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid`) and Grand Challenge (`https://idrid.grand-challenge.org/`).
   - **Access & License**: Open access on IEEE DataPort (requires a free IEEE account login); licensed under **CC BY 4.0** (cite Porwal et al., 2018).
   - **Subsets to Download**:
     - `A. Segmentation` (`A.%20Segmentation.zip`): **Required** for pixel-level lesion segmentation (`81` images + `MA`, `HE`, `EX`, `SE`, and `OD` masks).
     - `B. Disease Grading` (`B.%20Disease%20Grading.zip`): **Optional** `516` images suitable for unlabeled SimCLR self-supervised pretraining.

2. **DDR (Dataset for Diabetic Retinopathy)**:
   - **Official Source**: Nankai University Computer Vision Lab GitHub repository (`https://github.com/nkicsl/DDR-dataset`), which provides official Google Drive and Baidu Pan archive links.
   - **Access & License**: Academic/non-commercial research use (cite Li et al., *Information Sciences*, 2019).
   - **Subsets to Download**:
     - `lesion_segmentation`: **Required** for pixel-level lesion segmentation (`757` annotated images across `MA`, `HE`, `EX`, `SE`).
     - `DR_grading`: **Optional** `12,522` unannotated grading images suitable for unlabeled SimCLR self-supervised pretraining.

### 9.2 Expected Raw-Data Placement & Preservation Warning
Extract the downloaded archives into their respective roots under `data/raw/` while preserving the original dataset folder hierarchy:

```text
data/raw/
├── DDR/
│   └── <official extracted DDR structure, e.g., lesion_segmentation/...>
└── IDRiD/
    └── <official extracted IDRiD structure, e.g., 1. Original Images/... and 2. All Segmentation Groundtruths/...>
```

> **IMPORTANT — Do Not Modify Raw Data**:
> - **Never modify, rename, resize, or overwrite raw fundus images or raw ground-truth masks** inside `data/raw/`.
> - **Do not mix DDR and IDRiD files** in the same directory.
> - Keep `data/raw/` strictly read-only; all derived masks and cropped tensors belong in `data/processed/`.
> - `scripts/inspect_datasets.py` automatically traverses nested folders, so no manual folder flattening is needed.

### 9.3 Running the Inventory & Dataset Locator Utility (`scripts/inspect_datasets.py`)

```powershell
# Inspect configured data/raw/DDR and data/raw/IDRiD directories
.\.venv311\Scripts\python.exe scripts\inspect_datasets.py

# Scan a specific download directory for unextracted archives or extracted folders (read-only)
.\.venv311\Scripts\python.exe scripts\inspect_datasets.py --search-dir "C:\Users\Arushi\Downloads"

# Export full inventory and locator results to JSON
.\.venv311\Scripts\python.exe scripts\inspect_datasets.py --json-out logs\dataset_inventory.json
```

### 9.4 Canonical 4-Channel Mask Construction, Retinal FOV Cropping & Manifests (`scripts/build_manifests.py`)

- **Canonical Lesion Mask (`src/data/masks.py`)**: Constructs independent four-channel binary masks `(H, W, 4)` (`np.uint8`, values `{0, 1}`) in fixed `["MA", "HE", "EX", "SE"]` order while excluding `OD` (Optic Disc), synthesizing zero channels for omitted IDRiD masks (`1` `HE`, `41` `SE`), and safely binarizing both single-channel `{0, 1}` / `{0, 255}` masks and IDRiD's 4-channel RGBA `IDRiD_81_EX.tif` mask.
- **Retinal FOV Cropping (`src/data/fov.py`)**: Detects the retinal disk via `image[..., :3].max(axis=-1) > 10` with deterministic full-image fallback and applies identical integer crop coordinates to both image and `(H, W, 4)` mask without interpolation.
- **Reproducible Dataset Manifests (`src/data/manifests.py`)**: Generates validated segmentation manifests under `data/processed/manifests/` (`ddr_manifest.json` with `757` cases across `train: 383, valid: 149, test: 225`; `idrid_manifest.json` with `81` cases across `train: 54, test: 27`; and `combined_manifest.json` with `838` total cases).
- **Full Audit & Manifest Report**: Documented in [`docs/data_inventory.md`](file:///c:/Users/Arushi/OneDrive/Desktop/Fed_Morphis/docs/data_inventory.md).

```powershell
# Build and validate canonical DDR and IDRiD segmentation manifests
.\.venv311\Scripts\python.exe scripts\build_manifests.py

# Build federated client partitions (IID, Dirichlet, Cross-Dataset) and run real-data smoke test
.\.venv311\Scripts\python.exe scripts\build_partitions.py --smoke-test

# Run all Phase 1 data engineering unit tests
.\.venv311\Scripts\python.exe -m pytest tests\test_dataset_inventory.py tests\test_masks_fov_manifests.py tests\test_dataset_transforms_partitions.py -v
```

---

## 10. Current Implementation Status

- **Current Phase**: **Phase 1, Task 4 — PyTorch Dataset, DataLoader, Transforms, and Federated Client Partitions (Completed)**
  - [x] **Phase 0 Complete**: Environment inspection, modular structure, YAML configs, reproducibility (`set_seed`, `seed_worker`), device management (`resolve_device`, `get_device_info`), structured logging (`get_logger`), config validation (`validate_all_configs`), and CUDA 12.8 stack verification in `.venv311`.
  - [x] **Phase 1, Tasks 1, 2 & 3A Complete**: Dataset inventory, archive detector, candidate locator (`src/data/inventory.py`, `scripts/inspect_datasets.py`), and real-data audit report (`logs/dataset_inventory.json`, `docs/data_inventory.md`).
  - [x] **Phase 1, Task 3B Complete**: Canonical 4-channel `[MA, HE, EX, SE]` binary mask builder (`src/data/masks.py`), retinal FOV detector and synchronized cropper (`src/data/fov.py`), reproducible dataset manifests (`src/data/manifests.py`, `scripts/build_manifests.py`, `data/processed/manifests/*.json`), and unit tests (`tests/test_masks_fov_manifests.py`).
  - [x] **Phase 1, Task 4 Complete**: PyTorch `RetinalLesionSegmentationDataset` and `build_segmentation_dataloader` (`src/data/dataset.py`), Albumentations 2.0.8 transforms (`src/data/transforms.py`), IID / Dirichlet / Cross-Dataset federated client partitioning (`src/data/partitioning.py`, `scripts/build_partitions.py`, `data/partitions/*.json`), and unit tests (`tests/test_dataset_transforms_partitions.py`).
  - [ ] ResNet-50 U-Net & SimCLR implementation (Not started — Phase 2/3).
  - [ ] Flower Federated Learning implementation (Not started — Phase 4/5).

---

## 11. Future Implementation Phases

- **Phase 1 — Data Engineering & Partitioning**: DDR and IDRiD dataset adapters, Field-of-View (FOV) cropping, Albumentations pipelines, and IID / Dirichlet / Cross-Center federated split generators.
- **Phase 2 — Self-Supervised Pretraining (SimCLR)**: Contrastive data pairs, ResNet-50 encoder + MLP projection head, NT-Xent loss, and centralized SSL pretraining loop.
- **Phase 3 — Supervised Segmentation Fine-Tuning**: U-Net with ResNet-50 encoder (supporting random, ImageNet, or SimCLR weights), Dice + Focal loss, and centralized training/validation loop.
- **Phase 4 — Federated Supervised Segmentation**: Flower client/server orchestration implementing **FedAvg** and **FedProx** for decentralized U-Net training.
- **Phase 5 — Federated Self-Supervised Learning (FedSSL)**: Decentralized SimCLR pretraining across client silos followed by federated or local segmentation fine-tuning.
- **Phase 6 — Comparative Evaluation & Tracking**: Automated evaluation across Dice, IoU, AUPRC, Sensitivity, and Specificity with MLflow experiment tracking.
- **Phase 7 — Interactive Dashboard & Inference API**: FastAPI model serving endpoint and multi-page Streamlit dashboard for clinician-friendly lesion visualization and cross-regime comparison.


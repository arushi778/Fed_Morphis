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

## 10. Supervised U-Net + ResNet-50 Baseline (`Phase 2, Task 1`)

### 10.1 Architecture, Loss, Metrics & Trainer
- **U-Net + ResNet-50 (`src/models/unet.py`)**:
  - `ResNet50UNet` maps RGB fundus tensors `(B, 3, H, W)` (`torch.float32`) to raw segmentation logits `(B, 4, H, W)` in canonical `["MA", "HE", "EX", "SE"]` order (`32,521,028` trainable parameters).
  - Supports ImageNet pretrained weights (`pretrained=True`), offline/random initialization (`pretrained=False`), and SSL encoder checkpoints (`encoder_weights_source="ssl_checkpoint"`). Explicitly records `model.pretrained_weights_loaded` and `model.weights_status` without ever silently claiming pretrained weights were loaded when unavailable.
- **Multilabel BCE + Soft Dice Loss (`src/models/losses.py`)**:
  - `BCEDiceLoss` combines channel-independent `BCEWithLogitsLoss` and `SoftDiceLoss` (`bce_weight=0.5, dice_weight=0.5, smooth=1e-5`), remaining numerically finite even when all or individual lesion channels are empty (`0`).
- **Evaluation Metrics & Empty-Mask Policy (`src/models/metrics.py`)**:
  - `compute_segmentation_metrics` and `SegmentationMetricTracker` compute per-class (`MA`, `HE`, `EX`, `SE`) and macro-averaged (`macro` and `macro_present_only`) **Dice**, **IoU**, **Precision**, **Recall**, **Sensitivity**, and **Specificity** at configurable probability threshold (`0.5`).
  - Enforces an explicit, documented empty-mask policy (`both_empty_exact_match` $\rightarrow 1.0$; `target_empty_false_positive` $\rightarrow 0.0$; `prediction_empty_false_negative` $\rightarrow 0.0$) and reports `present_classes` and `absent_classes` so missing lesion classes in validation subsets are never hidden.
- **Supervised Trainer (`src/engine/trainer.py`)**:
  - `SupervisedSegmentationTrainer` supports `AdamW` / `Adam` / `SGD`, `torch.amp` mixed precision on CUDA, gradient norm clipping, `latest_checkpoint.pt` and `best_checkpoint.pt` persistence/resumption, and strict runtime enforcement (`TestSplitLeakageError`) that `test` split samples are never used during training or model selection.

### 10.2 Running the Baseline CLI & Smoke Test (`scripts/train_baseline.py`)

```powershell
# Run the real-data GPU smoke test on DDR + IDRiD (1 train step + 1 val step, batch_size=2, 512x512)
.\.venv311\Scripts\python.exe scripts\train_baseline.py --smoke-test --no-pretrained --batch-size 2 --image-size 512

# Run a single-batch dry-run verification pass
.\.venv311\Scripts\python.exe scripts\train_baseline.py --dry-run --no-pretrained --batch-size 2 --image-size 512

# Run the U-Net baseline unit test suite
.\.venv311\Scripts\python.exe -m pytest tests\test_unet_baseline.py -v
```

---

## 11. Flower Federated Learning (`Phase 4`)

### 11.1 Federated Learning Architecture
FedMorphis implements cross-institutional federated retinal lesion segmentation using **Flower (`flwr 1.43.0`)**:
- **Client Data Module (`src/federated/data.py`)**: Resolves client partition assignments from `data/partitions/combined_cross_dataset_2clients_seed42.json`, builds canonical 4-channel `[MA, HE, EX, SE]` segmentation datasets and `DataLoader`s, applies training augmentations only to training splits, and enforces zero overlap or test-split leakage.
- **Flower `NumPyClient` (`src/federated/client.py`)**: `FedMorphisNumPyClient` wraps `ResNet50UNet` and `SupervisedSegmentationTrainer`, serializing and validating `state_dict` parameters and BatchNorm buffers losslessly across Flower `Parameters` protobuf messages.
- **Custom FedAvg Strategy (`src/federated/strategy.py`)**: `FedMorphisFedAvg` subclasses `flwr.server.strategy.FedAvg` with strict shape, dtype, and finiteness validation, sample-weighted parameter aggregation, explicit integer buffer handling, and atomic global checkpointing.
- **Flower Server Orchestration (`src/federated/server.py`)**: `InProcessClientProxy`, `build_federated_server`, and `run_federated_training` orchestrate multi-round training and distributed client validation through `flwr.server.Server` and `SimpleClientManager`.

### 11.2 Client Assignments & Validation Policy (`official_and_carved`)
In the two-client cross-dataset configuration (`seed=42`):
- **Client 0 (`client_0` — DDR)**:
  - **Local Training**: `383` official DDR training records.
  - **Local Validation**: `149` official DDR validation records.
  - **Held-Out Test**: `225` official DDR test records (never used during federated training or checkpoint selection).
- **Client 1 (`client_1` — IDRiD)**:
  - **Local Training**: `44` IDRiD training records (after carving `20%` from the `54` official IDRiD training records).
  - **Local Validation**: `10` deterministically carved IDRiD validation records (`val_carve_ratio: 0.2`, `seed: 42`).
  - **Held-Out Test**: `27` official IDRiD test records (completely untouched during training and validation).

### 11.3 FedAvg Aggregation Policy
- **Floating-Point Parameters & BatchNorm Statistics (`weight`, `bias`, `running_mean`, `running_var`)**: Aggregated via sample-count-weighted averaging across participating clients:
  $$w^{(t)} = \sum_{k=1}^{K} \frac{n_k}{\sum_{j=1}^{K} n_j} w_k^{(t)}$$
- **Integer BatchNorm Counters (`num_batches_tracked`, `dtype=int64`)**: Aggregated using an explicit integer buffer policy (`"max"` across participating clients by default), preserving exact `np.int64` dtype without floating-point averaging or rounding.
- **Buffer Sharing (`share_batchnorm_buffers`)**: Configurable via `configs/federated.yaml` or `--share-batchnorm-buffers` / `--no-share-batchnorm-buffers`.
- **Metric Aggregation**: Computes sample-weighted averages across clients, unweighted client-macro averages (`macro_client_*`), and per-client metrics (`client_0_DDR_*`, `client_1_IDRiD_*`).

### 11.4 Global Model Initialization Options
Configured under `initialization` in `configs/federated.yaml` (or via CLI flags):
1. **ImageNet-Pretrained ResNet-50 Encoder** (default): `--encoder-weights-source imagenet`
2. **SimCLR Self-Supervised Encoder Checkpoint**: `--encoder-weights-source ssl_checkpoint --ssl-checkpoint-path checkpoints/ssl/ddr_simclr/best_simclr_encoder.pt`
3. **Full Segmentation Checkpoint**: `--initial-checkpoint-path <path/to/checkpoint.pt>`
4. **Random Initialization**: `--no-pretrained` (or `--encoder-weights-source random`)

### 11.5 Dataset Requirements
Before running federated training or evaluation on real data, ensure the following generated manifests exist:
- `data/processed/manifests/ddr_manifest.json`
- `data/processed/manifests/idrid_manifest.json`
- `data/processed/manifests/combined_manifest.json`
- `data/partitions/combined_cross_dataset_2clients_seed42.json`

### 11.6 Federated Training Command (`scripts/train_federated.py`)

```powershell
# Run federated training using configs/federated.yaml
python scripts/train_federated.py --config configs/federated.yaml

# Override rounds, local epochs, seed, device, or initialization source
python scripts/train_federated.py --config configs/federated.yaml --rounds 20 --local-epochs 3 --seed 42 --device cuda

# Initialize global encoder from the SimCLR SSL checkpoint
python scripts/train_federated.py --config configs/federated.yaml --encoder-weights-source ssl_checkpoint --ssl-checkpoint-path checkpoints/ssl/ddr_simclr/best_simclr_encoder.pt
```

### 11.7 Global Checkpoint Evaluation & Final Held-Out Test Evaluation (`scripts/evaluate_federated.py`)
- **Validation Split Evaluation**:
  ```powershell
  python scripts/evaluate_federated.py --checkpoint checkpoints/federated/best_checkpoint.pt --split valid
  ```
- **Final Held-Out Test Evaluation (Separate DDR and IDRiD Reporting)**:
  Held-out `test` data is never used during training or model selection. To evaluate the selected global checkpoint on the official DDR (`225` samples) and IDRiD (`27` samples) held-out test splits separately:
  ```powershell
  python scripts/evaluate_federated.py --checkpoint checkpoints/federated/best_checkpoint.pt --split test --datasets ddr idrid --report-path logs/federated_fedavg_test_evaluation_report.json
  ```
  Results are reported separately under `datasets.DDR` and `datasets.IDRiD` (never pooled across datasets), including loss, macro Dice, macro IoU, and per-lesion (`MA`, `HE`, `EX`, `SE`) Dice and IoU.

### 11.8 Checkpoint and Report Locations
- **Global Checkpoints**:
  - `checkpoints/federated/best_checkpoint.pt` (monitored on finite `val_dice_macro`)
  - `checkpoints/federated/latest_checkpoint.pt` (saved atomically after each round)
- **JSON Reports**:
  - Training report: `logs/federated_fedavg_ddr_idrid_report.json`
  - Held-out test evaluation report: `logs/federated_fedavg_test_evaluation_report.json`
  - Validation evaluation report: `logs/federated_fedavg_val_evaluation_report.json`

---

## 12. Current Implementation Status

- **Current Phase**: **Phase 4 — Flower Federated Learning (Tasks 1–5 Completed)**
  - [x] **Phase 0 Complete**: Environment inspection, modular structure, YAML configs, reproducibility (`set_seed`, `seed_worker`), device management (`resolve_device`, `get_device_info`), structured logging (`get_logger`), config validation (`validate_all_configs`), and CUDA stack verification.
  - [x] **Phase 1 Complete**: Dataset inventory (`src/data/inventory.py`), canonical 4-channel `[MA, HE, EX, SE]` mask builder (`src/data/masks.py`), retinal FOV cropper (`src/data/fov.py`), dataset manifests (`src/data/manifests.py`), `RetinalLesionSegmentationDataset` (`src/data/dataset.py`), Albumentations transforms (`src/data/transforms.py`), and federated partitioning (`src/data/partitioning.py`).
  - [x] **Phase 2 Complete**: Supervised `ResNet50UNet` baseline (`src/models/unet.py`), `BCEDiceLoss` (`src/models/losses.py`), `SegmentationMetricTracker` (`src/models/metrics.py`), `SupervisedSegmentationTrainer` (`src/engine/trainer.py`), and evaluator (`src/engine/evaluator.py`).
  - [x] **Phase 3 Complete**: SimCLR self-supervised representation pretraining (`src/ssl/`, `scripts/pretrain_simclr.py`) and segmentation fine-tuning.
  - [x] **Phase 4 Complete (Tasks 1–5)**: Flower 1.43.0 two-client DDR + IDRiD configuration (`configs/federated.yaml`), federated client data module (`src/federated/data.py`), `FedMorphisNumPyClient` (`src/federated/client.py`), custom `FedMorphisFedAvg` strategy (`src/federated/strategy.py`), Flower server orchestration (`src/federated/server.py`), training and evaluation CLIs (`scripts/train_federated.py`, `scripts/evaluate_federated.py`), and automated test suites (`tests/test_federated_*.py`).

---

## 12. Future Implementation Phases

- **Phase 1 — Data Engineering & Partitioning**: DDR and IDRiD dataset adapters, Field-of-View (FOV) cropping, Albumentations pipelines, and IID / Dirichlet / Cross-Center federated split generators.
- **Phase 2 — Self-Supervised Pretraining (SimCLR)**: Contrastive data pairs, ResNet-50 encoder + MLP projection head, NT-Xent loss, and centralized SSL pretraining loop.
- **Phase 3 — Supervised Segmentation Fine-Tuning**: U-Net with ResNet-50 encoder (supporting random, ImageNet, or SimCLR weights), Dice + Focal loss, and centralized training/validation loop.
- **Phase 4 — Federated Supervised Segmentation**: Flower client/server orchestration implementing **FedAvg** and **FedProx** for decentralized U-Net training.
- **Phase 5 — Federated Self-Supervised Learning (FedSSL)**: Decentralized SimCLR pretraining across client silos followed by federated or local segmentation fine-tuning.
- **Phase 6 — Comparative Evaluation & Tracking**: Automated evaluation across Dice, IoU, AUPRC, Sensitivity, and Specificity with MLflow experiment tracking.
- **Phase 7 — Interactive Dashboard & Inference API**: FastAPI model serving endpoint and multi-page Streamlit dashboard for clinician-friendly lesion visualization and cross-regime comparison.


## Task 16: Inference CLI and Demonstration App

- CLI: `python scripts/predict.py --image <path> --output-dir outputs/predictions` (best FedAvg checkpoint, round 17, frozen thresholds MA 0.25 / HE 0.40 / EX 0.35 / SE 0.35).
- Web demo: `python scripts/run_demo.py --host 127.0.0.1 --port 8000` with tabs for live inference, Flower federated architecture, 20-round training history, and the final held-out benchmark (read from `logs/federated/final_threshold_test_report.json`).
- Pre-loaded demo images come from the validation/training splits only; the held-out test images are not used by the demo.
- Prototype for research and demonstration only; not clinically certified.

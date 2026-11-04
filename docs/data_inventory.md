# DDR and IDRiD Dataset Inventory, Canonical Mask Construction, FOV Cropping & Manifest Report

**Audit Date**: 2026-11-04  
**Inventory & Locator Utility**: [`scripts/inspect_datasets.py`](file:///c:/Users/Arushi/OneDrive/Desktop/Fed_Morphis/scripts/inspect_datasets.py) / [`src/data/inventory.py`](file:///c:/Users/Arushi/OneDrive/Desktop/Fed_Morphis/src/data/inventory.py)  
**Canonical Mask, FOV & Manifest Pipeline**: [`src/data/masks.py`](file:///c:/Users/Arushi/OneDrive/Desktop/Fed_Morphis/src/data/masks.py), [`src/data/fov.py`](file:///c:/Users/Arushi/OneDrive/Desktop/Fed_Morphis/src/data/fov.py), [`src/data/manifests.py`](file:///c:/Users/Arushi/OneDrive/Desktop/Fed_Morphis/src/data/manifests.py), [`scripts/build_manifests.py`](file:///c:/Users/Arushi/OneDrive/Desktop/Fed_Morphis/scripts/build_manifests.py)  
**Machine-Readable Audit & Manifest Artifacts**:
- `logs/dataset_inventory.json`
- `data/processed/manifests/ddr_manifest.json`
- `data/processed/manifests/idrid_manifest.json`
- `data/processed/manifests/combined_manifest.json`  
**Configuration Reference**: [`configs/dataset.yaml`](file:///c:/Users/Arushi/OneDrive/Desktop/Fed_Morphis/configs/dataset.yaml)

---

## 1. Executive Summary

Both **DDR** and **IDRiD** have been audited and validated via `scripts/inspect_datasets.py` and `scripts/build_manifests.py` in `.venv311`.

| Metric | DDR (`data/raw/DDR`) | IDRiD (`data/raw/IDRiD`) |
| :--- | :--- | :--- |
| **Inventory Status** | `PRESENT` (`ddr_official`) | `PRESENT` (`idrid_official`) |
| **Segmentation Cases (Manifest)** | `757` (`train: 383`, `valid: 149`, `test: 225`) | `81` (`train: 54`, `test: 27`) |
| **Excluded Grading Images** | `13,673` (`.jpg` in `DR_grading`) | `0` (`B. Disease Grading` not placed) |
| **Canonical Lesion Channel Order** | `["MA", "HE", "EX", "SE"]` (`(H, W, 4)`) | `["MA", "HE", "EX", "SE"]` (`(H, W, 4)`) |
| **Excluded Anatomical Masks** | `0` | `81` (`OD` / Optic Disc excluded) |
| **Present Lesion Masks by Class** | `MA: 757, HE: 757, EX: 757, SE: 757` (`3,028` total) | `MA: 81, HE: 80, EX: 81, SE: 40` (`282` total) |
| **Missing Lesion Masks (Synthesized `0`)** | `MA: 0, HE: 0, EX: 0, SE: 0` | `MA: 0, HE: 1, EX: 0, SE: 41` (`42` total) |
| **Empty On-Disk Masks (`[0]`)** | `MA: 187, HE: 156, EX: 271, SE: 518` (`1,132` total) | `MA: 0, HE: 0, EX: 0, SE: 0` |
| **Positive Masks by Class** | `MA: 570, HE: 601, EX: 486, SE: 239` (`1,896` total) | `MA: 81, HE: 80, EX: 81, SE: 40` (`282` total) |
| **Source Mask Channel Counts** | `{1: 3028}` | `{1: 281, 4: 1}` (`IDRiD_81_EX.tif` is RGBA) |
| **Source Mask Pixel Encodings** | `[0, 255]` (`1,896` files), `[0]` (`1,132` files) | `[0, 1]` (`281` files), `[0, 255]` (`1` RGBA file) |
| **Samples w/ Multi-Lesion Overlap** | `141 / 757` (`18.6%`) | `30 / 81` (`37.0%`) |
| **Unreadable / Dimension Mismatches** | `0 / 0` | `0 / 0` |
| **FOV Cropped vs. Full-Size** | `654` cropped, `103` full-size (`fallback=0`) | `81` cropped, `0` full-size (`fallback=0`) |
| **Lesion Pixels Outside FOV Crop** | `0` (`1` boundary-expanded) | `0` (`1` boundary-expanded) |
| **Mean FOV Area Retention Ratio** | `0.7377` (`26.23%` background removed) | `0.8289` (`17.11%` background removed) |

---

## 2. Canonical Four-Channel Mask Construction (`src/data/masks.py`)

- **Canonical Shape & Channel Order**: Every sample's ground-truth segmentation mask is constructed with shape `(H, W, 4)`, dtype `np.uint8`, binary values `{0, 1}`, and fixed channel order `["MA", "HE", "EX", "SE"]`:
  - Channel `0`: `MA` (Microaneurysms)
  - Channel `1`: `HE` (Haemorrhages)
  - Channel `2`: `EX` (Hard Exudates)
  - Channel `3`: `SE` (Soft Exudates)
- **Independent Multi-Label Channels**: Channels are never collapsed into a single categorical map. Across the real datasets, `141 / 757` DDR cases and `30 / 81` IDRiD cases contain spatial pixels where $\ge 2$ lesion classes overlap simultaneously.
- **Optic Disc (`OD`) Exclusion**: IDRiD's `5. Optic Disc` (`81` files) is an anatomical structure and is excluded from the 4-channel lesion tensor.
- **Absent-Lesion & Encoding Harmonization**:
  - **Missing Files (IDRiD)**: Omitted lesion files (`1` `HE` mask in `IDRiD_28` and `41` `SE` masks) are synthesized as all-zero `(H, W)` channels and recorded in `missing_lesion_masks` and `synthesized_zero_channels`.
  - **Explicit All-Zero Files (DDR)**: DDR's `1,132` all-zero `.tif` masks are distinguished from omitted files via `empty_lesion_masks` vs. `positive_lesion_masks`.
  - **Multichannel RGBA Mask (`IDRiD_81_EX.tif`)**: Empirical channel inspection showed `IDRiD_81_EX.tif` has shape `(2848, 4288, 4)` (`RGBA`) with `R in {0, 255}` (`312,918` positive pixels), `G = 0`, `B = 0`, and opaque `A = 255`. `src/data/masks.py` inspects color channels separately from alpha and verifies spatial agreement across active color channels before thresholding `(plane > 0).astype(np.uint8)`.

---

## 3. Retinal Field-of-View (FOV) Detection & Cropping (`src/data/fov.py`)

- **Detection Method**: Computes per-pixel maximum RGB intensity `image[..., :3].max(axis=-1)` and thresholds `intensity > 10` (`DEFAULT_FOV_THRESHOLD = 10`) to isolate the circular retinal disk while ignoring low-amplitude JPEG compression noise ($\le 9$) in black corners.
- **Synchronized Image & Mask Slicing**: Image and canonical `(H, W, 4)` mask are sliced with identical integer coordinates `[y_min:y_max, x_min:x_max]` without interpolation.
- **Boundary Lesion Protection**: When `preserve_boundary_lesions=True`, if any positive lesion annotation lies on a dark peripheral border outside the initial intensity box (`1` case in DDR and `1` case in IDRiD), the shared crop box expands to include all positive mask pixels (`lesion_pixels_outside_fov = 0` across all `838` samples).
- **Fallback Behavior**: If an image has no foreground pixels above threshold or a degenerate crop area ($< 1\%$ of image area), FOV detection falls back deterministically to the full image `(0, H, 0, W)` with `fallback_used=True`.

---

## 4. PyTorch Dataset, Transforms, DataLoader & Federated Client Partitions (Phase 1, Task 4)

- **PyTorch Dataset (`src/data/dataset.py`)**: `RetinalLesionSegmentationDataset` reads manifest records, applies synchronized FOV cropping and Albumentations transforms, and yields CPU `torch.float32` image tensors `(3, H, W)` and binary `{0, 1}` mask tensors `(4, H, W)` in `[MA, HE, EX, SE]` channel order.
- **Transforms (`src/data/transforms.py`)**: Uses Albumentations `2.0.8` with bilinear RGB resizing (`cv2.INTER_LINEAR`) and nearest-neighbor mask resizing (`cv2.INTER_NEAREST`) to preserve binary `{0, 1}` mask values. Training splits apply conservative geometric flips/90° rotations jointly to image and 4-channel mask plus image-only `ColorJitter`; validation/test splits are strictly deterministic.
- **Federated Client Partitions (`src/data/partitioning.py`, `data/partitions/`)**: Partitions **only** the official `train` split (`383` DDR, `54` IDRiD, `437` combined) while holding out `valid` (`149` DDR) and `test` (`225` DDR, `27` IDRiD, `252` combined) with `0` split leakage and `0` lost samples:

| Partition Manifest (`data/partitions/`) | Strategy | Total Train | `client_0` | `client_1` | `client_2` | `client_3` | Held-Out Valid / Test |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `ddr_iid_4clients_seed42.json` | `iid` | `383` | `96` | `96` | `96` | `95` | `149` / `225` |
| `ddr_dirichlet_a0.5_4clients_seed42.json` | `dirichlet` ($\alpha=0.5$) | `383` | `173` | `47` | `119` | `44` | `149` / `225` |
| `idrid_iid_4clients_seed42.json` | `iid` | `54` | `14` | `14` | `13` | `13` | `0` / `27` |
| `idrid_dirichlet_a0.5_4clients_seed42.json` | `dirichlet` ($\alpha=0.5$) | `54` | `2` | `9` | `11` | `32` | `0` / `27` |
| `combined_iid_4clients_seed42.json` | `iid` | `437` | `110` | `109` | `109` | `109` | `149` / `252` |
| `combined_dirichlet_a0.5_4clients_seed42.json` | `dirichlet` ($\alpha=0.5$) | `437` | `180` | `63` | `136` | `58` | `149` / `252` |
| `combined_cross_dataset_4clients_seed42.json` | `cross_dataset` | `437` | `128` (DDR) | `128` (DDR) | `127` (DDR) | `54` (IDRiD) | `149` / `252` |


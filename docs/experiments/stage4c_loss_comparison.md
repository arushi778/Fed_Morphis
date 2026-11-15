# Stage 4C — Controlled Loss-Function Comparison (E0–E4)

This document records the experimental protocol, validation results, and exact reproducibility details for the Stage 4C controlled loss-function experiments on the combined **DDR + IDRiD** retinal lesion segmentation dataset.

## A. Experimental Protocol

- **Task**: Multi-label four-class retinal lesion segmentation (`[MA, HE, EX, SE]`, ResNet50 U-Net).
- **Dataset**: Combined `DDR + IDRiD` manifest (`data/processed/manifests/combined_manifest.json`).
- **Dataset split**: `427` training / `159` validation / `252` held-out test (`838` total records):
  - **DDR**: `383` train / `149` official validation / `225` held-out test.
  - **IDRiD**: `44` train / `10` carved validation / `27` official held-out test.
  - **Ten carved IDRiD validation IDs** (`split_seed=42`, `val_carve_ratio=0.2`): `IDRiD_06`, `IDRiD_08`, `IDRiD_18`, `IDRiD_24`, `IDRiD_28`, `IDRiD_35`, `IDRiD_38`, `IDRiD_41`, `IDRiD_52`, `IDRiD_54`.
  - **Held-out test isolation**: All `27` official IDRiD test images (and `225` DDR test images) remain untouched and un-evaluated (`test_split_evaluated: false`).
- **Controlled training hyperparameters**:
  - Seed: `42`
  - Epochs: `50`
  - Batch size: `4`
  - Input image size: `512 × 512` (resized full-image FOV-cropped training; `patch_training.enabled: false`)
  - Mixed precision (AMP): enabled (`true`)
  - Optimizer: `AdamW` (`learning_rate: 1e-4`, `weight_decay: 1e-4`, `scheduler: null`)
  - Checkpoint selection metric: `val_dice_macro` (`max`)
- **Common initialization checkpoint** (shared across all experiments; not tracked in Git):
  - Path: `checkpoints/loss_experiments/initialization/unet_seed42_init.pt`
  - File SHA-256: `7201a852b4820ea949b111d994df98242f60aeeb43222e7ce384325636d8d682`
  - Model-state SHA-256: `a23fdfd3f42b03f4dd70b2022980989a31ddbec23413c938444a77ec093217e2`

## B. Validation Results Summary

| Experiment | Loss | Best Epoch | Val Macro Dice | Val Macro IoU | Val Macro AP | Val Macro AUPR (Trap.) |
|---|---|---:|---:|---:|---:|---:|
| **E0** | BCE + Tversky ($\alpha=0.3, \beta=0.7$) | 47 | 0.4989 | 0.3418 | 0.3692 | 0.3730 |
| **E1** | Weighted BCE + Tversky (`pos_weight`) | 50 | 0.5564 | 0.3959 | 0.5327 | 0.5340 |
| **E2** | **BCE + Dice (unweighted)** | 42 | **0.5761** | **0.4143** | **0.5731** | **0.5731** |
| **E2b** | BCE + Balanced Dice (`class_weights`) | 42 | 0.5701 | 0.4084 | 0.5697 | 0.5697 |
| **E3** | BCE + Focal Tversky ($\gamma=0.75$) | 38 | 0.5066 | 0.3479 | 0.3674 | 0.3690 |
| **E4** | BCE + Focal Tversky ($\gamma=4/3$) | 47 | 0.5016 | 0.3473 | 0.3768 | 0.3808 |

- **Leading validation candidate**: **E2 (`bce_dice`)** achieved the highest validation macro Dice (`0.5761`), IoU (`0.4143`), AP (`0.5731`), and trapezoidal AUPR (`0.5731`).
- **Secondary comparator**: **E2b (`bce_balanced_dice`)** improved **HE** (`0.6231` Dice, `0.6477` AUPR vs E2 `0.5868` Dice, `0.5906` AUPR) and is retained as a useful HE-specific comparator, though it slightly trailed E2 on macro metrics.
- **Scope caveat**: These are **single-seed (`seed=42`) validation results** and do not constitute proof of statistical superiority or final held-out test performance.

## C. Reproducibility Command

`scripts/train_baseline.py` defaults `--manifest` to `data/processed/manifests/ddr_manifest.json` (DDR-only). All Stage 4C experiments **must explicitly pass** `--manifest data/processed/manifests/combined_manifest.json`; omitting this flag will not reproduce the combined DDR + IDRiD experiment.

Representative command for **E2 (`bce_dice`)**:

```bash
python scripts/train_baseline.py \
  --project-config configs/project.yaml \
  --dataset-config configs/loss_experiments/dataset_stage4_b4.yaml \
  --model-config configs/loss_experiments/E2_dice.yaml \
  --manifest data/processed/manifests/combined_manifest.json \
  --init-checkpoint checkpoints/loss_experiments/initialization/unet_seed42_init.pt \
  --output-dir checkpoints/loss_experiments/E2_dice \
  --seed 42 \
  --epochs 50 \
  --batch-size 4 \
  --image-size 512 \
  --device cuda
```

Shared configs live under `configs/loss_experiments/`:
- `dataset_stage4_b4.yaml`
- `E0_baseline.yaml`, `E1_weighted_bce.yaml`, `E2_dice.yaml`, `E2b_balanced_dice.yaml`, `E3_focal_gamma075.yaml`, `E4_focal_gamma133.yaml`, `E5_combined_conditional.yaml`

## D. Experiment Status

- **E0–E4**: 50-epoch training and validation AUPR evaluation completed.
- **E5 (`E5_combined_conditional.yaml`)**: Read-only audit completed (`CONDITIONAL PASS`), intentionally **not trained** due to weak scientific justification given E0–E4 evidence.
- **Official held-out test set**: **Not evaluated** (`test_split_evaluated: false` across all experiments).
- **Threshold tuning / test-based selection**: **Not performed**.

## E. Limitations

- **Single random seed (`seed=42`)**: Run-to-run variance is unquantified.
- **Validation sample size (`159` records)**: Includes only `10` carved IDRiD validation images alongside `149` DDR validation images.
- **Full-image resizing (`512 × 512`)**: Downsampling can attenuate small lesions such as Microaneurysms (`MA`) compared to high-resolution patch/sliding-window inference.
- **Small margin between E2 and E2b**: `0.0060` macro Dice and `0.0034` macro AUPR differences may fall within single-seed variance.
- **No calibration claim**: AP and trapezoidal AUPR measure threshold-free ranking quality, not probability calibration.

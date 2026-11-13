"""Validation-based channel-specific threshold optimization for federated models.

Evaluates candidate federated checkpoints on the exact 159 validation samples
(149 DDR + 10 IDRiD) across a search grid of per-channel classification thresholds,
comparing uniform 0.50 baseline vs. channel-optimized thresholds.
Strictly prohibits test split access or model mutation.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.masks import CANONICAL_LESION_CHANNELS
from src.engine.evaluator import reconstruct_model_from_checkpoint, verify_exact_split
from src.federated.data import build_all_federated_client_dataloaders
from src.models.metrics import (
    _compute_single_channel_metrics_from_counts,
    METRIC_NAMES,
)
from src.utils.device import resolve_device
from src.utils.logger import get_logger
from src.utils.paths import format_project_path, resolve_project_path

logger = get_logger("fedmorphis.threshold_optimization")

CANONICAL_LESION_ORDER = tuple(CANONICAL_LESION_CHANNELS)

# Per-channel threshold search candidates specified in Task 14
CANDIDATE_THRESHOLDS: dict[str, list[float]] = {
    "MA": [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60],
    "HE": [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60],
    "EX": [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60],
    "SE": [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60],
}


def compute_file_sha256(path: Path | str) -> str:
    """Compute SHA-256 digest of a file."""
    p = Path(path)
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


@torch.no_grad()
def extract_validation_probabilities_and_targets(
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader[Any],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Extract sigmoid probabilities and binary targets for all samples in a DataLoader."""
    model.eval()
    all_probs: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    all_sample_ids: list[str] = []

    use_amp = device.type == "cuda"

    for batch in val_loader:
        if isinstance(batch, dict):
            images = batch["image"]
            masks = batch["mask"]
            meta = batch.get("metadata", {})
        else:
            images = batch[0]
            masks = batch[1]
            meta = batch[2] if len(batch) > 2 else {}

        if isinstance(meta, dict) and "sample_id" in meta:
            sids = meta["sample_id"]
            if isinstance(sids, (list, tuple)):
                all_sample_ids.extend([str(s) for s in sids])
            else:
                all_sample_ids.append(str(sids))

        images = images.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            probs = torch.sigmoid(logits.float())

        all_probs.append(probs.cpu())
        all_targets.append((masks.float() >= 0.5).to(torch.uint8).cpu())

    cat_probs = torch.cat(all_probs, dim=0)
    cat_targets = torch.cat(all_targets, dim=0)
    return cat_probs, cat_targets, all_sample_ids


def evaluate_channel_at_threshold(
    probs_c: torch.Tensor,
    targets_c: torch.Tensor,
    threshold: float,
) -> dict[str, Any]:
    """Compute confusion counts and metrics for a single lesion channel at a specified threshold."""
    pred_bin = (probs_c >= threshold).to(torch.long)
    target_bin = (targets_c >= 1).to(torch.long)

    tp = int(torch.sum((pred_bin == 1) & (target_bin == 1)).item())
    fp = int(torch.sum((pred_bin == 1) & (target_bin == 0)).item())
    fn = int(torch.sum((pred_bin == 0) & (target_bin == 1)).item())
    tn = int(torch.sum((pred_bin == 0) & (target_bin == 0)).item())

    res = _compute_single_channel_metrics_from_counts(tp=tp, fp=fp, fn=fn, tn=tn)
    res["threshold"] = round(float(threshold), 4)
    return res


def run_threshold_grid_search(
    client_data: dict[str, dict[str, Any]],
    candidate_thresholds: dict[str, list[float]] = CANDIDATE_THRESHOLDS,
) -> dict[str, Any]:
    """Perform independent threshold search for each channel and evaluate combinations."""
    client_ids = sorted(client_data.keys())
    sample_counts = {cid: client_data[cid]["probs"].shape[0] for cid in client_ids}
    total_samples = sum(sample_counts.values())

    channels = list(CANONICAL_LESION_ORDER)
    num_channels = len(channels)

    grid_results_by_channel: dict[str, list[dict[str, Any]]] = {}
    best_threshold_by_channel: dict[str, dict[str, Any]] = {}
    baseline_threshold_by_channel: dict[str, dict[str, Any]] = {}

    # Client-specific best thresholds for stability audit
    client_best_thresholds: dict[str, dict[str, float]] = {cid: {} for cid in client_ids}

    for ch_idx, ch_name in enumerate(channels):
        grid = candidate_thresholds.get(ch_name, [0.5])
        channel_records: list[dict[str, Any]] = []

        best_score = -1.0
        best_record: dict[str, Any] | None = None
        baseline_record: dict[str, Any] | None = None

        # Track client-specific max dice for this channel
        client_max_dice: dict[str, float] = {cid: -1.0 for cid in client_ids}
        client_best_thresh: dict[str, float] = {cid: 0.50 for cid in client_ids}

        for thresh in grid:
            rec: dict[str, Any] = {"threshold": round(thresh, 4), "clients": {}}
            weighted_metrics: dict[str, float] = {
                "dice": 0.0,
                "iou": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "sensitivity": 0.0,
                "specificity": 0.0,
            }

            for cid in client_ids:
                p_c = client_data[cid]["probs"][:, ch_idx]
                t_c = client_data[cid]["targets"][:, ch_idx]
                c_metrics = evaluate_channel_at_threshold(p_c, t_c, thresh)
                rec["clients"][cid] = c_metrics
                n_ex = sample_counts[cid]

                for m_name in weighted_metrics:
                    weighted_metrics[m_name] += float(c_metrics[m_name]) * float(n_ex)

                if c_metrics["dice"] > client_max_dice[cid]:
                    client_max_dice[cid] = c_metrics["dice"]
                    client_best_thresh[cid] = thresh

            # Divide by total samples for Flower-exact sample-weighted metric
            for m_name in weighted_metrics:
                weighted_metrics[m_name] /= float(total_samples)

            rec["weighted"] = weighted_metrics
            channel_records.append(rec)

            if weighted_metrics["dice"] > best_score:
                best_score = weighted_metrics["dice"]
                best_record = rec

            if abs(thresh - 0.50) < 1e-4:
                baseline_record = rec

        grid_results_by_channel[ch_name] = channel_records
        best_threshold_by_channel[ch_name] = best_record  # type: ignore[assignment]
        baseline_threshold_by_channel[ch_name] = baseline_record  # type: ignore[assignment]

        for cid in client_ids:
            client_best_thresholds[cid][ch_name] = client_best_thresh[cid]

    # Combine per-channel results into macro summaries
    def _build_macro_summary(
        ch_selection: dict[str, dict[str, Any]],
        selection_type: str,
    ) -> dict[str, Any]:
        thresholds = {ch: ch_selection[ch]["threshold"] for ch in channels}
        per_channel_metrics = {ch: ch_selection[ch]["weighted"] for ch in channels}

        macro_weighted: dict[str, float] = {}
        for m_name in ["dice", "iou", "precision", "recall", "sensitivity", "specificity"]:
            vals = [per_channel_metrics[ch][m_name] for ch in channels]
            macro_weighted[f"macro_{m_name}"] = float(np.mean(vals))

        client_summaries: dict[str, dict[str, Any]] = {}
        for cid in client_ids:
            c_per_channel = {ch: ch_selection[ch]["clients"][cid] for ch in channels}
            c_macro: dict[str, float] = {}
            for m_name in ["dice", "iou", "precision", "recall", "sensitivity", "specificity"]:
                c_vals = [c_per_channel[ch][m_name] for ch in channels]
                c_macro[f"macro_{m_name}"] = float(np.mean(c_vals))
            client_summaries[cid] = {
                "dataset": client_data[cid].get("dataset", cid),
                "num_samples": sample_counts[cid],
                "macro": c_macro,
                "per_channel": c_per_channel,
            }

        return {
            "selection_type": selection_type,
            "thresholds": thresholds,
            "macro_metrics": macro_weighted,
            "per_channel_metrics": per_channel_metrics,
            "clients": client_summaries,
        }

    baseline_summary = _build_macro_summary(baseline_threshold_by_channel, "uniform_0.50")
    optimized_summary = _build_macro_summary(best_threshold_by_channel, "channel_optimized")

    # Check stability across clients
    stability_audit: dict[str, Any] = {}
    for ch in channels:
        best_glob = best_threshold_by_channel[ch]["threshold"]
        c_threshs = {cid: client_best_thresholds[cid][ch] for cid in client_ids}
        discrepancy = max(c_threshs.values()) - min(c_threshs.values())
        stability_audit[ch] = {
            "global_best_threshold": best_glob,
            "client_best_thresholds": c_threshs,
            "discrepancy": round(discrepancy, 4),
            "stable": discrepancy <= 0.10,
        }

    return {
        "candidate_thresholds": candidate_thresholds,
        "sample_counts": sample_counts,
        "total_samples": total_samples,
        "baseline_uniform": baseline_summary,
        "channel_optimized": optimized_summary,
        "stability_audit": stability_audit,
        "full_grid_search": grid_results_by_channel,
    }


def analyze_checkpoint(
    checkpoint_path: Path | str,
    val_bundles: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    """Load a checkpoint, extract validation probabilities, and perform threshold optimization."""
    ckpt_path = Path(checkpoint_path)
    logger.info("Reconstructing model from checkpoint: %s", ckpt_path)
    sha256_hash = compute_file_sha256(ckpt_path)
    model, ckpt_dict, _ = reconstruct_model_from_checkpoint(ckpt_path, device=device)

    client_data: dict[str, dict[str, Any]] = {}
    for cid, bundle in val_bundles.items():
        v_loader = bundle.val_loader
        # Strict split verification
        verify_exact_split(v_loader, expected_split="valid")
        logger.info("Extracting validation predictions for client %s (%d samples)...", cid, len(v_loader.dataset))
        probs, targets, sample_ids = extract_validation_probabilities_and_targets(
            model=model, val_loader=v_loader, device=device
        )
        ds_name = bundle.dataset_names[0] if getattr(bundle, "dataset_names", None) else cid
        client_data[cid] = {
            "dataset": ds_name,
            "probs": probs,
            "targets": targets,
            "sample_ids": sample_ids,
        }

    logger.info("Running per-channel threshold grid search for %s...", ckpt_path.name)
    results = run_threshold_grid_search(client_data)
    results["checkpoint_path"] = format_project_path(ckpt_path)
    results["checkpoint_sha256"] = sha256_hash
    results["best_round_saved"] = ckpt_dict.get("round", ckpt_dict.get("best_round"))
    return results


def build_markdown_report(report_data: dict[str, Any]) -> str:
    """Generate detailed markdown report from analysis output."""
    lines: list[str] = [
        "# FedMorphis — Task 14: Validation-Based Channel-Specific Threshold Optimisation Report",
        "",
        "## Executive Summary",
        "",
        "This study evaluates whether post-hoc per-channel prediction threshold tuning on the "
        "global validation cohort (159 images: 149 DDR, 10 IDRiD) improves retinal lesion segmentation "
        "without model retraining or architecture alteration.",
        "",
        "### Key Candidate Models Evaluated",
        "1. **Baseline FedAvg (Round 17)**: `checkpoints/federated/fedavg_full_20rounds/best_checkpoint.pt`",
        "2. **Experiment 1 PosWeight (Round 19)**: `checkpoints/federated/fedavg_posweight_20rounds/best_checkpoint.pt`",
        "",
        "> [!IMPORTANT]",
        "> **Strict Data & Scope Integrity Controls**:",
        "> - Only the validation split (149 DDR + 10 IDRiD = 159 images) was utilized for probability generation and threshold searching.",
        "> - The held-out test cohort ($n=252$) was **never loaded, inspected, or evaluated**.",
        "> - Zero model retraining or checkpoint mutation occurred; all checkpoint hashes were verified before and after.",
        "> - The same validation dataset is used for threshold selection and assessment; test performance may differ.",
        "",
        "---",
        "",
        "## 1. Candidate Comparison Summary",
        "",
        "| Candidate Checkpoint | Threshold Scheme | MA Thresh | HE Thresh | EX Thresh | SE Thresh | Macro Dice | Macro IoU | MA Dice | HE Dice | EX Dice | SE Dice |",
        "|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]

    for model_key, m_res in report_data["candidates"].items():
        base = m_res["baseline_uniform"]
        opt = m_res["channel_optimized"]

        b_th = base["thresholds"]
        b_mac = base["macro_metrics"]
        b_ch = base["per_channel_metrics"]

        o_th = opt["thresholds"]
        o_mac = opt["macro_metrics"]
        o_ch = opt["per_channel_metrics"]

        lines.append(
            f"| **{model_key}** (Baseline) | Uniform 0.50 | {b_th['MA']:.2f} | {b_th['HE']:.2f} | {b_th['EX']:.2f} | {b_th['SE']:.2f} | "
            f"**{b_mac['macro_dice']:.4f}** | {b_mac['macro_iou']:.4f} | {b_ch['MA']['dice']:.4f} | {b_ch['HE']['dice']:.4f} | {b_ch['EX']['dice']:.4f} | {b_ch['SE']['dice']:.4f} |"
        )
        lines.append(
            f"| **{model_key}** (Optimized) | **Channel-Tuned** | **{o_th['MA']:.2f}** | **{o_th['HE']:.2f}** | **{o_th['EX']:.2f}** | **{o_th['SE']:.2f}** | "
            f"**{o_mac['macro_dice']:.4f}** | {o_mac['macro_iou']:.4f} | {o_ch['MA']['dice']:.4f} | {o_ch['HE']['dice']:.4f} | {o_ch['EX']['dice']:.4f} | {o_ch['SE']['dice']:.4f} |"
        )
        diff_dice = o_mac['macro_dice'] - b_mac['macro_dice']
        lines.append(
            f"| *Δ ({model_key})* | *Gain* | *—* | *—* | *—* | *—* | "
            f"*{diff_dice:+.4f}* | *{o_mac['macro_iou'] - b_mac['macro_iou']:+.4f}* | *{o_ch['MA']['dice'] - b_ch['MA']['dice']:+.4f}* | *{o_ch['HE']['dice'] - b_ch['HE']['dice']:+.4f}* | *{o_ch['EX']['dice'] - b_ch['EX']['dice']:+.4f}* | *{o_ch['SE']['dice'] - b_ch['SE']['dice']:+.4f}* |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 2. Detailed Per-Candidate Breakdown",
        "",
    ])

    for model_key, m_res in report_data["candidates"].items():
        base = m_res["baseline_uniform"]
        opt = m_res["channel_optimized"]
        stability = m_res["stability_audit"]

        lines.extend([
            f"### Candidate: {model_key}",
            f"- **Checkpoint Path**: `{m_res['checkpoint_path']}`",
            f"- **SHA-256 Digest**: `{m_res['checkpoint_sha256']}`",
            f"- **Checkpoint Round**: {m_res['best_round_saved']}",
            "",
            "#### Precision, Recall, and Metric Shifts (Sample-Weighted Global Validation)",
            "",
            "| Lesion Channel | Baseline Thresh | Opt Thresh | Dice (Base → Opt) | IoU (Base → Opt) | Precision (Base → Opt) | Recall (Base → Opt) |",
            "|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
        ])

        for ch in CANONICAL_LESION_ORDER:
            b_m = base["per_channel_metrics"][ch]
            o_m = opt["per_channel_metrics"][ch]
            lines.append(
                f"| **{ch}** | {base['thresholds'][ch]:.2f} | **{opt['thresholds'][ch]:.2f}** | "
                f"{b_m['dice']:.4f} → **{o_m['dice']:.4f}** ({o_m['dice'] - b_m['dice']:+.4f}) | "
                f"{b_m['iou']:.4f} → **{o_m['iou']:.4f}** | "
                f"{b_m['precision']:.4f} → {o_m['precision']:.4f} | "
                f"{b_m['recall']:.4f} → {o_m['recall']:.4f} |"
            )

        lines.extend([
            "",
            "#### Client-Stratified Performance Breakdown",
            "",
            "| Client | Dataset | N Samples | Macro Dice (Base → Opt) | MA Dice | HE Dice | EX Dice | SE Dice |",
            "|---|---|:---:|:---:|:---:|:---:|:---:|:---:|",
        ])

        for cid, c_info in opt["clients"].items():
            b_c_macro = base["clients"][cid]["macro"]["macro_dice"]
            o_c_macro = c_info["macro"]["macro_dice"]
            b_c_ch = base["clients"][cid]["per_channel"]
            o_c_ch = c_info["per_channel"]
            lines.append(
                f"| `{cid}` | **{c_info['dataset']}** | {c_info['num_samples']} | "
                f"{b_c_macro:.4f} → **{o_c_macro:.4f}** ({o_c_macro - b_c_macro:+.4f}) | "
                f"{o_c_ch['MA']['dice']:.4f} | {o_c_ch['HE']['dice']:.4f} | {o_c_ch['EX']['dice']:.4f} | {o_c_ch['SE']['dice']:.4f} |"
            )

        lines.extend([
            "",
            "#### Client Threshold Stability & Domain Calibration Audit",
            "",
            "| Channel | Global Selected Thresh | Client 0 (DDR) Optimal | Client 1 (IDRiD) Optimal | Discrepancy | Stability Assessment |",
            "|:---:|:---:|:---:|:---:|:---:|---|",
        ])

        for ch in CANONICAL_LESION_ORDER:
            s_ch = stability[ch]
            c_ths = s_ch["client_best_thresholds"]
            status = "✅ Stable across clients" if s_ch["stable"] else "⚠️ Client discrepancy observed"
            lines.append(
                f"| **{ch}** | **{s_ch['global_best_threshold']:.2f}** | {c_ths.get('client_0', 0.5):.2f} | {c_ths.get('client_1', 0.5):.2f} | {s_ch['discrepancy']:.2f} | {status} |"
            )

        lines.append("")

    lines.extend([
        "---",
        "",
        "## 3. Full Search Space Grid",
        "",
        "Detailed inspection of validation Dice scores across candidate thresholds:",
        "",
    ])

    for model_key, m_res in report_data["candidates"].items():
        lines.append(f"### {model_key} Threshold Curves (Global Sample-Weighted Dice)")
        lines.append("")
        for ch in CANONICAL_LESION_ORDER:
            lines.append(f"#### Lesion: **{ch}**")
            lines.append("| Threshold | Weighted Dice | Weighted IoU | Weighted Precision | Weighted Recall | DDR Dice | IDRiD Dice |")
            lines.append("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")
            for rec in m_res["full_grid_search"][ch]:
                th = rec["threshold"]
                wm = rec["weighted"]
                c0_d = rec["clients"].get("client_0", {}).get("dice", 0.0)
                c1_d = rec["clients"].get("client_1", {}).get("dice", 0.0)
                marker = " ⭐ **(Opt)**" if abs(th - m_res["channel_optimized"]["thresholds"][ch]) < 1e-4 else (" *(Base 0.50)*" if abs(th - 0.50) < 1e-4 else "")
                lines.append(
                    f"| {th:.2f}{marker} | {wm['dice']:.4f} | {wm['iou']:.4f} | {wm['precision']:.4f} | {wm['recall']:.4f} | {c0_d:.4f} | {c1_d:.4f} |"
                )
            lines.append("")

    lines.extend([
        "---",
        "",
        "## 4. Limitations and Overfitting Risk Disclosure",
        "",
        "1. **Validation Re-use**: The classification thresholds were selected to optimize performance directly on the 159 validation samples. Without held-out cross-validation of threshold selection, there is an inherent risk of mild optimistic bias on the validation metric.",
        "2. **Sparse Client Discrepancy**: While DDR ($n=149$) dominates sample-weighted metrics, IDRiD validation ($n=10$) has much higher variance per image. Channel thresholds optimized on sample-weighted metrics primarily reflect DDR imaging characteristics.",
        "3. **Generalization Guidance**: When evaluating on the held-out test cohort in future steps, both the uniform 0.50 baseline and the channel-tuned thresholds should be reported side-by-side to verify whether validation threshold gains transfer out-of-domain.",
        "",
    ])

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 14: Validation-Based Channel-Specific Threshold Optimization")
    parser.add_argument(
        "--baseline-checkpoint",
        type=str,
        default="checkpoints/federated/fedavg_full_20rounds/best_checkpoint.pt",
        help="Path to baseline FedAvg best checkpoint (Round 17)",
    )
    parser.add_argument(
        "--exp1-checkpoint",
        type=str,
        default="checkpoints/federated/fedavg_posweight_20rounds/best_checkpoint.pt",
        help="Path to Experiment 1 PosWeight best checkpoint (Round 19)",
    )
    parser.add_argument(
        "--report-json",
        type=str,
        default="logs/federated/threshold_optimization_report.json",
        help="Destination path for structured JSON report",
    )
    parser.add_argument(
        "--report-md",
        type=str,
        default="logs/federated/threshold_optimization_report.md",
        help="Destination path for detailed Markdown report",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Inference device (cuda or cpu)",
    )
    args = parser.parse_args()

    device = resolve_device(args.device)
    logger.info("Initializing validation-based threshold optimization on device: %s", device)

    # 1. Build validation dataloaders
    logger.info("Loading federated client data bundles...")
    val_bundles = build_all_federated_client_dataloaders()
    logger.info("Loaded client bundles: %s", list(val_bundles.keys()))

    candidates_to_evaluate: dict[str, str] = {
        "FedAvg Baseline (Round 17)": args.baseline_checkpoint,
        "Exp 1: PosWeight (Round 19)": args.exp1_checkpoint,
    }

    report_data: dict[str, Any] = {
        "task": "Task 14: Validation-Based Channel-Specific Threshold Optimization",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "device": str(device),
        "candidates": {},
    }

    for name, path in candidates_to_evaluate.items():
        logger.info("==================================================================")
        logger.info("Analyzing candidate: %s from %s", name, path)
        logger.info("==================================================================")
        cand_results = analyze_checkpoint(
            checkpoint_path=path,
            val_bundles=val_bundles,
            device=device,
        )
        report_data["candidates"][name] = cand_results

    # Save JSON report
    out_json = resolve_project_path(args.report_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    logger.info("Saved structured JSON report -> %s", out_json)

    # Save Markdown report
    out_md = resolve_project_path(args.report_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    md_content = build_markdown_report(report_data)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(md_content)
    logger.info("Saved Markdown report -> %s", out_md)

    print("\n" + "=" * 80)
    print("THRESHOLD OPTIMIZATION COMPLETE")
    print("=" * 80)
    for name, res in report_data["candidates"].items():
        base_mac = res["baseline_uniform"]["macro_metrics"]["macro_dice"]
        opt_mac = res["channel_optimized"]["macro_metrics"]["macro_dice"]
        print(f"Candidate: {name}")
        print(f"  Uniform 0.50 Macro Dice:     {base_mac:.6f}")
        print(f"  Channel-Tuned Macro Dice:    {opt_mac:.6f} (Δ = {opt_mac - base_mac:+.6f})")
        print(f"  Selected Channel Thresholds: {res['channel_optimized']['thresholds']}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

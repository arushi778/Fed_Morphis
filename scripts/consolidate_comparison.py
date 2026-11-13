"""CLI script to consolidate evaluation reports and generate cross-method comparison artifacts.

Loads the evaluation test reports for:
1. Centralized Supervised Baseline
2. SimCLR Pretrained + Fine-Tuned Model
3. Flower FedAvg Global Checkpoint

Validates protocol integrity across all 252 held-out test samples and writes:
- logs/federated/consolidated_comparison_report.json
- logs/federated/consolidated_comparison_report.md
- Optional 5-panel visual comparison figures in outputs/federated/
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn

from src.data.dataset import RetinalLesionSegmentationDataset
from src.data.manifests import load_manifest_json
from src.data.masks import CANONICAL_LESION_CHANNELS
from src.data.transforms import build_segmentation_transforms
from src.engine.evaluator import reconstruct_model_from_checkpoint
from src.federated.comparison import (
    DEFAULT_REPORT_PATHS,
    generate_consolidated_reports,
    render_multi_model_comparison_panel,
)
from src.utils.config import load_yaml_config
from src.utils.device import resolve_device
from src.utils.logger import get_logger
from src.utils.paths import format_project_path, resolve_project_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for consolidated baseline comparison."""
    parser = argparse.ArgumentParser(
        description="Consolidate held-out test evaluations across Centralized, SimCLR, and Federated models."
    )
    parser.add_argument(
        "--supervised-report",
        type=str,
        default=DEFAULT_REPORT_PATHS["supervised_baseline"],
        help="Path to centralized supervised test report JSON.",
    )
    parser.add_argument(
        "--simclr-report",
        type=str,
        default=DEFAULT_REPORT_PATHS["simclr_fine_tuned"],
        help="Path to SimCLR fine-tuned test report JSON.",
    )
    parser.add_argument(
        "--federated-report",
        type=str,
        default=DEFAULT_REPORT_PATHS["federated_fedavg"],
        help="Path to Flower FedAvg test report JSON.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="logs/federated/consolidated_comparison_report.json",
        help="Destination path for consolidated JSON report.",
    )
    parser.add_argument(
        "--output-markdown",
        type=str,
        default="logs/federated/consolidated_comparison_report.md",
        help="Destination path for consolidated Markdown report.",
    )
    parser.add_argument(
        "--visualize-comparison",
        action="store_true",
        help="Generate side-by-side 5-panel multi-model qualitative comparison PNGs.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Execution device for visualization inference ('cuda', 'cpu', or 'auto').",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def _generate_multi_model_visualizations(
    output_dir: Path,
    device: torch.device,
) -> list[str]:
    """Generate side-by-side 5-panel comparison figures on representative DDR and IDRiD test samples."""
    logger = get_logger("fedmorphis.consolidate")
    output_dir.mkdir(parents=True, exist_ok=True)

    model_ckpts = {
        "supervised_baseline": resolve_project_path("checkpoints/segmentation/best_checkpoint.pt"),
        "simclr_fine_tuned": resolve_project_path("checkpoints/segmentation/simclr_finetuned_ddr/best_checkpoint.pt"),
        "federated_fedavg": resolve_project_path("checkpoints/federated/fedavg_full_20rounds/best_checkpoint.pt"),
    }

    models: dict[str, nn.Module] = {}
    for m_key, ckpt_p in model_ckpts.items():
        if not ckpt_p.is_file():
            logger.warning("Checkpoint %s not found; skipping multi-model visual comparisons.", ckpt_p)
            return []
        m, _, _ = reconstruct_model_from_checkpoint(ckpt_p, fallback_model_config="configs/model.yaml", device=device, strict=True)
        m.eval()
        models[m_key] = m

    ds_cfg = load_yaml_config("configs/dataset.yaml", validate=False)
    tform = build_segmentation_transforms(split="test", dataset_config=ds_cfg, image_size=(512, 512))

    cases_to_render = [
        ("DDR", "data/processed/manifests/ddr_manifest.json", "20170622080837660", "Complex Multi-Lesion Pathology", "multi_model_comparison_ddr.png"),
        ("IDRiD", "data/processed/manifests/idrid_manifest.json", "IDRiD_68", "Complex Multi-Lesion Pathology", "multi_model_comparison_idrid.png"),
    ]

    rendered_paths: list[str] = []
    amp_enabled = (device.type == "cuda")

    for ds_name, man_path, target_sid, cat_name, out_filename in cases_to_render:
        loaded = load_manifest_json(resolve_project_path(man_path))
        test_recs = [r for r in loaded.get("records", []) if str(r.get("split", "")).lower() == "test"]
        matched_idx = None
        for i, r in enumerate(test_recs):
            if str(r.get("sample_id", "")) == target_sid:
                matched_idx = i
                break
        if matched_idx is None:
            matched_idx = 0

        ds = RetinalLesionSegmentationDataset(
            manifest_or_records=test_recs,
            split="test",
            transform=tform,
            dataset_config=ds_cfg,
            image_size=(512, 512),
            return_metadata=True,
            cache_in_memory=False,
        )

        img_t, msk_t, meta = ds[matched_idx]
        sid = str(meta.get("sample_id", target_sid))

        preds_by_model: dict[str, torch.Tensor] = {}
        metrics_by_model: dict[str, dict[str, float]] = {}

        for m_key, model_inst in models.items():
            with torch.no_grad():
                with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                    logits = model_inst(img_t.unsqueeze(0).to(device))
                    probs = torch.sigmoid(logits[0])
                    pred_t = (probs >= 0.5).to(dtype=torch.uint8).cpu()

            preds_by_model[m_key] = pred_t

            dices = []
            for ch_idx in range(len(CANONICAL_LESION_CHANNELS)):
                gt_c = (msk_t[ch_idx] > 0).numpy()
                pr_c = (pred_t[ch_idx] > 0).numpy()
                tp = int((pr_c & gt_c).sum())
                denom = 2 * tp + int((pr_c & ~gt_c).sum()) + int((~pr_c & gt_c).sum())
                d_c = float(2.0 * tp / denom) if denom > 0 else (1.0 if not gt_c.any() and not pr_c.any() else 0.0)
                dices.append(d_c)
            metrics_by_model[m_key] = {"dice_macro": float(sum(dices) / len(dices))}

        out_path = output_dir / out_filename
        render_multi_model_comparison_panel(
            image=img_t,
            gt_mask=msk_t,
            preds_by_model=preds_by_model,
            metadata={"sample_id": sid, "dataset_name": ds_name, "category": cat_name},
            metrics_by_model=metrics_by_model,
            output_path=out_path,
        )
        rendered_paths.append(format_project_path(out_path))
        logger.info("Rendered multi-model comparison panel for %s (%s) -> %s", ds_name, sid, out_path)

    return rendered_paths


def main(argv: Sequence[str] | None = None) -> int:
    """CLI execution entry point."""
    args = parse_args(argv)
    logger = get_logger("fedmorphis.consolidate")

    report_paths = {
        "supervised_baseline": args.supervised_report,
        "simclr_fine_tuned": args.simclr_report,
        "federated_fedavg": args.federated_report,
    }

    logger.info("Generating consolidated evaluation comparison across 3 models...")
    comparison = generate_consolidated_reports(
        report_paths=report_paths,
        output_json_path=args.output_json,
        output_markdown_path=args.output_markdown,
    )

    vis_paths: list[str] = []
    if args.visualize_comparison:
        dev = resolve_device(args.device)
        logger.info("Generating multi-model comparison figures on device=%s...", dev)
        vis_paths = _generate_multi_model_visualizations(
            output_dir=resolve_project_path("outputs/federated"),
            device=dev,
        )

    # Print summary to stdout
    print("\n" + "=" * 95)
    print("FedMorphis — Consolidated Benchmark Comparison Across 252 Held-Out Test Samples")
    print("=" * 95)
    header = f"{'Methodology':<36} | {'Macro Dice':<11} | {'Macro IoU':<10} | {'Loss':<8} | {'MA':<7} | {'HE':<7} | {'EX':<7} | {'SE':<7}"
    print(header)
    print("-" * 95)

    sw = comparison["sample_weighted_overall"]
    for m_key, data in sw.items():
        pc = data["per_class"]
        print(
            f"{data['method_name']:<36} | "
            f"{data['dice_macro']:<11.4f} | "
            f"{data['iou_macro']:<10.4f} | "
            f"{data['loss']:<8.4f} | "
            f"{pc['MA']['dice']:<7.4f} | "
            f"{pc['HE']['dice']:<7.4f} | "
            f"{pc['EX']['dice']:<7.4f} | "
            f"{pc['SE']['dice']:<7.4f}"
        )
    print("-" * 95)
    print(f"JSON Report     : {comparison['json_report_path']}")
    print(f"Markdown Report : {comparison['markdown_report_path']}")
    if vis_paths:
        for vp in vis_paths:
            print(f"Visual Panel    : {vp}")
    print("=" * 95 + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


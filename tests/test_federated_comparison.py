"""Unit and regression tests for consolidated evaluation comparison across baseline models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scripts.consolidate_comparison import main as consolidate_main
from src.data.masks import CANONICAL_LESION_CHANNELS
from src.federated.comparison import (
    DEFAULT_REPORT_PATHS,
    build_consolidated_comparison,
    generate_comparison_markdown,
    generate_consolidated_reports,
    load_test_report,
    validate_test_report_integrity,
)


def test_load_and_validate_canonical_test_reports() -> None:
    """Verify that all three canonical evaluation reports load and pass integrity verification."""
    for m_key, r_path in DEFAULT_REPORT_PATHS.items():
        p = Path(r_path)
        assert p.is_file(), f"Report file missing for {m_key}: {p}"
        rep = load_test_report(p)
        assert rep["status"] == "completed"
        assert rep["split"] == "test"
        validate_test_report_integrity(rep, m_key)


def test_build_consolidated_comparison_metrics_and_channels() -> None:
    """Verify that consolidated comparison aggregates all 252 samples across canonical 4 channels."""
    reports = {k: load_test_report(v) for k, v in DEFAULT_REPORT_PATHS.items()}
    comp = build_consolidated_comparison(reports)

    assert comp["status"] == "completed"
    assert comp["sample_count_total"] == 252

    # Check sample-weighted overall
    sw = comp["sample_weighted_overall"]
    for m_key in ("supervised_baseline", "simclr_fine_tuned", "federated_fedavg"):
        assert m_key in sw
        m_data = sw[m_key]
        assert m_data["sample_count"] == 252
        assert 0.0 < m_data["dice_macro"] < 1.0
        assert 0.0 < m_data["iou_macro"] < 1.0
        assert np.isfinite(m_data["loss"])
        for ch in CANONICAL_LESION_CHANNELS:
            assert ch in m_data["per_class"]
            assert 0.0 <= m_data["per_class"][ch]["dice"] <= 1.0
            assert 0.0 <= m_data["per_class"][ch]["iou"] <= 1.0

    # Check client-macro overall
    cm = comp["client_macro_overall"]
    for m_key in ("supervised_baseline", "simclr_fine_tuned", "federated_fedavg"):
        assert m_key in cm
        assert 0.0 < cm[m_key]["dice_macro"] < 1.0
        assert 0.0 < cm[m_key]["iou_macro"] < 1.0

    # Check datasets breakdown
    ds = comp["dataset_breakdown"]
    assert "DDR" in ds and "IDRiD" in ds
    assert ds["DDR"]["federated_fedavg"]["sample_count"] == 225
    assert ds["IDRiD"]["federated_fedavg"]["sample_count"] == 27

    # Check checkpoint identities
    ck = comp["checkpoint_identities"]
    for m_key in ("supervised_baseline", "simclr_fine_tuned", "federated_fedavg"):
        assert ck[m_key]["verified_unmodified"] is True
        assert len(ck[m_key]["sha256"]) == 64

    # Check fairness audit
    fair = comp["fairness_audit"]
    assert fair["evaluation_cohort"]["total_samples"] == 252
    assert fair["evaluation_cohort"]["overlap_with_training"] == 0
    assert fair["evaluation_cohort"]["overlap_with_validation"] == 0


def test_generate_comparison_markdown_structure() -> None:
    """Verify that generated Markdown contains expected headers and comparison tables."""
    reports = {k: load_test_report(v) for k, v in DEFAULT_REPORT_PATHS.items()}
    comp = build_consolidated_comparison(reports)
    md = generate_comparison_markdown(comp)

    assert "# FedMorphis — Consolidated Test Evaluation & Benchmark Comparison" in md
    assert "## 1. Primary Benchmark: Sample-Weighted Test Performance ($N = 252$)" in md
    assert "## 2. Institutional Macro Benchmark" in md
    assert "## 3. Dataset-Specific Breakdown: DDR vs. IDRiD" in md
    assert "## 4. Checkpoint Identities" in md
    assert "## 5. Scientific Fairness and Methodological Audit" in md
    assert "Centralized Supervised Training" in md
    assert "SimCLR Pretraining + Supervised Fine-Tuning" in md
    assert "Flower FedAvg Federated Training" in md


def test_validation_rejects_incomplete_cohort() -> None:
    """Verify that report validation strictly rejects reports missing test samples."""
    dummy_report: dict[str, Any] = {
        "status": "completed",
        "isolation_verification": {
            "total_samples_evaluated": 225,  # missing IDRiD
            "per_dataset_sample_counts": {"DDR": 225},
            "test_data_leakage_detected": False,
        },
        "checkpoint": {"verified_unmodified": True},
        "overall": {"sample_weighted": {"loss": 0.2, "dice_macro": 0.4, "iou_macro": 0.3}},
    }
    with pytest.raises(ValueError, match="expected 252 test samples"):
        validate_test_report_integrity(dummy_report, "test_dummy")


def test_consolidate_comparison_cli(tmp_path: Path) -> None:
    """Verify that the consolidate CLI executes successfully and writes JSON and Markdown files."""
    out_json = tmp_path / "test_consolidated.json"
    out_md = tmp_path / "test_consolidated.md"

    code = consolidate_main([
        "--supervised-report", DEFAULT_REPORT_PATHS["supervised_baseline"],
        "--simclr-report", DEFAULT_REPORT_PATHS["simclr_fine_tuned"],
        "--federated-report", DEFAULT_REPORT_PATHS["federated_fedavg"],
        "--output-json", str(out_json),
        "--output-markdown", str(out_md),
    ])

    assert code == 0
    assert out_json.is_file()
    assert out_md.is_file()

    loaded_json = json.loads(out_json.read_text(encoding="utf-8"))
    assert loaded_json["status"] == "completed"
    assert loaded_json["sample_count_total"] == 252


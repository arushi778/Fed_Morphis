"""FedMorphis: Interactive Web Demonstration and Results Dashboard.

FastAPI-powered demonstration application featuring:
1. Live Retinal Lesion Segmentation Inference with frozen channel-tuned thresholds.
2. Flower 1.43.0 Federated Learning Architecture & Data Flow Visualizer.
3. 20-Round Federated Training History & Convergence Dashboard.
4. Final Held-Out Test Results (n=252) & Multi-Method Benchmark.

Usage:
    python scripts/run_demo.py --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path
import sys
import time
from typing import Any

import cv2
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
import numpy as np
from PIL import Image
import uvicorn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.engine.inference import (
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_FROZEN_THRESHOLDS,
    CANONICAL_LESION_COLORS,
    CANONICAL_LESION_NAMES,
    CANONICAL_LESION_ORDER,
    InferenceResult,
    LesionSegmentationPipeline,
)
from src.utils.logger import get_logger
from src.utils.paths import format_project_path, resolve_project_path

logger = get_logger("fedmorphis.demo_server")

# Sample images available for 1-click demonstration (validation and training splits only, never test split)
DEMO_SAMPLE_IMAGES: dict[str, dict[str, str]] = {
    "DDR_Sample_1": {
        "label": "DDR: 007-5797-300 (Validation split, all 4 lesion types)",
        "dataset": "DDR",
        "sample_id": "007-5797-300",
        "path": "data/raw/DDR/DDR-dataset/lesion_segmentation/valid/image/007-5797-300.jpg",
    },
    "DDR_Sample_2": {
        "label": "DDR: 007-5800-300 (Validation split, all 4 lesion types)",
        "dataset": "DDR",
        "sample_id": "007-5800-300",
        "path": "data/raw/DDR/DDR-dataset/lesion_segmentation/valid/image/007-5800-300.jpg",
    },
    "IDRiD_Sample_1": {
        "label": "IDRiD: IDRiD_03 (Training split, all 4 lesion types)",
        "dataset": "IDRiD",
        "sample_id": "IDRiD_03",
        "path": "data/raw/IDRiD/A. Segmentation/1. Original Images/a. Training Set/IDRiD_03.jpg",
    },
    "IDRiD_Sample_2": {
        "label": "IDRiD: IDRiD_08 (Training split, all 4 lesion types)",
        "dataset": "IDRiD",
        "sample_id": "IDRiD_08",
        "path": "data/raw/IDRiD/A. Segmentation/1. Original Images/a. Training Set/IDRiD_08.jpg",
    },
}

app = FastAPI(
    title="FedMorphis Demonstration Platform",
    description="Federated Multitask Retinal Lesion Segmentation Prototype (Flower 1.43.0)",
    version="1.0.0",
)

# Global pipeline instance
_pipeline: LesionSegmentationPipeline | None = None


def get_pipeline() -> LesionSegmentationPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = LesionSegmentationPipeline(
            checkpoint_path=DEFAULT_CHECKPOINT_PATH,
            thresholds=DEFAULT_FROZEN_THRESHOLDS,
            device="auto",
        )
    return _pipeline


def ndarray_to_base64_png(arr: np.ndarray) -> str:
    """Convert an RGB or grayscale uint8 NumPy array to a base64 data URI."""
    if arr.ndim == 3 and arr.shape[2] == 3:
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    else:
        bgr = arr
    success, buffer = cv2.imencode(".png", bgr)
    if not success:
        raise ValueError("Failed to encode image to PNG format.")
    b64_str = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"


def load_training_history() -> dict[str, Any]:
    """Load the recorded 20-round Flower FedAvg training report."""
    rep_p = resolve_project_path("logs/federated_fedavg_full_20rounds_report.json")
    if not rep_p.is_file():
        rep_p = resolve_project_path("logs/federated/fedavg_full_20rounds_report.json")
    if not rep_p.is_file():
        return {"error": "Training report file not found."}
    with open(rep_p, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {
        "experiment_name": data.get("experiment_name"),
        "best_round": data.get("best_round", 17),
        "best_metric_value": data.get("best_metric_value", 0.565677),
        "duration_sec": data.get("training_duration_sec", 4562.3),
        "per_round_val_metrics": data.get("per_round_val_metrics", []),
        "per_round_train_metrics": data.get("per_round_train_metrics", []),
    }


def load_final_test_benchmark() -> dict[str, Any]:
    """Load verified Task 15 held-out test report."""
    rep_p = resolve_project_path("logs/federated/final_threshold_test_report.json")
    if not rep_p.is_file():
        return {"error": "Final test report not found."}
    with open(rep_p, "r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/api/samples")
def get_sample_list() -> dict[str, Any]:
    """Return available sample images for 1-click demonstration."""
    return {"samples": DEMO_SAMPLE_IMAGES}


@app.get("/api/training-history")
def get_training_history_endpoint() -> dict[str, Any]:
    """Return round-by-round federated training metrics."""
    return load_training_history()


@app.get("/api/benchmark")
def get_benchmark_endpoint() -> dict[str, Any]:
    """Return held-out test benchmark metrics."""
    return load_final_test_benchmark()


@app.post("/api/predict")
async def run_predict(
    sample_key: str | None = Form(None),
    file: UploadFile | None = File(None),
) -> dict[str, Any]:
    """Run lesion segmentation inference on uploaded file or preloaded sample."""
    pipeline = get_pipeline()
    sample_name = "custom_upload"

    if file is not None and file.filename:
        contents = await file.read()
        sample_name = Path(file.filename).stem
        image_input: Any = contents
    elif sample_key and sample_key in DEMO_SAMPLE_IMAGES:
        sample_info = DEMO_SAMPLE_IMAGES[sample_key]
        sample_p = resolve_project_path(sample_info["path"])
        if not sample_p.is_file():
            raise HTTPException(status_code=404, detail=f"Sample file not found: {sample_p}")
        image_input = sample_p
        sample_name = sample_info["sample_id"]
    else:
        raise HTTPException(status_code=400, detail="Must provide either an uploaded file or valid sample_key.")

    try:
        res = pipeline.predict(image_input, sample_name=sample_name)
    except Exception as e:
        logger.exception("Prediction failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    # Encode images to base64 for direct browser rendering
    orig_b64 = ndarray_to_base64_png(res.original_image_rgb)
    overlay_b64 = ndarray_to_base64_png(res.combined_overlay_orig)

    channel_data: dict[str, Any] = {}
    for ch_code, cp in res.channels.items():
        channel_data[ch_code] = {
            "name": cp.channel_name,
            "threshold": cp.threshold,
            "color_rgb": cp.color_rgb,
            "positive_pixels": cp.positive_pixel_count,
            "area_pct": round(cp.area_percentage, 4),
            "detected": cp.detected,
            "mask_png_b64": ndarray_to_base64_png(cp.mask_orig),
            "colored_png_b64": ndarray_to_base64_png(cp.colored_mask_orig),
        }

    return {
        "status": "success",
        "sample_name": sample_name,
        "original_dimensions": {"height": res.original_dimensions[0], "width": res.original_dimensions[1]},
        "latency_ms": round(res.latency_ms, 2),
        "metadata": res.metadata,
        "images": {
            "original_b64": orig_b64,
            "overlay_b64": overlay_b64,
        },
        "channels": channel_data,
    }


@app.get("/", response_class=HTMLResponse)
def index_page() -> str:
    """Serve the complete, modern single-page demonstration dashboard."""
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>FedMorphis — Federated Retinal Lesion Segmentation Dashboard</title>
  <style>
    :root {
      --primary-navy: #142B45;
      --secondary-navy: #203D5B;
      --medical-teal: #12A89D;
      --medical-teal-dark: #0e877e;
      --medical-teal-light: #E6F7F5;
      --bg-main: #F5F7FA;
      --card-bg: #FFFFFF;
      --text-primary: #1E293B;
      --text-secondary: #64748B;
      --text-muted: #94A3B8;
      --border-color: #E2E8F0;
      --border-focus: #94A3B8;
      --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
      --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.07), 0 2px 4px -2px rgba(0, 0, 0, 0.05);
      --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.08), 0 4px 6px -4px rgba(0, 0, 0, 0.03);
      --ma-color: #E11D48;
      --he-color: #16A34A;
      --ex-color: #D97706;
      --se-color: #0284C7;
      --radius-sm: 6px;
      --radius-md: 10px;
      --radius-lg: 14px;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }
    
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
      background-color: var(--bg-main);
      color: var(--text-primary);
      line-height: 1.5;
      display: flex;
      min-height: 100vh;
      overflow-x: hidden;
    }

    /* Professional Sidebar */
    aside.sidebar {
      width: 280px;
      background: var(--primary-navy);
      color: #FFFFFF;
      display: flex;
      flex-direction: column;
      flex-shrink: 0;
      border-right: 1px solid rgba(255, 255, 255, 0.08);
      position: sticky;
      top: 0;
      height: 100vh;
      z-index: 50;
    }

    .brand-section {
      padding: 24px 20px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.1);
      display: flex;
      align-items: center;
      gap: 12px;
    }

    .brand-icon {
      width: 42px;
      height: 42px;
      border-radius: var(--radius-md);
      background: linear-gradient(135deg, var(--medical-teal) 0%, #0d6e67 100%);
      display: flex;
      align-items: center;
      justify-content: center;
      color: #FFFFFF;
      box-shadow: 0 4px 10px rgba(18, 168, 157, 0.35);
      flex-shrink: 0;
    }

    .brand-text h1 {
      font-size: 1.25rem;
      font-weight: 700;
      letter-spacing: -0.02em;
      color: #FFFFFF;
      line-height: 1.2;
    }

    .brand-text p {
      font-size: 0.75rem;
      color: #94A3B8;
      font-weight: 500;
      letter-spacing: 0.01em;
    }

    nav.nav-menu {
      padding: 20px 12px;
      flex: 1;
      display: flex;
      flex-direction: column;
      gap: 6px;
      overflow-y: auto;
    }

    .nav-label {
      font-size: 0.68rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: #64748B;
      padding: 10px 14px 6px;
      font-weight: 700;
    }

    .nav-item {
      display: flex;
      align-items: center;
      gap: 12px;
      padding: 12px 14px;
      border-radius: var(--radius-md);
      color: #CBD5E1;
      text-decoration: none;
      font-size: 0.88rem;
      font-weight: 500;
      cursor: pointer;
      border: 1px solid transparent;
      transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
      background: transparent;
      width: 100%;
      text-align: left;
    }

    .nav-item:hover {
      background: rgba(255, 255, 255, 0.06);
      color: #FFFFFF;
    }

    .nav-item.active {
      background: var(--secondary-navy);
      color: #FFFFFF;
      border-color: rgba(255, 255, 255, 0.12);
      box-shadow: inset 3px 0 0 var(--medical-teal);
    }

    .nav-item svg {
      width: 20px;
      height: 20px;
      stroke-width: 2;
      opacity: 0.85;
      flex-shrink: 0;
    }

    .nav-item.active svg {
      color: var(--medical-teal);
      opacity: 1;
    }

    .sidebar-footer {
      padding: 18px;
      border-top: 1px solid rgba(255, 255, 255, 0.08);
      background: rgba(0, 0, 0, 0.15);
    }

    .model-badge-card {
      background: rgba(255, 255, 255, 0.04);
      border: 1px solid rgba(255, 255, 255, 0.08);
      border-radius: var(--radius-sm);
      padding: 10px 12px;
      font-size: 0.76rem;
    }

    .model-badge-title {
      color: var(--medical-teal);
      font-weight: 600;
      display: flex;
      align-items: center;
      gap: 6px;
      margin-bottom: 2px;
    }

    .model-badge-sub {
      color: #94A3B8;
      font-size: 0.72rem;
    }

    /* Main Area Layout */
    main.main-viewport {
      flex: 1;
      display: flex;
      flex-direction: column;
      min-width: 0;
      background: var(--bg-main);
    }

    header.topbar {
      background: #FFFFFF;
      border-bottom: 1px solid var(--border-color);
      padding: 14px 32px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      position: sticky;
      top: 0;
      z-index: 40;
    }

    .breadcrumb {
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 0.84rem;
      color: var(--text-secondary);
    }

    .breadcrumb-active {
      color: var(--text-primary);
      font-weight: 600;
    }

    .topbar-meta {
      display: flex;
      align-items: center;
      gap: 12px;
    }

    .meta-tag {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 5px 12px;
      border-radius: 9999px;
      font-size: 0.76rem;
      font-weight: 600;
      background: #F1F5F9;
      color: var(--text-secondary);
      border: 1px solid var(--border-color);
    }

    .meta-tag.tag-teal {
      background: var(--medical-teal-light);
      color: var(--medical-teal-dark);
      border-color: rgba(18, 168, 157, 0.25);
    }

    .disclaimer-banner {
      background: #FEF2F2;
      border-bottom: 1px solid #FEE2E2;
      color: #991B1B;
      padding: 9px 32px;
      font-size: 0.80rem;
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .content-area {
      padding: 28px 32px 50px;
      max-width: 1500px;
      margin: 0 auto;
      width: 100%;
    }

    .tab-content { display: none; }
    .tab-content.active { display: block; }

    /* Component Card */
    .card {
      background: var(--card-bg);
      border: 1px solid var(--border-color);
      border-radius: var(--radius-lg);
      padding: 24px;
      margin-bottom: 24px;
      box-shadow: var(--shadow-sm);
    }

    .card-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 20px;
      padding-bottom: 14px;
      border-bottom: 1px solid var(--border-color);
    }

    .card-title {
      font-size: 1.08rem;
      font-weight: 700;
      color: var(--primary-navy);
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .card-subtitle {
      font-size: 0.82rem;
      color: var(--text-secondary);
      margin-top: 2px;
    }

    /* Grid Layouts */
    .grid-2col {
      display: grid;
      grid-template-columns: 380px 1fr;
      gap: 24px;
    }
    @media (max-width: 1100px) {
      .grid-2col { grid-template-columns: 1fr; }
    }

    .grid-4col {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 16px;
    }
    @media (max-width: 900px) {
      .grid-4col { grid-template-columns: repeat(2, 1fr); }
    }
    @media (max-width: 520px) {
      .grid-4col { grid-template-columns: 1fr; }
    }

    /* Form Controls */
    label.field-label {
      display: block;
      font-size: 0.82rem;
      font-weight: 600;
      color: var(--primary-navy);
      margin-bottom: 6px;
    }

    select.input-select {
      width: 100%;
      background: #FFFFFF;
      border: 1px solid var(--border-color);
      color: var(--text-primary);
      padding: 10px 12px;
      border-radius: var(--radius-md);
      font-size: 0.88rem;
      margin-bottom: 16px;
      outline: none;
      transition: border-color 0.15s;
    }
    select.input-select:focus {
      border-color: var(--medical-teal);
      box-shadow: 0 0 0 3px rgba(18, 168, 157, 0.15);
    }

    /* Drag and Drop Zone */
    .dropzone-container {
      border: 2px dashed #CBD5E1;
      border-radius: var(--radius-md);
      padding: 24px 16px;
      text-align: center;
      background: #FAFCFE;
      cursor: pointer;
      transition: all 0.2s ease;
      margin-bottom: 16px;
    }
    .dropzone-container:hover, .dropzone-container.drag-over {
      border-color: var(--medical-teal);
      background: var(--medical-teal-light);
    }
    .dropzone-icon {
      width: 36px;
      height: 36px;
      color: var(--medical-teal);
      margin: 0 auto 8px;
    }
    .dropzone-text-primary {
      font-size: 0.86rem;
      font-weight: 600;
      color: var(--primary-navy);
    }
    .dropzone-text-sub {
      font-size: 0.75rem;
      color: var(--text-secondary);
      margin-top: 4px;
    }
    .file-pill {
      display: none;
      align-items: center;
      justify-content: space-between;
      padding: 8px 12px;
      background: #EEF2F6;
      border-radius: var(--radius-sm);
      margin-bottom: 16px;
      font-size: 0.80rem;
      color: var(--primary-navy);
      font-weight: 500;
    }

    button.btn-primary {
      width: 100%;
      background: var(--medical-teal);
      color: #FFFFFF;
      border: none;
      padding: 13px 18px;
      border-radius: var(--radius-md);
      font-size: 0.94rem;
      font-weight: 600;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      transition: all 0.15s ease;
      box-shadow: 0 2px 4px rgba(18, 168, 157, 0.25);
    }
    button.btn-primary:hover {
      background: var(--medical-teal-dark);
      box-shadow: 0 4px 8px rgba(18, 168, 157, 0.35);
    }
    button.btn-primary:disabled {
      background: #CBD5E1;
      cursor: not-allowed;
      box-shadow: none;
    }

    .spinner {
      width: 18px;
      height: 18px;
      border: 2px solid rgba(255, 255, 255, 0.3);
      border-radius: 50%;
      border-top-color: #FFFFFF;
      animation: spin 0.8s linear infinite;
    }
    @keyframes spin { to { transform: rotate(360deg); } }

    /* Results Viewer Section */
    .viewer-toolbar {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 16px;
      flex-wrap: wrap;
      gap: 10px;
    }

    .view-mode-group {
      display: flex;
      background: #F1F5F9;
      border-radius: var(--radius-md);
      padding: 3px;
    }

    .view-mode-btn {
      border: none;
      background: transparent;
      padding: 6px 14px;
      font-size: 0.80rem;
      font-weight: 600;
      color: var(--text-secondary);
      border-radius: var(--radius-sm);
      cursor: pointer;
      transition: all 0.15s;
    }
    .view-mode-btn.active {
      background: #FFFFFF;
      color: var(--primary-navy);
      box-shadow: var(--shadow-sm);
    }

    .viewer-display {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
      margin-bottom: 24px;
    }
    @media (max-width: 768px) {
      .viewer-display { grid-template-columns: 1fr; }
    }

    .image-card {
      background: #090B10;
      border: 1px solid var(--border-color);
      border-radius: var(--radius-md);
      overflow: hidden;
      display: flex;
      flex-direction: column;
    }

    .image-card-bar {
      background: #151922;
      padding: 8px 14px;
      font-size: 0.80rem;
      font-weight: 600;
      color: #E2E8F0;
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-bottom: 1px solid #242936;
    }

    .image-viewport {
      width: 100%;
      aspect-ratio: 1 / 1;
      display: flex;
      align-items: center;
      justify-content: center;
      background: #040507;
      position: relative;
    }

    .image-viewport img {
      max-width: 100%;
      max-height: 100%;
      object-fit: contain;
    }

    /* Channel Cards */
    .channel-card {
      background: #FFFFFF;
      border: 1px solid var(--border-color);
      border-radius: var(--radius-md);
      padding: 14px;
      display: flex;
      flex-direction: column;
      box-shadow: var(--shadow-sm);
    }

    .channel-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 10px;
    }

    .channel-pill {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 3px 9px;
      border-radius: var(--radius-sm);
      font-size: 0.75rem;
      font-weight: 700;
    }

    .pill-ma { background: #FFE4E6; color: var(--ma-color); }
    .pill-he { background: #DCFCE7; color: var(--he-color); }
    .pill-ex { background: #FEF3C7; color: var(--ex-color); }
    .pill-se { background: #E0F2FE; color: var(--se-color); }

    .channel-img-frame {
      width: 100%;
      aspect-ratio: 1 / 1;
      background: #090B10;
      border-radius: var(--radius-sm);
      overflow: hidden;
      display: flex;
      align-items: center;
      justify-content: center;
      margin-bottom: 10px;
    }

    .channel-img-frame img {
      max-width: 100%;
      max-height: 100%;
      object-fit: contain;
    }

    .channel-stats {
      font-size: 0.76rem;
      color: var(--text-secondary);
      display: flex;
      flex-direction: column;
      gap: 3px;
    }

    /* Clean Medical Tables */
    table.data-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.85rem;
    }
    table.data-table th, table.data-table td {
      padding: 12px 14px;
      text-align: left;
      border-bottom: 1px solid var(--border-color);
    }
    table.data-table th {
      background: #F8FAFC;
      color: var(--text-secondary);
      font-weight: 600;
      font-size: 0.78rem;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }
    table.data-table tr:hover {
      background: #FAFCFE;
    }
    table.data-table td.num {
      text-align: right;
      font-variant-numeric: tabular-nums;
    }
    table.data-table tr.highlight-best {
      background: #F0FDF4;
      font-weight: 600;
    }

    /* KPI Summary Cards */
    .kpi-row {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 16px;
      margin-bottom: 24px;
    }
    @media (max-width: 960px) {
      .kpi-row { grid-template-columns: repeat(2, 1fr); }
    }

    .kpi-card {
      background: #FFFFFF;
      border: 1px solid var(--border-color);
      border-radius: var(--radius-md);
      padding: 18px 20px;
      box-shadow: var(--shadow-sm);
    }
    .kpi-card.teal-accent {
      border-top: 3px solid var(--medical-teal);
    }
    .kpi-card.navy-accent {
      border-top: 3px solid var(--primary-navy);
    }
    .kpi-label {
      font-size: 0.78rem;
      font-weight: 600;
      color: var(--text-secondary);
      text-transform: uppercase;
      letter-spacing: 0.04em;
      margin-bottom: 6px;
    }
    .kpi-value {
      font-size: 1.8rem;
      font-weight: 700;
      color: var(--primary-navy);
      line-height: 1.1;
      font-variant-numeric: tabular-nums;
    }
    .kpi-sub {
      font-size: 0.75rem;
      color: var(--text-muted);
      margin-top: 4px;
    }

    /* Architecture Flow Visualizer */
    .flow-diagram-container {
      background: #F8FAFC;
      border: 1px solid var(--border-color);
      border-radius: var(--radius-lg);
      padding: 28px 20px;
      margin-bottom: 24px;
    }

    .flow-nodes-wrapper {
      display: flex;
      align-items: center;
      justify-content: space-around;
      gap: 16px;
      flex-wrap: wrap;
    }

    .flow-client-node {
      background: #FFFFFF;
      border: 1px solid #CBD5E1;
      border-radius: var(--radius-md);
      padding: 18px 22px;
      text-align: center;
      width: 240px;
      box-shadow: var(--shadow-sm);
      position: relative;
    }
    .flow-client-node.active-ddr {
      border-left: 4px solid #3B82F6;
    }
    .flow-client-node.active-idrid {
      border-left: 4px solid #10B981;
    }

    .flow-server-node {
      background: #142B45;
      color: #FFFFFF;
      border-radius: var(--radius-lg);
      padding: 24px 28px;
      text-align: center;
      width: 290px;
      box-shadow: var(--shadow-md);
      position: relative;
    }

    .flow-arrow-col {
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      color: var(--medical-teal);
      font-weight: 600;
      font-size: 0.78rem;
      gap: 6px;
    }

    .callout-box {
      background: #F0FDF4;
      border: 1px solid #BBF7D0;
      border-radius: var(--radius-md);
      padding: 16px 20px;
      color: #166534;
      font-size: 0.88rem;
      line-height: 1.6;
    }

    .callout-box.info {
      background: #EFF6FF;
      border-color: #BFDBFE;
      color: #1E40AF;
    }

    /* Interactive Chart Canvas */
    .chart-container {
      width: 100%;
      height: 280px;
      position: relative;
      margin-top: 10px;
    }

    /* Mobile drawer trigger */
    .mobile-menu-btn {
      display: none;
      background: none;
      border: none;
      cursor: pointer;
      color: var(--primary-navy);
    }
    @media (max-width: 900px) {
      aside.sidebar {
        position: fixed;
        left: -280px;
        transition: left 0.3s ease;
      }
      aside.sidebar.open {
        left: 0;
      }
      .mobile-menu-btn {
        display: block;
      }
      header.topbar {
        padding: 14px 18px;
      }
      .content-area {
        padding: 18px 16px 40px;
      }
    }
  </style>
</head>
<body>

<!-- Sidebar Navigation -->
<aside class="sidebar" id="app-sidebar">
  <div class="brand-section">
    <div class="brand-icon">
      <!-- Neural Retinal Icon -->
      <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7Z"/>
        <circle cx="12" cy="12" r="3"/>
        <path d="M12 2v3M12 19v3M2 12h3M19 12h3"/>
      </svg>
    </div>
    <div class="brand-text">
      <h1>FedMorphis</h1>
      <p>Federated Retinal AI</p>
    </div>
  </div>

  <nav class="nav-menu">
    <div class="nav-label">Core Modules</div>
    
    <button class="nav-item active" onclick="switchNav('inference-tab', 'Live Lesion Inference')">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor">
        <circle cx="12" cy="12" r="10"/>
        <line x1="22" y1="12" x2="18" y2="12"/>
        <line x1="6" y1="12" x2="2" y2="12"/>
        <line x1="12" y1="6" x2="12" y2="2"/>
        <line x1="12" y1="22" x2="12" y2="18"/>
      </svg>
      <span>Live Lesion Inference</span>
    </button>

    <button class="nav-item" onclick="switchNav('architecture-tab', 'Federated Architecture')">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor">
        <rect x="2" y="2" width="8" height="8" rx="2"/>
        <rect x="14" y="2" width="8" height="8" rx="2"/>
        <rect x="8" y="14" width="8" height="8" rx="2"/>
        <path d="M6 10v2a2 2 0 0 0 2 2h4m0 0h4a2 2 0 0 0 2-2v-2"/>
      </svg>
      <span>Flower Architecture</span>
    </button>

    <button class="nav-item" onclick="switchNav('training-tab', '20-Round Training History')">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor">
        <path d="M3 3v18h18"/>
        <path d="m19 9-5 5-4-4-3 3"/>
      </svg>
      <span>Training History</span>
    </button>

    <button class="nav-item" onclick="switchNav('benchmark-tab', 'Research Benchmark')">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor">
        <path d="M6 9H4.5a2.5 2.5 0 0 1 0-5H6"/>
        <path d="M18 9h1.5a2.5 2.5 0 0 0 0-5H18"/>
        <path d="M4 22h16"/>
        <path d="M10 14.66V17c0 .55-.47.98-.97 1.21C7.85 18.75 7 20.24 7 22"/>
        <path d="M14 14.66V17c0 .55.47.98.97 1.21C16.15 18.75 17 20.24 17 22"/>
        <path d="M18 2H6v7a6 6 0 0 0 12 0V2Z"/>
      </svg>
      <span>Research Benchmark</span>
    </button>
  </nav>

  <div class="sidebar-footer">
    <div class="model-badge-card">
      <div class="model-badge-title">
        <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor">
          <circle cx="12" cy="12" r="10"/>
        </svg>
        <span>Global Checkpoint: R17</span>
      </div>
      <div class="model-badge-sub">ResNet-50 U-Net &bull; Flower 1.43.0</div>
    </div>
  </div>
</aside>

<!-- Main Viewport Area -->
<main class="main-viewport">

  <!-- Top Application Bar -->
  <header class="topbar">
    <div style="display:flex; align-items:center; gap:12px;">
      <button class="mobile-menu-btn" onclick="toggleSidebar()" aria-label="Toggle Navigation">
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <line x1="3" y1="12" x2="21" y2="12"/><line x1="3" y1="6" x2="21" y2="6"/><line x1="3" y1="18" x2="21" y2="18"/>
        </svg>
      </button>
      <div class="breadcrumb">
        <span>FedMorphis</span>
        <span>/</span>
        <span class="breadcrumb-active" id="current-breadcrumb">Live Lesion Inference</span>
      </div>
    </div>

    <div class="topbar-meta">
      <span class="meta-tag tag-teal">Flower 1.43.0 FedAvg</span>
      <span class="meta-tag" id="top-device-badge">Device: CUDA</span>
      <span class="meta-tag">Frozen Thresholds</span>
    </div>
  </header>

  <!-- Global Clinical Research Disclaimer Banner -->
  <div class="disclaimer-banner">
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
      <path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/>
      <line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>
    </svg>
    <span><strong>Research &amp; Demonstration Disclaimer:</strong> FedMorphis is an academic research prototype developed for university engineering demonstration. Predictions are generated for exploratory research and are not certified for medical diagnostic use.</span>
  </div>

  <div class="content-area">

    <!-- ================================================================= -->
    <!-- TAB 1: Live Lesion Inference                                      -->
    <!-- ================================================================= -->
    <section id="inference-tab" class="tab-content active">
      <div class="grid-2col">

        <!-- Controls Column -->
        <div>
          <div class="card">
            <div class="card-header">
              <div>
                <div class="card-title">Retinal Image Input</div>
                <div class="card-subtitle">Select an authorized split sample or upload an image</div>
              </div>
            </div>

            <label class="field-label" for="sample-select">Pre-Loaded Sample Images (Non-Test Splits)</label>
            <select id="sample-select" class="input-select" onchange="onSampleSelected()">
              <option value="DDR_Sample_1">DDR: 007-5797-300 (validation split)</option>
              <option value="DDR_Sample_2">DDR: 007-5800-300 (validation split)</option>
              <option value="IDRiD_Sample_1" selected>IDRiD: IDRiD_03 (training split)</option>
              <option value="IDRiD_Sample_2">IDRiD: IDRiD_08 (training split)</option>
            </select>

            <label class="field-label">Custom Fundus Photograph Upload</label>
            <div class="dropzone-container" id="dropzone" onclick="triggerFileInput()">
              <input type="file" id="file-upload" accept="image/jpeg,image/png,image/tiff" style="display:none;" onchange="handleFileSelected(event)">
              <svg class="dropzone-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                <polyline points="17 8 12 3 7 8"/>
                <line x1="12" y1="3" x2="12" y2="15"/>
              </svg>
              <div class="dropzone-text-primary" id="dropzone-prompt">Drag &amp; drop fundus image here, or <span style="color:var(--medical-teal); text-decoration:underline;">browse</span></div>
              <div class="dropzone-text-sub">Supported formats: JPG, PNG, TIF &bull; Auto-scaled to 512&times;512</div>
            </div>

            <div class="file-pill" id="file-details-pill">
              <span id="file-name-label">filename.jpg</span>
              <span id="file-meta-label" style="color:var(--text-secondary); font-size:0.75rem;">—</span>
            </div>

            <button class="btn-primary" id="btn-run" onclick="executeInference()">
              <span id="btn-text">Execute Multi-Task Segmentation</span>
              <div id="btn-spinner" class="spinner" style="display:none;"></div>
            </button>
          </div>

          <!-- Frozen Threshold Settings Card -->
          <div class="card">
            <div class="card-header">
              <div>
                <div class="card-title">Calibrated Operating Points</div>
                <div class="card-subtitle">Channel-tuned decision thresholds (Round 17 Global Model)</div>
              </div>
            </div>

            <table class="data-table">
              <thead>
                <tr>
                  <th>Lesion Channel</th>
                  <th class="num">Threshold</th>
                  <th>Visual Overlay</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td><strong>MA</strong> (Microaneurysms)</td>
                  <td class="num"><strong>0.25</strong></td>
                  <td><span class="channel-pill pill-ma">&FilledSmallSquare; Crimson Red</span></td>
                </tr>
                <tr>
                  <td><strong>HE</strong> (Haemorrhages)</td>
                  <td class="num"><strong>0.40</strong></td>
                  <td><span class="channel-pill pill-he">&FilledSmallSquare; Lime Green</span></td>
                </tr>
                <tr>
                  <td><strong>EX</strong> (Hard Exudates)</td>
                  <td class="num"><strong>0.35</strong></td>
                  <td><span class="channel-pill pill-ex">&FilledSmallSquare; Gold Yellow</span></td>
                </tr>
                <tr>
                  <td><strong>SE</strong> (Soft Exudates)</td>
                  <td class="num"><strong>0.35</strong></td>
                  <td><span class="channel-pill pill-se">&FilledSmallSquare; Cyan Turquoise</span></td>
                </tr>
              </tbody>
            </table>
          </div>

          <!-- Pipeline Architecture Card -->
          <div class="card">
            <div class="card-header">
              <div class="card-title">Inference Execution Engine</div>
            </div>
            <table class="data-table" style="font-size:0.80rem;">
              <tr><td>Model Architecture</td><td class="num" style="font-weight:600;">ResNet-50 U-Net</td></tr>
              <tr><td>Framework</td><td class="num">PyTorch 2.x</td></tr>
              <tr><td>Input Preprocessing</td><td class="num">ImageNet Standardization (512&times;512)</td></tr>
              <tr><td>Interpolation</td><td class="num">Bilinear (Prob) &bull; Nearest (Mask)</td></tr>
              <tr><td>Execution Device</td><td class="num" id="info-device-label">—</td></tr>
              <tr><td>Measured Latency</td><td class="num" id="info-latency-label" style="font-weight:700; color:var(--medical-teal);">—</td></tr>
            </table>
          </div>
        </div>

        <!-- Displays Column -->
        <div>
          <div class="card">
            <div class="viewer-toolbar">
              <div>
                <div class="card-title">Clinical Segmentation Viewer</div>
                <div class="card-subtitle" id="viewer-sample-id">Sample: Initializing...</div>
              </div>
              <div class="view-mode-group">
                <button class="view-mode-btn active" id="vm-dual" onclick="setViewMode('dual')">Dual View</button>
                <button class="view-mode-btn" id="vm-overlay" onclick="setViewMode('overlay')">Overlay Only</button>
                <button class="view-mode-btn" id="vm-orig" onclick="setViewMode('orig')">Original Only</button>
              </div>
            </div>

            <!-- Side-by-side or Single Viewport -->
            <div class="viewer-display" id="viewer-display-grid">
              <div class="image-card" id="img-card-orig">
                <div class="image-card-bar">
                  <span>Input Fundus Photograph</span>
                  <span id="orig-dim" style="font-size:0.72rem; color:#94A3B8;">—</span>
                </div>
                <div class="image-viewport">
                  <img id="img-orig" alt="Original Fundus Image" src="">
                </div>
              </div>

              <div class="image-card" id="img-card-overlay">
                <div class="image-card-bar">
                  <span>Multi-Lesion Composite Overlay</span>
                  <span id="latency-badge" style="font-size:0.72rem; color:var(--medical-teal); font-weight:700;">—</span>
                </div>
                <div class="image-viewport">
                  <img id="img-overlay" alt="Composite Lesion Overlay" src="">
                </div>
              </div>
            </div>

            <!-- 4-Channel Decomposition Grid -->
            <div style="margin-top:20px; margin-bottom:12px; display:flex; justify-content:space-between; align-items:center;">
              <div class="card-title" style="font-size:0.96rem;">Decomposed Lesion Channel Masks</div>
              <span style="font-size:0.75rem; color:var(--text-secondary);">Restored to native acquisition resolution</span>
            </div>

            <div class="grid-4col">
              <!-- MA Card -->
              <div class="channel-card">
                <div class="channel-header">
                  <span class="channel-pill pill-ma">MA &bull; Microaneurysms</span>
                  <span style="font-size:0.72rem; color:var(--text-secondary);">&tau;=0.25</span>
                </div>
                <div class="channel-img-frame">
                  <img id="img-ma" alt="MA Channel Mask" src="">
                </div>
                <div class="channel-stats">
                  <div style="display:flex; justify-content:space-between;">
                    <span>Positive Area:</span>
                    <strong id="stat-ma-px">—</strong>
                  </div>
                  <div style="display:flex; justify-content:space-between;">
                    <span>Retinal %:</span>
                    <strong id="stat-ma-pct">—</strong>
                  </div>
                  <div id="stat-ma-status" style="margin-top:2px;">—</div>
                </div>
              </div>

              <!-- HE Card -->
              <div class="channel-card">
                <div class="channel-header">
                  <span class="channel-pill pill-he">HE &bull; Haemorrhages</span>
                  <span style="font-size:0.72rem; color:var(--text-secondary);">&tau;=0.40</span>
                </div>
                <div class="channel-img-frame">
                  <img id="img-he" alt="HE Channel Mask" src="">
                </div>
                <div class="channel-stats">
                  <div style="display:flex; justify-content:space-between;">
                    <span>Positive Area:</span>
                    <strong id="stat-he-px">—</strong>
                  </div>
                  <div style="display:flex; justify-content:space-between;">
                    <span>Retinal %:</span>
                    <strong id="stat-he-pct">—</strong>
                  </div>
                  <div id="stat-he-status" style="margin-top:2px;">—</div>
                </div>
              </div>

              <!-- EX Card -->
              <div class="channel-card">
                <div class="channel-header">
                  <span class="channel-pill pill-ex">EX &bull; Hard Exudates</span>
                  <span style="font-size:0.72rem; color:var(--text-secondary);">&tau;=0.35</span>
                </div>
                <div class="channel-img-frame">
                  <img id="img-ex" alt="EX Channel Mask" src="">
                </div>
                <div class="channel-stats">
                  <div style="display:flex; justify-content:space-between;">
                    <span>Positive Area:</span>
                    <strong id="stat-ex-px">—</strong>
                  </div>
                  <div style="display:flex; justify-content:space-between;">
                    <span>Retinal %:</span>
                    <strong id="stat-ex-pct">—</strong>
                  </div>
                  <div id="stat-ex-status" style="margin-top:2px;">—</div>
                </div>
              </div>

              <!-- SE Card -->
              <div class="channel-card">
                <div class="channel-header">
                  <span class="channel-pill pill-se">SE &bull; Soft Exudates</span>
                  <span style="font-size:0.72rem; color:var(--text-secondary);">&tau;=0.35</span>
                </div>
                <div class="channel-img-frame">
                  <img id="img-se" alt="SE Channel Mask" src="">
                </div>
                <div class="channel-stats">
                  <div style="display:flex; justify-content:space-between;">
                    <span>Positive Area:</span>
                    <strong id="stat-se-px">—</strong>
                  </div>
                  <div style="display:flex; justify-content:space-between;">
                    <span>Retinal %:</span>
                    <strong id="stat-se-pct">—</strong>
                  </div>
                  <div id="stat-se-status" style="margin-top:2px;">—</div>
                </div>
              </div>
            </div>
          </div>

          <!-- Quantification Summary Table -->
          <div class="card">
            <div class="card-header">
              <div>
                <div class="card-title">Retinal Lesion Morphometry &amp; Quantification</div>
                <div class="card-subtitle">Pixel-level positive counts and proportional surface burden</div>
              </div>
            </div>

            <table class="data-table" id="quant-table">
              <thead>
                <tr>
                  <th>Lesion Pathology</th>
                  <th class="num">Operating &tau;</th>
                  <th class="num">Segmented Pixels</th>
                  <th class="num">Fundus Area Burden</th>
                  <th>Clinical Detection Status</th>
                </tr>
              </thead>
              <tbody id="quant-tbody">
                <tr><td colspan="5" style="text-align:center; color:var(--text-secondary); padding:20px;">Executing automated inference pipeline...</td></tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </section>

    <!-- ================================================================= -->
    <!-- TAB 2: Federated Architecture                                    -->
    <!-- ================================================================= -->
    <section id="architecture-tab" class="tab-content">
      <div class="card">
        <div class="card-header">
          <div>
            <div class="card-title">Flower 1.43.0 Federated Topology &amp; Aggregation Workflow</div>
            <div class="card-subtitle">Single-machine simulation across dataset-partitioned clinical cohorts</div>
          </div>
          <span class="meta-tag tag-teal">Protocol: FedAvg &bull; 20 Rounds</span>
        </div>

        <p style="color:var(--text-secondary); font-size:0.90rem; margin-bottom:24px; line-height:1.6;">
          FedMorphis trains a shared multi-task ResNet-50 U-Net across two dataset-defined client partitions (DDR and IDRiD) using <strong>Flower 1.43.0 (flwr)</strong>. Client nodes independently optimize local loss on private fundus imagery and communicate only parameter weight updates (&Delta;W) to the central server, preserving strict data isolation.
        </p>

        <!-- Diagram Container -->
        <div class="flow-diagram-container">
          <div class="flow-nodes-wrapper">
            <!-- Client 0 Node -->
            <div class="flow-client-node active-ddr">
              <div style="font-size:1.8rem; margin-bottom:4px;">🏥</div>
              <strong style="color:var(--primary-navy); font-size:0.95rem;">Client 0: DDR Cohort</strong>
              <div style="font-size:0.78rem; color:var(--text-secondary); margin-top:4px;">383 Train &bull; 149 Val</div>
              <div style="margin-top:8px; padding:3px 8px; background:#EFF6FF; border-radius:4px; font-size:0.74rem; color:#1D4ED8; font-weight:600;">
                Federated Weight = 89.7%
              </div>
              <div style="font-size:0.70rem; color:#64748B; margin-top:6px;">3 Epochs / Round (AdamW)</div>
            </div>

            <!-- Upward / Downward Sync Arrows -->
            <div class="flow-arrow-col">
              <span>&rarr; Local &Delta;W &rarr;</span>
              <svg width="40" height="24" viewBox="0 0 40 24" fill="none" stroke="currentColor" stroke-width="2.5">
                <path d="M5 6h30M28 2l7 4-7 4M35 18H5M12 14l-7 4 7 4"/>
              </svg>
              <span>&larr; Global W &larr;</span>
            </div>

            <!-- Central Flower Server Node -->
            <div class="flow-server-node">
              <div style="font-size:2rem; margin-bottom:4px;">🌐</div>
              <strong style="font-size:1.05rem; letter-spacing:0.02em;">Flower Central Server</strong>
              <div style="font-size:0.80rem; color:#94A3B8; margin-top:4px;">Flower 1.43.0 &bull; PyTorch AMP</div>
              <div style="margin-top:10px; padding:6px 12px; background:rgba(18, 168, 157, 0.2); border:1px solid var(--medical-teal); border-radius:var(--radius-sm); font-size:0.78rem; color:#34D399; font-weight:600;">
                Strategy: FedAvg + BN Buffer Sharing
              </div>
              <div style="font-size:0.72rem; color:#94A3B8; margin-top:8px;">20 Iterative Communication Rounds</div>
            </div>

            <!-- Upward / Downward Sync Arrows -->
            <div class="flow-arrow-col">
              <span>&larr; Local &Delta;W &larr;</span>
              <svg width="40" height="24" viewBox="0 0 40 24" fill="none" stroke="currentColor" stroke-width="2.5">
                <path d="M35 6H5M12 2l-7 4 7 4M5 18h30M28 14l7 4-7 4"/>
              </svg>
              <span>&rarr; Global W &rarr;</span>
            </div>

            <!-- Client 1 Node -->
            <div class="flow-client-node active-idrid">
              <div style="font-size:1.8rem; margin-bottom:4px;">🏥</div>
              <strong style="color:var(--primary-navy); font-size:0.95rem;">Client 1: IDRiD Cohort</strong>
              <div style="font-size:0.78rem; color:var(--text-secondary); margin-top:4px;">44 Train &bull; 10 Val</div>
              <div style="margin-top:8px; padding:3px 8px; background:#ECFDF5; border-radius:4px; font-size:0.74rem; color:#047857; font-weight:600;">
                Federated Weight = 10.3%
              </div>
              <div style="font-size:0.70rem; color:#64748B; margin-top:6px;">3 Epochs / Round (AdamW)</div>
            </div>
          </div>
        </div>

        <!-- Simulation Transparency Callout -->
        <div class="callout-box info" style="margin-bottom:24px;">
          <strong>Simulation Disclosure:</strong> This federated learning implementation was executed as a single-machine Flower 1.43.0 simulation with two distinct dataset-defined client partitions (DDR and IDRiD), exchanging solely model weight parameters. No physical wide-area network or live multi-site hospital transmission was deployed. All reported metrics reflect historical training logs.
        </div>

        <div style="display:grid; grid-template-columns:1fr 1fr; gap:24px;">
          <div class="card" style="margin-bottom:0; box-shadow:none; border:1px solid var(--border-color);">
            <div class="card-title" style="font-size:0.98rem; margin-bottom:12px;">Federated Hyperparameters</div>
            <table class="data-table" style="font-size:0.82rem;">
              <tr><td>Federated Framework</td><td class="num">Flower 1.43.0 (flwr)</td></tr>
              <tr><td>Aggregation Strategy</td><td class="num">FedAvg (Weighted by Client Samples)</td></tr>
              <tr><td>BatchNorm Strategy</td><td class="num">Synchronized Global BN Sharing</td></tr>
              <tr><td>Communication Rounds</td><td class="num">20 Full Rounds</td></tr>
              <tr><td>Local Epochs / Round</td><td class="num">3 Epochs per client</td></tr>
              <tr><td>Local Batch Size</td><td class="num">8 images / batch</td></tr>
              <tr><td>Optimizer</td><td class="num">AdamW (lr=1e-4, weight_decay=1e-4)</td></tr>
              <tr><td>Loss Function</td><td class="num">BCE (0.50) + Dice (0.50)</td></tr>
            </table>
          </div>

          <div class="card" style="margin-bottom:0; box-shadow:none; border:1px solid var(--border-color);">
            <div class="card-title" style="font-size:0.98rem; margin-bottom:12px;">Data Isolation &amp; Privacy Safeguards</div>
            <ul style="color:var(--text-secondary); font-size:0.84rem; padding-left:18px; line-height:1.8;">
              <li><strong>Zero Image Transfer:</strong> Raw fundus images, annotations, and patient masks remain exclusively within client partition boundaries.</li>
              <li><strong>Strict Partitioning:</strong> DDR and IDRiD records are completely isolated in separate manifests with verified disjoint indices.</li>
              <li><strong>Held-Out Cohort Protection:</strong> The 252-sample test cohort was quarantined and never accessed during federated rounds or threshold tuning.</li>
              <li><strong>Deterministic Reproducibility:</strong> Global PRNG seeds (seed=42) fixed for data sampling, model initialization, and batch ordering.</li>
            </ul>
          </div>
        </div>
      </div>
    </section>

    <!-- ================================================================= -->
    <!-- TAB 3: 20-Round Training History                                 -->
    <!-- ================================================================= -->
    <section id="training-tab" class="tab-content">
      <div class="card">
        <div class="card-header">
          <div>
            <div class="card-title">20-Round Federated Convergence Profile</div>
            <div class="card-subtitle">Validation cohort metrics (n=159) evaluated at each communication round</div>
          </div>
          <span class="meta-tag tag-teal">Best Model: Round 17 (Macro Dice 0.5657)</span>
        </div>

        <div class="callout-box info" style="margin-bottom:20px; font-size:0.82rem;">
          <strong>Historical Report Log:</strong> These values reflect recorded validation metrics from <code>logs/federated_fedavg_full_20rounds_report.json</code> across 20 communication rounds. They are historical experiment results, not live training.
        </div>

        <!-- KPI Row for Convergence -->
        <div class="kpi-row">
          <div class="kpi-card teal-accent">
            <div class="kpi-label">Peak Val Macro Dice</div>
            <div class="kpi-value" id="kpi-best-dice">0.5657</div>
            <div class="kpi-sub">Round 17 Global Model Checkpoint</div>
          </div>
          <div class="kpi-card navy-accent">
            <div class="kpi-label">Peak Val Macro IoU</div>
            <div class="kpi-value" id="kpi-best-iou">0.4064</div>
            <div class="kpi-sub">Round 17 Global Model Checkpoint</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-label">Final Round 20 Loss</div>
            <div class="kpi-value">0.2452</div>
            <div class="kpi-sub">BCE + Dice Composite Loss</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-label">Training Duration</div>
            <div class="kpi-value">76.9 min</div>
            <div class="kpi-sub">4,613s on NVIDIA CUDA GPU</div>
          </div>
        </div>

        <!-- Interactive SVG Line Charts -->
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:20px; margin-bottom:24px;">
          <div class="card" style="box-shadow:none; border:1px solid var(--border-color); margin-bottom:0;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
              <span class="card-title" style="font-size:0.92rem;">Validation Macro Dice Trajectory</span>
              <span style="font-size:0.75rem; color:var(--medical-teal); font-weight:600;">Peak = Round 17</span>
            </div>
            <div class="chart-container">
              <svg id="svg-dice-chart" viewBox="0 0 500 220" style="width:100%; height:100%; overflow:visible;"></svg>
            </div>
          </div>

          <div class="card" style="box-shadow:none; border:1px solid var(--border-color); margin-bottom:0;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
              <span class="card-title" style="font-size:0.92rem;">Validation Loss Convergence</span>
              <span style="font-size:0.75rem; color:var(--text-secondary);">Min Loss = 0.2375 (R17)</span>
            </div>
            <div class="chart-container">
              <svg id="svg-loss-chart" viewBox="0 0 500 220" style="width:100%; height:100%; overflow:visible;"></svg>
            </div>
          </div>
        </div>

        <!-- Complete 20-Round Table -->
        <div class="card-header" style="margin-top:10px;">
          <div class="card-title" style="font-size:0.98rem;">Round-by-Round Validation Telemetry</div>
        </div>
        <table class="data-table">
          <thead>
            <tr>
              <th>Round</th>
              <th class="num">Global Val Loss</th>
              <th class="num">Macro Dice</th>
              <th class="num">Macro IoU</th>
              <th class="num">MA Dice</th>
              <th class="num">HE Dice</th>
              <th class="num">EX Dice</th>
              <th class="num">SE Dice</th>
              <th>Checkpoint Selection</th>
            </tr>
          </thead>
          <tbody id="history-tbody">
            <tr><td colspan="9" style="text-align:center; padding:20px; color:var(--text-secondary);">Loading telemetry logs...</td></tr>
          </tbody>
        </table>
      </div>
    </section>

    <!-- ================================================================= -->
    <!-- TAB 4: Final Test Benchmark                                      -->
    <!-- ================================================================= -->
    <section id="benchmark-tab" class="tab-content">
      <div class="card">
        <div class="card-header">
          <div>
            <div class="card-title">Multi-Method Benchmark on Held-Out Test Cohort (n=252)</div>
            <div class="card-subtitle">Sample-weighted evaluation across 225 DDR test images + 27 IDRiD test images</div>
          </div>
          <span class="meta-tag tag-teal">Frozen Test Set Evaluation</span>
        </div>

        <!-- Benchmark Summary KPIs -->
        <div class="kpi-row">
          <div class="kpi-card teal-accent">
            <div class="kpi-label">FedAvg (Tuned Thresholds)</div>
            <div class="kpi-value" style="color:var(--medical-teal);">0.4760</div>
            <div class="kpi-sub">Macro Dice (+1.05% vs Uniform)</div>
          </div>
          <div class="kpi-card navy-accent">
            <div class="kpi-label">Centralized Supervised</div>
            <div class="kpi-value">0.4660</div>
            <div class="kpi-sub">Macro Dice (Pooled Centralized)</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-label">FedAvg (Uniform 0.50)</div>
            <div class="kpi-value">0.4655</div>
            <div class="kpi-sub">Macro Dice (Standard Baseline)</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-label">SimCLR Fine-Tuned</div>
            <div class="kpi-value">0.4376</div>
            <div class="kpi-sub">Macro Dice (SSL Pretrained)</div>
          </div>
        </div>

        <!-- Main Comparative Benchmark Table -->
        <table class="data-table" style="margin-bottom:24px;">
          <thead>
            <tr>
              <th>Learning Paradigm</th>
              <th>Data Governance &amp; Operating Thresholds</th>
              <th class="num">Macro Dice</th>
              <th class="num">Macro IoU</th>
              <th class="num">MA Dice</th>
              <th class="num">HE Dice</th>
              <th class="num">EX Dice</th>
              <th class="num">SE Dice</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td><strong>Centralized Supervised</strong></td>
              <td>Pooled Centralized Training (ResNet-50 U-Net &bull; Uniform 0.50)</td>
              <td class="num">0.4660</td><td class="num">0.3137</td>
              <td class="num">0.2490</td><td class="num"><strong>0.4679</strong></td>
              <td class="num">0.6089</td><td class="num">0.5380</td>
            </tr>
            <tr>
              <td><strong>SimCLR Fine-Tuned</strong></td>
              <td>Self-Supervised Pretraining + Supervised Fine-Tuning (&tau;=0.50)</td>
              <td class="num">0.4376</td><td class="num">0.2907</td>
              <td class="num">0.2032</td><td class="num">0.4367</td>
              <td class="num">0.5786</td><td class="num">0.5316</td>
            </tr>
            <tr>
              <td><strong>Flower FedAvg (Baseline)</strong></td>
              <td>Federated 2 Clients &bull; Standard Uniform 0.50 Thresholds</td>
              <td class="num">0.4655</td><td class="num">0.3134</td>
              <td class="num">0.2525</td><td class="num">0.4519</td>
              <td class="num">0.6074</td><td class="num">0.5502</td>
            </tr>
            <tr class="highlight-best">
              <td><strong>Flower FedAvg (Tuned)</strong></td>
              <td><strong>Federated 2 Clients &bull; Frozen Channel-Calibrated Thresholds</strong></td>
              <td class="num" style="color:var(--medical-teal); font-weight:700;">0.4760</td>
              <td class="num" style="color:var(--medical-teal); font-weight:700;">0.3229</td>
              <td class="num" style="font-weight:700;">0.2584</td>
              <td class="num">0.4664</td>
              <td class="num" style="font-weight:700;">0.6251</td>
              <td class="num" style="font-weight:700;">0.5541</td>
            </tr>
          </tbody>
        </table>

        <!-- Detailed Statistical & Clinical Analysis -->
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:20px; margin-bottom:20px;">
          <div class="card" style="box-shadow:none; border:1px solid var(--border-color); margin-bottom:0;">
            <div class="card-title" style="font-size:0.96rem; margin-bottom:8px;">Precision vs. Recall Operating Trade-Off</div>
            <p style="font-size:0.82rem; color:var(--text-secondary); line-height:1.6; margin-bottom:12px;">
              Optimizing thresholds specifically for microaneurysms (&tau;=0.25) and exudates (&tau;=0.35) shifted the classification operating point toward higher clinical recall:
            </p>
            <table class="data-table" style="font-size:0.80rem;">
              <tr><th>Operating Regime</th><th class="num">Precision</th><th class="num">Recall (Sensitivity)</th><th class="num">Specificity</th></tr>
              <tr><td>Uniform 0.50 Baseline</td><td class="num">60.02%</td><td class="num">38.96%</td><td class="num">99.95%</td></tr>
              <tr class="highlight-best"><td><strong>Channel-Tuned Model</strong></td><td class="num">55.92%</td><td class="num"><strong>43.56% (+4.60%)</strong></td><td class="num">99.93%</td></tr>
            </table>
          </div>

          <div class="card" style="box-shadow:none; border:1px solid var(--border-color); margin-bottom:0;">
            <div class="card-title" style="font-size:0.96rem; margin-bottom:8px;">Client Breakdown on Held-Out Cohort</div>
            <p style="font-size:0.82rem; color:var(--text-secondary); line-height:1.6; margin-bottom:12px;">
              Evaluation across both independent acquisition centers confirms consistent performance without cross-client degradation:
            </p>
            <table class="data-table" style="font-size:0.80rem;">
              <tr><th>Dataset Client</th><th class="num">Test Samples</th><th class="num">Uniform Dice</th><th class="num">Tuned Dice</th></tr>
              <tr><td>DDR Testing Partition</td><td class="num">n = 225</td><td class="num">0.4571</td><td class="num"><strong>0.4688</strong></td></tr>
              <tr><td>IDRiD Testing Partition</td><td class="num">n = 27</td><td class="num">0.5350</td><td class="num"><strong>0.5361</strong></td></tr>
            </table>
          </div>
        </div>

        <!-- Academic Discussion & Conclusion -->
        <div class="callout-box">
          <h4 style="color:#15803D; margin-bottom:6px; font-size:0.92rem;">Research Finding &amp; Clinical Context</h4>
          <p style="font-size:0.84rem; line-height:1.6;">
            With frozen channel-tuned thresholds, the federated model achieves <strong>0.4760</strong> Macro Dice versus <strong>0.4660</strong> for the centralized baseline. When evaluated at the exact same uniform 0.50 threshold, FedAvg (0.4655) and centralized supervised training (0.4660) are essentially equivalent (&Delta; &lt; 0.001). This proves that privacy-preserving federated aggregation can match centralized models without aggregating sensitive patient records in a central database.
          </p>
        </div>
      </div>
    </section>

  </div>
</main>

<script>
  let selectedFileObj = null;

  function switchNav(tabId, breadcrumbTitle) {
    document.querySelectorAll('.nav-item').forEach(btn => btn.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    
    // Highlight sidebar
    const clickedBtn = Array.from(document.querySelectorAll('.nav-item')).find(b => b.getAttribute('onclick').includes(tabId));
    if (clickedBtn) clickedBtn.classList.add('active');

    const targetTab = document.getElementById(tabId);
    if (targetTab) targetTab.classList.add('active');

    document.getElementById('current-breadcrumb').innerText = breadcrumbTitle;

    // Lazy load training history charts if selected
    if (tabId === 'training-tab') {
      loadTrainingHistoryData();
    }

    // Close mobile drawer if open
    const sidebar = document.getElementById('app-sidebar');
    if (sidebar.classList.contains('open')) {
      sidebar.classList.remove('open');
    }
  }

  function toggleSidebar() {
    const sidebar = document.getElementById('app-sidebar');
    sidebar.classList.toggle('open');
  }

  function triggerFileInput() {
    document.getElementById('file-upload').click();
  }

  function handleFileSelected(event) {
    const file = event.target.files[0];
    if (!file) return;
    setUploadedFile(file);
  }

  function setUploadedFile(file) {
    selectedFileObj = file;
    // Clear select dropdown
    document.getElementById('sample-select').value = "";
    
    // Show details pill
    const pill = document.getElementById('file-details-pill');
    pill.style.display = 'flex';
    document.getElementById('file-name-label').innerText = file.name;
    const sizeKb = (file.size / 1024).toFixed(1);
    document.getElementById('file-meta-label').innerText = sizeKb + ' KB (Custom Upload)';

    document.getElementById('dropzone-prompt').innerHTML = "Selected: <strong>" + file.name + "</strong>";
  }

  function onSampleSelected() {
    selectedFileObj = null;
    document.getElementById('file-upload').value = "";
    document.getElementById('file-details-pill').style.display = 'none';
    document.getElementById('dropzone-prompt').innerHTML = 'Drag &amp; drop fundus image here, or <span style="color:var(--medical-teal); text-decoration:underline;">browse</span>';
  }

  // Drag and drop handlers
  const dropzone = document.getElementById('dropzone');
  ['dragenter', 'dragover'].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.add('drag-over');
    }, false);
  });

  ['dragleave', 'drop'].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.remove('drag-over');
    }, false);
  });

  dropzone.addEventListener('drop', (e) => {
    const dt = e.dataTransfer;
    const files = dt.files;
    if (files && files.length > 0) {
      setUploadedFile(files[0]);
    }
  });

  // View mode switcher
  function setViewMode(mode) {
    const dualBtn = document.getElementById('vm-dual');
    const overlayBtn = document.getElementById('vm-overlay');
    const origBtn = document.getElementById('vm-orig');
    const origCard = document.getElementById('img-card-orig');
    const overlayCard = document.getElementById('img-card-overlay');
    const grid = document.getElementById('viewer-display-grid');

    [dualBtn, overlayBtn, origBtn].forEach(b => b.classList.remove('active'));

    if (mode === 'dual') {
      dualBtn.classList.add('active');
      origCard.style.display = 'flex';
      overlayCard.style.display = 'flex';
      grid.style.gridTemplateColumns = '1fr 1fr';
    } else if (mode === 'overlay') {
      overlayBtn.classList.add('active');
      origCard.style.display = 'none';
      overlayCard.style.display = 'flex';
      grid.style.gridTemplateColumns = '1fr';
    } else if (mode === 'orig') {
      origBtn.classList.add('active');
      origCard.style.display = 'flex';
      overlayCard.style.display = 'none';
      grid.style.gridTemplateColumns = '1fr';
    }
  }

  // Execute inference pipeline via FastAPI POST /api/predict
  async function executeInference() {
    const btn = document.getElementById('btn-run');
    const spinner = document.getElementById('btn-spinner');
    const btnText = document.getElementById('btn-text');
    
    btn.disabled = true;
    spinner.style.display = 'block';
    btnText.innerText = "Analyzing Lesions...";

    const formData = new FormData();
    const sampleSelect = document.getElementById('sample-select');

    if (selectedFileObj) {
      formData.append('file', selectedFileObj);
    } else if (sampleSelect.value) {
      formData.append('sample_key', sampleSelect.value);
    } else {
      alert("Please select a pre-loaded sample or upload an image file.");
      btn.disabled = false;
      spinner.style.display = 'none';
      btnText.innerText = "Execute Multi-Task Segmentation";
      return;
    }

    try {
      const resp = await fetch('/api/predict', { method: 'POST', body: formData });
      if (!resp.ok) {
        const err = await resp.json();
        alert('Inference error: ' + (err.detail || 'Segmentation failed.'));
        return;
      }
      const data = await resp.json();
      renderInferenceResults(data);
    } catch (e) {
      alert('Network or server error executing inference: ' + e);
    } finally {
      btn.disabled = false;
      spinner.style.display = 'none';
      btnText.innerText = "Execute Multi-Task Segmentation";
    }
  }

  function renderInferenceResults(data) {
    document.getElementById('img-orig').src = data.images.original_b64;
    document.getElementById('img-overlay').src = data.images.overlay_b64;

    const dimStr = data.original_dimensions.width + ' \u00d7 ' + data.original_dimensions.height + ' px';
    document.getElementById('orig-dim').innerText = dimStr;
    document.getElementById('latency-badge').innerText = data.latency_ms + ' ms (' + data.metadata.device.toUpperCase() + ')';
    document.getElementById('viewer-sample-id').innerText = 'Sample: ' + data.sample_name + ' \u2022 ' + dimStr;
    document.getElementById('info-device-label').innerText = data.metadata.device.toUpperCase();
    document.getElementById('info-latency-label').innerText = data.latency_ms.toFixed(1) + ' ms';
    document.getElementById('top-device-badge').innerText = 'Device: ' + data.metadata.device.toUpperCase();

    const chs = data.channels;

    // MA
    document.getElementById('img-ma').src = chs.MA.colored_png_b64;
    document.getElementById('stat-ma-px').innerText = chs.MA.positive_pixels.toLocaleString() + ' px';
    document.getElementById('stat-ma-pct').innerText = chs.MA.area_pct.toFixed(4) + '%';
    document.getElementById('stat-ma-status').innerHTML = chs.MA.detected 
      ? '<span style="color:#16A34A; font-weight:700;">&bull; DETECTED</span>' 
      : '<span style="color:#94A3B8;">&bull; Not Detected</span>';

    // HE
    document.getElementById('img-he').src = chs.HE.colored_png_b64;
    document.getElementById('stat-he-px').innerText = chs.HE.positive_pixels.toLocaleString() + ' px';
    document.getElementById('stat-he-pct').innerText = chs.HE.area_pct.toFixed(4) + '%';
    document.getElementById('stat-he-status').innerHTML = chs.HE.detected 
      ? '<span style="color:#16A34A; font-weight:700;">&bull; DETECTED</span>' 
      : '<span style="color:#94A3B8;">&bull; Not Detected</span>';

    // EX
    document.getElementById('img-ex').src = chs.EX.colored_png_b64;
    document.getElementById('stat-ex-px').innerText = chs.EX.positive_pixels.toLocaleString() + ' px';
    document.getElementById('stat-ex-pct').innerText = chs.EX.area_pct.toFixed(4) + '%';
    document.getElementById('stat-ex-status').innerHTML = chs.EX.detected 
      ? '<span style="color:#16A34A; font-weight:700;">&bull; DETECTED</span>' 
      : '<span style="color:#94A3B8;">&bull; Not Detected</span>';

    // SE
    document.getElementById('img-se').src = chs.SE.colored_png_b64;
    document.getElementById('stat-se-px').innerText = chs.SE.positive_pixels.toLocaleString() + ' px';
    document.getElementById('stat-se-pct').innerText = chs.SE.area_pct.toFixed(4) + '%';
    document.getElementById('stat-se-status').innerHTML = chs.SE.detected 
      ? '<span style="color:#16A34A; font-weight:700;">&bull; DETECTED</span>' 
      : '<span style="color:#94A3B8;">&bull; Not Detected</span>';

    // Populate Quantification Table
    const tbody = document.getElementById('quant-tbody');
    tbody.innerHTML = '';
    for (const [code, info] of Object.entries(chs)) {
      const tr = document.createElement('tr');
      const statusBadge = info.detected 
        ? '<span style="color:#16A34A; font-weight:700; background:#DCFCE7; padding:2px 8px; border-radius:4px;">DETECTED</span>'
        : '<span style="color:#64748B; background:#F1F5F9; padding:2px 8px; border-radius:4px;">Not Detected</span>';
      
      tr.innerHTML = `
        <td><strong>${code}</strong> &bull; ${info.name}</td>
        <td class="num">${info.threshold.toFixed(2)}</td>
        <td class="num" style="font-weight:600;">${info.positive_pixels.toLocaleString()}</td>
        <td class="num">${info.area_pct.toFixed(4)}%</td>
        <td>${statusBadge}</td>
      `;
      tbody.appendChild(tr);
    }
  }

  // Training History Telemetry & SVG Chart Rendering
  let trainingDataLoaded = false;
  async function loadTrainingHistoryData() {
    if (trainingDataLoaded) return;
    try {
      const resp = await fetch('/api/training-history');
      const data = await resp.json();
      if (!data.per_round_val_metrics) return;

      const tbody = document.getElementById('history-tbody');
      tbody.innerHTML = '';

      const rounds = [];
      const diceScores = [];
      const losses = [];

      data.per_round_val_metrics.forEach(vm => {
        const isBest = vm.round === data.best_round;
        const rNum = vm.round;
        const lossVal = vm.loss ? vm.loss : (vm.global_val_loss || 0);
        const diceVal = vm.dice_macro ? vm.dice_macro : (vm.dice || 0);
        const iouVal = vm.iou_macro ? vm.iou_macro : (vm.iou || 0);

        rounds.push(rNum);
        diceScores.push(diceVal);
        losses.push(lossVal);

        const tr = document.createElement('tr');
        if (isBest) tr.classList.add('highlight-best');

        const statusHtml = isBest 
          ? '<span style="color:#15803D; font-weight:700;">&#9733; Best Global Model (R17)</span>' 
          : (rNum === 20 ? 'Final Round' : 'Completed');

        const maStr = vm.dice_MA != null ? vm.dice_MA.toFixed(4) : '—';
        const heStr = vm.dice_HE != null ? vm.dice_HE.toFixed(4) : '—';
        const exStr = vm.dice_EX != null ? vm.dice_EX.toFixed(4) : '—';
        const seStr = vm.dice_SE != null ? vm.dice_SE.toFixed(4) : '—';

        tr.innerHTML = `
          <td><strong>Round ${rNum}</strong></td>
          <td class="num">${lossVal.toFixed(4)}</td>
          <td class="num" style="${isBest ? 'color:#15803D; font-weight:700;' : ''}">${diceVal.toFixed(4)}</td>
          <td class="num">${iouVal.toFixed(4)}</td>
          <td class="num">${maStr}</td>
          <td class="num">${heStr}</td>
          <td class="num">${exStr}</td>
          <td class="num">${seStr}</td>
          <td>${statusHtml}</td>
        `;
        tbody.appendChild(tr);
      });

      renderSvgLineChart('svg-dice-chart', rounds, diceScores, 0.40, 0.60, '#12A89D', 'Round 17: 0.5657', 17);
      renderSvgLineChart('svg-loss-chart', rounds, losses, 0.20, 0.50, '#E11D48', 'Round 17: 0.2375', 17);

      trainingDataLoaded = true;
    } catch (e) {
      console.error("Failed to load training telemetry:", e);
    }
  }

  function renderSvgLineChart(svgId, xVals, yVals, yMin, yMax, strokeColor, peakLabel, peakX) {
    const svg = document.getElementById(svgId);
    if (!svg || xVals.length === 0) return;

    const width = 500;
    const height = 220;
    const padL = 45;
    const padR = 25;
    const padT = 25;
    const padB = 30;

    const plotW = width - padL - padR;
    const plotH = height - padT - padB;

    function getX(val) {
      return padL + ((val - 1) / (xVals.length - 1)) * plotW;
    }
    function getY(val) {
      const clamped = Math.max(yMin, Math.min(yMax, val));
      return padT + plotH - ((clamped - yMin) / (yMax - yMin)) * plotH;
    }

    let pathD = '';
    let peakCoords = { x: 0, y: 0 };

    xVals.forEach((x, i) => {
      const cx = getX(x);
      const cy = getY(yVals[i]);
      if (i === 0) pathD += `M ${cx} ${cy} `;
      else pathD += `L ${cx} ${cy} `;

      if (x === peakX) {
        peakCoords = { x: cx, y: cy };
      }
    });

    // Area fill
    const areaD = `${pathD} L ${getX(xVals[xVals.length-1])} ${getY(yMin)} L ${getX(1)} ${getY(yMin)} Z`;

    let html = `
      <!-- Grid lines -->
      <line x1="${padL}" y1="${getY(yMin)}" x2="${width-padR}" y2="${getY(yMin)}" stroke="#E2E8F0" stroke-width="1"/>
      <line x1="${padL}" y1="${getY((yMin+yMax)/2)}" x2="${width-padR}" y2="${getY((yMin+yMax)/2)}" stroke="#E2E8F0" stroke-width="1" stroke-dasharray="4,4"/>
      <line x1="${padL}" y1="${getY(yMax)}" x2="${width-padR}" y2="${getY(yMax)}" stroke="#E2E8F0" stroke-width="1" stroke-dasharray="4,4"/>

      <!-- Y axis labels -->
      <text x="${padL - 8}" y="${getY(yMin) + 4}" fill="#94A3B8" font-size="10" text-anchor="end">${yMin.toFixed(2)}</text>
      <text x="${padL - 8}" y="${getY((yMin+yMax)/2) + 4}" fill="#94A3B8" font-size="10" text-anchor="end">${((yMin+yMax)/2).toFixed(2)}</text>
      <text x="${padL - 8}" y="${getY(yMax) + 4}" fill="#94A3B8" font-size="10" text-anchor="end">${yMax.toFixed(2)}</text>

      <!-- X axis labels -->
      <text x="${getX(1)}" y="${height - 10}" fill="#94A3B8" font-size="10" text-anchor="middle">R1</text>
      <text x="${getX(5)}" y="${height - 10}" fill="#94A3B8" font-size="10" text-anchor="middle">R5</text>
      <text x="${getX(10)}" y="${height - 10}" fill="#94A3B8" font-size="10" text-anchor="middle">R10</text>
      <text x="${getX(15)}" y="${height - 10}" fill="#94A3B8" font-size="10" text-anchor="middle">R15</text>
      <text x="${getX(20)}" y="${height - 10}" fill="#94A3B8" font-size="10" text-anchor="middle">R20</text>

      <!-- Area gradient -->
      <path d="${areaD}" fill="${strokeColor}" fill-opacity="0.08"/>

      <!-- Path line -->
      <path d="${pathD}" fill="none" stroke="${strokeColor}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
    `;

    // Highlight peak
    if (peakX) {
      html += `
        <circle cx="${peakCoords.x}" cy="${peakCoords.y}" r="5" fill="#FFFFFF" stroke="${strokeColor}" stroke-width="3"/>
        <rect x="${peakCoords.x - 45}" y="${peakCoords.y - 24}" width="90" height="18" rx="4" fill="#142B45"/>
        <text x="${peakCoords.x}" y="${peakCoords.y - 12}" fill="#FFFFFF" font-size="9" font-weight="bold" text-anchor="middle">${peakLabel}</text>
      `;
    }

    svg.innerHTML = html;
  }

  // Auto-run initial verification sample on DOM load
  window.addEventListener('DOMContentLoaded', () => {
    executeInference();
  });
</script>

</body>
</html>
"""
    return html_content


def main() -> None:
    parser = argparse.ArgumentParser(description="FedMorphis Demonstration Platform")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    parser.add_argument("--device", type=str, default="auto", help="Device ('auto', 'cuda', 'cpu')")
    args = parser.parse_args()

    # Pre-initialize pipeline
    _ = get_pipeline()

    print("\n" + "=" * 80)
    print("FEDMORPHIS DEMONSTRATION PLATFORM")
    print("=" * 80)
    print(f"Server URL:     http://{args.host}:{args.port}")
    print(f"Device:         {_pipeline.device if _pipeline else 'auto'}")
    print(f"Checkpoint:     {DEFAULT_CHECKPOINT_PATH}")
    print(f"Thresholds:     {DEFAULT_FROZEN_THRESHOLDS}")
    print("-" * 80)
    print("Open your browser and navigate to the Server URL to explore the interactive demo.")
    print("=" * 80 + "\n")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()

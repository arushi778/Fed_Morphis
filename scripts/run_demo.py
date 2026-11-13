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

# Sample images available for 1-click demonstration
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
  <title>FedMorphis — Federated Retinal Lesion Segmentation Prototype</title>
  <style>
    :root {
      --bg-dark: #0f1117;
      --card-bg: #181b24;
      --card-border: #262a36;
      --accent-blue: #3b82f6;
      --accent-green: #10b981;
      --text-main: #f3f4f6;
      --text-muted: #9ca3af;
      --ma-color: #ff3232;
      --he-color: #32cd32;
      --ex-color: #ffd700;
      --se-color: #00ced1;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background-color: var(--bg-dark);
      color: var(--text-main);
      line-height: 1.5;
      padding-bottom: 50px;
    }
    header {
      background: linear-gradient(180deg, #161922 0%, var(--bg-dark) 100%);
      padding: 24px 32px;
      border-bottom: 1px solid var(--card-border);
    }
    .header-content {
      max-width: 1400px;
      margin: 0 auto;
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 16px;
    }
    .title-group h1 { font-size: 1.6rem; font-weight: 700; color: #fff; display: flex; align-items: center; gap: 10px; }
    .badge {
      display: inline-block;
      padding: 3px 9px;
      font-size: 0.75rem;
      font-weight: 600;
      border-radius: 9999px;
      background: rgba(59, 130, 246, 0.2);
      color: #60a5fa;
      border: 1px solid rgba(59, 130, 246, 0.4);
    }
    .subtitle { font-size: 0.9rem; color: var(--text-muted); margin-top: 4px; }
    .disclaimer-banner {
      background: rgba(239, 68, 68, 0.1);
      border: 1px solid rgba(239, 68, 68, 0.3);
      color: #fca5a5;
      padding: 10px 16px;
      border-radius: 8px;
      font-size: 0.82rem;
      max-width: 1400px;
      margin: 16px auto 0;
    }
    .container { max-width: 1400px; margin: 24px auto; padding: 0 24px; }
    
    /* Navigation Tabs */
    .tabs {
      display: flex;
      gap: 8px;
      border-bottom: 1px solid var(--card-border);
      margin-bottom: 24px;
    }
    .tab-btn {
      background: transparent;
      border: none;
      color: var(--text-muted);
      padding: 12px 20px;
      font-size: 0.95rem;
      font-weight: 600;
      cursor: pointer;
      border-bottom: 2px solid transparent;
      transition: all 0.2s;
    }
    .tab-btn:hover { color: #fff; }
    .tab-btn.active {
      color: var(--accent-blue);
      border-bottom-color: var(--accent-blue);
    }
    .tab-content { display: none; }
    .tab-content.active { display: block; }
    
    /* Grid & Cards */
    .grid-2 { display: grid; grid-template-columns: 360px 1fr; gap: 24px; }
    @media (max-width: 960px) { .grid-2 { grid-template-columns: 1fr; } }
    .card {
      background-color: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 12px;
      padding: 20px;
      margin-bottom: 20px;
    }
    .card-title {
      font-size: 1.1rem;
      font-weight: 600;
      margin-bottom: 16px;
      color: #fff;
      display: flex;
      align-items: center;
      gap: 8px;
    }
    
    /* Form inputs */
    label { display: block; font-size: 0.85rem; color: var(--text-muted); margin-bottom: 6px; font-weight: 500; }
    select, input[type="file"] {
      width: 100%;
      background: #11141c;
      border: 1px solid var(--card-border);
      color: #fff;
      padding: 10px 12px;
      border-radius: 8px;
      font-size: 0.9rem;
      margin-bottom: 16px;
    }
    button.btn-primary {
      width: 100%;
      background: var(--accent-blue);
      color: #fff;
      border: none;
      padding: 12px;
      border-radius: 8px;
      font-size: 0.95rem;
      font-weight: 600;
      cursor: pointer;
      transition: background 0.2s;
    }
    button.btn-primary:hover { background: #2563eb; }
    
    /* Image Display Area */
    .viewer-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
      margin-bottom: 20px;
    }
    @media (max-width: 768px) { .viewer-grid { grid-template-columns: 1fr; } }
    .img-box {
      background: #090a0f;
      border: 1px solid var(--card-border);
      border-radius: 8px;
      overflow: hidden;
      display: flex;
      flex-direction: column;
    }
    .img-header {
      padding: 8px 12px;
      background: #13161f;
      font-size: 0.85rem;
      font-weight: 600;
      border-bottom: 1px solid var(--card-border);
      display: flex;
      justify-content: space-between;
    }
    .img-container {
      width: 100%;
      aspect-ratio: 1 / 1;
      background: #050608;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    .img-container img {
      max-width: 100%;
      max-height: 100%;
      object-fit: contain;
    }
    
    /* 4-Channel Lesion Panels */
    .channels-grid {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 12px;
    }
    @media (max-width: 800px) { .channels-grid { grid-template-columns: repeat(2, 1fr); } }
    .channel-box {
      background: #0e1118;
      border: 1px solid var(--card-border);
      border-radius: 8px;
      padding: 10px;
      text-align: center;
    }
    .channel-pill {
      display: inline-block;
      font-size: 0.75rem;
      font-weight: 700;
      padding: 2px 8px;
      border-radius: 4px;
      margin-bottom: 8px;
    }
    .pill-ma { background: rgba(255, 50, 50, 0.2); color: var(--ma-color); border: 1px solid var(--ma-color); }
    .pill-he { background: rgba(50, 205, 50, 0.2); color: var(--he-color); border: 1px solid var(--he-color); }
    .pill-ex { background: rgba(255, 215, 0, 0.2); color: var(--ex-color); border: 1px solid var(--ex-color); }
    .pill-se { background: rgba(0, 206, 209, 0.2); color: var(--se-color); border: 1px solid var(--se-color); }
    
    /* Tables */
    table { width: 100%; border-collapse: collapse; font-size: 0.88rem; }
    th, td { padding: 10px 12px; text-align: left; border-bottom: 1px solid var(--card-border); }
    th { color: var(--text-muted); font-weight: 600; background: rgba(255, 255, 255, 0.02); }
    tr:last-child td { border-bottom: none; }
    .num { text-align: right; font-variant-numeric: tabular-nums; }
    .highlight-row { background: rgba(59, 130, 246, 0.1); font-weight: 600; color: #93c5fd; }
    
    /* Architecture diagram */
    .arch-flow {
      display: flex;
      align-items: center;
      justify-content: space-around;
      padding: 30px 10px;
      gap: 16px;
      flex-wrap: wrap;
    }
    .arch-node {
      background: #1e2330;
      border: 1px solid #333a4d;
      border-radius: 10px;
      padding: 18px 24px;
      text-align: center;
      min-width: 200px;
    }
    .arch-arrow { font-size: 1.5rem; color: var(--accent-blue); font-weight: bold; }
    
    .loading-spinner {
      display: none;
      text-align: center;
      padding: 20px;
      color: var(--accent-blue);
      font-weight: 600;
    }
  </style>
</head>
<body>

<header>
  <div class="header-content">
    <div class="title-group">
      <h1>FedMorphis <span class="badge">Flower 1.43.0 FedAvg</span></h1>
      <div class="subtitle">Multi-Lesion Retinal Fundus Segmentation Prototype with Channel-Calibrated Thresholding</div>
    </div>
    <div>
      <span class="badge" style="background: rgba(16, 185, 129, 0.2); color: #34d399; border-color: rgba(16, 185, 129, 0.4);">
        Best Global Checkpoint: Round 17 (ResNet50 U-Net)
      </span>
    </div>
  </div>
  <div class="disclaimer-banner">
    ⚠️ <strong>Demonstration Disclaimer:</strong> This software is an academic engineering prototype developed for university research and demonstration purposes. Model predictions are not clinically certified or intended for independent medical diagnostic use.
  </div>
</header>

<div class="container">
  <!-- Navigation Tabs -->
  <div class="tabs">
    <button class="tab-btn active" onclick="switchTab('inference-tab')">🔍 Live Lesion Inference</button>
    <button class="tab-btn" onclick="switchTab('architecture-tab')">🌐 Flower Federated Architecture</button>
    <button class="tab-btn" onclick="switchTab('training-tab')">📈 20-Round Training History</button>
    <button class="tab-btn" onclick="switchTab('benchmark-tab')">🏆 Final Held-Out Benchmark (n=252)</button>
  </div>

  <!-- TAB 1: Live Inference -->
  <div id="inference-tab" class="tab-content active">
    <div class="grid-2">
      <!-- Left Column: Controls -->
      <div>
        <div class="card">
          <div class="card-title">Select or Upload Image</div>
          <label>Pre-Loaded Sample Images (non-test splits)</label>
          <select id="sample-select" onchange="onSampleSelected()">
            <option value="DDR_Sample_1">DDR: 007-5797-300 (validation)</option>
            <option value="DDR_Sample_2">DDR: 007-5800-300 (validation)</option>
            <option value="IDRiD_Sample_1" selected>IDRiD: IDRiD_03 (training split)</option>
            <option value="IDRiD_Sample_2">IDRiD: IDRiD_08 (training split)</option>
          </select>

          <label>Or Upload Custom Fundus Image (JPG, PNG)</label>
          <input type="file" id="file-upload" accept="image/*" onchange="onFileUploaded()">

          <button class="btn-primary" id="btn-run" onclick="executeInference()">Segment Retinal Lesions</button>
          <div id="spinner" class="loading-spinner">Running ResNet-50 U-Net Inference...</div>
        </div>

        <div class="card">
          <div class="card-title">Frozen Threshold Settings</div>
          <table>
            <thead><tr><th>Lesion Channel</th><th class="num">Threshold</th><th class="num">Color</th></tr></thead>
            <tbody>
              <tr><td><strong>MA</strong> (Microaneurysms)</td><td class="num">0.25</td><td class="num" style="color:var(--ma-color)">■ Vivid Red</td></tr>
              <tr><td><strong>HE</strong> (Hemorrhages)</td><td class="num">0.40</td><td class="num" style="color:var(--he-color)">■ Lime Green</td></tr>
              <tr><td><strong>EX</strong> (Hard Exudates)</td><td class="num">0.35</td><td class="num" style="color:var(--ex-color)">■ Gold / Yellow</td></tr>
              <tr><td><strong>SE</strong> (Soft Exudates)</td><td class="num">0.35</td><td class="num" style="color:var(--se-color)">■ Cyan</td></tr>
            </tbody>
          </table>
        </div>
      </div>

      <!-- Right Column: Visual Displays -->
      <div>
        <div class="card">
          <div class="card-title">Segmentation Results</div>
          <div class="viewer-grid">
            <div class="img-box">
              <div class="img-header">
                <span>Original Fundus Image</span>
                <span id="orig-dim" style="color:var(--text-muted); font-size:0.75rem;">—</span>
              </div>
              <div class="img-container">
                <img id="img-orig" alt="Original Image" src="">
              </div>
            </div>
            <div class="img-box">
              <div class="img-header">
                <span>Multi-Lesion Composite Overlay</span>
                <span id="latency-badge" style="color:var(--accent-green); font-size:0.75rem;">—</span>
              </div>
              <div class="img-container">
                <img id="img-overlay" alt="Segmentation Overlay" src="">
              </div>
            </div>
          </div>

          <!-- 4 Separate Lesion Panels -->
          <div class="card-title" style="font-size:0.95rem; margin-top:10px;">Individual Lesion Channels</div>
          <div class="channels-grid">
            <div class="channel-box">
              <span class="channel-pill pill-ma">MA (Microaneurysms)</span>
              <div class="img-container" style="aspect-ratio:1/1;">
                <img id="img-ma" alt="MA Mask" src="">
              </div>
              <div id="stat-ma" style="font-size:0.75rem; color:var(--text-muted); margin-top:4px;">—</div>
            </div>
            <div class="channel-box">
              <span class="channel-pill pill-he">HE (Hemorrhages)</span>
              <div class="img-container" style="aspect-ratio:1/1;">
                <img id="img-he" alt="HE Mask" src="">
              </div>
              <div id="stat-he" style="font-size:0.75rem; color:var(--text-muted); margin-top:4px;">—</div>
            </div>
            <div class="channel-box">
              <span class="channel-pill pill-ex">EX (Hard Exudates)</span>
              <div class="img-container" style="aspect-ratio:1/1;">
                <img id="img-ex" alt="EX Mask" src="">
              </div>
              <div id="stat-ex" style="font-size:0.75rem; color:var(--text-muted); margin-top:4px;">—</div>
            </div>
            <div class="channel-box">
              <span class="channel-pill pill-se">SE (Soft Exudates)</span>
              <div class="img-container" style="aspect-ratio:1/1;">
                <img id="img-se" alt="SE Mask" src="">
              </div>
              <div id="stat-se" style="font-size:0.75rem; color:var(--text-muted); margin-top:4px;">—</div>
            </div>
          </div>
        </div>

        <div class="card">
          <div class="card-title">Lesion Pixel Quantification</div>
          <table id="quant-table">
            <thead>
              <tr>
                <th>Channel</th>
                <th>Threshold</th>
                <th class="num">Positive Pixels</th>
                <th class="num">Retinal Area %</th>
                <th>Detection Status</th>
              </tr>
            </thead>
            <tbody id="quant-tbody">
              <tr><td colspan="5" style="text-align:center; color:var(--text-muted);">Run inference to view pixel quantification.</td></tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  </div>

  <!-- TAB 2: Federated Architecture -->
  <div id="architecture-tab" class="tab-content">
    <div class="card">
      <div class="card-title">Flower 1.43.0 Federated Topology & Client Partitions</div>
      <p style="color:var(--text-muted); margin-bottom:20px;">
        FedMorphis trains a shared multi-task ResNet-50 U-Net across two dataset-defined clients (DDR and IDRiD), exchanging only model weights rather than fundus images. The recorded run was a single-machine Flower simulation, not a live multi-site deployment. This page shows recorded results; no training runs here.
      </p>

      <div class="arch-flow">
        <div class="arch-node">
          <div style="font-size:1.8rem; margin-bottom:6px;">🏥</div>
          <strong>Client 0: DDR</strong>
          <div style="color:var(--text-muted); font-size:0.8rem; margin-top:4px;">383 Train | 149 Val</div>
          <div style="color:#60a5fa; font-size:0.75rem; margin-top:4px;">Weight = 383/427 (89.7%)</div>
        </div>

        <div class="arch-arrow">➔ Updates ➔<br>⮌ Global ⮌</div>

        <div class="arch-node" style="border-color:var(--accent-blue); background:#152238;">
          <div style="font-size:1.8rem; margin-bottom:6px;">🌐</div>
          <strong>Flower Central Server</strong>
          <div style="color:var(--accent-green); font-size:0.8rem; margin-top:4px;">Strategy: FedAvg</div>
          <div style="color:#94a3b8; font-size:0.75rem; margin-top:4px;">20 Communication Rounds</div>
        </div>

        <div class="arch-arrow">➔ Updates ➔<br>⮌ Global ⮌</div>

        <div class="arch-node">
          <div style="font-size:1.8rem; margin-bottom:6px;">🏥</div>
          <strong>Client 1: IDRiD</strong>
          <div style="color:var(--text-muted); font-size:0.8rem; margin-top:4px;">44 Train | 10 Val</div>
          <div style="color:#60a5fa; font-size:0.75rem; margin-top:4px;">Weight = 44/427 (10.3%)</div>
        </div>
      </div>

      <div style="display:grid; grid-template-columns:1fr 1fr; gap:20px; margin-top:20px;">
        <div class="card" style="margin-bottom:0;">
          <div class="card-title" style="font-size:1rem;">Key Federated Hyperparameters</div>
          <table>
            <tr><td>Federated Framework</td><td class="num">Flower 1.43.0 (flwr)</td></tr>
            <tr><td>Aggregation Strategy</td><td class="num">FedAvg with BN Sharing</td></tr>
            <tr><td>Total Communication Rounds</td><td class="num">20 Rounds</td></tr>
            <tr><td>Client Local Epochs</td><td class="num">3 Epochs / Round</td></tr>
            <tr><td>Local Batch Size / Optimizer</td><td class="num">Batch 8 / AdamW (lr=1e-4)</td></tr>
            <tr><td>Mixed Precision</td><td class="num">PyTorch AMP (CUDA)</td></tr>
          </table>
        </div>
        <div class="card" style="margin-bottom:0;">
          <div class="card-title" style="font-size:1rem;">Privacy & Compliance Safeguards</div>
          <ul style="color:var(--text-muted); font-size:0.88rem; padding-left:20px; line-height:1.8;">
            <li>Zero raw retinal image or mask transmission across network boundaries.</li>
            <li>Client data partitions strictly isolated across independent directories.</li>
            <li>Deterministic seeds (42) and verified non-overlapping client records.</li>
            <li>Held-out test cohort (252 samples) strictly isolated and never accessed during training.</li>
          </ul>
        </div>
      </div>
    </div>
  </div>

  <!-- TAB 3: 20-Round Training History -->
  <div id="training-tab" class="tab-content">
    <div class="card">
      <div class="card-title">20-Round FedAvg Global Convergence Profile</div>
      <p style="color:var(--text-muted); margin-bottom:16px;">
        Recorded (historical) metrics read from logs/federated_fedavg_full_20rounds_report.json; these are not live. Validation split only (159 images). Best global checkpoint was selected at Round 17 by validation macro Dice.
      </p>
      <table>
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
            <th>Checkpoint Status</th>
          </tr>
        </thead>
        <tbody id="history-tbody">
          <!-- Dynamically populated -->
        </tbody>
      </table>
    </div>
  </div>

  <!-- TAB 4: Final Test Benchmark -->
  <div id="benchmark-tab" class="tab-content">
    <div class="card">
      <div class="card-title">Final Held-Out Benchmark on 252 Untouched Test Images</div>
      <p style="color:var(--text-muted); margin-bottom:16px;">
        Sample-weighted evaluation over 225 DDR test images + 27 IDRiD test images with frozen thresholds:
      </p>

      <table>
        <thead>
          <tr>
            <th>Method</th>
            <th>Supervision / Federated Setup</th>
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
            <td>Centralized Supervised</td>
            <td>Pooled Centralized Training (ResNet50 U-Net)</td>
            <td class="num">0.4660</td><td class="num">0.3137</td>
            <td class="num">0.2490</td><td class="num"><strong>0.4679</strong></td>
            <td class="num">0.6089</td><td class="num">0.5380</td>
          </tr>
          <tr>
            <td>SimCLR Fine-Tuned</td>
            <td>Self-Supervised Pretraining + Fine-Tuning</td>
            <td class="num">0.4376</td><td class="num">0.2907</td>
            <td class="num">0.2032</td><td class="num">0.4367</td>
            <td class="num">0.5786</td><td class="num">0.5316</td>
          </tr>
          <tr>
            <td>Flower FedAvg (Baseline)</td>
            <td>Federated (2 Clients, Uniform 0.50 Threshold)</td>
            <td class="num">0.4655</td><td class="num">0.3134</td>
            <td class="num">0.2525</td><td class="num">0.4519</td>
            <td class="num">0.6074</td><td class="num">0.5502</td>
          </tr>
          <tr class="highlight-row">
            <td><strong>Flower FedAvg (Tuned)</strong></td>
            <td><strong>Federated (Frozen Channel-Tuned Thresholds)</strong></td>
            <td class="num"><strong>0.4760</strong></td><td class="num"><strong>0.3229</strong></td>
            <td class="num"><strong>0.2584</strong></td><td class="num">0.4664</td>
            <td class="num"><strong>0.6251</strong></td><td class="num"><strong>0.5541</strong></td>
          </tr>
        </tbody>
      </table>

      <div style="margin-top:24px; padding:16px; background:rgba(16, 185, 129, 0.1); border:1px solid rgba(16, 185, 129, 0.3); border-radius:8px;">
        <h4 style="color:#34d399; margin-bottom:8px;">Key Benchmark Finding</h4>
        <p style="font-size:0.9rem; color:#d1fae5;">
          With frozen channel-tuned thresholds, the federated model scores <strong>0.4760</strong> Macro Dice versus <strong>0.4660</strong> for the centralized baseline. Caveat: the thresholds were tuned only for the federated model, while the other rows use a uniform 0.50 threshold, and a 0.01 Dice gap on 252 images is small and has no significance test. With the same 0.50 threshold, FedAvg (0.4655) and centralized (0.4660) are essentially equal. These are research-prototype results, not clinical validation.
        </p>
      </div>
    </div>
  </div>
</div>

<script>
  function switchTab(tabId) {
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
    event.target.classList.add('active');
    document.getElementById(tabId).classList.add('active');

    if (tabId === 'training-tab') loadTrainingHistoryData();
  }

  function onFileUploaded() {
    document.getElementById('sample-select').value = "";
  }

  function onSampleSelected() {
    document.getElementById('file-upload').value = "";
  }

  async function executeInference() {
    const spinner = document.getElementById('spinner');
    const btn = document.getElementById('btn-run');
    spinner.style.display = 'block';
    btn.disabled = true;

    const fileInput = document.getElementById('file-upload');
    const sampleSelect = document.getElementById('sample-select');

    const formData = new FormData();
    if (fileInput.files.length > 0) {
      formData.append('file', fileInput.files[0]);
    } else {
      formData.append('sample_key', sampleSelect.value);
    }

    try {
      const resp = await fetch('/api/predict', { method: 'POST', body: formData });
      if (!resp.ok) {
        const err = await resp.json();
        alert('Inference error: ' + (err.detail || 'Unknown error'));
        return;
      }
      const data = await resp.json();
      renderInferenceResults(data);
    } catch (e) {
      alert('Network or server error: ' + e);
    } finally {
      spinner.style.display = 'none';
      btn.disabled = false;
    }
  }

  function renderInferenceResults(data) {
    document.getElementById('img-orig').src = data.images.original_b64;
    document.getElementById('img-overlay').src = data.images.overlay_b64;

    document.getElementById('orig-dim').innerText = data.original_dimensions.width + ' × ' + data.original_dimensions.height;
    document.getElementById('latency-badge').innerText = data.latency_ms + ' ms (' + data.metadata.device + ')';

    const chs = data.channels;
    document.getElementById('img-ma').src = chs.MA.colored_png_b64;
    document.getElementById('stat-ma').innerText = chs.MA.detected ? chs.MA.positive_pixels + ' px (' + chs.MA.area_pct + '%)' : 'None detected';

    document.getElementById('img-he').src = chs.HE.colored_png_b64;
    document.getElementById('stat-he').innerText = chs.HE.detected ? chs.HE.positive_pixels + ' px (' + chs.HE.area_pct + '%)' : 'None detected';

    document.getElementById('img-ex').src = chs.EX.colored_png_b64;
    document.getElementById('stat-ex').innerText = chs.EX.detected ? chs.EX.positive_pixels + ' px (' + chs.EX.area_pct + '%)' : 'None detected';

    document.getElementById('img-se').src = chs.SE.colored_png_b64;
    document.getElementById('stat-se').innerText = chs.SE.detected ? chs.SE.positive_pixels + ' px (' + chs.SE.area_pct + '%)' : 'None detected';

    // Populate table
    const tbody = document.getElementById('quant-tbody');
    tbody.innerHTML = '';
    for (const [code, info] of Object.entries(chs)) {
      const tr = document.createElement('tr');
      const statusColor = info.detected ? '#10b981' : '#6b7280';
      const statusText = info.detected ? 'DETECTED' : 'Not Detected';
      tr.innerHTML = `
        <td><strong>${code}</strong> (${info.name})</td>
        <td>${info.threshold}</td>
        <td class="num">${info.positive_pixels.toLocaleString()}</td>
        <td class="num">${info.area_pct.toFixed(4)}%</td>
        <td><span style="color:${statusColor}; font-weight:600;">● ${statusText}</span></td>
      `;
      tbody.appendChild(tr);
    }
  }

  let trainingDataLoaded = false;
  async function loadTrainingHistoryData() {
    if (trainingDataLoaded) return;
    try {
      const resp = await fetch('/api/training-history');
      const data = await resp.json();
      const tbody = document.getElementById('history-tbody');
      tbody.innerHTML = '';
      if (!data.per_round_val_metrics) return;

      data.per_round_val_metrics.forEach(vm => {
        const isBest = vm.round === data.best_round;
        const tr = document.createElement('tr');
        if (isBest) tr.classList.add('highlight-row');
        const loss = vm.loss ? vm.loss.toFixed(4) : vm.global_val_loss?.toFixed(4) || '—';
        const dice = vm.dice_macro ? vm.dice_macro.toFixed(4) : vm.dice?.toFixed(4) || '—';
        const iou = vm.iou_macro ? vm.iou_macro.toFixed(4) : vm.iou?.toFixed(4) || '—';
        const ma = vm.dice_MA != null ? vm.dice_MA.toFixed(4) : '—';
        const he = vm.dice_HE != null ? vm.dice_HE.toFixed(4) : '—';
        const ex = vm.dice_EX != null ? vm.dice_EX.toFixed(4) : '—';
        const se = vm.dice_SE != null ? vm.dice_SE.toFixed(4) : '—';
        const status = isBest ? '⭐ <strong>Best Global Model (Round 17)</strong>' : (vm.round === 20 ? 'Final Round' : 'Completed');

        tr.innerHTML = `
          <td><strong>Round ${vm.round}</strong></td>
          <td class="num">${loss}</td>
          <td class="num"><strong>${dice}</strong></td>
          <td class="num">${iou}</td>
          <td class="num">${ma}</td>
          <td class="num">${he}</td>
          <td class="num">${ex}</td>
          <td class="num">${se}</td>
          <td>${status}</td>
        `;
        tbody.appendChild(tr);
      });
      trainingDataLoaded = true;
    } catch (e) {
      console.error('Failed to load training data', e);
    }
  }

  // Load initial demo inference on page load
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


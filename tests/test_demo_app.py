"""Fast tests for the FedMorphis demonstration app (no model weights loaded)."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def demo_module():
    spec = importlib.util.spec_from_file_location("run_demo", PROJECT_ROOT / "scripts" / "run_demo.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakePipeline:
    def predict(self, image, sample_name="x"):
        h, w = 16, 20
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[2:4, 2:4] = 255
        channels = {
            code: SimpleNamespace(
                channel_name=name,
                threshold=thr,
                color_rgb=(1, 2, 3),
                positive_pixel_count=4,
                area_percentage=1.25,
                detected=True,
                mask_orig=mask,
                colored_mask_orig=rgb,
            )
            for code, name, thr in [
                ("MA", "Microaneurysms", 0.25),
                ("HE", "Hemorrhages", 0.40),
                ("EX", "Hard Exudates", 0.35),
                ("SE", "Soft Exudates", 0.35),
            ]
        }
        return SimpleNamespace(
            original_image_rgb=rgb,
            combined_overlay_orig=rgb,
            original_dimensions=(h, w),
            latency_ms=1.0,
            metadata={"device": "cpu"},
            channels=channels,
        )


@pytest.fixture()
def client(demo_module, monkeypatch):
    monkeypatch.setattr(demo_module, "get_pipeline", lambda: _FakePipeline())
    return TestClient(demo_module.app)


def test_demo_samples_never_reference_held_out_test_split(demo_module):
    for info in demo_module.DEMO_SAMPLE_IMAGES.values():
        p = info["path"].replace("\\", "/").lower()
        assert "/test/" not in p and "testing set" not in p


def test_demo_sample_files_exist(demo_module):
    for info in demo_module.DEMO_SAMPLE_IMAGES.values():
        assert (PROJECT_ROOT / info["path"]).is_file(), info["path"]


def test_index_page_has_all_sections(client):
    r = client.get("/")
    assert r.status_code == 200
    for marker in ("inference-tab", "architecture-tab", "training-tab", "benchmark-tab", "Disclaimer"):
        assert marker in r.text


def test_samples_endpoint(client):
    assert len(client.get("/api/samples").json()["samples"]) == 4


def test_training_history_endpoint(client):
    data = client.get("/api/training-history").json()
    assert data["best_round"] == 17
    assert len(data["per_round_val_metrics"]) == 20


def test_benchmark_endpoint_matches_html_headline(client):
    data = client.get("/api/benchmark").json()
    tuned = data["channel_tuned"]["sample_weighted"]["macro"]["dice"]
    assert round(tuned, 4) == 0.4760
    assert "0.4760" in client.get("/").text


def test_predict_requires_input(client):
    assert client.post("/api/predict").status_code == 400


def test_predict_with_sample_key_uses_pipeline(client):
    r = client.post("/api/predict", data={"sample_key": "DDR_Sample_1"})
    assert r.status_code == 200
    body = r.json()
    assert set(body["channels"]) == {"MA", "HE", "EX", "SE"}
    assert body["images"]["overlay_b64"].startswith("data:image/png;base64,")


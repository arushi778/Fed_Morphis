"""Unit and integration tests for the Flower federated server and global initialization (src/federated/server.py)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from flwr.common import (
    Code,
    EvaluateRes,
    FitRes,
    Status,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server import Server
from flwr.server.client_proxy import ClientProxy
import numpy as np
from PIL import Image
import pytest
import torch
import torch.nn as nn

from src.data.masks import CANONICAL_LESION_CHANNELS
from src.data.partitioning import partition_training_records
from src.engine.evaluator import reconstruct_model_from_checkpoint
from src.federated.client import (
    FedMorphisNumPyClient,
    build_federated_client,
    get_model_parameters,
    get_model_state_keys,
    set_model_parameters,
)
from src.federated.data import build_all_federated_client_dataloaders
from src.federated.server import (
    FederatedServerRunResult,
    InProcessClientProxy,
    build_federated_server,
    run_federated_training,
)
from src.federated.strategy import (
    FedMorphisFedAvg,
    initialize_global_model,
    load_global_checkpoint,
    save_global_checkpoint,
)
from src.models.unet import PretrainedWeightsError, ResNet50UNet


class _TinyBNUNet(nn.Module):
    """Lightweight segmentation model with Conv2d + BatchNorm2d for fast integration tests."""

    def __init__(self, in_channels: int = 3, out_channels: int = 4) -> None:
        super().__init__()
        self.enc_conv = nn.Conv2d(in_channels, 8, kernel_size=3, padding=1, bias=False)
        self.enc_bn = nn.BatchNorm2d(8)
        self.act = nn.ReLU(inplace=True)
        self.head = nn.Conv2d(8, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.act(self.enc_bn(self.enc_conv(x))))


class _DummyProxy(ClientProxy):
    """Minimal ClientProxy stub for testing strategy round checkpointing."""

    def __init__(self, cid: str) -> None:
        super().__init__(cid=cid)

    def get_properties(self, ins, timeout, group_id):
        raise NotImplementedError

    def get_parameters(self, ins, timeout, group_id):
        raise NotImplementedError

    def fit(self, ins, timeout, group_id):
        raise NotImplementedError

    def evaluate(self, ins, timeout, group_id):
        raise NotImplementedError

    def reconnect(self, ins, timeout, group_id):
        raise NotImplementedError


def _create_synthetic_sample(
    root_dir: Path,
    sample_id: str,
    split: str = "train",
    dataset_name: str = "DDR",
    height: int = 32,
    width: int = 32,
) -> dict[str, Any]:
    """Create a synthetic retinal image and 4-channel lesion masks on disk."""
    img_arr = np.zeros((height, width, 3), dtype=np.uint8)
    img_arr[4 : height - 4, 4 : width - 4, 0] = 175
    img_arr[4 : height - 4, 4 : width - 4, 1] = 125

    boxes = {
        "MA": (6, 12, 6, 12),
        "HE": (12, 18, 12, 18),
        "EX": (18, 24, 18, 24),
        "SE": (22, 28, 8, 16),
    }

    if dataset_name == "DDR":
        mask_dir = "segmentation label" if split == "valid" else "label"
        img_path = (
            root_dir
            / f"DDR-dataset/lesion_segmentation/{split}/image/{sample_id}.png"
        )
    else:
        split_folder = "a. Training Set" if split == "train" else "b. Testing Set"
        img_path = (
            root_dir
            / f"A. Segmentation/1. Original Images/{split_folder}/{sample_id}.png"
        )

    img_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img_arr, mode="RGB").save(img_path)

    mask_paths: dict[str, str | None] = {}
    pixel_counts: dict[str, int] = {}
    present_list: list[str] = []

    for cls_name in CANONICAL_LESION_CHANNELS:
        m_arr = np.zeros((height, width), dtype=np.uint8)
        y0, y1, x0, x1 = boxes[cls_name]
        m_arr[y0:y1, x0:x1] = 255 if dataset_name == "DDR" else 1

        if dataset_name == "DDR":
            m_path = (
                root_dir
                / f"DDR-dataset/lesion_segmentation/{split}/{mask_dir}/{cls_name}/{sample_id}.tif"
            )
        else:
            folder_map = {
                "MA": "1. Microaneurysms",
                "HE": "2. Haemorrhages",
                "EX": "3. Hard Exudates",
                "SE": "4. Soft Exudates",
            }
            m_path = (
                root_dir
                / f"A. Segmentation/2. All Segmentation Groundtruths/{split_folder}/{folder_map[cls_name]}/{sample_id}_{cls_name}.tif"
            )

        m_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(m_arr, mode="L").save(m_path)

        mask_paths[cls_name] = m_path.as_posix()
        pixel_counts[cls_name] = int((m_arr > 0).sum())
        present_list.append(cls_name)

    return {
        "dataset_name": dataset_name,
        "sample_id": sample_id,
        "split": split,
        "image_path": str(img_path),
        "mask_paths": mask_paths,
        "channel_order": list(CANONICAL_LESION_CHANNELS),
        "present_lesion_masks": present_list,
        "missing_lesion_masks": [],
        "synthesized_zero_channels": [],
        "empty_lesion_masks": [],
        "positive_lesion_masks": list(present_list),
        "has_missing_masks": False,
        "lesion_pixel_counts": pixel_counts,
        "image_dimensions": {"height": height, "width": width, "channels": 3},
        "canonical_mask_dimensions": [height, width, 4],
    }


def _build_synthetic_two_client_workspace(
    tmp_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Create synthetic DDR + IDRiD records and a 2-client cross_dataset partition manifest."""
    records: list[dict[str, Any]] = []
    for i in range(6):
        records.append(
            _create_synthetic_sample(tmp_path, f"ddr_tr_{i:02d}", split="train", dataset_name="DDR")
        )
    for i in range(2):
        records.append(
            _create_synthetic_sample(tmp_path, f"ddr_val_{i:02d}", split="valid", dataset_name="DDR")
        )
    records.append(
        _create_synthetic_sample(tmp_path, "ddr_test_00", split="test", dataset_name="DDR")
    )

    for i in range(10):
        records.append(
            _create_synthetic_sample(tmp_path, f"IDRiD_{i + 1:02d}", split="train", dataset_name="IDRiD")
        )
    for i in range(2):
        records.append(
            _create_synthetic_sample(tmp_path, f"IDRiD_{i + 55:02d}", split="test", dataset_name="IDRiD")
        )

    partition_manifest = partition_training_records(
        records,
        num_clients=2,
        strategy="cross_dataset",
        seed=42,
    )
    return records, partition_manifest


def test_global_model_initialization_modes_and_missing_checkpoint_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify random, SimCLR encoder checkpoint, full checkpoint init, and missing checkpoint errors."""
    # 1. Random initialization
    model_rand_a, info_a = initialize_global_model(
        pretrained_override=False,
        encoder_weights_source_override="random",
        seed=42,
    )
    model_rand_b, _ = initialize_global_model(
        pretrained_override=False,
        encoder_weights_source_override="random",
        seed=42,
    )
    assert info_a["initialization_mode"] == "random"
    assert model_rand_a.pretrained_weights_loaded is False
    for (k1, t1), (k2, t2) in zip(
        model_rand_a.state_dict().items(), model_rand_b.state_dict().items()
    ):
        assert k1 == k2
        assert torch.equal(t1, t2)

    # 2. SimCLR encoder checkpoint initialization
    donor_model = ResNet50UNet(pretrained=False)
    with torch.no_grad():
        donor_model.encoder_conv1.weight.fill_(0.1234)
        donor_model.encoder_bn1.running_mean.fill_(0.5678)

    simclr_ckpt_path = tmp_path / "mock_simclr_encoder.pt"
    torch.save(
        {
            "epoch": 50,
            "encoder_state_dict": donor_model.export_encoder_state_dict(),
        },
        simclr_ckpt_path,
    )

    model_ssl, info_ssl = initialize_global_model(
        pretrained_override=True,
        encoder_weights_source_override="ssl_checkpoint",
        ssl_checkpoint_path_override=simclr_ckpt_path,
        seed=42,
    )
    assert info_ssl["initialization_mode"] == "ssl_checkpoint"
    assert model_ssl.pretrained_weights_loaded is True
    assert torch.allclose(
        model_ssl.encoder_conv1.weight,
        torch.full_like(model_ssl.encoder_conv1.weight, 0.1234),
    )
    assert torch.allclose(
        model_ssl.encoder_bn1.running_mean,
        torch.full_like(model_ssl.encoder_bn1.running_mean, 0.5678),
    )

    # 3. Full segmentation checkpoint initialization
    with torch.no_grad():
        donor_model.segmentation_head.weight.fill_(-0.3333)
        donor_model.segmentation_head.bias.fill_(0.2222)

    full_ckpt_path = tmp_path / "mock_full_unet_checkpoint.pt"
    torch.save(
        {
            "epoch": 12,
            "model_state_dict": donor_model.state_dict(),
        },
        full_ckpt_path,
    )

    model_full, info_full = initialize_global_model(
        initial_checkpoint_path_override=full_ckpt_path,
        seed=42,
    )
    assert info_full["initialization_mode"] == "full_checkpoint"
    assert model_full.pretrained_weights_loaded is True
    for (k1, t1), (k2, t2) in zip(
        donor_model.state_dict().items(), model_full.state_dict().items()
    ):
        assert k1 == k2
        assert torch.equal(t1, t2)

    # 4. Missing checkpoint path errors
    missing_path = tmp_path / "does_not_exist.pt"
    with pytest.raises(FileNotFoundError, match="Initial global segmentation checkpoint not found"):
        initialize_global_model(initial_checkpoint_path_override=missing_path)

    with pytest.raises(FileNotFoundError, match="Pretrained encoder checkpoint not found"):
        initialize_global_model(
            pretrained_override=True,
            encoder_weights_source_override="ssl_checkpoint",
            ssl_checkpoint_path_override=missing_path,
        )

    # Incompatible checkpoint error
    bad_ckpt_path = tmp_path / "incompatible_checkpoint.pt"
    torch.save({"model_state_dict": {"bad_layer.weight": torch.zeros(2, 2)}}, bad_ckpt_path)
    with pytest.raises(PretrainedWeightsError, match="Incompatible state_dict"):
        initialize_global_model(initial_checkpoint_path_override=bad_ckpt_path)

    # 5. ImageNet initialization refuses to silently fall back to random when weights fail to load
    import src.federated.strategy as strat_mod

    def _mock_unloaded_unet(*args, **kwargs):
        m = ResNet50UNet(pretrained=False)
        m.pretrained_requested = True
        m.pretrained_weights_loaded = False
        m.weights_status = "fallback_random_imagenet_unavailable:URLError"
        return m

    monkeypatch.setattr(strat_mod, "build_unet_from_config", _mock_unloaded_unet)
    with pytest.raises(PretrainedWeightsError, match="Refusing to silently fall back to random initialization"):
        initialize_global_model(
            pretrained_override=True,
            encoder_weights_source_override="imagenet",
            strict_pretrained_override=False,
        )


def test_atomic_checkpoint_saving_reloading_and_best_metric_guard(
    tmp_path: Path,
) -> None:
    """Verify atomic checkpoint save/reload, evaluator compatibility, and NaN/missing best-checkpoint guard."""
    model = ResNet50UNet(pretrained=False)
    with torch.no_grad():
        model.segmentation_head.weight.fill_(0.42)

    ckpt_dir = tmp_path / "checkpoints_test"
    saved_path = save_global_checkpoint(
        checkpoint_path=ckpt_dir / "global_round_3.pt",
        model=model,
        server_round=3,
        metrics={"val_dice_macro": 0.612, "val_iou_macro": 0.451},
        best_metric_value=0.612,
        best_round=3,
        seed=42,
    )
    assert saved_path.is_file()
    # Verify no leftover temporary files remain
    assert list(ckpt_dir.glob("*.tmp")) == []

    # Reload via load_global_checkpoint and reconstruct_model_from_checkpoint
    reloaded_model, payload = load_global_checkpoint(saved_path)
    assert payload["round"] == 3
    assert payload["epoch"] == 3
    assert payload["best_round"] == 3
    assert np.isclose(payload["best_metric_value"], 0.612)
    for (k1, t1), (k2, t2) in zip(
        model.state_dict().items(), reloaded_model.state_dict().items()
    ):
        assert k1 == k2
        assert torch.equal(t1, t2)

    # Verify best_checkpoint.pt guard in FedMorphisFedAvg across multiple rounds
    tiny_model = _TinyBNUNet()
    strategy = FedMorphisFedAvg(
        global_model=tiny_model,
        checkpoint_dir=ckpt_dir / "strategy_ckpts",
        save_checkpoints=True,
        monitor_metric="val_dice_macro",
        monitor_mode="max",
        accept_failures=True,
        min_fit_clients=1,
        min_evaluate_clients=1,
    )

    params = ndarrays_to_parameters(get_model_parameters(tiny_model, include_buffers=True))
    proxy_0 = _DummyProxy("client_0")

    # Round 1: val_dice_macro is missing -> latest_checkpoint.pt saved, best_checkpoint.pt NOT saved
    strategy.aggregate_fit(
        server_round=1,
        results=[
            (
                proxy_0,
                FitRes(
                    status=Status(code=Code.OK, message="OK"),
                    parameters=params,
                    num_examples=20,
                    metrics={"client_id": "client_0", "loss": 0.6},
                ),
            )
        ],
        failures=[],
    )
    strategy._finalize_round(
        server_round=1,
        val_loss=0.5,
        val_metrics={"val_dice_macro": float("nan")},
    )
    assert strategy.latest_checkpoint_path is not None and strategy.latest_checkpoint_path.is_file()
    assert strategy.best_checkpoint_path is None
    assert not (ckpt_dir / "strategy_ckpts" / "best_checkpoint.pt").exists()

    # Round 2: finite val_dice_macro = 0.52 -> best_checkpoint.pt saved
    strategy._finalize_round(
        server_round=2,
        val_loss=0.4,
        val_metrics={"val_dice_macro": 0.52},
    )
    assert strategy.best_round == 2
    assert np.isclose(strategy.best_metric_value, 0.52)
    assert strategy.best_checkpoint_path is not None and strategy.best_checkpoint_path.is_file()

    # Round 3: lower val_dice_macro = 0.48 -> latest updated (round 3), best remains round 2
    strategy._finalize_round(
        server_round=3,
        val_loss=0.45,
        val_metrics={"val_dice_macro": 0.48},
    )
    assert strategy.best_round == 2
    best_payload = torch.load(strategy.best_checkpoint_path, map_location="cpu", weights_only=False)
    latest_payload = torch.load(strategy.latest_checkpoint_path, map_location="cpu", weights_only=False)
    assert best_payload["round"] == 2
    assert latest_payload["round"] == 3


def test_one_round_two_client_flower_server_integration_smoke(
    tmp_path: Path,
) -> None:
    """Run a 1-round 2-client Flower orchestration smoke test with FedMorphisNumPyClient."""
    records, partition_manifest = _build_synthetic_two_client_workspace(tmp_path)

    bundles = build_all_federated_client_dataloaders(
        partition_manifest=partition_manifest,
        source_manifest=records,
        batch_size=2,
        val_batch_size=2,
        num_workers=0,
        seed=42,
        image_size=(32, 32),
        fov_crop=False,
    )
    assert set(bundles.keys()) == {"client_0", "client_1"}

    global_model = _TinyBNUNet()
    initial_global_params = get_model_parameters(global_model, include_buffers=True)

    # Create the two FedMorphisNumPyClient instances with distinct initial weights prior to Round 1
    client_0_model = _TinyBNUNet()
    client_1_model = _TinyBNUNet()
    with torch.no_grad():
        client_0_model.head.weight.fill_(9.0)
        client_1_model.head.weight.fill_(-9.0)

    received_round1_params: dict[str, list[np.ndarray]] = {}

    class _TrackingFedClient(FedMorphisNumPyClient):
        def fit(self, parameters, config):
            # Capture the exact global parameters received from the server at the start of fit()
            received_round1_params[self.client_id] = [arr.copy() for arr in parameters]
            return super().fit(parameters, config)

    client_0 = _TrackingFedClient(
        client_id="client_0",
        client_index=0,
        model=client_0_model,
        train_loader=bundles["client_0"].train_loader,
        val_loader=bundles["client_0"].val_loader,
        dataset_names=bundles["client_0"].dataset_names,
        data_bundle=bundles["client_0"],
        device="cpu",
        local_epochs=1,
        learning_rate=1e-3,
        include_buffers=True,
        seed=42,
    )
    client_1 = _TrackingFedClient(
        client_id="client_1",
        client_index=1,
        model=client_1_model,
        train_loader=bundles["client_1"].train_loader,
        val_loader=bundles["client_1"].val_loader,
        dataset_names=bundles["client_1"].dataset_names,
        data_bundle=bundles["client_1"],
        device="cpu",
        local_epochs=1,
        learning_rate=1e-3,
        include_buffers=True,
        seed=42,
    )

    ckpt_dir = tmp_path / "fed_smoke_ckpts"
    report_path = tmp_path / "fed_smoke_report.json"

    run_result = run_federated_training(
        num_rounds=1,
        clients=[client_0, client_1],
        global_model=global_model,
        checkpoint_dir=ckpt_dir,
        save_checkpoints=True,
        report_path=report_path,
        save_report=True,
        local_epochs=1,
        learning_rate=1e-3,
        include_buffers=True,
        seed=42,
        device="cpu",
        max_workers=1,
    )

    assert isinstance(run_result, FederatedServerRunResult)
    assert isinstance(run_result.server, Server)
    assert run_result.num_rounds == 1
    assert len(run_result.round_history) == 1

    # 1. Verify both clients received identical initial global parameters at Round 1
    assert set(received_round1_params.keys()) == {"client_0", "client_1"}
    for a_init, a_c0, a_c1 in zip(
        initial_global_params,
        received_round1_params["client_0"],
        received_round1_params["client_1"],
    ):
        assert np.array_equal(a_init, a_c0)
        assert np.array_equal(a_init, a_c1)

    # 2. Verify global model parameters were updated by aggregation and remain finite
    updated_global_params = get_model_parameters(global_model, include_buffers=True)
    state_keys = get_model_state_keys(global_model, include_buffers=True)
    any_float_changed = any(
        not np.allclose(a0, a1)
        for a0, a1 in zip(initial_global_params, updated_global_params)
        if np.issubdtype(a0.dtype, np.floating)
    )
    assert any_float_changed

    # Verify BatchNorm num_batches_tracked integer buffer is int64 and == max(3 batches, 4 batches) == 4
    bn_idx = state_keys.index("enc_bn.num_batches_tracked")
    assert updated_global_params[bn_idx].dtype == np.int64
    assert int(updated_global_params[bn_idx]) == 4

    # 3. Verify round metrics and sample counts (DDR: 6 train / 2 val; IDRiD: 8 train / 2 val)
    round_1 = run_result.round_history[0]
    assert round_1["round"] == 1
    assert round_1["train"]["num_samples_total"] == 14
    assert round_1["val"]["num_samples_total"] == 4
    assert "val_dice_macro" in round_1["val"]
    assert "macro_client_val_dice_macro" in round_1["val"]
    assert "client_0_DDR_val_dice_macro" in round_1["val"]
    assert "client_1_IDRiD_val_dice_macro" in round_1["val"]

    # 4. Verify evaluation did not mutate global model parameters after aggregate_fit
    params_after_round = get_model_parameters(global_model, include_buffers=True)
    eval_res = run_result.server.evaluate_round(server_round=1, timeout=None)
    assert eval_res is not None
    params_after_extra_eval = get_model_parameters(global_model, include_buffers=True)
    for a_before, a_after in zip(params_after_round, params_after_extra_eval):
        assert np.array_equal(a_before, a_after)

    # 5. Verify checkpoints and JSON report exist and are valid
    assert run_result.latest_checkpoint_path is not None and run_result.latest_checkpoint_path.is_file()
    assert run_result.best_checkpoint_path is not None and run_result.best_checkpoint_path.is_file()
    assert report_path.is_file()
    report_data = json.loads(report_path.read_text(encoding="utf-8"))
    assert report_data["num_rounds"] == 1
    assert report_data["best_round"] == 1
    assert len(report_data["round_history"]) == 1

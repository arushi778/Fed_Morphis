"""Unit tests for the custom Flower FedAvg aggregation strategy (src/federated/strategy.py)."""

from __future__ import annotations

from pathlib import Path

from flwr.common import (
    Code,
    EvaluateRes,
    FitRes,
    Parameters,
    Status,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
import numpy as np
import pytest
import torch
import torch.nn as nn

from src.federated.client import (
    get_model_parameters,
    get_model_state_keys,
    set_model_parameters,
)
from src.federated.strategy import (
    FedMorphisFedAvg,
    FederatedStrategyError,
    aggregate_client_metrics,
    aggregate_parameters_with_policy,
)
from src.models.unet import ResNet50UNet


class _TinyBNUNet(nn.Module):
    """Lightweight segmentation model with Conv2d + BatchNorm2d for fast unit tests."""

    def __init__(self, in_channels: int = 3, out_channels: int = 4) -> None:
        super().__init__()
        self.enc_conv = nn.Conv2d(in_channels, 8, kernel_size=3, padding=1, bias=False)
        self.enc_bn = nn.BatchNorm2d(8)
        self.act = nn.ReLU(inplace=True)
        self.head = nn.Conv2d(8, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.act(self.enc_bn(self.enc_conv(x))))


class _DummyClientProxy(ClientProxy):
    """Minimal ClientProxy stub for testing strategy aggregate_fit / aggregate_evaluate."""

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


def test_weighted_averaging_and_integer_buffer_policy() -> None:
    """Verify exact sample-weighted float averaging and int64 max policy for num_batches_tracked."""
    ref_model = _TinyBNUNet()
    state_keys = get_model_state_keys(ref_model, include_buffers=True)

    # Create two distinct client parameter sets (e.g., DDR n_0=383, IDRiD n_1=44)
    params_c0 = get_model_parameters(ref_model, include_buffers=True)
    params_c1 = get_model_parameters(ref_model, include_buffers=True)

    for idx, key in enumerate(state_keys):
        if np.issubdtype(params_c0[idx].dtype, np.floating):
            params_c0[idx] = np.full_like(params_c0[idx], 2.0, dtype=np.float32)
            params_c1[idx] = np.full_like(params_c1[idx], 10.0, dtype=np.float32)
        elif key.endswith("num_batches_tracked"):
            params_c0[idx] = np.asarray(144, dtype=np.int64)
            params_c1[idx] = np.asarray(18, dtype=np.int64)

    n0, n1 = 383, 44
    expected_float_val = np.float32((383.0 * 2.0 + 44.0 * 10.0) / (383.0 + 44.0))

    # 1. Default "max" integer buffer policy
    agg_max = aggregate_parameters_with_policy(
        results=[(params_c0, n0), (params_c1, n1)],
        reference_model=ref_model,
        include_buffers=True,
        integer_buffer_policy="max",
    )
    for idx, key in enumerate(state_keys):
        if np.issubdtype(agg_max[idx].dtype, np.floating):
            assert agg_max[idx].dtype == np.float32
            assert np.allclose(agg_max[idx], expected_float_val, atol=1e-6)
        elif key.endswith("num_batches_tracked"):
            assert agg_max[idx].dtype == np.int64
            assert agg_max[idx].shape == ()
            assert int(agg_max[idx]) == 144

    # 2. "sum" and "first" integer buffer policies
    agg_sum = aggregate_parameters_with_policy(
        results=[(params_c0, n0), (params_c1, n1)],
        reference_model=ref_model,
        include_buffers=True,
        integer_buffer_policy="sum",
    )
    agg_first = aggregate_parameters_with_policy(
        results=[(params_c1, n1), (params_c0, n0)],
        reference_model=ref_model,
        include_buffers=True,
        integer_buffer_policy="first",
    )
    for idx, key in enumerate(state_keys):
        if key.endswith("num_batches_tracked"):
            assert agg_sum[idx].dtype == np.int64
            assert int(agg_sum[idx]) == 144 + 18
            assert agg_first[idx].dtype == np.int64
            assert int(agg_first[idx]) == 18


def test_resnet50_unet_buffer_sharing_true_and_false() -> None:
    """Verify ResNet50UNet aggregation with share_batchnorm_buffers=True and False."""
    model = ResNet50UNet(pretrained=False)

    # Set custom BatchNorm buffers on global model
    with torch.no_grad():
        model.encoder_bn1.running_mean.fill_(5.5)
        model.encoder_bn1.running_var.fill_(2.5)
        model.encoder_bn1.num_batches_tracked.fill_(77)

    # Case A: include_buffers=False (only named_parameters aggregated; BN buffers untouched)
    p0_no_buf = get_model_parameters(model, include_buffers=False)
    p1_no_buf = get_model_parameters(model, include_buffers=False)
    assert len(p0_no_buf) == len(list(model.named_parameters()))

    p0_no_buf[0] = np.full_like(p0_no_buf[0], 1.0, dtype=np.float32)
    p1_no_buf[0] = np.full_like(p1_no_buf[0], 3.0, dtype=np.float32)

    strategy_no_buf = FedMorphisFedAvg(
        global_model=model,
        include_buffers=False,
        save_checkpoints=False,
    )
    fit_results_no_buf = [
        (
            _DummyClientProxy("client_0"),
            FitRes(
                status=Status(code=Code.OK, message="OK"),
                parameters=ndarrays_to_parameters(p0_no_buf),
                num_examples=100,
                metrics={"client_id": "client_0", "dataset": "DDR", "loss": 0.5, "dice_macro": 0.4},
            ),
        ),
        (
            _DummyClientProxy("client_1"),
            FitRes(
                status=Status(code=Code.OK, message="OK"),
                parameters=ndarrays_to_parameters(p1_no_buf),
                num_examples=300,
                metrics={"client_id": "client_1", "dataset": "IDRiD", "loss": 0.3, "dice_macro": 0.6},
            ),
        ),
    ]
    agg_params_proto, _ = strategy_no_buf.aggregate_fit(
        server_round=1,
        results=fit_results_no_buf,
        failures=[],
    )
    assert isinstance(agg_params_proto, Parameters)
    agg_arrays_no_buf = parameters_to_ndarrays(agg_params_proto)
    assert np.allclose(agg_arrays_no_buf[0], 2.5, atol=1e-6)
    # Verify global model's BatchNorm buffers were preserved untouched
    assert torch.allclose(model.encoder_bn1.running_mean, torch.full_like(model.encoder_bn1.running_mean, 5.5))
    assert int(model.encoder_bn1.num_batches_tracked.item()) == 77

    # Case B: include_buffers=True (all 375 state_dict entries aggregated)
    p0_with_buf = get_model_parameters(model, include_buffers=True)
    p1_with_buf = get_model_parameters(model, include_buffers=True)
    full_keys = get_model_state_keys(model, include_buffers=True)
    bn_mean_idx = full_keys.index("encoder_bn1.running_mean")
    bn_cnt_idx = full_keys.index("encoder_bn1.num_batches_tracked")

    p0_with_buf[bn_mean_idx] = np.full_like(p0_with_buf[bn_mean_idx], 1.0, dtype=np.float32)
    p1_with_buf[bn_mean_idx] = np.full_like(p1_with_buf[bn_mean_idx], 5.0, dtype=np.float32)
    p0_with_buf[bn_cnt_idx] = np.asarray(120, dtype=np.int64)
    p1_with_buf[bn_cnt_idx] = np.asarray(45, dtype=np.int64)

    strategy_with_buf = FedMorphisFedAvg(
        global_model=model,
        include_buffers=True,
        save_checkpoints=False,
    )
    fit_results_with_buf = [
        (
            _DummyClientProxy("client_0"),
            FitRes(
                status=Status(code=Code.OK, message="OK"),
                parameters=ndarrays_to_parameters(p0_with_buf),
                num_examples=50,
                metrics={"client_id": "client_0", "dataset": "DDR", "loss": 0.4},
            ),
        ),
        (
            _DummyClientProxy("client_1"),
            FitRes(
                status=Status(code=Code.OK, message="OK"),
                parameters=ndarrays_to_parameters(p1_with_buf),
                num_examples=50,
                metrics={"client_id": "client_1", "dataset": "IDRiD", "loss": 0.2},
            ),
        ),
    ]
    agg_params_full, _ = strategy_with_buf.aggregate_fit(
        server_round=1,
        results=fit_results_with_buf,
        failures=[],
    )
    assert agg_params_full is not None
    assert torch.allclose(model.encoder_bn1.running_mean, torch.full_like(model.encoder_bn1.running_mean, 3.0))
    assert model.encoder_bn1.num_batches_tracked.dtype == torch.int64
    assert int(model.encoder_bn1.num_batches_tracked.item()) == 120


def test_invalid_and_non_finite_client_updates_rejected() -> None:
    """Verify NaN, Inf, shape mismatch, dtype mismatch, and non-positive sample counts fail clearly."""
    ref_model = _TinyBNUNet()
    valid_params = get_model_parameters(ref_model, include_buffers=True)
    state_keys = get_model_state_keys(ref_model, include_buffers=True)

    # 1. Empty results list
    with pytest.raises(FederatedStrategyError, match="Cannot aggregate empty"):
        aggregate_parameters_with_policy([], reference_model=ref_model)

    # 2. Non-positive sample count
    with pytest.raises(FederatedStrategyError, match="non-positive num_examples"):
        aggregate_parameters_with_policy([(valid_params, 0)], reference_model=ref_model)

    # 3. NaN in floating-point tensor
    nan_params = [arr.copy() for arr in valid_params]
    nan_params[0].flat[0] = np.nan
    with pytest.raises(FederatedStrategyError, match="Non-finite"):
        aggregate_parameters_with_policy(
            [(valid_params, 10), (nan_params, 10)],
            reference_model=ref_model,
        )

    # 4. Inf in floating-point tensor
    inf_params = [arr.copy() for arr in valid_params]
    inf_params[0].flat[0] = np.inf
    with pytest.raises(FederatedStrategyError, match="Non-finite"):
        aggregate_parameters_with_policy(
            [(valid_params, 10), (inf_params, 10)],
            reference_model=ref_model,
        )

    # 5. Shape mismatch
    bad_shape_params = [arr.copy() for arr in valid_params]
    bad_shape_params[0] = np.zeros((2, 2), dtype=np.float32)
    with pytest.raises(FederatedStrategyError, match="Shape mismatch"):
        aggregate_parameters_with_policy(
            [(valid_params, 10), (bad_shape_params, 10)],
            reference_model=ref_model,
        )

    # 6. Float64 conversion of int64 num_batches_tracked buffer
    bad_dtype_params = [arr.copy() for arr in valid_params]
    cnt_idx = state_keys.index("enc_bn.num_batches_tracked")
    bad_dtype_params[cnt_idx] = np.asarray(10.0, dtype=np.float64)
    with pytest.raises(FederatedStrategyError, match="Dtype mismatch"):
        aggregate_parameters_with_policy(
            [(valid_params, 10), (bad_dtype_params, 10)],
            reference_model=ref_model,
        )

    # 7. Strategy accept_failures=False vs accept_failures=True
    strategy_strict = FedMorphisFedAvg(
        global_model=ref_model,
        accept_failures=False,
        save_checkpoints=False,
    )
    with pytest.raises(FederatedStrategyError, match="fit failed with 1 client failure"):
        strategy_strict.aggregate_fit(
            server_round=1,
            results=[],
            failures=[RuntimeError("Simulated client disconnect")],
        )

    # With accept_failures=True, a corrupt client update is skipped while valid client updates succeed
    strategy_tolerant = FedMorphisFedAvg(
        global_model=ref_model,
        accept_failures=True,
        min_fit_clients=1,
        save_checkpoints=False,
    )
    good_params = [arr.copy() for arr in valid_params]
    good_params[0] = np.full_like(good_params[0], 4.25, dtype=np.float32)
    agg_proto, agg_metrics = strategy_tolerant.aggregate_fit(
        server_round=1,
        results=[
            (
                _DummyClientProxy("client_0"),
                FitRes(
                    status=Status(code=Code.OK, message="OK"),
                    parameters=ndarrays_to_parameters(good_params),
                    num_examples=25,
                    metrics={"client_id": "client_0", "dataset": "DDR", "loss": 0.25},
                ),
            ),
            (
                _DummyClientProxy("client_1"),
                FitRes(
                    status=Status(code=Code.OK, message="OK"),
                    parameters=ndarrays_to_parameters(nan_params),
                    num_examples=25,
                    metrics={"client_id": "client_1", "dataset": "IDRiD", "loss": 0.50},
                ),
            ),
        ],
        failures=[RuntimeError("Extra failure ignored")],
    )
    assert agg_proto is not None
    assert np.allclose(parameters_to_ndarrays(agg_proto)[0], 4.25, atol=1e-6)
    assert agg_metrics["num_clients"] == 1


def test_metric_aggregation_weighted_macro_and_per_client() -> None:
    """Verify weighted average, unweighted client-macro, and per-client metric aggregation."""
    c0_metrics = {
        "client_id": "client_0",
        "client_index": 0,
        "dataset": "DDR",
        "server_round": 2,
        "loss": 0.40,
        "val_loss": 0.40,
        "dice_macro": 0.50,
        "val_dice_macro": 0.50,
        "iou_macro": 0.35,
        "val_iou_macro": 0.35,
        "dice_MA": 0.30,
        "dice_HE": 0.55,
        "dice_EX": 0.65,
        "dice_SE": 0.50,
    }
    c1_metrics = {
        "client_id": "client_1",
        "client_index": 1,
        "dataset": "IDRiD",
        "server_round": 2,
        "loss": 0.20,
        "val_loss": 0.20,
        "dice_macro": 0.70,
        "val_dice_macro": 0.70,
        "iou_macro": 0.55,
        "val_iou_macro": 0.55,
        "dice_MA": 0.50,
        "dice_HE": 0.75,
        "dice_EX": 0.85,
        "dice_SE": 0.70,
    }

    n0, n1 = 149, 10
    agg = aggregate_client_metrics(
        [(n1, c1_metrics), (n0, c0_metrics)],
        stage="val",
    )

    assert agg["num_clients"] == 2
    assert agg["num_samples_total"] == 159
    assert agg["participating_clients"] == "client_0,client_1"

    expected_weighted_dice = (149.0 * 0.50 + 10.0 * 0.70) / 159.0
    expected_macro_client_dice = (0.50 + 0.70) / 2.0

    assert np.isclose(float(agg["val_dice_macro"]), expected_weighted_dice, atol=1e-6)
    assert np.isclose(float(agg["macro_client_val_dice_macro"]), expected_macro_client_dice, atol=1e-6)
    assert np.isclose(float(agg["client_0_DDR_val_dice_macro"]), 0.50, atol=1e-6)
    assert np.isclose(float(agg["client_1_IDRiD_val_dice_macro"]), 0.70, atol=1e-6)
    assert agg["client_0_DDR_num_samples"] == 149
    assert agg["client_1_IDRiD_num_samples"] == 10

    # Non-finite metric should fail clearly when strict_finite=True
    bad_metrics = dict(c0_metrics)
    bad_metrics["val_dice_macro"] = float("nan")
    with pytest.raises(FederatedStrategyError, match="Non-finite val metric"):
        aggregate_client_metrics([(10, bad_metrics)], stage="val", strict_finite=True)

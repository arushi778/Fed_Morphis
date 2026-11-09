"""Unit and integration tests for the federated training and evaluation CLIs (Task 5)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import pytest
import torch

import scripts.evaluate_federated as eval_fed_cli
import scripts.train_federated as train_fed_cli
from src.data.masks import CANONICAL_LESION_CHANNELS
from src.data.partitioning import partition_training_records, save_partition_manifest
from src.engine.evaluator import reconstruct_model_from_checkpoint
from src.utils.config import ConfigValidationError


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

        mask_paths[cls_name] = str(m_path)
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


def _create_synthetic_manifests_on_disk(
    tmp_path: Path,
) -> tuple[Path, Path, Path, Path]:
    """Write synthetic combined, DDR, IDRiD, and 2-client partition manifests to disk."""
    ddr_records: list[dict[str, Any]] = []
    for i in range(4):
        ddr_records.append(
            _create_synthetic_sample(tmp_path, f"ddr_tr_{i:02d}", split="train", dataset_name="DDR")
        )
    for i in range(2):
        ddr_records.append(
            _create_synthetic_sample(tmp_path, f"ddr_val_{i:02d}", split="valid", dataset_name="DDR")
        )
    for i in range(2):
        ddr_records.append(
            _create_synthetic_sample(tmp_path, f"ddr_test_{i:02d}", split="test", dataset_name="DDR")
        )

    idrid_records: list[dict[str, Any]] = []
    for i in range(5):
        idrid_records.append(
            _create_synthetic_sample(tmp_path, f"IDRiD_{i + 1:02d}", split="train", dataset_name="IDRiD")
        )
    for i in range(2):
        idrid_records.append(
            _create_synthetic_sample(tmp_path, f"IDRiD_{i + 55:02d}", split="test", dataset_name="IDRiD")
        )

    combined_records = ddr_records + idrid_records

    ddr_manifest_path = tmp_path / "ddr_manifest.json"
    idrid_manifest_path = tmp_path / "idrid_manifest.json"
    combined_manifest_path = tmp_path / "combined_manifest.json"
    partition_manifest_path = tmp_path / "partition_2clients.json"

    ddr_manifest_path.write_text(
        json.dumps({"dataset_name": "DDR", "records": ddr_records}, indent=2),
        encoding="utf-8",
    )
    idrid_manifest_path.write_text(
        json.dumps({"dataset_name": "IDRiD", "records": idrid_records}, indent=2),
        encoding="utf-8",
    )
    combined_manifest_path.write_text(
        json.dumps({"dataset_name": "COMBINED", "records": combined_records}, indent=2),
        encoding="utf-8",
    )

    partition = partition_training_records(
        combined_records,
        num_clients=2,
        strategy="cross_dataset",
        seed=42,
    )
    save_partition_manifest(partition, partition_manifest_path)

    return (
        combined_manifest_path,
        ddr_manifest_path,
        idrid_manifest_path,
        partition_manifest_path,
    )


def test_train_and_evaluate_cli_argument_parsing_and_validation() -> None:
    """Verify CLI argument parsing, config override merging, and invalid override rejection."""
    # 1. Training CLI argument parsing
    train_args = train_fed_cli.parse_args(
        [
            "--config",
            "configs/federated.yaml",
            "--rounds",
            "5",
            "--local-epochs",
            "2",
            "--client-fraction",
            "1.0",
            "--batch-size",
            "4",
            "--learning-rate",
            "0.0002",
            "--image-size",
            "64",
            "--seed",
            "123",
            "--device",
            "cpu",
            "--no-pretrained",
            "--no-share-batchnorm-buffers",
            "--experiment-name",
            "cli_parse_test",
        ]
    )
    assert train_args.config == "configs/federated.yaml"
    assert train_args.rounds == 5
    assert train_args.local_epochs == 2
    assert train_args.client_fraction == 1.0
    assert train_args.batch_size == 4
    assert np.isclose(train_args.learning_rate, 0.0002)
    assert train_args.image_size == 64
    assert train_args.seed == 123
    assert train_args.device == "cpu"
    assert train_args.no_pretrained is True
    assert train_args.share_batchnorm_buffers is False
    assert train_args.experiment_name == "cli_parse_test"

    validated_cfgs = train_fed_cli.load_and_validate_federated_cli_configs(train_args)
    assert validated_cfgs["federated"]["rounds"] == 5
    assert validated_cfgs["federated"]["local_epochs"] == 2
    assert validated_cfgs["federated"]["seed"] == 123
    assert validated_cfgs["federated"]["share_batchnorm_buffers"] is False
    assert validated_cfgs["federated"]["initialization"]["encoder_weights_source"] == "random"
    assert validated_cfgs["dataset"]["image_size"] == {"height": 64, "width": 64}
    assert validated_cfgs["project"]["device"] == "cpu"

    # 2. Invalid training CLI overrides fail before training starts
    bad_rounds = train_fed_cli.parse_args(["--rounds", "0"])
    with pytest.raises(ConfigValidationError):
        train_fed_cli.load_and_validate_federated_cli_configs(bad_rounds)

    bad_fraction = train_fed_cli.parse_args(["--client-fraction", "1.5"])
    with pytest.raises(ConfigValidationError):
        train_fed_cli.load_and_validate_federated_cli_configs(bad_fraction)

    bad_img_size = train_fed_cli.parse_args(["--image-size", "50"])
    with pytest.raises(ConfigValidationError, match="divisible by 32"):
        train_fed_cli.load_and_validate_federated_cli_configs(bad_img_size)

    # 3. Evaluation CLI argument parsing and split/dataset validation
    eval_args = eval_fed_cli.parse_args(
        [
            "--checkpoint",
            "checkpoints/federated/best_checkpoint.pt",
            "--split",
            "test",
            "--datasets",
            "ddr",
            "idrid",
            "--batch-size",
            "4",
            "--image-size",
            "64",
            "--threshold",
            "0.45",
            "--device",
            "cpu",
        ]
    )
    assert eval_args.checkpoint == "checkpoints/federated/best_checkpoint.pt"
    assert eval_args.split == "test"
    assert eval_args.datasets == ["ddr", "idrid"]
    assert eval_args.batch_size == 4
    assert eval_args.image_size == 64
    assert np.isclose(eval_args.threshold, 0.45)
    assert eval_args.device == "cpu"

    # Disallow evaluating on 'train' split
    with pytest.raises(ConfigValidationError, match="only 'test' and 'valid' are permitted"):
        eval_fed_cli._resolve_eval_split("train")

    # Disallow unknown dataset name
    with pytest.raises(ConfigValidationError, match="Unsupported dataset"):
        eval_fed_cli._resolve_requested_datasets(["unknown_ds"])


def _extract_trailing_json(stdout_text: str) -> dict[str, Any]:
    """Extract the trailing JSON object printed to stdout after any logger lines."""
    idx = stdout_text.find("\n{")
    if idx != -1:
        return json.loads(stdout_text[idx + 1 :])
    start = stdout_text.find("{")
    if start == -1:
        raise ValueError(f"No JSON object found in stdout: {stdout_text!r}")
    return json.loads(stdout_text[start:])


def test_end_to_end_federated_train_cli_reproducibility_and_eval_cli(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Run 1-round 2-client training via train_federated CLI, verify seed reproducibility, and evaluate via evaluate_federated CLI."""
    (
        combined_manifest_path,
        ddr_manifest_path,
        idrid_manifest_path,
        partition_manifest_path,
    ) = _create_synthetic_manifests_on_disk(tmp_path)

    ckpt_dir_1 = tmp_path / "run1_ckpts"
    report_path_1 = tmp_path / "run1_report.json"

    cli_args_1 = [
        "--config",
        "configs/federated.yaml",
        "--partition-manifest",
        str(partition_manifest_path),
        "--source-manifest",
        str(combined_manifest_path),
        "--rounds",
        "1",
        "--local-epochs",
        "1",
        "--batch-size",
        "2",
        "--val-batch-size",
        "2",
        "--image-size",
        "32",
        "--seed",
        "42",
        "--device",
        "cpu",
        "--no-pretrained",
        "--checkpoint-dir",
        str(ckpt_dir_1),
        "--report-path",
        str(report_path_1),
        "--experiment-name",
        "synthetic_cli_e2e_test",
    ]

    rc1 = train_fed_cli.main(cli_args_1)
    assert rc1 == 0
    captured_train = capsys.readouterr().out
    stdout_summary = _extract_trailing_json(captured_train)
    assert stdout_summary["status"] == "completed"
    assert stdout_summary["experiment_name"] == "synthetic_cli_e2e_test"
    assert stdout_summary["num_rounds"] == 1
    assert stdout_summary["rounds_completed"] == 1

    # Verify structured JSON report contents (Step 4 requirements)
    assert report_path_1.is_file()
    report_1 = json.loads(report_path_1.read_text(encoding="utf-8"))
    assert report_1["status"] == "completed"
    assert report_1["experiment_name"] == "synthetic_cli_e2e_test"
    assert report_1["strategy"] == "FedAvg"
    assert report_1["number_of_clients"] == 2
    assert report_1["num_rounds"] == 1
    assert report_1["local_epochs"] == 1
    assert report_1["seed"] == 42
    assert report_1["global_initialization_source"] == "random"
    assert report_1["client_sample_counts"]["total_train_samples"] == 8  # 4 DDR + 4 IDRiD
    assert report_1["client_sample_counts"]["total_val_samples"] == 3  # 2 DDR + 1 carved IDRiD
    assert len(report_1["per_round_train_metrics"]) == 1
    assert len(report_1["per_round_val_metrics"]) == 1
    assert report_1["best_round"] == 1
    assert report_1["best_metric_value"] is not None
    assert (ckpt_dir_1 / "best_checkpoint.pt").is_file()
    assert (ckpt_dir_1 / "latest_checkpoint.pt").is_file()
    assert isinstance(report_1["training_duration_sec"], float)
    assert "federated" in report_1["config"]
    assert "model" in report_1["config"]

    # 2. Verify seed reproducibility: a second run with seed=42 produces identical global weights
    ckpt_dir_2 = tmp_path / "run2_ckpts"
    report_path_2 = tmp_path / "run2_report.json"
    cli_args_2 = [
        "--config",
        "configs/federated.yaml",
        "--partition-manifest",
        str(partition_manifest_path),
        "--source-manifest",
        str(combined_manifest_path),
        "--rounds",
        "1",
        "--local-epochs",
        "1",
        "--batch-size",
        "2",
        "--val-batch-size",
        "2",
        "--image-size",
        "32",
        "--seed",
        "42",
        "--device",
        "cpu",
        "--no-pretrained",
        "--checkpoint-dir",
        str(ckpt_dir_2),
        "--report-path",
        str(report_path_2),
        "--experiment-name",
        "synthetic_cli_e2e_test",
    ]
    rc2 = train_fed_cli.main(cli_args_2)
    assert rc2 == 0
    _ = capsys.readouterr()

    model_1, _, _ = reconstruct_model_from_checkpoint(ckpt_dir_1 / "best_checkpoint.pt")
    model_2, _, _ = reconstruct_model_from_checkpoint(ckpt_dir_2 / "best_checkpoint.pt")
    for (k1, t1), (k2, t2) in zip(model_1.state_dict().items(), model_2.state_dict().items()):
        assert k1 == k2
        assert torch.allclose(t1, t2, atol=1e-6)

    # 3. Evaluate the saved global checkpoint on held-out 'test' split via evaluate_federated CLI
    test_eval_report_path = tmp_path / "eval_test_report.json"
    rc_eval_test = eval_fed_cli.main(
        [
            "--checkpoint",
            str(ckpt_dir_1 / "best_checkpoint.pt"),
            "--split",
            "test",
            "--datasets",
            "ddr",
            "idrid",
            "--ddr-manifest",
            str(ddr_manifest_path),
            "--idrid-manifest",
            str(idrid_manifest_path),
            "--partition-manifest",
            str(partition_manifest_path),
            "--batch-size",
            "2",
            "--image-size",
            "32",
            "--device",
            "cpu",
            "--report-path",
            str(test_eval_report_path),
        ]
    )
    assert rc_eval_test == 0
    eval_stdout = _extract_trailing_json(capsys.readouterr().out)
    assert eval_stdout["status"] == "completed"
    assert eval_stdout["split"] == "test"
    assert eval_stdout["evaluated_datasets"] == ["DDR", "IDRiD"]
    assert eval_stdout["checkpoint"]["round"] == 1
    assert eval_stdout["datasets"]["DDR"]["sample_count"] == 2
    assert eval_stdout["datasets"]["IDRiD"]["sample_count"] == 2

    assert test_eval_report_path.is_file()
    test_eval_json = json.loads(test_eval_report_path.read_text(encoding="utf-8"))
    assert test_eval_json["separate_dataset_reporting"] is True
    assert set(test_eval_json["datasets"].keys()) == {"DDR", "IDRiD"}
    for ds_key in ("DDR", "IDRiD"):
        ds_metrics = test_eval_json["datasets"][ds_key]["metrics"]
        assert ds_metrics["loss"] is not None
        assert "dice_macro" in ds_metrics
        assert "iou_macro" in ds_metrics
        for ch in CANONICAL_LESION_CHANNELS:
            assert ch in ds_metrics["per_class"]
            assert "dice" in ds_metrics["per_class"][ch]
            assert "iou" in ds_metrics["per_class"][ch]

    # 4. Evaluate the saved global checkpoint on 'valid' split via evaluate_federated CLI
    val_eval_report_path = tmp_path / "eval_val_report.json"
    rc_eval_val = eval_fed_cli.main(
        [
            "--checkpoint",
            str(ckpt_dir_1 / "best_checkpoint.pt"),
            "--split",
            "valid",
            "--datasets",
            "ddr",
            "idrid",
            "--manifest",
            str(combined_manifest_path),
            "--partition-manifest",
            str(partition_manifest_path),
            "--batch-size",
            "2",
            "--image-size",
            "32",
            "--device",
            "cpu",
            "--report-path",
            str(val_eval_report_path),
        ]
    )
    assert rc_eval_val == 0
    _ = capsys.readouterr()
    assert val_eval_report_path.is_file()
    val_eval_json = json.loads(val_eval_report_path.read_text(encoding="utf-8"))
    assert val_eval_json["split"] == "valid"
    assert val_eval_json["datasets"]["DDR"]["sample_count"] == 2
    assert val_eval_json["datasets"]["DDR"]["val_source"] == "official"
    assert val_eval_json["datasets"]["IDRiD"]["sample_count"] == 1
    assert val_eval_json["datasets"]["IDRiD"]["val_source"] == "carved_from_train"


def test_failed_training_report_and_test_split_leakage_guard(
    tmp_path: Path,
) -> None:
    """Verify failed runs record status='failed' and test leakage in partition manifests is rejected."""
    from flwr.common import FitIns, FitRes, Parameters, Status, Code
    from flwr.server.client_proxy import ClientProxy
    from src.federated.data import FederatedDataError
    from src.federated.server import run_federated_training
    from src.federated.strategy import FederatedStrategyError
    from src.models.unet import ResNet50UNet

    class _FailingProxy(ClientProxy):
        def get_properties(self, ins, timeout, group_id):
            raise NotImplementedError

        def get_parameters(self, ins, timeout, group_id):
            raise NotImplementedError

        def fit(self, ins: FitIns, timeout, group_id) -> FitRes:
            return FitRes(
                status=Status(code=Code.FIT_NOT_IMPLEMENTED, message="Simulated client failure"),
                parameters=Parameters(tensors=[], tensor_type="numpy.ndarray"),
                num_examples=0,
                metrics={},
            )

        def evaluate(self, ins, timeout, group_id):
            raise NotImplementedError

        def reconnect(self, ins, timeout, group_id):
            raise NotImplementedError

    failed_report_path = tmp_path / "failed_run_report.json"
    with pytest.raises(FederatedStrategyError):
        run_federated_training(
            num_rounds=1,
            clients=[_FailingProxy("client_0"), _FailingProxy("client_1")],
            global_model=ResNet50UNet(pretrained=False),
            checkpoint_dir=tmp_path / "failed_ckpts",
            save_checkpoints=False,
            report_path=failed_report_path,
            save_report=True,
            accept_failures=False,
            seed=42,
            device="cpu",
        )

    assert failed_report_path.is_file()
    failed_report = json.loads(failed_report_path.read_text(encoding="utf-8"))
    assert failed_report["status"] == "failed"
    assert "error" in failed_report

    # Verify evaluate_federated test-split leakage guard rejects contaminated partition manifest
    bad_part_path = tmp_path / "contaminated_partition.json"
    bad_part_path.write_text(
        json.dumps(
            {
                "clients": {
                    "client_0": [
                        {
                            "dataset_name": "DDR",
                            "sample_id": "ddr_test_00",
                            "split": "train",
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(FederatedDataError, match="Test split leakage detected"):
        eval_fed_cli._verify_partition_manifest_disjoint_from_test(
            partition_manifest_path=bad_part_path,
            test_records=[
                {
                    "dataset_name": "DDR",
                    "sample_id": "ddr_test_00",
                    "split": "test",
                }
            ],
            dataset_name="DDR",
        )


def test_federated_realtime_logging_and_log_file_output(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify real-time console and persistent file logging, client attribution, round headers, and determinism."""
    (
        combined_manifest_path,
        _ddr_manifest_path,
        _idrid_manifest_path,
        partition_manifest_path,
    ) = _create_synthetic_manifests_on_disk(tmp_path)

    ckpt_dir = tmp_path / "logged_ckpts"
    report_path = tmp_path / "logged_report.json"
    log_path = tmp_path / "logged_federated.log"

    cli_args = [
        "--config",
        "configs/federated.yaml",
        "--partition-manifest",
        str(partition_manifest_path),
        "--source-manifest",
        str(combined_manifest_path),
        "--rounds",
        "1",
        "--local-epochs",
        "1",
        "--batch-size",
        "2",
        "--val-batch-size",
        "2",
        "--image-size",
        "32",
        "--seed",
        "42",
        "--device",
        "cpu",
        "--no-pretrained",
        "--checkpoint-dir",
        str(ckpt_dir),
        "--report-path",
        str(report_path),
        "--log-path",
        str(log_path),
        "--log-level",
        "INFO",
        "--experiment-name",
        "realtime_logging_test",
    ]

    rc = train_fed_cli.main(cli_args)
    assert rc == 0
    captured_stdout = capsys.readouterr().out

    # 1. Persistent log file is created and non-empty
    assert log_path.is_file()
    log_text = log_path.read_text(encoding="utf-8")
    assert len(log_text.strip()) > 0

    # 2. Both console output and persistent log file contain all required lifecycle markers
    required_markers = [
        "[FEDERATED INIT]",
        "========== FEDERATED ROUND 1 / 1 ==========",
        "[ROUND START] Round 1 / 1",
        "Client 0 [DDR]",
        "Client 1 [IDRiD]",
        "[CLIENT TRAIN START] Client 0 [DDR]",
        "[CLIENT TRAIN] Client 0 [DDR] | Round 1 | Epoch 1/1 | batch",
        "[CLIENT EPOCH DONE] Client 0 [DDR]",
        "[CLIENT TRAIN COMPLETE] Client 0 [DDR]",
        "[CLIENT TRAIN START] Client 1 [IDRiD]",
        "[CLIENT TRAIN COMPLETE] Client 1 [IDRiD]",
        "[SERVER AGGREGATION] Starting FedAvg parameter aggregation (Round 1 / 1)",
        "[SERVER AGGREGATION] Completed FedAvg parameter aggregation (Round 1 / 1)",
        "[CLIENT VALIDATION] Client 0 [DDR]",
        "[CLIENT VALIDATION] Client 1 [IDRiD]",
        "[GLOBAL VALIDATION] Round 1 / 1",
        "[CHECKPOINT] Saved latest global checkpoint",
        "[CHECKPOINT] New best val_dice_macro=",
        "[ROUND COMPLETE] Round 1 / 1 finished",
        "==================== FEDERATED TRAINING SUMMARY ====================",
    ]
    for marker in required_markers:
        assert marker in log_text, f"Missing marker in persistent log file: {marker!r}"
        assert marker in captured_stdout, f"Missing marker in real-time console output: {marker!r}"

    # 3. Verify that logging does not change random state, model updates, or evaluation metrics
    ckpt_dir_nolog = tmp_path / "nolog_ckpts"
    report_path_nolog = tmp_path / "nolog_report.json"
    cli_args_nolog = [
        "--config",
        "configs/federated.yaml",
        "--partition-manifest",
        str(partition_manifest_path),
        "--source-manifest",
        str(combined_manifest_path),
        "--rounds",
        "1",
        "--local-epochs",
        "1",
        "--batch-size",
        "2",
        "--val-batch-size",
        "2",
        "--image-size",
        "32",
        "--seed",
        "42",
        "--device",
        "cpu",
        "--no-pretrained",
        "--checkpoint-dir",
        str(ckpt_dir_nolog),
        "--report-path",
        str(report_path_nolog),
        "--experiment-name",
        "realtime_logging_test",
    ]
    rc_nolog = train_fed_cli.main(cli_args_nolog)
    assert rc_nolog == 0
    _ = capsys.readouterr()

    model_logged, _, _ = reconstruct_model_from_checkpoint(ckpt_dir / "best_checkpoint.pt")
    model_nolog, _, _ = reconstruct_model_from_checkpoint(ckpt_dir_nolog / "best_checkpoint.pt")
    for (k1, t1), (k2, t2) in zip(model_logged.state_dict().items(), model_nolog.state_dict().items()):
        assert k1 == k2
        assert torch.allclose(t1, t2, atol=1e-6)

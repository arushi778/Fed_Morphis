"""Stage 4B regression and infrastructure tests for reproducible loss experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import pytest
import torch

from scripts.train_baseline import (
    compute_state_dict_sha256,
    create_initial_model_checkpoint,
    load_initial_model_checkpoint,
    main as train_baseline_main,
    resolve_supervised_train_val_records,
    verify_output_directory_safe,
)
from src.data.manifests import load_manifest_json
from src.engine.trainer import (
    SupervisedSegmentationTrainer,
    inspect_dataloader_batch_distribution,
)
from src.federated.data import carve_client_validation_records
from src.models.losses import (
    BCEDiceLoss,
    BCEFocalTverskyLoss,
    BCETverskyLoss,
    build_loss_from_config,
)
from src.models.unet import build_unet_from_config
from src.utils.config import (
    load_yaml_config,
    validate_dataset_config,
    validate_model_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMBINED_MANIFEST_PATH = PROJECT_ROOT / "data/processed/manifests/combined_manifest.json"
DDR_MANIFEST_PATH = PROJECT_ROOT / "data/processed/manifests/ddr_manifest.json"


def _create_synthetic_manifest(root: Path) -> Path:
    """Create a tiny valid non-test manifest with synthetic PNG images and masks."""
    records: list[dict[str, Any]] = []
    for ds_name, splits in [("DDR", ["train"] * 4 + ["valid"] * 2), ("IDRiD", ["train"] * 4)]:
        for idx, split in enumerate(splits):
            sample_id = f"{ds_name}_SYN_{idx:02d}"
            img_path = root / "images" / f"{sample_id}.png"
            img_path.parent.mkdir(parents=True, exist_ok=True)
            rng = np.random.default_rng(100 + len(records))
            img_arr = rng.integers(30, 220, size=(64, 64, 3), dtype=np.uint8)
            Image.fromarray(img_arr, mode="RGB").save(img_path)

            mask_paths: dict[str, str | None] = {}
            lesion_counts: dict[str, int] = {}
            for ch_idx, ch in enumerate(["MA", "HE", "EX", "SE"]):
                if (idx + ch_idx) % 2 == 0:
                    m_path = root / "masks" / f"{sample_id}_{ch}.png"
                    m_path.parent.mkdir(parents=True, exist_ok=True)
                    m_arr = np.zeros((64, 64), dtype=np.uint8)
                    m_arr[16:32, 16:32] = 255
                    Image.fromarray(m_arr, mode="L").save(m_path)
                    mask_paths[ch] = str(m_path)
                    lesion_counts[ch] = 256
                else:
                    mask_paths[ch] = None
                    lesion_counts[ch] = 0

            records.append(
                {
                    "sample_id": sample_id,
                    "dataset_name": ds_name,
                    "split": split,
                    "image_path": str(img_path),
                    "mask_paths": mask_paths,
                    "lesion_pixel_counts": lesion_counts,
                    "image_dimensions": {"height": 64, "width": 64},
                    "fov_crop": {"y_min": 0, "y_max": 64, "x_min": 0, "x_max": 64},
                }
            )

    manifest_path = root / "synthetic_manifest.json"
    manifest_path.write_text(
        json.dumps({"records": records}, indent=2), encoding="utf-8"
    )
    return manifest_path


def _write_minimal_configs(root: Path, *, loss_cfg: dict[str, Any] | None = None) -> tuple[Path, Path, Path]:
    """Write minimal valid project, dataset, and random-init model configs."""
    project_cfg_path = root / "project.yaml"
    project_cfg_path.write_text(
        "project_name: FedMorphis\nrandom_seed: 42\ndeterministic: true\ndevice: cpu\noutput_directory: outputs\n",
        encoding="utf-8",
    )

    dataset_cfg_path = root / "dataset.yaml"
    dataset_cfg_path.write_text(
        "\n".join(
            [
                "dataset_names: ['DDR', 'IDRiD']",
                "image_size: {height: 64, width: 64}",
                "batch_size: 4",
                "number_of_workers: 0",
                "pin_memory: false",
                "fov_cropping: {enabled: false}",
                "normalization: {enabled: true, mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225], fov_crop: false}",
                "augmentation: {enabled: false}",
                "train_validation_test_split: {train: 0.70, validation: 0.15, test: 0.15}",
            ]
        ),
        encoding="utf-8",
    )

    effective_loss = loss_cfg or {
        "type": "bce_tversky",
        "bce_weight": 0.5,
        "tversky_weight": 0.5,
        "alpha": 0.3,
        "beta": 0.7,
        "smooth": 1.0e-5,
        "pos_weight": None,
    }
    model_cfg = {
        "architecture": "UNet",
        "encoder": "resnet50",
        "pretrained": False,
        "strict_pretrained": False,
        "encoder_weights_source": "random",
        "input_channels": 3,
        "number_of_classes": 4,
        "output_channels": 4,
        "output_channel_order": ["MA", "HE", "EX", "SE"],
        "decoder": {
            "channels": [256, 128, 64, 32, 16],
            "use_batchnorm": True,
            "dropout": 0.0,
            "activation": None,
        },
        "loss": effective_loss,
        "training": {
            "optimizer": "adamw",
            "learning_rate": 0.0001,
            "weight_decay": 0.0001,
            "epochs": 1,
            "mixed_precision": False,
            "gradient_clip_max_norm": 1.0,
            "metric_threshold": 0.5,
            "monitor_metric": "val_dice_macro",
            "monitor_mode": "max",
        },
    }
    model_cfg_path = root / "model.yaml"
    import yaml

    model_cfg_path.write_text(yaml.safe_dump(model_cfg, sort_keys=False), encoding="utf-8")
    return project_cfg_path, dataset_cfg_path, model_cfg_path


# ---------------------------------------------------------------------------
# 1-5: Split resolution tests (S4A-01)
# ---------------------------------------------------------------------------


def test_01_combined_manifest_split_counts_and_per_dataset_carving() -> None:
    manifest = load_manifest_json(COMBINED_MANIFEST_PATH)
    train_recs, val_recs, split_info = resolve_supervised_train_val_records(
        manifest["records"],
        val_split_mode="official_and_carved",
        val_carve_ratio=0.2,
        split_seed=42,
    )

    assert len(train_recs) == 427
    assert len(val_recs) == 159
    assert split_info["test_samples_held_out"] == 252
    assert split_info["per_dataset"]["DDR"] == {
        "train_samples": 383,
        "validation_samples": 149,
        "test_samples_held_out": 225,
        "val_source": "official_valid",
    }
    assert split_info["per_dataset"]["IDRiD"] == {
        "train_samples": 44,
        "validation_samples": 10,
        "test_samples_held_out": 27,
        "val_source": "carved_from_train",
    }


def test_02_carved_idrid_validation_ids_match_federated_carver() -> None:
    manifest = load_manifest_json(COMBINED_MANIFEST_PATH)
    idrid_train_raw = [
        r
        for r in manifest["records"]
        if r["dataset_name"] == "IDRiD" and r["split"] == "train"
    ]
    _, expected_carved = carve_client_validation_records(
        idrid_train_raw, val_carve_ratio=0.2, seed=42
    )
    expected_ids = [r["sample_id"] for r in expected_carved]

    _, val_recs, split_info = resolve_supervised_train_val_records(manifest["records"])
    actual_idrid_val_ids = [
        r["sample_id"] for r in val_recs if r["dataset_name"] == "IDRiD"
    ]
    assert actual_idrid_val_ids == expected_ids
    assert split_info["carved_validation_sample_ids"]["IDRiD"] == expected_ids
    assert len(expected_ids) == 10


def test_03_split_resolution_is_deterministic() -> None:
    manifest = load_manifest_json(COMBINED_MANIFEST_PATH)
    tr1, va1, info1 = resolve_supervised_train_val_records(manifest["records"])
    tr2, va2, info2 = resolve_supervised_train_val_records(
        list(reversed(manifest["records"]))
    )

    assert [(r["dataset_name"], r["sample_id"]) for r in tr1] == [
        (r["dataset_name"], r["sample_id"]) for r in tr2
    ]
    assert [(r["dataset_name"], r["sample_id"]) for r in va1] == [
        (r["dataset_name"], r["sample_id"]) for r in va2
    ]
    assert info1["train_sample_ids_sha256"] == info2["train_sample_ids_sha256"]
    assert info1["val_sample_ids_sha256"] == info2["val_sample_ids_sha256"]


def test_04_zero_overlap_and_zero_test_leakage() -> None:
    manifest = load_manifest_json(COMBINED_MANIFEST_PATH)
    train_recs, val_recs, _ = resolve_supervised_train_val_records(manifest["records"])

    train_keys = {(r["dataset_name"], r["sample_id"]) for r in train_recs}
    val_keys = {(r["dataset_name"], r["sample_id"]) for r in val_recs}
    test_keys = {
        (r["dataset_name"], r["sample_id"])
        for r in manifest["records"]
        if str(r["split"]).lower() == "test"
    }

    assert train_keys.isdisjoint(val_keys)
    assert train_keys.isdisjoint(test_keys)
    assert val_keys.isdisjoint(test_keys)
    assert all(r["split"] == "train" for r in train_recs)
    assert all(r["split"] == "valid" for r in val_recs)


def test_05_ddr_only_behavior_preserved() -> None:
    ddr_manifest = load_manifest_json(DDR_MANIFEST_PATH)
    train_recs, val_recs, split_info = resolve_supervised_train_val_records(
        ddr_manifest["records"], allowed_datasets=["DDR"]
    )
    assert len(train_recs) == 383
    assert len(val_recs) == 149
    assert split_info["test_samples_held_out"] == 225
    assert list(split_info["per_dataset"].keys()) == ["DDR"]


# ---------------------------------------------------------------------------
# 6-8: Initialization checkpoint & experiment isolation (S4A-04)
# ---------------------------------------------------------------------------


def test_06_create_and_load_init_checkpoint_identical_across_runs(tmp_path: Path) -> None:
    _, _, model_cfg_path = _write_minimal_configs(tmp_path)
    model_cfg = load_yaml_config(model_cfg_path, validate=False)
    init_path = tmp_path / "init" / "unet_seed42_init.pt"

    meta = create_initial_model_checkpoint(
        model_cfg=model_cfg,
        output_path=init_path,
        seed=42,
        deterministic=True,
        strict_pretrained_override=False,
        overwrite=False,
    )
    assert init_path.is_file()
    assert len(meta["file_sha256"]) == 64
    assert len(meta["model_state_sha256"]) == 64

    # Refuses silent overwrite of existing initialization checkpoint
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        create_initial_model_checkpoint(
            model_cfg=model_cfg,
            output_path=init_path,
            seed=42,
            overwrite=False,
        )

    model_a = build_unet_from_config(model_cfg, strict_pretrained_override=False)
    model_b = build_unet_from_config(model_cfg, strict_pretrained_override=False)
    info_a = load_initial_model_checkpoint(model_a, init_path)
    info_b = load_initial_model_checkpoint(model_b, init_path)

    assert info_a["initial_model_state_sha256"] == meta["model_state_sha256"]
    assert info_b["initial_model_state_sha256"] == meta["model_state_sha256"]
    assert compute_state_dict_sha256(model_a.state_dict()) == compute_state_dict_sha256(
        model_b.state_dict()
    )


def test_07_invalid_init_checkpoint_and_resume_conflict_fail_clearly(tmp_path: Path) -> None:
    proj_p, data_p, model_p = _write_minimal_configs(tmp_path)
    model_cfg = load_yaml_config(model_p, validate=False)
    model = build_unet_from_config(model_cfg, strict_pretrained_override=False)

    # Missing file
    with pytest.raises(FileNotFoundError):
        load_initial_model_checkpoint(model, tmp_path / "missing.pt")

    # Wrong tensor shape
    bad_ckpt = tmp_path / "bad_shape.pt"
    bad_state = dict(model.state_dict())
    first_key = next(iter(bad_state))
    bad_state[first_key] = torch.zeros((1, 1, 1, 1))
    torch.save({"model_state_dict": bad_state}, bad_ckpt)
    with pytest.raises(ValueError, match="shape mismatch"):
        load_initial_model_checkpoint(model, bad_ckpt)

    # --init-checkpoint + --resume mutual exclusion
    with pytest.raises(ValueError, match="mutually exclusive"):
        train_baseline_main(
            [
                "--project-config",
                str(proj_p),
                "--dataset-config",
                str(data_p),
                "--model-config",
                str(model_p),
                "--init-checkpoint",
                str(bad_ckpt),
                "--resume",
                str(bad_ckpt),
            ]
        )


def test_08_output_dir_isolation_and_collision_prevention(tmp_path: Path) -> None:
    manifest_path = _create_synthetic_manifest(tmp_path / "data")
    proj_p, data_p, model_p = _write_minimal_configs(tmp_path)
    out_dir = tmp_path / "exp_isolated"

    rc = train_baseline_main(
        [
            "--project-config",
            str(proj_p),
            "--dataset-config",
            str(data_p),
            "--model-config",
            str(model_p),
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(out_dir),
            "--epochs",
            "1",
            "--no-pretrained",
            "--no-cache",
            "--device",
            "cpu",
        ]
    )
    assert rc == 0
    assert (out_dir / "best_checkpoint.pt").is_file()
    assert (out_dir / "latest_checkpoint.pt").is_file()
    assert (out_dir / "effective_config.yaml").is_file()
    assert (out_dir / "validation_aupr_report.json").is_file()
    assert (out_dir / "experiment_report.json").is_file()

    # Re-running into the same --output-dir without --resume must fail safely
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        verify_output_directory_safe(
            ckpt_dir=out_dir,
            report_path=out_dir / "experiment_report.json",
            is_resume=False,
            is_dry_run=False,
        )


# ---------------------------------------------------------------------------
# 9-10: Validation AUPR & loss-component per-channel logging
# ---------------------------------------------------------------------------


def test_09_validation_aupr_and_loss_component_logging_in_report(tmp_path: Path) -> None:
    manifest_path = _create_synthetic_manifest(tmp_path / "data")
    proj_p, data_p, model_p = _write_minimal_configs(
        tmp_path,
        loss_cfg={
            "type": "bce_balanced_dice",
            "bce_weight": 0.5,
            "dice_weight": 0.5,
            "smooth": 1.0e-5,
            "pos_weight": None,
            "class_weights": [1.7, 1.0, 1.0, 1.2],
        },
    )
    out_dir = tmp_path / "exp_e2b"

    rc = train_baseline_main(
        [
            "--project-config",
            str(proj_p),
            "--dataset-config",
            str(data_p),
            "--model-config",
            str(model_p),
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(out_dir),
            "--epochs",
            "1",
            "--no-pretrained",
            "--no-cache",
            "--device",
            "cpu",
        ]
    )
    assert rc == 0

    report = json.loads((out_dir / "experiment_report.json").read_text(encoding="utf-8"))
    aupr_report = json.loads((out_dir / "validation_aupr_report.json").read_text(encoding="utf-8"))

    assert report["validation_aupr"]["split"] == "valid"
    assert report["best_validation_metrics"]["aupr"]["split"] == "valid"
    assert set(aupr_report["aupr_metrics"]["per_class"].keys()) == {"MA", "HE", "EX", "SE"}
    assert "aupr_macro" in aupr_report
    assert "aupr_macro_present_only" in aupr_report
    assert "ap_macro" in aupr_report
    assert "ap_macro_present_only" in aupr_report

    hist0 = report["epoch_history"][0]
    assert hist0["primary_loss_component"] == "bce"
    assert hist0["secondary_loss_component"] == "balanced_dice"
    assert hist0["overlap_component_name"] == "balanced_dice"
    assert set(hist0["train_per_channel_bce"].keys()) == {"MA", "HE", "EX", "SE"}
    assert set(hist0["train_per_channel_overlap"].keys()) == {"MA", "HE", "EX", "SE"}
    assert set(hist0["val_per_channel_bce"].keys()) == {"MA", "HE", "EX", "SE"}
    assert set(hist0["val_per_channel_overlap"].keys()) == {"MA", "HE", "EX", "SE"}


# ---------------------------------------------------------------------------
# 11: Batch size 4 tail-batch safety & Stage 4 config suite validation
# ---------------------------------------------------------------------------


def test_10_batch_size_4_tail_batch_safety() -> None:
    ddr = inspect_dataloader_batch_distribution(383, batch_size=4, drop_last=False)
    idrid = inspect_dataloader_batch_distribution(44, batch_size=4, drop_last=False)
    combined = inspect_dataloader_batch_distribution(427, batch_size=4, drop_last=False)
    val_combined = inspect_dataloader_batch_distribution(159, batch_size=4, drop_last=False)

    assert ddr["tail_batch_size"] == 3 and not ddr["has_singleton_tail_batch"]
    assert idrid["tail_batch_size"] == 4 and not idrid["has_singleton_tail_batch"]
    assert combined["tail_batch_size"] == 3 and not combined["has_singleton_tail_batch"]
    assert val_combined["tail_batch_size"] == 3 and not val_combined["has_singleton_tail_batch"]


def test_11_stage4_experiment_config_suite_validates_and_builds() -> None:
    ds_cfg = load_yaml_config(
        PROJECT_ROOT / "configs/loss_experiments/dataset_stage4_b4.yaml",
        validate=False,
    )
    validate_dataset_config(ds_cfg)
    assert ds_cfg["batch_size"] == 4
    assert ds_cfg["number_of_workers"] == 0
    assert ds_cfg["image_size"] == {"height": 512, "width": 512}

    expected = [
        ("E0_baseline.yaml", BCETverskyLoss, "tversky"),
        ("E1_weighted_bce.yaml", BCETverskyLoss, "tversky"),
        ("E2_dice.yaml", BCEDiceLoss, "dice"),
        ("E2b_balanced_dice.yaml", BCEDiceLoss, "balanced_dice"),
        ("E3_focal_gamma075.yaml", BCEFocalTverskyLoss, "focal_tversky"),
        ("E4_focal_gamma133.yaml", BCEFocalTverskyLoss, "focal_tversky"),
        ("E5_combined_conditional.yaml", BCEFocalTverskyLoss, "focal_tversky"),
    ]
    for fname, expected_cls, expected_sec_name in expected:
        cfg = load_yaml_config(
            PROJECT_ROOT / "configs/loss_experiments" / fname, validate=False
        )
        validate_model_config(cfg)
        criterion = build_loss_from_config(cfg)
        assert type(criterion) is expected_cls
        assert criterion.secondary_component_name == expected_sec_name

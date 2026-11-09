"""Unit tests for configuration loading, deep-merging, and validation (src/utils/config.py)."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from src.utils.config import (
    CONFIG_FILES,
    ConfigValidationError,
    deep_merge_configs,
    load_all_configs,
    load_yaml_config,
    validate_all_configs,
    validate_config,
)


@pytest.mark.parametrize("config_filename", CONFIG_FILES)
def test_existing_default_yaml_files_load_and_validate(config_filename: str) -> None:
    """Verify every existing default YAML configuration file loads and passes validation."""
    cfg = load_yaml_config(config_filename, validate=True)
    assert isinstance(cfg, dict)
    validated = validate_config(cfg, config_filename)
    assert validated == cfg


def test_validate_all_configs_passes_on_defaults() -> None:
    """Verify validate_all_configs succeeds on all 6 repository configurations."""
    all_cfgs = validate_all_configs()
    assert set(all_cfgs.keys()) == {
        "project",
        "dataset",
        "model",
        "ssl",
        "federated",
        "experiment",
    }


def test_load_yaml_config_missing_file_raises_file_not_found(tmp_path: Path) -> None:
    """Verify loading a non-existent file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="Configuration file not found"):
        load_yaml_config(tmp_path / "does_not_exist.yaml")


def test_load_yaml_config_malformed_yaml_raises_clear_error(tmp_path: Path) -> None:
    """Verify malformed YAML syntax raises ConfigValidationError."""
    bad_yaml = tmp_path / "malformed.yaml"
    bad_yaml.write_text("project_name: [unclosed_list\n  broken: : :", encoding="utf-8")

    with pytest.raises(ConfigValidationError, match="Malformed YAML"):
        load_yaml_config(bad_yaml)


@pytest.mark.parametrize("non_mapping_content", ["- item1\n- item2\n", "just_a_string\n", "42\n", ""])
def test_load_yaml_config_rejects_non_mapping_root(
    tmp_path: Path, non_mapping_content: str
) -> None:
    """Verify YAML files whose root is a list, scalar, or empty raise ConfigValidationError."""
    non_map_file = tmp_path / "non_mapping.yaml"
    non_map_file.write_text(non_mapping_content, encoding="utf-8")

    with pytest.raises(ConfigValidationError, match="Expected YAML mapping"):
        load_yaml_config(non_map_file)


def test_deep_merge_preserves_unrelated_nested_values_and_overrides_matching() -> None:
    """Verify deep_merge_configs merges nested dicts recursively without losing sibling keys."""
    base: dict[str, Any] = {
        "project_name": "FedMorphis",
        "random_seed": 42,
        "normalization": {
            "enabled": True,
            "mean": [0.485, 0.456, 0.406],
            "fov_crop": True,
        },
        "augmentation": {
            "enabled": True,
            "color_jitter": {
                "brightness": 0.15,
                "contrast": 0.15,
            },
        },
    }
    overrides: dict[str, Any] = {
        "random_seed": 99,
        "normalization": {
            "fov_crop": False,
        },
        "augmentation": {
            "color_jitter": {
                "brightness": 0.30,
            },
        },
    }

    base_snapshot = copy.deepcopy(base)
    overrides_snapshot = copy.deepcopy(overrides)

    merged = deep_merge_configs(base, overrides)

    # Matching keys overridden
    assert merged["random_seed"] == 99
    assert merged["normalization"]["fov_crop"] is False
    assert merged["augmentation"]["color_jitter"]["brightness"] == 0.30

    # Unrelated nested values preserved
    assert merged["project_name"] == "FedMorphis"
    assert merged["normalization"]["enabled"] is True
    assert merged["normalization"]["mean"] == [0.485, 0.456, 0.406]
    assert merged["augmentation"]["enabled"] is True
    assert merged["augmentation"]["color_jitter"]["contrast"] == 0.15

    # Inputs were not mutated
    assert base == base_snapshot
    assert overrides == overrides_snapshot

    # Mutating returned dict does not mutate base or overrides
    merged["normalization"]["mean"].append(0.999)
    assert base["normalization"]["mean"] == [0.485, 0.456, 0.406]


@pytest.mark.parametrize(
    ("bad_base", "bad_overrides"),
    [
        (["not_a_dict"], {"a": 1}),
        ({"a": 1}, ["not_a_dict"]),
        (None, {"a": 1}),
    ],
)
def test_deep_merge_rejects_non_mapping_arguments(
    bad_base: object, bad_overrides: object
) -> None:
    """Verify deep_merge_configs raises ConfigValidationError for non-mapping inputs."""
    with pytest.raises(ConfigValidationError, match="mapping/dict"):
        deep_merge_configs(bad_base, bad_overrides)  # type: ignore[arg-type]


def test_validate_config_does_not_mutate_input() -> None:
    """Verify validate_config returns a copy without mutating the caller's dictionary."""
    project_cfg = load_yaml_config("project.yaml")
    snapshot = copy.deepcopy(project_cfg)

    returned = validate_config(project_cfg, "project")
    assert project_cfg == snapshot
    returned["project_name"] = "ModifiedName"
    assert project_cfg["project_name"] == "FedMorphis"


def test_missing_required_fields_are_reported_clearly() -> None:
    """Verify missing required keys raise ConfigValidationError naming the section and field."""
    incomplete_project = {"project_name": "FedMorphis", "device": "auto"}
    with pytest.raises(
        ConfigValidationError,
        match=r"\[project\.yaml\].*Missing required configuration field\(s\):.*random_seed.*output_directory",
    ):
        validate_config(incomplete_project, "project.yaml")

    incomplete_federated = {"strategy": "FedAvg"}
    with pytest.raises(
        ConfigValidationError,
        match=r"\[federated\.yaml\].*Missing required configuration field\(s\):.*number_of_clients",
    ):
        validate_config(incomplete_federated, "federated")


def test_invalid_project_config_types_and_values_rejected() -> None:
    """Verify invalid types or choices in project.yaml raise ConfigValidationError."""
    base = load_yaml_config("project.yaml")

    with pytest.raises(ConfigValidationError, match=r"\[project\.yaml\].*random_seed"):
        validate_config({**base, "random_seed": "forty-two"}, "project")

    with pytest.raises(ConfigValidationError, match=r"\[project\.yaml\].*random_seed"):
        validate_config({**base, "random_seed": True}, "project")

    with pytest.raises(ConfigValidationError, match=r"\[project\.yaml\].*device"):
        validate_config({**base, "device": "quantum_gpu"}, "project")

    with pytest.raises(ConfigValidationError, match=r"\[project\.yaml\].*deterministic"):
        validate_config({**base, "deterministic": "true"}, "project")


def test_invalid_dataset_config_values_rejected() -> None:
    """Verify invalid batch_size, image_size, or split ratios in dataset.yaml fail clearly."""
    base = load_yaml_config("dataset.yaml")

    with pytest.raises(ConfigValidationError, match=r"\[dataset\.yaml\].*batch_size"):
        validate_config({**base, "batch_size": 0}, "dataset")

    with pytest.raises(ConfigValidationError, match=r"\[dataset\.yaml\].*dataset_names"):
        validate_config({**base, "dataset_names": []}, "dataset")

    bad_splits = {"train": 0.5, "validation": 0.2, "test": 0.1}
    with pytest.raises(
        ConfigValidationError,
        match=r"\[dataset\.yaml\].*train_validation_test_split.*sum to 1\.0",
    ):
        validate_config({**base, "train_validation_test_split": bad_splits}, "dataset")


def test_invalid_model_ssl_federated_experiment_configs_rejected() -> None:
    """Verify invalid values across model, ssl, federated, and experiment configs fail clearly."""
    model_cfg = load_yaml_config("model.yaml")
    with pytest.raises(ConfigValidationError, match=r"\[model\.yaml\].*number_of_classes"):
        validate_config({**model_cfg, "number_of_classes": 1}, "model")

    ssl_cfg = load_yaml_config("ssl.yaml")
    with pytest.raises(ConfigValidationError, match=r"\[ssl\.yaml\].*temperature"):
        validate_config({**ssl_cfg, "temperature": -0.5}, "ssl")

    fed_cfg = load_yaml_config("federated.yaml")
    with pytest.raises(ConfigValidationError, match=r"\[federated\.yaml\].*strategy"):
        validate_config({**fed_cfg, "strategy": "UnsupportedStrategy"}, "federated")

    with pytest.raises(ConfigValidationError, match=r"\[federated\.yaml\].*client_fraction"):
        validate_config({**fed_cfg, "client_fraction": 1.5}, "federated")

    exp_cfg = load_yaml_config("experiment.yaml")
    with pytest.raises(
        ConfigValidationError, match=r"\[experiment\.yaml\].*evaluation_metrics"
    ):
        validate_config({**exp_cfg, "evaluation_metrics": []}, "experiment")


def test_unknown_config_section_rejected() -> None:
    """Verify validate_config rejects unknown configuration section names."""
    with pytest.raises(ConfigValidationError, match="Unknown configuration section"):
        validate_config({"a": 1}, "nonexistent_section")


def test_default_simclr_checkpoint_path_and_validation() -> None:
    """Verify model.yaml ssl_checkpoint_path points to the canonical SimCLR encoder checkpoint and is validated (L-3)."""
    model_cfg = load_yaml_config("model.yaml", validate=True)
    assert (
        model_cfg["ssl_checkpoint_path"]
        == "checkpoints/ssl/ddr_simclr/best_simclr_encoder.pt"
    )

    with pytest.raises(ConfigValidationError, match=r"\[model\.yaml\].*ssl_checkpoint_path"):
        validate_config({**model_cfg, "ssl_checkpoint_path": ""}, "model")

    with pytest.raises(ConfigValidationError, match=r"\[model\.yaml\].*ssl_checkpoint_path"):
        validate_config({**model_cfg, "ssl_checkpoint_path": 123}, "model")


def test_portable_report_paths_and_format_project_path() -> None:
    """Verify format_project_path normalizes paths and tracked JSON artifacts contain no machine-specific paths (H-1)."""
    from src.utils.paths import format_project_path, get_project_root

    root = get_project_root()
    abs_target = root / "checkpoints" / "segmentation" / "best_checkpoint.pt"
    assert (
        format_project_path(abs_target)
        == "checkpoints/segmentation/best_checkpoint.pt"
    )
    assert (
        format_project_path(r"checkpoints\ssl\ddr_simclr\best_simclr_encoder.pt")
        == "checkpoints/ssl/ddr_simclr/best_simclr_encoder.pt"
    )
    assert format_project_path(None) is None

    tracked_json_files = [
        *list((root / "data/processed/manifests").glob("*.json")),
        *list((root / "data/partitions").glob("*.json")),
        root / "logs/dataset_inventory.json",
        root / "logs/supervised_baseline_ddr_report.json",
        root / "logs/supervised_baseline_ddr_test_report.json",
        root / "logs/simclr_preflight_report.json",
        root / "logs/simclr_ddr_pretraining_report.json",
        root / "logs/simclr_finetuned_ddr_report.json",
    ]
    for json_file in tracked_json_files:
        assert json_file.is_file(), f"Expected tracked JSON artifact missing: {json_file}"
        content = json_file.read_text(encoding="utf-8")
        assert "\\" not in content, f"Backslash found in {json_file.name}"
        assert "C:" not in content and "c:" not in content, (
            f"Machine-specific drive path found in {json_file.name}"
        )


def test_two_client_federated_config_defaults_and_validation() -> None:
    """Verify two-client Flower federated.yaml settings load and validate properly."""
    fed_cfg = load_yaml_config("federated.yaml", validate=True)

    assert fed_cfg["framework"] == "flower"
    assert fed_cfg["aggregation_strategy"] == "fedavg"
    assert fed_cfg["strategy"] == "FedAvg"
    assert fed_cfg["number_of_clients"] == 2
    assert fed_cfg["rounds"] >= 1
    assert fed_cfg["local_epochs"] >= 1
    assert fed_cfg["client_fraction"] == 1.0
    assert fed_cfg["min_fit_clients"] == 2
    assert fed_cfg["min_evaluate_clients"] == 2
    assert fed_cfg["min_available_clients"] == 2
    assert fed_cfg["seed"] == 42
    assert fed_cfg["val_split_mode"] == "official_and_carved"
    assert fed_cfg["val_carve_ratio"] == pytest.approx(0.2)
    assert fed_cfg["share_batchnorm_buffers"] is True
    assert fed_cfg["partitioning"]["scheme"] == "cross_dataset"
    assert fed_cfg["partitioning"]["seed"] == 42
    assert (
        fed_cfg["partitioning"]["manifest_path"]
        == "data/partitions/combined_cross_dataset_2clients_seed42.json"
    )
    assert fed_cfg["initialization"]["encoder_weights_source"] == "imagenet"
    assert fed_cfg["checkpoint_directory"] == "checkpoints/federated"
    assert fed_cfg["logs_directory"] == "logs"
    assert fed_cfg["report_path"].startswith("logs/")

    with pytest.raises(ConfigValidationError, match=r"\[federated\.yaml\].*framework"):
        validate_config({**fed_cfg, "framework": "nvflare"}, "federated")

    with pytest.raises(
        ConfigValidationError, match=r"\[federated\.yaml\].*min_fit_clients"
    ):
        validate_config({**fed_cfg, "min_fit_clients": 5}, "federated")

    with pytest.raises(
        ConfigValidationError, match=r"\[federated\.yaml\].*val_split_mode"
    ):
        validate_config({**fed_cfg, "val_split_mode": "invalid_mode"}, "federated")

    with pytest.raises(
        ConfigValidationError, match=r"\[federated\.yaml\].*val_carve_ratio"
    ):
        validate_config({**fed_cfg, "val_carve_ratio": 1.0}, "federated")

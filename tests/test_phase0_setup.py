"""Phase 0 verification tests for FedMorphis project initialization."""

import importlib
from pathlib import Path

import pytest

from src.utils.config import CONFIG_FILES, load_all_configs, load_yaml_config
from src.utils.paths import REQUIRED_PROJECT_DIRS, get_project_root

PACKAGE_MODULES: tuple[str, ...] = (
    "src",
    "src.data",
    "src.models",
    "src.ssl",
    "src.segmentation",
    "src.federated",
    "src.evaluation",
    "src.experiments",
    "src.utils",
    "app",
    "app.pages",
    "tests",
)

EXPECTED_CONFIG_KEYS: dict[str, set[str]] = {
    "project.yaml": {
        "project_name",
        "random_seed",
        "device",
        "output_directory",
    },
    "dataset.yaml": {
        "dataset_names",
        "image_size",
        "batch_size",
        "number_of_workers",
        "normalization",
        "augmentation",
        "train_validation_test_split",
    },
    "model.yaml": {
        "architecture",
        "encoder",
        "pretrained",
        "number_of_classes",
        "input_channels",
    },
    "ssl.yaml": {
        "method",
        "projection_dimension",
        "temperature",
        "learning_rate",
        "epochs",
    },
    "federated.yaml": {
        "strategy",
        "number_of_clients",
        "communication_rounds",
        "local_epochs",
        "client_fraction",
        "aggregation_method",
    },
    "experiment.yaml": {
        "experiment_name",
        "tracking_enabled",
        "checkpoint_directory",
        "evaluation_metrics",
    },
}

FOUNDATIONAL_FILES: tuple[str, ...] = (
    "README.md",
    "requirements.txt",
    ".gitignore",
    ".env.example",
    "pyproject.toml",
    "docs/architecture.md",
)


@pytest.mark.parametrize("module_name", PACKAGE_MODULES)
def test_python_package_imports(module_name: str) -> None:
    """Verify that all FedMorphis Python packages can be imported cleanly."""
    module = importlib.import_module(module_name)
    assert module is not None


@pytest.mark.parametrize("third_party", ["yaml", "numpy", "pandas", "dotenv"])
def test_foundational_dependency_imports(third_party: str) -> None:
    """Verify that installed foundational dependencies import cleanly."""
    module = importlib.import_module(third_party)
    assert module is not None


@pytest.mark.parametrize("relative_dir", REQUIRED_PROJECT_DIRS)
def test_required_project_directories_exist(relative_dir: str) -> None:
    """Verify that every required directory in the project structure exists."""
    root = get_project_root()
    target_dir = root / Path(relative_dir)
    assert target_dir.exists(), f"Missing required directory: {relative_dir}"
    assert target_dir.is_dir(), f"Expected directory, found file: {relative_dir}"


@pytest.mark.parametrize("relative_file", FOUNDATIONAL_FILES)
def test_foundational_files_exist(relative_file: str) -> None:
    """Verify that all foundational documentation and setup files exist and are non-empty."""
    root = get_project_root()
    target_file = root / Path(relative_file)
    assert target_file.exists(), f"Missing foundational file: {relative_file}"
    assert target_file.is_file(), f"Expected file, found directory: {relative_file}"
    assert target_file.stat().st_size > 0, f"Foundational file is empty: {relative_file}"


@pytest.mark.parametrize("config_filename", CONFIG_FILES)
def test_yaml_configuration_loading_and_keys(config_filename: str) -> None:
    """Verify that each YAML configuration file loads into a dict with required keys."""
    cfg = load_yaml_config(config_filename)
    assert isinstance(cfg, dict)

    required_keys = EXPECTED_CONFIG_KEYS[config_filename]
    missing_keys = required_keys - set(cfg.keys())
    assert not missing_keys, f"{config_filename} is missing required keys: {missing_keys}"


def test_load_all_configs_values() -> None:
    """Verify domain-specific defaults across all loaded YAML configurations."""
    all_configs = load_all_configs()
    assert set(all_configs.keys()) == {
        "project",
        "dataset",
        "model",
        "ssl",
        "federated",
        "experiment",
    }

    assert "DDR" in all_configs["dataset"]["dataset_names"]
    assert "IDRiD" in all_configs["dataset"]["dataset_names"]
    assert all_configs["model"]["architecture"] == "UNet"
    assert all_configs["model"]["encoder"] == "resnet50"
    assert all_configs["ssl"]["method"] == "SimCLR"
    assert all_configs["federated"]["strategy"] in {"FedAvg", "FedProx"}


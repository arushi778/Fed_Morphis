"""Path resolution utilities using pathlib without hardcoded machine paths."""

from pathlib import Path

REQUIRED_PROJECT_DIRS: tuple[str, ...] = (
    "configs",
    "data",
    "data/raw",
    "data/raw/DDR",
    "data/raw/IDRiD",
    "data/processed",
    "data/partitions",
    "checkpoints",
    "checkpoints/ssl",
    "checkpoints/segmentation",
    "checkpoints/federated",
    "experiments",
    "logs",
    "docs",
    "scripts",
    "src",
    "src/data",
    "src/models",
    "src/ssl",
    "src/segmentation",
    "src/federated",
    "src/evaluation",
    "src/experiments",
    "src/utils",
    "app",
    "app/pages",
    "tests",
)


def get_project_root() -> Path:
    """Resolve the root directory of the FedMorphis repository dynamically."""
    return Path(__file__).resolve().parents[2]


def get_configs_dir() -> Path:
    """Return the path to the `configs/` directory."""
    return get_project_root() / "configs"


def get_logs_dir() -> Path:
    """Return the path to the `logs/` directory."""
    return get_project_root() / "logs"


def resolve_project_path(relative_path: str | Path) -> Path:
    """Resolve a project-relative path against the dynamically discovered project root."""
    rel = Path(relative_path)
    if rel.is_absolute():
        return rel
    return get_project_root() / rel


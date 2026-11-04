"""Centralized structured logging utilities for FedMorphis.

Usage in application modules:
    >>> from src.utils.logger import get_logger
    >>> logger = get_logger(__name__)
    >>> logger.info("Initialized preprocessing pipeline.")
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from src.utils.paths import get_logs_dir, resolve_project_path

DEFAULT_LOGGER_NAME: str = "fedmorphis"
DEFAULT_LOG_FILENAME: str = "fedmorphis.log"
DEFAULT_LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DEFAULT_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"
DEFAULT_MAX_BYTES: int = 5 * 1024 * 1024  # 5 MB per log file
DEFAULT_BACKUP_COUNT: int = 3

VALID_LOG_LEVELS: dict[str, int] = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "WARN": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


def resolve_log_level(level: str | int) -> int:
    """Validate and convert a log level name or integer to a standard logging constant.

    Args:
        level: Log level string (e.g., ``"DEBUG"``, ``"INFO"``, ``"WARNING"``,
            ``"ERROR"``, ``"CRITICAL"``) or standard ``logging`` integer constant.

    Returns:
        Integer logging level (e.g., ``logging.INFO``).

    Raises:
        TypeError: If ``level`` is not a ``str`` or ``int`` (or is a ``bool``).
        ValueError: If ``level`` is not a recognized log level name or valid positive level.
    """
    if isinstance(level, bool) or not isinstance(level, (str, int)):
        raise TypeError(
            f"Invalid log level type {type(level).__name__} ({level!r}). "
            "Expected a level name string or integer."
        )

    if isinstance(level, str):
        normalized = level.strip().upper()
        if normalized not in VALID_LOG_LEVELS:
            supported = ", ".join(sorted(VALID_LOG_LEVELS.keys()))
            raise ValueError(
                f"Invalid log level {level!r}. Supported levels are: {supported}."
            )
        return VALID_LOG_LEVELS[normalized]

    if level not in set(VALID_LOG_LEVELS.values()):
        supported_ints = sorted(set(VALID_LOG_LEVELS.values()))
        raise ValueError(
            f"Invalid integer log level {level}. Supported integer levels are: {supported_ints}."
        )
    return level


def _create_formatter() -> logging.Formatter:
    """Create the standard FedMorphis log formatter."""
    return logging.Formatter(fmt=DEFAULT_LOG_FORMAT, datefmt=DEFAULT_DATE_FORMAT)


def get_logger(
    name: str = DEFAULT_LOGGER_NAME,
    level: str | int = "INFO",
    log_file: str | Path | None = DEFAULT_LOG_FILENAME,
    log_dir: str | Path | None = None,
    console: bool = True,
    file_logging: bool = True,
    max_bytes: int = DEFAULT_MAX_BYTES,
    backup_count: int = DEFAULT_BACKUP_COUNT,
) -> logging.Logger:
    """Obtain a configured logger with console and rotating file handlers.

    Repeated calls with the same logger ``name`` and file target will update the
    logger's level without attaching duplicate handlers. Propagation to the root
    logger is disabled (``logger.propagate = False``) to prevent duplicate output
    and avoid interfering with third-party loggers.

    Args:
        name: Logger name (e.g., ``__name__`` or ``"fedmorphis.data"``).
            Empty strings are rejected to protect the root logger.
        level: Logging level name (``"DEBUG"``, ``"INFO"``, etc.) or integer constant.
        log_file: Filename for the rotating log file inside ``log_dir`` (default:
            ``"fedmorphis.log"``). Ignored if ``file_logging=False``.
        log_dir: Directory where log files are stored. Defaults to the project
            ``logs/`` directory resolved via ``get_logs_dir()``.
        console: Whether to attach a console ``StreamHandler``.
        file_logging: Whether to attach a ``RotatingFileHandler``.
        max_bytes: Maximum size in bytes before rotating the log file.
        backup_count: Number of rotated backup log files to retain.

    Returns:
        Configured ``logging.Logger`` instance.

    Raises:
        ValueError: If ``name`` is empty or ``level`` / ``max_bytes`` / ``backup_count``
            are invalid.
        TypeError: If ``name`` is not a string or ``level`` has an invalid type.
    """
    if not isinstance(name, str):
        raise TypeError(f"Logger name must be a string, got {type(name).__name__}.")
    cleaned_name = name.strip()
    if not cleaned_name:
        raise ValueError(
            "Logger name must be a non-empty string (configuring the root logger is not permitted)."
        )

    resolved_level = resolve_log_level(level)

    if max_bytes <= 0:
        raise ValueError(f"'max_bytes' must be positive, got {max_bytes}.")
    if backup_count < 0:
        raise ValueError(f"'backup_count' must be non-negative, got {backup_count}.")

    logger = logging.getLogger(cleaned_name)
    logger.setLevel(resolved_level)
    logger.propagate = False

    formatter = _create_formatter()

    # --- Manage Console Handler ---
    existing_console_handlers = [
        h
        for h in logger.handlers
        if getattr(h, "_fedmorphis_kind", None) == "console"
    ]
    if console:
        if existing_console_handlers:
            for h in existing_console_handlers:
                h.setLevel(resolved_level)
                h.setFormatter(formatter)
        else:
            stream_handler = logging.StreamHandler(sys.stdout)
            stream_handler.setLevel(resolved_level)
            stream_handler.setFormatter(formatter)
            setattr(stream_handler, "_fedmorphis_kind", "console")
            logger.addHandler(stream_handler)
    else:
        for h in existing_console_handlers:
            logger.removeHandler(h)

    # --- Manage Rotating File Handler ---
    existing_file_handlers = [
        h for h in logger.handlers if getattr(h, "_fedmorphis_kind", None) == "file"
    ]
    if file_logging and log_file is not None:
        base_dir = (
            get_logs_dir()
            if log_dir is None
            else resolve_project_path(log_dir)
        )
        log_file_path = Path(log_file)
        target_path = (
            log_file_path.resolve()
            if log_file_path.is_absolute()
            else (base_dir / log_file_path).resolve()
        )
        target_path.parent.mkdir(parents=True, exist_ok=True)

        matching_handler: RotatingFileHandler | None = None
        for h in existing_file_handlers:
            if (
                isinstance(h, RotatingFileHandler)
                and Path(h.baseFilename).resolve() == target_path
            ):
                matching_handler = h
            else:
                logger.removeHandler(h)
                h.close()

        if matching_handler is not None:
            matching_handler.setLevel(resolved_level)
            matching_handler.setFormatter(formatter)
        else:
            file_handler = RotatingFileHandler(
                filename=target_path,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding="utf-8",
            )
            file_handler.setLevel(resolved_level)
            file_handler.setFormatter(formatter)
            setattr(file_handler, "_fedmorphis_kind", "file")
            logger.addHandler(file_handler)
    else:
        for h in existing_file_handlers:
            logger.removeHandler(h)
            h.close()

    return logger


def configure_logging(
    level: str | int = "INFO",
    log_file: str | Path | None = DEFAULT_LOG_FILENAME,
    log_dir: str | Path | None = None,
    console: bool = True,
    file_logging: bool = True,
) -> logging.Logger:
    """Configure and return the default project-level ``fedmorphis`` logger."""
    return get_logger(
        name=DEFAULT_LOGGER_NAME,
        level=level,
        log_file=log_file,
        log_dir=log_dir,
        console=console,
        file_logging=file_logging,
    )


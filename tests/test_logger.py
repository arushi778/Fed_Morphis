"""Unit tests for centralized structured logging (src/utils/logger.py)."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Iterator

import pytest

from src.utils.logger import (
     configure_logging,
    get_logger,
    resolve_log_level,
)


@pytest.fixture(autouse=True)
def _cleanup_test_loggers() -> Iterator[None]:
    """Ensure any test loggers close file handles after each test."""
    yield
    for name, logger_obj in list(logging.Logger.manager.loggerDict.items()):
        if isinstance(logger_obj, logging.Logger) and name.startswith("test_"):
            for handler in list(logger_obj.handlers):
                logger_obj.removeHandler(handler)
                handler.close()


@pytest.mark.parametrize(
    ("level_input", "expected_level"),
    [
        ("DEBUG", logging.DEBUG),
        ("debug", logging.DEBUG),
        ("  info ", logging.INFO),
        ("WARNING", logging.WARNING),
        ("WARN", logging.WARNING),
        ("ERROR", logging.ERROR),
        ("CRITICAL", logging.CRITICAL),
        (logging.DEBUG, logging.DEBUG),
        (logging.INFO, logging.INFO),
        (logging.ERROR, logging.ERROR),
    ],
)
def test_resolve_log_level_valid_inputs(
    level_input: str | int, expected_level: int
) -> None:
    """Verify resolve_log_level converts valid level strings and ints accurately."""
    assert resolve_log_level(level_input) == expected_level


@pytest.mark.parametrize("bad_level", ["VERBOSE", "TRACE", "", "NOT_A_LEVEL", 15, -10, 999])
def test_resolve_log_level_rejects_invalid_values(bad_level: str | int) -> None:
    """Verify invalid log level strings or integers raise ValueError."""
    with pytest.raises(ValueError, match="log level"):
        resolve_log_level(bad_level)


@pytest.mark.parametrize("bad_type", [None, True, False, 3.14, ["INFO"]])
def test_resolve_log_level_rejects_invalid_types(bad_type: object) -> None:
    """Verify non-str/non-int log levels (including booleans) raise TypeError."""
    with pytest.raises(TypeError, match="Invalid log level type"):
        resolve_log_level(bad_type)  # type: ignore[arg-type]


def test_logger_creation_and_expected_level(tmp_path: Path) -> None:
    """Verify logger creation sets level, disables propagation, and attaches handlers."""
    logger = get_logger(
        name="test_logger.basic",
        level="DEBUG",
        log_dir=tmp_path,
        log_file="unit_test.log",
    )
    assert logger.name == "test_logger.basic"
    assert logger.level == logging.DEBUG
    assert logger.propagate is False
    assert len(logger.handlers) == 2


def test_console_and_file_logging_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify messages are formatted and written to both stdout console and log file."""
    log_dir = tmp_path / "nested_logs"
    logger = get_logger(
        name="test_logger.io",
        level="INFO",
        log_dir=log_dir,
        log_file="app_run.log",
        console=True,
        file_logging=True,
    )

    logger.info("Biomedical pipeline initialized.")
    logger.debug("This debug message should be filtered out at INFO level.")
    logger.warning("Low contrast fundus image detected.")

    for handler in logger.handlers:
        handler.flush()

    captured = capsys.readouterr()
    assert "Biomedical pipeline initialized." in captured.out
    assert "INFO" in captured.out
    assert "test_logger.io" in captured.out
    assert "This debug message should be filtered out" not in captured.out

    expected_file = log_dir / "app_run.log"
    assert expected_file.exists()
    content = expected_file.read_text(encoding="utf-8")
    assert "Biomedical pipeline initialized." in content
    assert "Low contrast fundus image detected." in content
    assert "WARNING" in content
    assert "This debug message should be filtered out" not in content


def test_repeated_get_logger_does_not_duplicate_handlers(tmp_path: Path) -> None:
    """Verify calling get_logger repeatedly on the same logger does not add duplicate handlers."""
    logger_1 = get_logger(
        name="test_logger.idempotent",
        level="INFO",
        log_dir=tmp_path,
        log_file="repeat.log",
    )
    initial_handler_count = len(logger_1.handlers)
    assert initial_handler_count == 2

    logger_2 = get_logger(
        name="test_logger.idempotent",
        level="DEBUG",
        log_dir=tmp_path,
        log_file="repeat.log",
    )
    assert logger_1 is logger_2
    assert len(logger_2.handlers) == initial_handler_count
    assert logger_2.level == logging.DEBUG
    for handler in logger_2.handlers:
        assert handler.level == logging.DEBUG


def test_different_logger_names_behave_independently(tmp_path: Path) -> None:
    """Verify distinct logger names maintain independent levels and log files."""
    logger_ssl = get_logger(
        name="test_logger.ssl",
        level="DEBUG",
        log_dir=tmp_path,
        log_file="ssl.log",
    )
    logger_fed = get_logger(
        name="test_logger.federated",
        level="ERROR",
        log_dir=tmp_path,
        log_file="federated.log",
    )

    assert logger_ssl is not logger_fed
    assert logger_ssl.level == logging.DEBUG
    assert logger_fed.level == logging.ERROR

    logger_ssl.debug("SimCLR projection head initialized.")
    logger_fed.info("This INFO message should not appear in ERROR-level federated log.")
    logger_fed.error("Client 2 connection timeout.")

    for h in logger_ssl.handlers + logger_fed.handlers:
        h.flush()

    ssl_text = (tmp_path / "ssl.log").read_text(encoding="utf-8")
    fed_text = (tmp_path / "federated.log").read_text(encoding="utf-8")

    assert "SimCLR projection head initialized." in ssl_text
    assert "Client 2 connection timeout." not in ssl_text

    assert "Client 2 connection timeout." in fed_text
    assert "This INFO message should not appear" not in fed_text


def test_console_only_and_file_only_modes(tmp_path: Path) -> None:
    """Verify toggling console=False or file_logging=False attaches only the requested handler."""
    console_only = get_logger(
        name="test_logger.console_only",
        log_dir=tmp_path,
        console=True,
        file_logging=False,
    )
    assert len(console_only.handlers) == 1
    assert not isinstance(console_only.handlers[0], RotatingFileHandler)

    file_only = get_logger(
        name="test_logger.file_only",
        log_dir=tmp_path,
        log_file="only_file.log",
        console=False,
        file_logging=True,
    )
    assert len(file_only.handlers) == 1
    assert isinstance(file_only.handlers[0], RotatingFileHandler)


def test_invalid_logger_arguments_raise_clear_errors(tmp_path: Path) -> None:
    """Verify invalid logger names, levels, or rotation settings raise clear exceptions."""
    with pytest.raises(ValueError, match="non-empty string"):
        get_logger(name="   ", log_dir=tmp_path)

    with pytest.raises(TypeError, match="Logger name must be a string"):
        get_logger(name=123, log_dir=tmp_path)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Invalid log level"):
        get_logger(name="test_logger.bad_level", level="INVALID", log_dir=tmp_path)

    with pytest.raises(ValueError, match="max_bytes"):
        get_logger(name="test_logger.bad_bytes", max_bytes=0, log_dir=tmp_path)

    with pytest.raises(ValueError, match="backup_count"):
        get_logger(name="test_logger.bad_backups", backup_count=-1, log_dir=tmp_path)


def test_configure_logging_helper_uses_isolated_dir(tmp_path: Path) -> None:
    """Verify configure_logging configures the default project logger in an isolated directory."""
    logger = configure_logging(
        level="INFO",
        log_file="project_test.log",
        log_dir=tmp_path,
        console=True,
        file_logging=True,
    )
    logger.info("Configured project logger test.")
    for h in list(logger.handlers):
        h.flush()
        if isinstance(h, RotatingFileHandler):
            logger.removeHandler(h)
            h.close()

    assert (tmp_path / "project_test.log").exists()


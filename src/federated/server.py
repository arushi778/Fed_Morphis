"""Flower federated server orchestration for FedMorphis retinal lesion segmentation.

Provides:
- :class:`InProcessClientProxy`: Flower :class:`flwr.server.client_proxy.ClientProxy`
  bridging :class:`flwr.server.Server` with :class:`src.federated.client.FedMorphisNumPyClient`
  (via :meth:`NumPyClient.to_client`) using standard Flower protobuf :class:`Parameters`,
  :class:`FitIns`, :class:`FitRes`, :class:`EvaluateIns`, and :class:`EvaluateRes` messages.
- :class:`FederatedServerRunResult`: Structured container returned after executing
  federated rounds.
- :func:`build_federated_server`: Constructs the global model, :class:`FedMorphisFedAvg`
  strategy, :class:`SimpleClientManager`, and :class:`flwr.server.Server`.
- :func:`run_federated_training` (and alias :func:`run_federated_rounds`): End-to-end
  federated server orchestration across the configured clients and communication rounds.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import json
import logging
import os
from pathlib import Path
from typing import Any
import uuid

import flwr
from flwr.client import Client, NumPyClient
from flwr.common import (
    DisconnectRes,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    GetPropertiesIns,
    GetPropertiesRes,
    Parameters,
    ReconnectIns,
)
from flwr.server import History, Server, ServerConfig, SimpleClientManager
from flwr.server.client_proxy import ClientProxy
import torch
import torch.nn as nn

from src.data.dataset import _extract_records_from_input
from src.data.masks import CANONICAL_LESION_CHANNELS
from src.engine.evaluator import verify_exact_split
from src.engine.trainer import verify_no_test_split_leakage
from src.federated.client import (
    FedMorphisNumPyClient,
    build_federated_client,
)
from src.federated.data import (
    ClientDataBundle,
    FederatedDataError,
    build_all_federated_client_dataloaders,
)
from src.federated.strategy import (
    FedMorphisFedAvg,
    FederatedStrategyError,
    build_fedavg_strategy,
    initialize_global_model,
    load_global_checkpoint,
    save_global_checkpoint,
)
from src.utils.config import load_yaml_config
from src.utils.device import resolve_device
from src.utils.logger import configure_federated_logging, get_logger
from src.utils.paths import format_project_path, resolve_project_path
from src.utils.seed import set_seed

logger = get_logger("fedmorphis.federated.server")


class FederatedPreflightError(FederatedStrategyError):
    """Raised when a federated training preflight check fails."""


class InProcessClientProxy(ClientProxy):
    """Flower :class:`ClientProxy` executing a local :class:`Client` or :class:`NumPyClient`.

    Routes all :meth:`get_properties`, :meth:`get_parameters`, :meth:`fit`, and
    :meth:`evaluate` instructions through Flower's :class:`flwr.client.Client`
    interface so parameter serialization (`Parameters` protobufs) and status codes
    behave identically to networked Flower deployments.
    """

    def __init__(
        self,
        cid: str,
        client: NumPyClient | Client,
        node_id: int | None = None,
    ) -> None:
        super().__init__(cid=str(cid))
        if isinstance(client, NumPyClient):
            self.numpy_client: NumPyClient | None = client
            self.flower_client: Client = client.to_client()
        elif isinstance(client, Client):
            self.numpy_client = None
            self.flower_client = client
        else:
            raise TypeError(
                f"Expected flwr.client.NumPyClient or flwr.client.Client for cid={cid!r}, "
                f"got {type(client).__name__}."
            )
        self.node_id = int(node_id if node_id is not None else abs(hash(str(cid))) % (2**31))

    def get_properties(
        self,
        ins: GetPropertiesIns,
        timeout: float | None,
        group_id: int | None,
    ) -> GetPropertiesRes:
        """Return the client's static properties."""
        _ = (timeout, group_id)
        return self.flower_client.get_properties(ins)

    def get_parameters(
        self,
        ins: GetParametersIns,
        timeout: float | None,
        group_id: int | None,
    ) -> GetParametersRes:
        """Return the current local model parameters from the client."""
        _ = (timeout, group_id)
        return self.flower_client.get_parameters(ins)

    def fit(
        self,
        ins: FitIns,
        timeout: float | None,
        group_id: int | None,
    ) -> FitRes:
        """Execute local training on the client for one communication round."""
        _ = (timeout, group_id)
        return self.flower_client.fit(ins)

    def evaluate(
        self,
        ins: EvaluateIns,
        timeout: float | None,
        group_id: int | None,
    ) -> EvaluateRes:
        """Execute local validation on the client for one communication round."""
        _ = (timeout, group_id)
        return self.flower_client.evaluate(ins)

    def reconnect(
        self,
        ins: ReconnectIns,
        timeout: float | None,
        group_id: int | None,
    ) -> DisconnectRes:
        """Handle server disconnect/reconnect instruction."""
        _ = (ins, timeout, group_id)
        return DisconnectRes(reason="RECONNECT")


@dataclass
class FederatedServerRunResult:
    """Structured output of a completed Flower federated server run."""

    num_rounds: int
    elapsed_sec: float
    history: History
    round_history: list[dict[str, Any]]
    global_model: nn.Module
    strategy: FedMorphisFedAvg
    server: Server
    best_round: int
    best_metric_value: float | None
    monitor_metric: str
    monitor_mode: str
    latest_checkpoint_path: Path | None = None
    best_checkpoint_path: Path | None = None
    report_path: Path | None = None
    log_path: Path | None = None
    initialization_metadata: dict[str, Any] = field(default_factory=dict)
    experiment_name: str = "fedmorphis_federated_fedavg"
    status: str = "completed"
    number_of_clients: int = 2
    local_epochs: int = 3
    seed: int = 42
    device: str = "cpu"
    client_sample_counts: dict[str, Any] = field(default_factory=dict)
    config_metadata: dict[str, Any] = field(default_factory=dict)

    def to_report_dict(self) -> dict[str, Any]:
        """Serialize run summary and round history into a JSON-compatible dictionary."""
        init_meta = dict(self.initialization_metadata)
        global_init_source = str(
            init_meta.get(
                "initialization_mode",
                init_meta.get("encoder_weights_source", "unknown"),
            )
        )
        per_round_train = [
            {"round": int(r.get("round", idx + 1)), **dict(r.get("train", {}))}
            for idx, r in enumerate(self.round_history)
        ]
        per_round_val = [
            {"round": int(r.get("round", idx + 1)), **dict(r.get("val", {}))}
            for idx, r in enumerate(self.round_history)
        ]
        best_val = (
            float(self.best_metric_value)
            if self.best_metric_value is not None
            and self.best_metric_value not in {float("-inf"), float("inf")}
            else None
        )
        latest_ckpt_str = (
            format_project_path(self.latest_checkpoint_path)
            if self.latest_checkpoint_path is not None
            else None
        )
        best_ckpt_str = (
            format_project_path(self.best_checkpoint_path)
            if self.best_checkpoint_path is not None
            else None
        )
        log_path_str = (
            format_project_path(self.log_path)
            if self.log_path is not None
            else None
        )
        duration = round(float(self.elapsed_sec), 4)

        return {
            "status": str(self.status),
            "experiment_name": str(self.experiment_name),
            "framework": "flower",
            "strategy": str(
                self.config_metadata.get("federated", {}).get("strategy", "FedAvg")
            ),
            "number_of_clients": int(self.number_of_clients),
            "num_rounds": int(self.num_rounds),
            "rounds_completed": int(len(self.round_history)),
            "local_epochs": int(self.local_epochs),
            "seed": int(self.seed),
            "device": str(self.device),
            "client_sample_counts": dict(self.client_sample_counts),
            "global_initialization_source": global_init_source,
            "initialization": init_meta,
            "monitor_metric": self.monitor_metric,
            "monitor_mode": self.monitor_mode,
            "best_round": int(self.best_round),
            "best_metric_value": best_val,
            "best_checkpoint_path": best_ckpt_str,
            "latest_checkpoint_path": latest_ckpt_str,
            "log_path": log_path_str,
            "training_duration_sec": duration,
            "elapsed_sec": duration,
            "per_round_train_metrics": per_round_train,
            "per_round_val_metrics": per_round_val,
            "round_history": list(self.round_history),
            "config": dict(self.config_metadata),
        }


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    """Write ``payload`` as formatted JSON atomically using a temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.parent / f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    try:
        with tmp_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=False)
            fh.write("\n")
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise
    return path


def register_clients_with_manager(
    clients: Sequence[FedMorphisNumPyClient | NumPyClient | Client | ClientProxy]
    | Mapping[str, FedMorphisNumPyClient | NumPyClient | Client | ClientProxy],
    client_manager: SimpleClientManager | None = None,
) -> tuple[SimpleClientManager, list[ClientProxy]]:
    """Wrap and register federated clients with a Flower :class:`SimpleClientManager`."""
    manager = client_manager if client_manager is not None else SimpleClientManager()
    proxies: list[ClientProxy] = []

    if isinstance(clients, Mapping):
        items: list[tuple[str, Any]] = [(str(k), v) for k, v in clients.items()]
    else:
        items = []
        for idx, client_obj in enumerate(clients):
            if isinstance(client_obj, ClientProxy):
                cid = str(client_obj.cid)
            elif hasattr(client_obj, "client_id"):
                cid = str(getattr(client_obj, "client_id"))
            else:
                cid = f"client_{idx}"
            items.append((cid, client_obj))

    if not items:
        raise FederatedStrategyError("Cannot register an empty client collection with server.")

    for idx, (cid, client_obj) in enumerate(items):
        if isinstance(client_obj, ClientProxy):
            proxy = client_obj
        else:
            proxy = InProcessClientProxy(cid=cid, client=client_obj, node_id=idx)

        registered = manager.register(proxy)
        if not registered:
            raise FederatedStrategyError(
                f"Failed to register client '{cid}' with SimpleClientManager."
            )
        proxies.append(proxy)

    return manager, proxies


def build_federated_server(
    federated_config: Mapping[str, Any] | str | Path = "federated.yaml",
    dataset_config: Mapping[str, Any] | str | Path = "dataset.yaml",
    model_config: Mapping[str, Any] | str | Path = "model.yaml",
    project_config: Mapping[str, Any] | str | Path = "project.yaml",
    *,
    clients: (
        Sequence[FedMorphisNumPyClient | NumPyClient | Client | ClientProxy]
        | Mapping[str, FedMorphisNumPyClient | NumPyClient | Client | ClientProxy]
        | None
    ) = None,
    data_bundles: Mapping[str, ClientDataBundle] | Sequence[ClientDataBundle] | None = None,
    partition_manifest: str | Path | Mapping[str, Any] | None = None,
    source_manifest: str | Path | Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    global_model: nn.Module | None = None,
    strategy: FedMorphisFedAvg | None = None,
    initial_parameters: Parameters | None = None,
    checkpoint_dir: str | Path | None = None,
    save_checkpoints: bool = True,
    accept_failures: bool = False,
    include_buffers: bool | None = None,
    integer_buffer_policy: str = "max",
    local_epochs: int | None = None,
    batch_size: int | None = None,
    val_batch_size: int | None = None,
    num_workers: int | None = None,
    learning_rate: float | None = None,
    mixed_precision: bool | None = None,
    seed: int | None = None,
    device: torch.device | str | None = None,
    image_size: tuple[int, int] | None = None,
    fov_crop: bool | None = None,
    cache_in_memory: bool = False,
    verify_files_exist: bool = False,
    max_train_batches: int | None = None,
    max_val_batches: int | None = None,
    num_rounds: int | None = None,
    max_workers: int = 1,
) -> tuple[Server, FedMorphisFedAvg, nn.Module, list[ClientProxy]]:
    """Construct a Flower :class:`Server`, :class:`FedMorphisFedAvg` strategy, global model, and client proxies.

    Defaults to ``max_workers=1`` on the Flower :class:`Server` so in-process GPU
    clients execute sequentially and deterministically without GPU contention or
    RNG interleaving.
    """
    fed_cfg = (
        load_yaml_config(federated_config, validate=True)
        if isinstance(federated_config, (str, Path))
        else dict(federated_config)
    )
    ds_cfg = (
        load_yaml_config(dataset_config, validate=True)
        if isinstance(dataset_config, (str, Path))
        else dict(dataset_config)
    )
    mdl_cfg = (
        load_yaml_config(model_config, validate=True)
        if isinstance(model_config, (str, Path))
        else dict(model_config)
    )
    proj_cfg = (
        load_yaml_config(project_config, validate=True)
        if isinstance(project_config, (str, Path))
        else dict(project_config)
    )

    resolved_seed = int(
        seed
        if seed is not None
        else fed_cfg.get(
            "seed",
            fed_cfg.get("partitioning", {}).get("seed", proj_cfg.get("random_seed", 42)),
        )
    )
    deterministic = bool(proj_cfg.get("deterministic", True))
    set_seed(resolved_seed, deterministic=deterministic)

    if device is None:
        resolved_device = resolve_device(str(proj_cfg.get("device", "auto")))
    elif isinstance(device, str):
        resolved_device = resolve_device(device)
    else:
        resolved_device = torch.device(device)

    # 1. Initialize global model and strategy
    if strategy is None:
        strategy, global_model = build_fedavg_strategy(
            federated_config=fed_cfg,
            model_config=mdl_cfg,
            dataset_config=ds_cfg,
            project_config=proj_cfg,
            global_model=global_model,
            initial_parameters=initial_parameters,
            checkpoint_dir=checkpoint_dir,
            save_checkpoints=save_checkpoints,
            accept_failures=accept_failures,
            include_buffers=include_buffers,
            integer_buffer_policy=integer_buffer_policy,
            local_epochs=local_epochs,
            learning_rate=learning_rate,
            max_train_batches=max_train_batches,
            max_val_batches=max_val_batches,
            seed=resolved_seed,
            num_rounds=num_rounds,
            device="cpu",
        )
    else:
        if num_rounds is not None:
            strategy.num_rounds = int(num_rounds)
        if global_model is None:
            global_model = strategy.global_model

    # 2. Construct federated clients if not pre-supplied
    if clients is None:
        if data_bundles is None:
            bundles_map = build_all_federated_client_dataloaders(
                partition_manifest=partition_manifest,
                source_manifest=source_manifest,
                federated_config=fed_cfg,
                dataset_config=ds_cfg,
                batch_size=batch_size,
                val_batch_size=val_batch_size,
                num_workers=num_workers,
                seed=resolved_seed,
                image_size=image_size,
                fov_crop=fov_crop,
                cache_in_memory=cache_in_memory,
                verify_files_exist=verify_files_exist,
            )
            ordered_bundles = [bundles_map[k] for k in sorted(bundles_map.keys())]
        elif isinstance(data_bundles, Mapping):
            ordered_bundles = [data_bundles[k] for k in sorted(data_bundles.keys())]
        else:
            ordered_bundles = list(data_bundles)

        built_clients: list[FedMorphisNumPyClient] = []
        for bundle in ordered_bundles:
            c = build_federated_client(
                client_id=bundle.client_id,
                federated_config=fed_cfg,
                dataset_config=ds_cfg,
                model_config=mdl_cfg,
                project_config=proj_cfg,
                data_bundle=bundle,
                pretrained=False,
                strict_pretrained=False,
                device=resolved_device,
                local_epochs=strategy.local_epochs,
                learning_rate=strategy.learning_rate,
                include_buffers=strategy.include_buffers,
                mixed_precision=mixed_precision,
                seed=resolved_seed,
                max_train_batches=max_train_batches,
                max_val_batches=max_val_batches,
                as_flower_client=False,
            )
            assert isinstance(c, FedMorphisNumPyClient)
            built_clients.append(c)
        clients_to_register: Any = built_clients
    else:
        clients_to_register = clients

    client_manager, proxies = register_clients_with_manager(clients_to_register)

    server = Server(
        client_manager=client_manager,
        strategy=strategy,
    )
    if max_workers is not None:
        if int(max_workers) < 1:
            raise ValueError(f"max_workers must be >= 1, got {max_workers}.")
        server.set_max_workers(int(max_workers))

    return server, strategy, global_model, proxies


def _extract_client_sample_counts(
    proxies: Sequence[ClientProxy],
) -> dict[str, Any]:
    """Collect per-client and total train/validation sample counts from registered client proxies."""
    per_client: dict[str, Any] = {}
    total_train = 0
    total_val = 0

    for proxy in sorted(proxies, key=lambda p: str(p.cid)):
        cid = str(proxy.cid)
        entry: dict[str, Any] = {"client_id": cid}

        np_client = getattr(proxy, "numpy_client", None)
        if np_client is not None and hasattr(np_client, "get_properties"):
            try:
                props = np_client.get_properties({})
                n_tr = int(props.get("num_train_samples", 0))
                n_val = int(props.get("num_val_samples", 0))
                ds_str = str(props.get("dataset", ""))
                entry["dataset"] = ds_str
                entry["dataset_names"] = [d for d in ds_str.split(",") if d]
                entry["num_train_samples"] = n_tr
                entry["num_val_samples"] = n_val
                bundle = getattr(np_client, "data_bundle", None)
                if bundle is not None:
                    entry["val_split_mode"] = getattr(bundle, "val_split_mode", None)
                total_train += n_tr
                total_val += n_val
            except Exception:
                pass
        else:
            try:
                res = proxy.get_properties(
                    GetPropertiesIns(config={}),
                    timeout=None,
                    group_id=0,
                )
                props = dict(res.properties or {})
                n_tr = int(props.get("num_train_samples", 0))
                n_val = int(props.get("num_val_samples", 0))
                ds_str = str(props.get("dataset", ""))
                entry["dataset"] = ds_str
                entry["dataset_names"] = [d for d in ds_str.split(",") if d]
                entry["num_train_samples"] = n_tr
                entry["num_val_samples"] = n_val
                total_train += n_tr
                total_val += n_val
            except Exception:
                pass

        per_client[cid] = entry

    return {
        "clients": per_client,
        "total_train_samples": int(total_train),
        "total_val_samples": int(total_val),
    }


def _verify_directory_writable(directory: Path, label: str) -> None:
    """Ensure ``directory`` can be created and written to on disk."""
    try:
        directory.mkdir(parents=True, exist_ok=True)
        probe_path = directory / f".fedmorphis_write_probe_{os.getpid()}_{uuid.uuid4().hex}.tmp"
        with probe_path.open("w", encoding="utf-8") as fh:
            fh.write("ok\n")
        probe_path.unlink()
    except Exception as exc:
        raise FederatedPreflightError(
            f"Preflight failed: {label} directory '{directory}' is not writable: {exc}"
        ) from exc


def _verify_output_paths_isolated(
    *,
    checkpoint_dir: Path | None,
    report_path: Path | None,
    log_path: Path | None,
    protect_pilot_outputs: bool = True,
) -> dict[str, Any]:
    """Verify output paths are writable and will not overwrite pilot checkpoints or reports."""
    paths_summary: dict[str, Any] = {
        "checkpoint_dir": format_project_path(checkpoint_dir) if checkpoint_dir is not None else None,
        "report_path": format_project_path(report_path) if report_path is not None else None,
        "log_path": format_project_path(log_path) if log_path is not None else None,
        "writable": True,
        "pilot_isolated": True,
    }

    if checkpoint_dir is not None:
        _verify_directory_writable(checkpoint_dir, "checkpoint")
    if report_path is not None:
        _verify_directory_writable(report_path.parent, "report")
    if log_path is not None:
        _verify_directory_writable(log_path.parent, "log")

    if not protect_pilot_outputs:
        return paths_summary

    for label, path_obj in (
        ("checkpoint_dir", checkpoint_dir),
        ("report_path", report_path),
        ("log_path", log_path),
    ):
        if path_obj is None:
            continue
        norm_str = path_obj.as_posix().lower()
        if "pilot" in norm_str:
            raise FederatedPreflightError(
                f"Preflight failed: {label} '{format_project_path(path_obj)}' targets a pilot "
                "path while protect_pilot_outputs=True. Use a dedicated non-pilot path."
            )

    if checkpoint_dir is not None:
        default_root = resolve_project_path("checkpoints/federated").resolve()
        if checkpoint_dir.resolve() == default_root:
            existing_ckpts = [
                p.name
                for p in (
                    default_root / "best_checkpoint.pt",
                    default_root / "latest_checkpoint.pt",
                )
                if p.exists()
            ]
            if existing_ckpts:
                raise FederatedPreflightError(
                    f"Preflight failed: checkpoint_dir '{format_project_path(checkpoint_dir)}' "
                    f"already contains checkpoint file(s) {existing_ckpts}. Use a dedicated "
                    "subdirectory so existing/pilot checkpoints are not overwritten."
                )

    if report_path is not None and report_path.exists():
        default_rep = resolve_project_path(
            "logs/federated_fedavg_ddr_idrid_report.json"
        ).resolve()
        if report_path.resolve() == default_rep:
            raise FederatedPreflightError(
                f"Preflight failed: report_path '{format_project_path(report_path)}' already "
                "exists at the default report location. Use a dedicated report path to avoid "
                "overwriting pilot or prior experiment reports."
            )

    return paths_summary


def run_federated_preflight(
    federated_config: Mapping[str, Any] | str | Path = "federated.yaml",
    dataset_config: Mapping[str, Any] | str | Path = "dataset.yaml",
    model_config: Mapping[str, Any] | str | Path = "model.yaml",
    project_config: Mapping[str, Any] | str | Path = "project.yaml",
    *,
    experiment_config: Mapping[str, Any] | str | Path | None = None,
    experiment_name: str | None = None,
    num_rounds: int | None = None,
    partition_manifest: str | Path | Mapping[str, Any] | None = None,
    source_manifest: str | Path | Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    checkpoint_dir: str | Path | None = None,
    report_path: str | Path | None = None,
    log_path: str | Path | None = None,
    log_level: str = "INFO",
    include_buffers: bool | None = None,
    integer_buffer_policy: str = "max",
    local_epochs: int | None = None,
    batch_size: int | None = None,
    val_batch_size: int | None = None,
    num_workers: int | None = None,
    learning_rate: float | None = None,
    mixed_precision: bool | None = None,
    seed: int | None = None,
    device: torch.device | str | None = None,
    image_size: tuple[int, int] | None = None,
    fov_crop: bool | None = None,
    verify_files_exist: bool = True,
    protect_pilot_outputs: bool = True,
    require_cuda: bool = False,
    expected_client_counts: Mapping[str, Mapping[str, int]] | None = None,
) -> dict[str, Any]:
    """Execute preflight validation for Flower federated segmentation training.

    Verifies before training begins:
    1. DDR and IDRiD training and validation datasets can be loaded and all underlying
       image and lesion mask files exist on disk (when ``verify_files_exist=True``).
    2. Both clients (`client_0` = DDR, `client_1` = IDRiD) are constructed, registered,
       and eligible to participate in every round.
    3. The global :class:`ResNet50UNet` model initializes cleanly with finite parameters
       and the configured pretrained weights (`imagenet`, `ssl_checkpoint`, or `random`).
    4. CUDA availability and mixed-precision (`AMP`) settings are valid and operational.
    5. Target checkpoint, report, and log directories are writable and isolated from
       Task 7B pilot checkpoints/reports.
    6. Zero held-out test split samples enter any client training or validation dataset.

    Raises:
        FederatedPreflightError: If any required preflight condition is not met.
    """
    try:
        fed_cfg = (
            load_yaml_config(federated_config, validate=True)
            if isinstance(federated_config, (str, Path))
            else dict(federated_config)
        )
        ds_cfg = (
            load_yaml_config(dataset_config, validate=True)
            if isinstance(dataset_config, (str, Path))
            else dict(dataset_config)
        )
        mdl_cfg = (
            load_yaml_config(model_config, validate=True)
            if isinstance(model_config, (str, Path))
            else dict(model_config)
        )
        proj_cfg = (
            load_yaml_config(project_config, validate=True)
            if isinstance(project_config, (str, Path))
            else dict(project_config)
        )
        if experiment_config is not None:
            exp_cfg: dict[str, Any] = (
                load_yaml_config(experiment_config, validate=True)
                if isinstance(experiment_config, (str, Path))
                else dict(experiment_config)
            )
        else:
            try:
                exp_cfg = load_yaml_config("experiment.yaml", validate=False)
            except Exception:
                exp_cfg = {}
    except Exception as exc:
        raise FederatedPreflightError(
            f"Preflight failed during configuration validation: {exc}"
        ) from exc

    resolved_experiment_name = str(
        experiment_name
        if experiment_name is not None
        else fed_cfg.get(
            "experiment_name",
            exp_cfg.get("experiment_name", "fedmorphis_federated_fedavg"),
        )
    ).strip() or "fedmorphis_federated_fedavg"

    resolved_rounds = int(
        num_rounds
        if num_rounds is not None
        else fed_cfg.get("rounds", fed_cfg.get("communication_rounds", 20))
    )
    if resolved_rounds < 1:
        raise FederatedPreflightError(
            f"Preflight failed: num_rounds must be >= 1, got {resolved_rounds}."
        )

    resolved_seed = int(
        seed
        if seed is not None
        else fed_cfg.get(
            "seed",
            fed_cfg.get("partitioning", {}).get("seed", proj_cfg.get("random_seed", 42)),
        )
    )
    deterministic = bool(proj_cfg.get("deterministic", True))
    set_seed(resolved_seed, deterministic=deterministic)

    # 1. Resolve and validate target device & mixed precision
    requested_device_str = (
        str(device)
        if device is not None
        else str(proj_cfg.get("device", "auto"))
    ).strip().lower()

    try:
        if device is None:
            resolved_device = resolve_device(str(proj_cfg.get("device", "auto")))
        elif isinstance(device, str):
            resolved_device = resolve_device(device)
        else:
            resolved_device = torch.device(device)
    except Exception as exc:
        raise FederatedPreflightError(
            f"Preflight failed: invalid device configuration ({requested_device_str!r}): {exc}"
        ) from exc

    cuda_available = bool(torch.cuda.is_available())
    if (require_cuda or requested_device_str == "cuda") and (
        resolved_device.type != "cuda" or not cuda_available
    ):
        raise FederatedPreflightError(
            f"Preflight failed: CUDA execution is required (requested={requested_device_str!r}, "
            f"require_cuda={require_cuda}), but resolved_device={resolved_device} and "
            f"torch.cuda.is_available()={cuda_available}."
        )

    resolved_amp = bool(
        mixed_precision
        if mixed_precision is not None
        else mdl_cfg.get("training", {}).get("mixed_precision", False)
    )
    if not isinstance(resolved_amp, bool):
        raise FederatedPreflightError(
            f"Preflight failed: mixed_precision must be a boolean, got {resolved_amp!r}."
        )

    gpu_name: str | None = None
    gpu_vram_gb: float | None = None
    amp_verified = False
    if resolved_device.type == "cuda":
        if not cuda_available:
            raise FederatedPreflightError(
                "Preflight failed: resolved device is 'cuda', but torch.cuda.is_available() is False."
            )
        try:
            dev_idx = (
                resolved_device.index
                if resolved_device.index is not None
                else torch.cuda.current_device()
            )
            props = torch.cuda.get_device_properties(dev_idx)
            gpu_name = str(props.name)
            gpu_vram_gb = round(float(props.total_memory / (1024**3)), 2)
            # Execute a lightweight CUDA + AMP probe
            probe_x = torch.ones((2, 4), device=resolved_device, dtype=torch.float32, requires_grad=True)
            probe_w = torch.ones((4, 2), device=resolved_device, dtype=torch.float32, requires_grad=True)
            scaler = torch.amp.GradScaler("cuda", enabled=resolved_amp)
            with torch.amp.autocast("cuda", enabled=resolved_amp):
                probe_out = (probe_x @ probe_w).sum()
            scaler.scale(probe_out).backward()
            if not bool(torch.isfinite(probe_out).item()):
                raise FederatedPreflightError("CUDA/AMP probe produced non-finite output.")
            amp_verified = True
        except FederatedPreflightError:
            raise
        except Exception as exc:
            raise FederatedPreflightError(
                f"Preflight failed during CUDA / mixed-precision verification on {resolved_device}: {exc}"
            ) from exc
    else:
        amp_verified = not resolved_amp

    # 2. Resolve and verify output directories and pilot isolation
    ckpt_cfg = dict(fed_cfg.get("checkpointing", {}))
    raw_ckpt_dir = (
        checkpoint_dir
        if checkpoint_dir is not None
        else ckpt_cfg.get(
            "save_dir",
            fed_cfg.get("checkpoint_directory", "checkpoints/federated"),
        )
    )
    resolved_ckpt_dir = (
        resolve_project_path(raw_ckpt_dir) if raw_ckpt_dir is not None else None
    )

    raw_rep = (
        report_path
        if report_path is not None
        else fed_cfg.get("logging", {}).get(
            "report_path", fed_cfg.get("report_path")
        )
    )
    resolved_report_path = (
        resolve_project_path(raw_rep) if raw_rep is not None else None
    )

    raw_log = (
        log_path
        if log_path is not None
        else fed_cfg.get("logging", {}).get(
            "log_path", fed_cfg.get("log_path")
        )
    )
    resolved_log_path = (
        resolve_project_path(raw_log) if raw_log is not None else None
    )
    if resolved_log_path is not None:
        configure_federated_logging(
            level=str(log_level or "INFO").upper(),
            log_path=resolved_log_path,
            console=True,
        )

    output_paths_check = _verify_output_paths_isolated(
        checkpoint_dir=resolved_ckpt_dir,
        report_path=resolved_report_path,
        log_path=resolved_log_path,
        protect_pilot_outputs=protect_pilot_outputs,
    )

    # 3. Load and verify client datasets, file existence, and zero test leakage
    using_default_partition = partition_manifest is None and (
        fed_cfg.get("partitioning", {}).get("manifest_path")
        in {None, "data/partitions/combined_cross_dataset_2clients_seed42.json"}
    )
    try:
        bundles_map = build_all_federated_client_dataloaders(
            partition_manifest=partition_manifest,
            source_manifest=source_manifest,
            federated_config=fed_cfg,
            dataset_config=ds_cfg,
            batch_size=batch_size,
            val_batch_size=val_batch_size,
            num_workers=num_workers,
            seed=resolved_seed,
            image_size=image_size,
            fov_crop=fov_crop,
            cache_in_memory=False,
            verify_files_exist=verify_files_exist,
        )
    except Exception as exc:
        raise FederatedPreflightError(
            f"Preflight failed while loading federated client datasets: {exc}"
        ) from exc

    expected_num_clients = int(fed_cfg.get("number_of_clients", 2))
    if len(bundles_map) != expected_num_clients:
        raise FederatedPreflightError(
            f"Preflight failed: expected {expected_num_clients} federated clients, "
            f"but partition manifest produced {len(bundles_map)} ({list(bundles_map.keys())})."
        )

    # Collect held-out test records to double-check zero test-split overlap across all clients
    held_out_test_keys: set[tuple[str, str]] = set()
    if source_manifest is not None:
        src_recs = _extract_records_from_input(source_manifest)
        for r in src_recs:
            if str(r.get("split", "")).strip().lower() == "test":
                held_out_test_keys.add((str(r.get("dataset_name")), str(r.get("sample_id"))))
    elif using_default_partition:
        manifest_dir = resolve_project_path(
            ds_cfg.get("paths", {}).get("manifests", "data/processed/manifests")
        )
        default_combined = manifest_dir / "combined_manifest.json"
        if default_combined.is_file():
            src_recs = _extract_records_from_input(default_combined)
            for r in src_recs:
                if str(r.get("split", "")).strip().lower() == "test":
                    held_out_test_keys.add(
                        (str(r.get("dataset_name")), str(r.get("sample_id")))
                    )

    active_train_keys: set[tuple[str, str]] = set()
    active_val_keys: set[tuple[str, str]] = set()
    by_dataset_counts: dict[str, dict[str, int]] = {}

    for cid, bundle in sorted(bundles_map.items()):
        if bundle.num_train_samples <= 0:
            raise FederatedPreflightError(
                f"Preflight failed: client '{cid}' has 0 training samples."
            )
        if bundle.num_val_samples <= 0:
            raise FederatedPreflightError(
                f"Preflight failed: client '{cid}' has 0 validation samples."
            )
        try:
            verify_no_test_split_leakage(bundle.train_loader, stage_name=f"{cid}:train")
            verify_no_test_split_leakage(bundle.val_loader, stage_name=f"{cid}:validation")
            verify_exact_split(bundle.val_loader, expected_split="valid")
        except Exception as exc:
            raise FederatedPreflightError(
                f"Preflight failed split-hygiene check for client '{cid}': {exc}"
            ) from exc

        for rec in bundle.train_records:
            sp = str(rec.get("split", "")).strip().lower()
            if sp != "train":
                raise FederatedPreflightError(
                    f"Preflight failed: non-train record {rec.get('sample_id')!r} (split={sp!r}) "
                    f"in {cid} training set."
                )
            key = (str(rec["dataset_name"]), str(rec["sample_id"]))
            active_train_keys.add(key)
            ds_entry = by_dataset_counts.setdefault(
                str(rec["dataset_name"]), {"train": 0, "val": 0}
            )
            ds_entry["train"] += 1

        for rec in bundle.val_records:
            sp = str(rec.get("split", "")).strip().lower()
            if sp not in {"valid", "val"}:
                raise FederatedPreflightError(
                    f"Preflight failed: non-validation record {rec.get('sample_id')!r} (split={sp!r}) "
                    f"in {cid} validation set."
                )
            key = (str(rec["dataset_name"]), str(rec["sample_id"]))
            active_val_keys.add(key)
            ds_entry = by_dataset_counts.setdefault(
                str(rec["dataset_name"]), {"train": 0, "val": 0}
            )
            ds_entry["val"] += 1

    leaked_test = sorted((active_train_keys | active_val_keys) & held_out_test_keys)
    if leaked_test:
        raise FederatedPreflightError(
            f"Preflight failed: {len(leaked_test)} held-out test sample(s) leaked into "
            f"training/validation: {leaked_test[:5]}"
        )

    resolved_expected_counts: Mapping[str, Mapping[str, int]] | None = expected_client_counts
    if (
        resolved_expected_counts is None
        and using_default_partition
        and source_manifest is None
        and str(fed_cfg.get("val_split_mode", "official_and_carved")).strip().lower()
        == "official_and_carved"
        and abs(float(fed_cfg.get("val_carve_ratio", 0.2)) - 0.2) < 1e-6
    ):
        resolved_expected_counts = {
            "DDR": {"train": 383, "val": 149},
            "IDRiD": {"train": 44, "val": 10},
        }

    if resolved_expected_counts is not None:
        for ds_name, exp_map in resolved_expected_counts.items():
            actual_map = by_dataset_counts.get(ds_name)
            if actual_map is None:
                raise FederatedPreflightError(
                    f"Preflight failed: expected dataset '{ds_name}' in client bundles, "
                    f"found datasets {sorted(by_dataset_counts.keys())}."
                )
            exp_tr = int(exp_map["train"])
            exp_val = int(exp_map["val"])
            if actual_map["train"] != exp_tr or actual_map["val"] != exp_val:
                raise FederatedPreflightError(
                    f"Preflight failed: dataset '{ds_name}' sample counts mismatch: "
                    f"expected train={exp_tr}, val={exp_val}, got train={actual_map['train']}, "
                    f"val={actual_map['val']}."
                )

    # 4. Construct global model, strategy, and Flower server and verify client participation
    try:
        server, fed_strategy, model, proxies = build_federated_server(
            federated_config=fed_cfg,
            dataset_config=ds_cfg,
            model_config=mdl_cfg,
            project_config=proj_cfg,
            data_bundles=bundles_map,
            checkpoint_dir=resolved_ckpt_dir,
            save_checkpoints=True,
            accept_failures=False,
            include_buffers=include_buffers,
            integer_buffer_policy=integer_buffer_policy,
            local_epochs=local_epochs,
            batch_size=batch_size,
            val_batch_size=val_batch_size,
            num_workers=num_workers,
            learning_rate=learning_rate,
            mixed_precision=resolved_amp,
            seed=resolved_seed,
            device=resolved_device,
            image_size=image_size,
            fov_crop=fov_crop,
            cache_in_memory=False,
            verify_files_exist=False,
            num_rounds=resolved_rounds,
            max_workers=1,
        )
    except Exception as exc:
        raise FederatedPreflightError(
            f"Preflight failed during global model / Flower server initialization: {exc}"
        ) from exc

    num_registered = server._client_manager.num_available()
    if len(proxies) != expected_num_clients or num_registered != expected_num_clients:
        raise FederatedPreflightError(
            f"Preflight failed: expected {expected_num_clients} registered clients, "
            f"got proxies={len(proxies)}, available={num_registered}."
        )
    if (
        fed_strategy.min_fit_clients != expected_num_clients
        or fed_strategy.min_evaluate_clients != expected_num_clients
        or fed_strategy.min_available_clients != expected_num_clients
    ):
        raise FederatedPreflightError(
            f"Preflight failed: full client participation required ({expected_num_clients} clients), "
            f"but min_fit_clients={fed_strategy.min_fit_clients}, "
            f"min_evaluate_clients={fed_strategy.min_evaluate_clients}, "
            f"min_available_clients={fed_strategy.min_available_clients}."
        )

    # 5. Verify global model architecture, pretrained weights, and parameter finiteness
    out_ch = getattr(model, "output_channels", getattr(model, "num_classes", None))
    if out_ch is not None and int(out_ch) != len(CANONICAL_LESION_CHANNELS):
        raise FederatedPreflightError(
            f"Preflight failed: global model output_channels={out_ch}, "
            f"expected {len(CANONICAL_LESION_CHANNELS)} ({list(CANONICAL_LESION_CHANNELS)})."
        )

    for key, tensor in model.state_dict().items():
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all().item()):
            raise FederatedPreflightError(
                f"Preflight failed: global model contains non-finite values in '{key}'."
            )

    init_meta = dict(fed_strategy.initialization_metadata)
    init_mode = str(
        init_meta.get(
            "initialization_mode",
            init_meta.get("encoder_weights_source", "unknown"),
        )
    )
    pretrained_req = bool(
        fed_cfg.get("initialization", {}).get("pretrained", mdl_cfg.get("pretrained", True))
    )
    pretrained_loaded = bool(
        init_meta.get(
            "pretrained_weights_loaded",
            getattr(model, "pretrained_weights_loaded", False),
        )
    )
    if pretrained_req and init_mode != "random" and not pretrained_loaded:
        raise FederatedPreflightError(
            f"Preflight failed: pretrained global encoder weights were requested "
            f"(source={init_mode!r}), but pretrained_weights_loaded is False."
        )

    client_sample_counts = _extract_client_sample_counts(proxies)

    preflight_report: dict[str, Any] = {
        "status": "passed",
        "experiment_name": resolved_experiment_name,
        "framework": "flower",
        "flower_version": str(getattr(flwr, "__version__", "unknown")),
        "strategy": str(fed_cfg.get("strategy", "FedAvg")),
        "num_rounds": int(resolved_rounds),
        "local_epochs": int(fed_strategy.local_epochs),
        "batch_size": int(
            batch_size
            if batch_size is not None
            else fed_cfg.get("local_batch_size", ds_cfg.get("batch_size", 8))
        ),
        "image_size": list(
            image_size
            if image_size is not None
            else (
                int(ds_cfg.get("image_size", {}).get("height", 512)),
                int(ds_cfg.get("image_size", {}).get("width", 512)),
            )
        ),
        "seed": int(resolved_seed),
        "device": str(resolved_device),
        "cuda_available": cuda_available,
        "gpu_name": gpu_name,
        "gpu_vram_gb": gpu_vram_gb,
        "mixed_precision": resolved_amp,
        "amp_verified": amp_verified,
        "number_of_clients": int(len(proxies)),
        "both_clients_registered": bool(len(proxies) == expected_num_clients),
        "client_sample_counts": client_sample_counts,
        "dataset_sample_counts": by_dataset_counts,
        "held_out_test_samples_excluded": int(len(held_out_test_keys)),
        "zero_test_leakage": True,
        "files_verified": bool(verify_files_exist),
        "global_model": {
            "architecture": type(model).__name__,
            "output_channels": int(out_ch) if out_ch is not None else 4,
            "channel_order": list(CANONICAL_LESION_CHANNELS),
            "initialization_mode": init_mode,
            "pretrained_weights_loaded": pretrained_loaded,
            "weights_status": str(
                init_meta.get("weights_status", getattr(model, "weights_status", "unknown"))
            ),
        },
        "output_paths": output_paths_check,
    }

    logger.info("==================== FEDERATED PREFLIGHT PASSED ====================")
    logger.info(
        "[PREFLIGHT] Framework: Flower %s | Strategy: %s | Rounds: %d | Local Epochs: %d | Seed: %d",
        preflight_report["flower_version"],
        preflight_report["strategy"],
        preflight_report["num_rounds"],
        preflight_report["local_epochs"],
        preflight_report["seed"],
    )
    logger.info(
        "[PREFLIGHT] Device: %s | CUDA Available: %s | GPU: %s | Mixed Precision: %s (verified=%s)",
        preflight_report["device"],
        preflight_report["cuda_available"],
        f"{gpu_name} ({gpu_vram_gb} GB)" if gpu_name else "CPU",
        preflight_report["mixed_precision"],
        preflight_report["amp_verified"],
    )
    logger.info(
        "[PREFLIGHT] Global Model: %s | Init Source: %s | Pretrained Loaded: %s | Channels: %s",
        preflight_report["global_model"]["architecture"],
        preflight_report["global_model"]["initialization_mode"],
        preflight_report["global_model"]["pretrained_weights_loaded"],
        preflight_report["global_model"]["channel_order"],
    )
    logger.info(
        "[PREFLIGHT] Clients Registered: %d | Total Train: %d | Total Val: %d | Held-Out Test Excluded: %d | Files Verified: %s",
        preflight_report["number_of_clients"],
        client_sample_counts.get("total_train_samples", 0),
        client_sample_counts.get("total_val_samples", 0),
        preflight_report["held_out_test_samples_excluded"],
        preflight_report["files_verified"],
    )
    for ds_name, counts in sorted(by_dataset_counts.items()):
        logger.info(
            "[PREFLIGHT]   Dataset %s: train=%d samples, val=%d samples",
            ds_name,
            counts["train"],
            counts["val"],
        )
    logger.info(
        "[PREFLIGHT] Output Paths: checkpoints=%s | report=%s | log=%s (writable=%s, pilot_isolated=%s)",
        output_paths_check["checkpoint_dir"],
        output_paths_check["report_path"],
        output_paths_check["log_path"],
        output_paths_check["writable"],
        output_paths_check["pilot_isolated"],
    )
    logger.info("====================================================================")

    return preflight_report


def run_federated_training(
    federated_config: Mapping[str, Any] | str | Path = "federated.yaml",
    dataset_config: Mapping[str, Any] | str | Path = "dataset.yaml",
    model_config: Mapping[str, Any] | str | Path = "model.yaml",
    project_config: Mapping[str, Any] | str | Path = "project.yaml",
    *,
    experiment_config: Mapping[str, Any] | str | Path | None = None,
    experiment_name: str | None = None,
    num_rounds: int | None = None,
    clients: (
        Sequence[FedMorphisNumPyClient | NumPyClient | Client | ClientProxy]
        | Mapping[str, FedMorphisNumPyClient | NumPyClient | Client | ClientProxy]
        | None
    ) = None,
    data_bundles: Mapping[str, ClientDataBundle] | Sequence[ClientDataBundle] | None = None,
    partition_manifest: str | Path | Mapping[str, Any] | None = None,
    source_manifest: str | Path | Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    global_model: nn.Module | None = None,
    strategy: FedMorphisFedAvg | None = None,
    initial_parameters: Parameters | None = None,
    checkpoint_dir: str | Path | None = None,
    save_checkpoints: bool = True,
    report_path: str | Path | None = None,
    save_report: bool = True,
    log_path: str | Path | None = None,
    log_level: str = "INFO",
    accept_failures: bool = False,
    include_buffers: bool | None = None,
    integer_buffer_policy: str = "max",
    local_epochs: int | None = None,
    batch_size: int | None = None,
    val_batch_size: int | None = None,
    num_workers: int | None = None,
    learning_rate: float | None = None,
    mixed_precision: bool | None = None,
    seed: int | None = None,
    device: torch.device | str | None = None,
    image_size: tuple[int, int] | None = None,
    fov_crop: bool | None = None,
    cache_in_memory: bool = False,
    verify_files_exist: bool = False,
    max_train_batches: int | None = None,
    max_val_batches: int | None = None,
    max_workers: int = 1,
    timeout: float | None = None,
) -> FederatedServerRunResult:
    """Run Flower federated segmentation training and distributed validation across ``num_rounds``.

    Orchestrates:
    1. Global segmentation model initialization and parameter serialization.
    2. :class:`FedMorphisFedAvg` strategy construction.
    3. Registration of client proxies (`client_0` = DDR, `client_1` = IDRiD).
    4. Execution of :meth:`flwr.server.Server.fit` for ``num_rounds`` rounds.
    5. Synchronization of final global parameters into ``global_model``.
    6. Atomic checkpoint persistence (``latest_checkpoint.pt`` and ``best_checkpoint.pt``)
       and optional JSON report generation.
    """
    fed_cfg = (
        load_yaml_config(federated_config, validate=True)
        if isinstance(federated_config, (str, Path))
        else dict(federated_config)
    )
    ds_cfg = (
        load_yaml_config(dataset_config, validate=True)
        if isinstance(dataset_config, (str, Path))
        else dict(dataset_config)
    )
    mdl_cfg = (
        load_yaml_config(model_config, validate=True)
        if isinstance(model_config, (str, Path))
        else dict(model_config)
    )
    proj_cfg = (
        load_yaml_config(project_config, validate=True)
        if isinstance(project_config, (str, Path))
        else dict(project_config)
    )
    if experiment_config is not None:
        exp_cfg: dict[str, Any] = (
            load_yaml_config(experiment_config, validate=True)
            if isinstance(experiment_config, (str, Path))
            else dict(experiment_config)
        )
    else:
        try:
            exp_cfg = load_yaml_config("experiment.yaml", validate=False)
        except Exception:
            exp_cfg = {}

    resolved_experiment_name = str(
        experiment_name
        if experiment_name is not None
        else fed_cfg.get(
            "experiment_name",
            exp_cfg.get("experiment_name", "fedmorphis_federated_fedavg"),
        )
    ).strip() or "fedmorphis_federated_fedavg"

    resolved_rounds = int(
        num_rounds
        if num_rounds is not None
        else fed_cfg.get("rounds", fed_cfg.get("communication_rounds", 20))
    )
    if resolved_rounds < 1:
        raise ValueError(f"num_rounds must be >= 1, got {resolved_rounds}.")

    if device is None:
        resolved_device = resolve_device(str(proj_cfg.get("device", "auto")))
    elif isinstance(device, str):
        resolved_device = resolve_device(device)
    else:
        resolved_device = torch.device(device)

    resolved_report_path: Path | None = None
    if save_report:
        raw_rep = (
            report_path
            if report_path is not None
            else fed_cfg.get("logging", {}).get(
                "report_path", fed_cfg.get("report_path")
            )
        )
        if raw_rep is not None:
            resolved_report_path = resolve_project_path(raw_rep)

    raw_log_path = (
        log_path
        if log_path is not None
        else fed_cfg.get("logging", {}).get(
            "log_path", fed_cfg.get("log_path")
        )
    )
    resolved_log_level = str(
        log_level
        if log_level
        else fed_cfg.get("logging", {}).get(
            "log_level", fed_cfg.get("log_level", "INFO")
        )
    ).upper()
    resolved_log_path: Path | None = None
    if raw_log_path is not None:
        resolved_log_path = configure_federated_logging(
            level=resolved_log_level,
            log_path=resolve_project_path(raw_log_path),
            console=True,
        )

    server, fed_strategy, model, proxies = build_federated_server(
        federated_config=fed_cfg,
        dataset_config=ds_cfg,
        model_config=mdl_cfg,
        project_config=proj_cfg,
        clients=clients,
        data_bundles=data_bundles,
        partition_manifest=partition_manifest,
        source_manifest=source_manifest,
        global_model=global_model,
        strategy=strategy,
        initial_parameters=initial_parameters,
        checkpoint_dir=checkpoint_dir,
        save_checkpoints=save_checkpoints,
        accept_failures=accept_failures,
        include_buffers=include_buffers,
        integer_buffer_policy=integer_buffer_policy,
        local_epochs=local_epochs,
        batch_size=batch_size,
        val_batch_size=val_batch_size,
        num_workers=num_workers,
        learning_rate=learning_rate,
        mixed_precision=mixed_precision,
        seed=seed,
        device=resolved_device,
        image_size=image_size,
        fov_crop=fov_crop,
        cache_in_memory=cache_in_memory,
        verify_files_exist=verify_files_exist,
        max_train_batches=max_train_batches,
        max_val_batches=max_val_batches,
        num_rounds=resolved_rounds,
        max_workers=max_workers,
    )

    client_sample_counts = _extract_client_sample_counts(proxies)
    config_metadata = {
        "federated": dict(fed_cfg),
        "model": dict(mdl_cfg),
        "dataset": dict(ds_cfg),
        "project": dict(proj_cfg),
        "experiment": dict(exp_cfg),
    }

    gpu_info_str = "CPU"
    if resolved_device.type == "cuda" and torch.cuda.is_available():
        dev_idx = (
            resolved_device.index
            if resolved_device.index is not None
            else torch.cuda.current_device()
        )
        props = torch.cuda.get_device_properties(dev_idx)
        vram_gb = props.total_memory / (1024**3)
        gpu_info_str = f"{props.name} ({vram_gb:.2f} GB VRAM, CUDA {torch.version.cuda})"

    init_meta = dict(fed_strategy.initialization_metadata)
    init_source = str(
        init_meta.get(
            "initialization_mode",
            init_meta.get("encoder_weights_source", "unknown"),
        )
    )
    weights_status = str(
        init_meta.get("weights_status", getattr(model, "weights_status", "unknown"))
    )
    pretrained_loaded = bool(
        init_meta.get(
            "pretrained_weights_loaded",
            getattr(model, "pretrained_weights_loaded", False),
        )
    )

    logger.info("======================================================================")
    logger.info("[FEDERATED INIT] Experiment: %s", resolved_experiment_name)
    logger.info(
        "[FEDERATED INIT] Framework: Flower %s | Strategy: %s",
        getattr(flwr, "__version__", "unknown"),
        fed_cfg.get("strategy", "FedAvg"),
    )
    logger.info(
        "[FEDERATED INIT] Device: %s | Hardware: %s",
        resolved_device,
        gpu_info_str,
    )
    logger.info(
        "[FEDERATED INIT] Global Model: %s | Init Source: %s | Pretrained Loaded: %s | Status: %s",
        type(model).__name__,
        init_source,
        pretrained_loaded,
        weights_status,
    )
    logger.info(
        "[FEDERATED INIT] Clients: %d | Total Train Samples: %d | Total Val Samples: %d",
        len(proxies),
        client_sample_counts.get("total_train_samples", 0),
        client_sample_counts.get("total_val_samples", 0),
    )
    for idx, (cid, c_info) in enumerate(client_sample_counts.get("clients", {}).items()):
        logger.info(
            "[FEDERATED INIT]   Client %d [%s] (id=%s): train=%d samples, val=%d samples (val_mode=%s)",
            idx,
            c_info.get("dataset", "unknown"),
            cid,
            int(c_info.get("num_train_samples", 0)),
            int(c_info.get("num_val_samples", 0)),
            c_info.get("val_split_mode", "official"),
        )
    logger.info(
        "[FEDERATED INIT] Training Config: rounds=%d | local_epochs=%d | batch_size=%s | val_batch_size=%s | lr=%.6g | share_bn_buffers=%s | mixed_precision=%s | seed=%d",
        resolved_rounds,
        fed_strategy.local_epochs,
        batch_size
        if batch_size is not None
        else fed_cfg.get(
            "local_batch_size",
            ds_cfg.get("dataloader", {}).get("batch_size", 4),
        ),
        val_batch_size
        if val_batch_size is not None
        else fed_cfg.get(
            "local_val_batch_size",
            ds_cfg.get("dataloader", {}).get("val_batch_size", 4),
        ),
        fed_strategy.learning_rate,
        fed_strategy.include_buffers,
        mixed_precision
        if mixed_precision is not None
        else mdl_cfg.get("training", {}).get("mixed_precision", False),
        fed_strategy.seed,
    )
    if resolved_log_path is not None:
        logger.info(
            "[FEDERATED INIT] Persistent Log File: %s",
            format_project_path(resolved_log_path),
        )
    logger.info("======================================================================")

    server_cfg = ServerConfig(num_rounds=resolved_rounds, round_timeout=timeout)
    set_seed(fed_strategy.seed, deterministic=bool(proj_cfg.get("deterministic", True)))
    try:
        history, elapsed_sec = server.fit(
            num_rounds=server_cfg.num_rounds,
            timeout=server_cfg.round_timeout,
        )
        if not accept_failures and len(fed_strategy.round_history) < resolved_rounds:
            raise FederatedStrategyError(
                f"Federated training terminated with only {len(fed_strategy.round_history)} "
                f"of {resolved_rounds} completed round(s)."
            )
    except Exception as exc:
        logger.exception(
            "[FEDERATED ERROR] Federated training failed for experiment '%s'",
            resolved_experiment_name,
        )
        if resolved_report_path is not None:
            failed_payload = {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "experiment_name": resolved_experiment_name,
                "framework": "flower",
                "strategy": str(fed_cfg.get("strategy", "FedAvg")),
                "number_of_clients": int(len(proxies)),
                "num_rounds": int(resolved_rounds),
                "rounds_completed": int(len(fed_strategy.round_history)),
                "local_epochs": int(fed_strategy.local_epochs),
                "seed": int(fed_strategy.seed),
                "device": str(resolved_device),
                "client_sample_counts": client_sample_counts,
                "global_initialization_source": str(
                    fed_strategy.initialization_metadata.get(
                        "initialization_mode",
                        fed_strategy.initialization_metadata.get(
                            "encoder_weights_source", "unknown"
                        ),
                    )
                ),
                "initialization": dict(fed_strategy.initialization_metadata),
                "log_path": (
                    format_project_path(resolved_log_path)
                    if resolved_log_path is not None
                    else None
                ),
                "round_history": list(fed_strategy.round_history),
                "config": config_metadata,
            }
            _atomic_write_json(resolved_report_path, failed_payload)
        raise

    best_val = (
        float(fed_strategy.best_metric_value)
        if fed_strategy.best_round > 0
        else None
    )

    result = FederatedServerRunResult(
        num_rounds=resolved_rounds,
        elapsed_sec=float(elapsed_sec),
        history=history,
        round_history=list(fed_strategy.round_history),
        global_model=model,
        strategy=fed_strategy,
        server=server,
        best_round=int(fed_strategy.best_round),
        best_metric_value=best_val,
        monitor_metric=fed_strategy.monitor_metric,
        monitor_mode=fed_strategy.monitor_mode,
        latest_checkpoint_path=fed_strategy.latest_checkpoint_path,
        best_checkpoint_path=fed_strategy.best_checkpoint_path,
        report_path=resolved_report_path,
        log_path=resolved_log_path,
        initialization_metadata=dict(fed_strategy.initialization_metadata),
        experiment_name=resolved_experiment_name,
        status="completed",
        number_of_clients=int(len(proxies)),
        local_epochs=int(fed_strategy.local_epochs),
        seed=int(fed_strategy.seed),
        device=str(resolved_device),
        client_sample_counts=client_sample_counts,
        config_metadata=config_metadata,
    )

    if resolved_report_path is not None:
        _atomic_write_json(resolved_report_path, result.to_report_dict())

    logger.info("==================== FEDERATED TRAINING SUMMARY ====================")
    logger.info(
        "[SUMMARY] Experiment: %s | Status: completed | Rounds Completed: %d / %d | Total Duration: %.2fs",
        resolved_experiment_name,
        len(fed_strategy.round_history),
        resolved_rounds,
        float(elapsed_sec),
    )
    logger.info(
        "[SUMMARY] Best Round: %d | Best %s: %s",
        fed_strategy.best_round,
        fed_strategy.monitor_metric,
        f"{best_val:.4f}" if best_val is not None else "N/A",
    )
    logger.info(
        "[SUMMARY] Best Checkpoint: %s",
        format_project_path(fed_strategy.best_checkpoint_path)
        if fed_strategy.best_checkpoint_path is not None
        else "None",
    )
    logger.info(
        "[SUMMARY] Latest Checkpoint: %s",
        format_project_path(fed_strategy.latest_checkpoint_path)
        if fed_strategy.latest_checkpoint_path is not None
        else "None",
    )
    logger.info(
        "[SUMMARY] Training Report: %s",
        format_project_path(resolved_report_path)
        if resolved_report_path is not None
        else "None",
    )
    if resolved_log_path is not None:
        logger.info(
            "[SUMMARY] Persistent Log File: %s",
            format_project_path(resolved_log_path),
        )
    logger.info("====================================================================")

    return result


# Convenience alias
run_federated_rounds = run_federated_training


__all__ = [
    "FederatedPreflightError",
    "FederatedServerRunResult",
    "InProcessClientProxy",
    "build_federated_server",
    "initialize_global_model",
    "load_global_checkpoint",
    "register_clients_with_manager",
    "run_federated_preflight",
    "run_federated_rounds",
    "run_federated_training",
    "save_global_checkpoint",
]

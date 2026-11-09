"""Federated client dataset and DataLoader construction for DDR and IDRiD.

Responsibilities & Guarantees:
- Loads a configured partition manifest (defaulting to the 2-client cross-dataset
  manifest ``data/partitions/combined_cross_dataset_2clients_seed42.json``).
- Resolves client identifiers by integer index (``0``, ``1``), canonical client ID
  (``"client_0"``, ``"client_1"``), or unambiguous dataset name (``"DDR"``, ``"IDRiD"``).
- Isolates each client's local training records and pairs them with the appropriate
  validation split:
  - **Client 0 (DDR)**: Retains all ``383`` official DDR training records and uses the
    ``149`` official DDR validation records.
  - **Client 1 (IDRiD)**: Because IDRiD provides only ``54`` official training records
    and ``27`` held-out test records (with ``0`` official validation records),
    deterministically carves ``20%`` (``10`` records) from the ``54`` IDRiD training
    records under ``seed=42`` for local validation, leaving ``44`` local training
    records while keeping the ``27`` held-out test records completely untouched.
- Reuses :class:`src.data.dataset.RetinalLesionSegmentationDataset`,
  :func:`src.data.dataset.build_segmentation_dataloader`, and
  :func:`src.data.transforms.build_segmentation_transforms` without duplicating
  preprocessing, FOV cropping, or mask canonicalization logic.
- Enforces strict split hygiene: training augmentations are applied only to training
  data, validation transforms are deterministic, and zero train/validation overlap,
  cross-client duplication, or test-split leakage is permitted.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import (
    DatasetSampleError,
    RetinalLesionSegmentationDataset,
    _extract_records_from_input,
    _validate_manifest_record,
    build_segmentation_dataloader,
)
from src.data.masks import CANONICAL_LESION_CHANNELS
from src.data.partitioning import load_partition_manifest
from src.data.transforms import build_segmentation_transforms
from src.engine.evaluator import verify_exact_split
from src.engine.trainer import verify_no_test_split_leakage
from src.utils.config import SUPPORTED_VAL_SPLIT_MODES, load_yaml_config
from src.utils.paths import resolve_project_path
from src.utils.seed import MAX_UINT32_SEED


class FederatedDataError(ValueError):
    """Raised when federated client data resolution, carving, or validation fails."""


@dataclass(frozen=True)
class ClientDataBundle:
    """Container holding a federated client's datasets, DataLoaders, and split metadata."""

    client_id: str
    client_index: int
    dataset_names: tuple[str, ...]
    train_records: list[dict[str, Any]]
    val_records: list[dict[str, Any]]
    train_dataset: RetinalLesionSegmentationDataset
    val_dataset: RetinalLesionSegmentationDataset
    train_loader: DataLoader[Any]
    val_loader: DataLoader[Any]
    val_split_mode: str
    val_source: str
    seed: int
    batch_size: int
    val_batch_size: int
    num_workers: int
    channel_order: tuple[str, ...] = CANONICAL_LESION_CHANNELS

    @property
    def num_train_samples(self) -> int:
        """Return the number of local training samples for this client."""
        return len(self.train_dataset)

    @property
    def num_val_samples(self) -> int:
        """Return the number of local validation samples for this client."""
        return len(self.val_dataset)

    def summary(self) -> dict[str, Any]:
        """Return a JSON-serializable summary of the client's data bundle."""
        train_ds_counts: dict[str, int] = {}
        for rec in self.train_records:
            ds_name = str(rec["dataset_name"])
            train_ds_counts[ds_name] = train_ds_counts.get(ds_name, 0) + 1

        val_ds_counts: dict[str, int] = {}
        for rec in self.val_records:
            ds_name = str(rec["dataset_name"])
            val_ds_counts[ds_name] = val_ds_counts.get(ds_name, 0) + 1

        return {
            "client_id": self.client_id,
            "client_index": self.client_index,
            "dataset_names": list(self.dataset_names),
            "num_train_samples": self.num_train_samples,
            "num_val_samples": self.num_val_samples,
            "train_dataset_counts": dict(sorted(train_ds_counts.items())),
            "val_dataset_counts": dict(sorted(val_ds_counts.items())),
            "val_split_mode": self.val_split_mode,
            "val_source": self.val_source,
            "seed": self.seed,
            "batch_size": self.batch_size,
            "val_batch_size": self.val_batch_size,
            "num_workers": self.num_workers,
            "channel_order": list(self.channel_order),
        }


def resolve_client_id(
    client_id: int | str,
    clients_mapping: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[str, int]:
    """Resolve a client identifier to ``(canonical_client_id, client_index)``.

    Supports:
    - Integer client index (e.g., ``0`` -> ``("client_0", 0)``).
    - Numeric string (e.g., ``"0"`` -> ``("client_0", 0)``).
    - Canonical key (e.g., ``"client_0"`` -> ``("client_0", 0)``).
    - Unambiguous dataset name (e.g., ``"DDR"`` -> ``("client_0", 0)``,
      ``"IDRiD"`` -> ``("client_1", 1)`` when exactly one client holds that dataset).

    Raises:
        FederatedDataError: If ``client_id`` is invalid, out of range, or not found.
    """
    if not isinstance(clients_mapping, Mapping) or len(clients_mapping) == 0:
        raise FederatedDataError(
            "Partition manifest contains no client entries in 'clients'."
        )

    ordered_keys = list(clients_mapping.keys())

    if isinstance(client_id, bool) or client_id is None:
        raise FederatedDataError(
            f"Invalid client_id {client_id!r}; expected an integer index or client string."
        )

    if isinstance(client_id, int):
        if client_id < 0:
            raise FederatedDataError(
                f"Invalid negative client_id {client_id}; must be >= 0."
            )
        candidate_key = f"client_{client_id}"
        if candidate_key in clients_mapping:
            idx = ordered_keys.index(candidate_key)
            return candidate_key, idx
        if client_id < len(ordered_keys):
            return ordered_keys[client_id], client_id
        raise FederatedDataError(
            f"Client index {client_id} out of range for available clients {ordered_keys}."
        )

    if isinstance(client_id, str):
        raw = client_id.strip()
        if not raw:
            raise FederatedDataError("client_id cannot be an empty string.")

        if raw in clients_mapping:
            return raw, ordered_keys.index(raw)

        lowered = raw.lower()
        for idx, k in enumerate(ordered_keys):
            if k.lower() == lowered:
                return k, idx

        if raw.isdigit() or (raw.startswith("-") and raw[1:].isdigit()):
            int_id = int(raw)
            return resolve_client_id(int_id, clients_mapping)

        # Check unambiguous dataset name match (e.g. "DDR" or "IDRiD")
        matching_clients: list[tuple[str, int]] = []
        for idx, (cid, recs) in enumerate(clients_mapping.items()):
            if not isinstance(recs, Sequence):
                continue
            ds_names = {
                str(r.get("dataset_name", "")).strip().lower()
                for r in recs
                if isinstance(r, Mapping)
            }
            if lowered in ds_names and len(ds_names) == 1:
                matching_clients.append((cid, idx))

        if len(matching_clients) == 1:
            return matching_clients[0]
        if len(matching_clients) > 1:
            matched_ids = [m[0] for m in matching_clients]
            raise FederatedDataError(
                f"Dataset alias {client_id!r} is ambiguous across multiple clients: {matched_ids}."
            )

        raise FederatedDataError(
            f"Unknown client_id {client_id!r}. Available clients: {ordered_keys}."
        )

    raise FederatedDataError(
        f"Unsupported client_id type {type(client_id).__name__} ({client_id!r})."
    )


def verify_client_record_files(
    records: Sequence[Mapping[str, Any]],
    client_id: str | None = None,
) -> None:
    """Verify that every record's image file and present lesion mask files exist on disk.

    Raises:
        FederatedDataError: If a record is malformed.
        FileNotFoundError: If any referenced image or non-null lesion mask file is missing.
    """
    prefix = f"[{client_id}] " if client_id else ""
    for idx, rec in enumerate(records):
        try:
            _validate_manifest_record(rec, idx)
        except DatasetSampleError as exc:
            raise FederatedDataError(f"{prefix}{exc}") from exc

        sample_id = str(rec["sample_id"])
        img_path = resolve_project_path(rec["image_path"])
        if not img_path.is_file():
            raise FileNotFoundError(
                f"{prefix}Missing image file for sample '{sample_id}': {img_path}"
            )

        mask_paths = rec["mask_paths"]
        for cls_name in CANONICAL_LESION_CHANNELS:
            m_val = mask_paths.get(cls_name)
            if m_val is not None:
                m_path = resolve_project_path(m_val)
                if not m_path.is_file():
                    raise FileNotFoundError(
                        f"{prefix}Missing '{cls_name}' mask file for sample "
                        f"'{sample_id}': {m_path}"
                    )


def carve_client_validation_records(
    train_records: Sequence[Mapping[str, Any]],
    val_carve_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Deterministically carve a validation subset from a client's training records.

    For IDRiD's ``54`` official training records with ``val_carve_ratio=0.2`` and
    ``seed=42``, this carves ``10`` records for validation (re-labeled with
    ``split="valid"``) and leaves ``44`` records for local training (``split="train"``),
    without ever touching held-out ``test`` records.

    Args:
        train_records: Client training records (all must have ``split == "train"``).
        val_carve_ratio: Fraction in ``(0.0, 1.0)`` of training records to carve out
            for validation (default ``0.2``).
        seed: Random seed in ``[0, 2**32 - 1]`` for deterministic selection.

    Returns:
        Tuple ``(remaining_train_records, carved_val_records)``, both canonically
        sorted by ``(dataset_name, sample_id)``.
    """
    if (
        isinstance(val_carve_ratio, bool)
        or not isinstance(val_carve_ratio, (int, float))
        or not math.isfinite(float(val_carve_ratio))
        or float(val_carve_ratio) <= 0.0
        or float(val_carve_ratio) >= 1.0
    ):
        raise FederatedDataError(
            f"val_carve_ratio must be a finite number in (0.0, 1.0), got {val_carve_ratio!r}."
        )

    if isinstance(seed, bool) or not isinstance(seed, int) or not (0 <= seed <= MAX_UINT32_SEED):
        raise FederatedDataError(
            f"seed must be an integer in [0, {MAX_UINT32_SEED}], got {seed!r}."
        )

    if len(train_records) < 2:
        raise FederatedDataError(
            f"Cannot carve a non-empty validation split from {len(train_records)} training "
            "record(s); at least 2 training records are required."
        )

    validated_pool: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str]] = set()
    for idx, rec in enumerate(train_records):
        try:
            _validate_manifest_record(rec, idx)
        except DatasetSampleError as exc:
            raise FederatedDataError(str(exc)) from exc

        sp = str(rec.get("split", "")).strip().lower()
        if sp != "train":
            raise FederatedDataError(
                f"Cannot carve validation from non-train record "
                f"'{rec.get('dataset_name')}:{rec.get('sample_id')}' with split={sp!r}."
            )
        key = (str(rec["dataset_name"]), str(rec["sample_id"]))
        if key in seen_keys:
            raise FederatedDataError(
                f"Duplicate training record detected before validation carve: {key}."
            )
        seen_keys.add(key)
        validated_pool.append(dict(rec))

    # Canonical sort so input ordering never alters the carved subset
    validated_pool.sort(key=lambda r: (str(r["dataset_name"]), str(r["sample_id"])))

    total = len(validated_pool)
    val_count = int(math.floor(total * float(val_carve_ratio) + 1e-9))
    val_count = max(1, min(total - 1, val_count))

    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(total).tolist()

    val_index_set = set(perm[:val_count])
    remaining_train: list[dict[str, Any]] = []
    carved_val: list[dict[str, Any]] = []

    for idx, rec in enumerate(validated_pool):
        if idx in val_index_set:
            val_rec = dict(rec)
            val_rec["split"] = "valid"
            val_rec["original_split"] = "train"
            val_rec["val_source"] = "carved_from_train"
            carved_val.append(val_rec)
        else:
            train_rec = dict(rec)
            train_rec["split"] = "train"
            remaining_train.append(train_rec)

    remaining_train.sort(key=lambda r: (str(r["dataset_name"]), str(r["sample_id"])))
    carved_val.sort(key=lambda r: (str(r["dataset_name"]), str(r["sample_id"])))

    return remaining_train, carved_val


def verify_client_split_integrity(
    client_id: str,
    train_records: Sequence[Mapping[str, Any]],
    val_records: Sequence[Mapping[str, Any]],
    held_out_test_records: Sequence[Mapping[str, Any]] = (),
    other_client_train_records: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Verify training/validation disjointness, cross-client disjointness, and zero test leakage."""
    if not train_records:
        raise FederatedDataError(
            f"Client '{client_id}' has an empty local training split (0 records)."
        )
    if not val_records:
        raise FederatedDataError(
            f"Client '{client_id}' has an empty local validation split (0 records)."
        )

    train_keys: set[tuple[str, str]] = set()
    for idx, rec in enumerate(train_records):
        try:
            _validate_manifest_record(rec, idx)
        except DatasetSampleError as exc:
            raise FederatedDataError(f"[{client_id}:train] {exc}") from exc
        sp = str(rec.get("split", "")).strip().lower()
        if sp != "train":
            raise FederatedDataError(
                f"[{client_id}:train] Forbidden split {sp!r} on sample '{rec.get('sample_id')}'."
            )
        key = (str(rec["dataset_name"]), str(rec["sample_id"]))
        if key in train_keys:
            raise FederatedDataError(
                f"[{client_id}:train] Duplicate sample in training split: {key}."
            )
        train_keys.add(key)

    val_keys: set[tuple[str, str]] = set()
    for idx, rec in enumerate(val_records):
        try:
            _validate_manifest_record(rec, idx)
        except DatasetSampleError as exc:
            raise FederatedDataError(f"[{client_id}:valid] {exc}") from exc
        sp = str(rec.get("split", "")).strip().lower()
        if sp not in {"valid", "val"}:
            raise FederatedDataError(
                f"[{client_id}:valid] Forbidden split {sp!r} on sample '{rec.get('sample_id')}'; "
                "expected 'valid'."
            )
        key = (str(rec["dataset_name"]), str(rec["sample_id"]))
        if key in val_keys:
            raise FederatedDataError(
                f"[{client_id}:valid] Duplicate sample in validation split: {key}."
            )
        val_keys.add(key)

    overlap_train_val = sorted(train_keys & val_keys)
    if overlap_train_val:
        raise FederatedDataError(
            f"[{client_id}] Training and validation splits overlap on {len(overlap_train_val)} "
            f"sample(s): {overlap_train_val[:5]}."
        )

    test_keys = {
        (str(r["dataset_name"]), str(r["sample_id"])) for r in held_out_test_records
    }
    train_test_leak = sorted(train_keys & test_keys)
    val_test_leak = sorted(val_keys & test_keys)
    if train_test_leak or val_test_leak:
        raise FederatedDataError(
            f"[{client_id}] Test split leakage detected: train_test_leak={len(train_test_leak)}, "
            f"val_test_leak={len(val_test_leak)}."
        )

    cross_client_duplicates: list[tuple[str, tuple[str, str]]] = []
    if other_client_train_records is not None:
        for other_cid, other_recs in other_client_train_records.items():
            if other_cid == client_id:
                continue
            other_keys = {
                (str(r["dataset_name"]), str(r["sample_id"])) for r in other_recs
            }
            shared = sorted(train_keys & other_keys)
            for k in shared:
                cross_client_duplicates.append((other_cid, k))

    if cross_client_duplicates:
        raise FederatedDataError(
            f"[{client_id}] Cross-client training record duplication detected: "
            f"{cross_client_duplicates[:5]}."
        )

    return {
        "client_id": client_id,
        "num_train": len(train_keys),
        "num_val": len(val_keys),
        "train_val_disjoint": True,
        "zero_test_leakage": True,
        "cross_client_disjoint": True,
    }


def _load_and_validate_partition_dict(
    partition_manifest: str | Path | Mapping[str, Any],
) -> dict[str, Any]:
    """Load and validate a partition manifest from path or mapping."""
    if isinstance(partition_manifest, (str, Path)):
        part_dict = load_partition_manifest(partition_manifest)
    elif isinstance(partition_manifest, Mapping):
        part_dict = dict(partition_manifest)
    else:
        raise FederatedDataError(
            f"Unsupported partition_manifest type {type(partition_manifest).__name__}."
        )

    if "clients" not in part_dict or not isinstance(part_dict["clients"], Mapping):
        raise FederatedDataError(
            "Partition manifest is missing a valid 'clients' mapping."
        )

    clients_map = part_dict["clients"]
    if len(clients_map) == 0:
        raise FederatedDataError("Partition manifest 'clients' mapping is empty.")

    # Validate every client's record list and check cross-client disjointness
    seen_global: dict[tuple[str, str], str] = {}
    for cid, recs in clients_map.items():
        if not isinstance(recs, Sequence) or isinstance(recs, (str, bytes)):
            raise FederatedDataError(
                f"Partition manifest entry for client '{cid}' must be a sequence of records."
            )
        if len(recs) == 0:
            raise FederatedDataError(
                f"Partition manifest entry for client '{cid}' is empty (0 records)."
            )
        for idx, rec in enumerate(recs):
            if not isinstance(rec, Mapping):
                raise FederatedDataError(
                    f"Record {idx} in client '{cid}' must be a mapping."
                )
            try:
                _validate_manifest_record(rec, idx)
            except DatasetSampleError as exc:
                raise FederatedDataError(f"[{cid}] {exc}") from exc

            sp = str(rec.get("split", "")).strip().lower()
            if sp != "train":
                raise FederatedDataError(
                    f"Partition manifest client '{cid}' contains non-train record "
                    f"'{rec.get('dataset_name')}:{rec.get('sample_id')}' (split={sp!r})."
                )
            key = (str(rec["dataset_name"]), str(rec["sample_id"]))
            if key in seen_global:
                prev_cid = seen_global[key]
                raise FederatedDataError(
                    f"Duplicate training sample {key} assigned to both '{prev_cid}' and '{cid}'."
                )
            seen_global[key] = str(cid)

    return part_dict


def resolve_client_train_val_records(
    client_id: int | str,
    partition_manifest: str | Path | Mapping[str, Any] | None = None,
    source_manifest: str | Path | Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    validation_records: Sequence[Mapping[str, Any]] | None = None,
    federated_config: Mapping[str, Any] | str | Path = "federated.yaml",
    dataset_config: Mapping[str, Any] | str | Path = "dataset.yaml",
    val_split_mode: str | None = None,
    val_carve_ratio: float | None = None,
    seed: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Resolve a client's ``(train_records, val_records, info)`` with split-hygiene verification."""
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

    part_cfg = fed_cfg.get("partitioning", {})
    resolved_seed = int(
        seed
        if seed is not None
        else fed_cfg.get("seed", part_cfg.get("seed", 42))
    )
    resolved_mode = str(
        val_split_mode
        if val_split_mode is not None
        else fed_cfg.get("val_split_mode", "official_and_carved")
    ).strip().lower()
    if resolved_mode not in SUPPORTED_VAL_SPLIT_MODES:
        raise FederatedDataError(
            f"Unsupported val_split_mode {resolved_mode!r}; expected one of "
            f"{SUPPORTED_VAL_SPLIT_MODES}."
        )

    resolved_carve_ratio = float(
        val_carve_ratio
        if val_carve_ratio is not None
        else fed_cfg.get("val_carve_ratio", 0.2)
    )

    using_default_partition = partition_manifest is None
    if partition_manifest is None:
        default_part_rel = part_cfg.get("manifest_path")
        if not default_part_rel:
            out_dir = part_cfg.get("output_directory", "data/partitions")
            scheme = str(part_cfg.get("scheme", "cross_dataset")).strip().lower()
            if scheme == "by_dataset":
                scheme = "cross_dataset"
            elif scheme == "non_iid":
                scheme = "dirichlet"
            n_clients = int(fed_cfg.get("number_of_clients", 2))
            if scheme == "dirichlet":
                alpha = float(part_cfg.get("dirichlet_alpha", 0.5))
                default_part_rel = (
                    f"{out_dir}/combined_dirichlet_a{alpha}_{n_clients}clients_seed{resolved_seed}.json"
                )
            else:
                default_part_rel = (
                    f"{out_dir}/combined_{scheme}_{n_clients}clients_seed{resolved_seed}.json"
                )
        partition_manifest = resolve_project_path(default_part_rel)

    part_dict = _load_and_validate_partition_dict(partition_manifest)
    clients_map: Mapping[str, Sequence[Mapping[str, Any]]] = part_dict["clients"]
    canonical_cid, client_idx = resolve_client_id(client_id, clients_map)
    raw_client_train = [dict(r) for r in clients_map[canonical_cid]]

    client_datasets = tuple(
        sorted({str(r["dataset_name"]) for r in raw_client_train})
    )

    # Resolve source manifest records (for official validation & held-out test records)
    all_source_records: list[dict[str, Any]] = []
    if source_manifest is not None:
        try:
            all_source_records = _extract_records_from_input(source_manifest)
        except DatasetSampleError as exc:
            raise FederatedDataError(str(exc)) from exc
    elif validation_records is None:
        # Load default combined_manifest.json if it exists and matches the partition's datasets
        manifest_dir = resolve_project_path(
            ds_cfg.get("paths", {}).get("manifests", "data/processed/manifests")
        )
        default_combined = manifest_dir / "combined_manifest.json"
        if default_combined.is_file():
            candidate_records = _extract_records_from_input(default_combined)
            if using_default_partition:
                all_source_records = candidate_records
            else:
                # Only attach default combined_manifest records when sample IDs belong to it
                candidate_train_keys = {
                    (str(r.get("dataset_name")), str(r.get("sample_id")))
                    for r in candidate_records
                    if str(r.get("split", "")).strip().lower() == "train"
                }
                client_train_keys = {
                    (str(r.get("dataset_name")), str(r.get("sample_id")))
                    for r in raw_client_train
                }
                if client_train_keys and client_train_keys.issubset(candidate_train_keys):
                    all_source_records = candidate_records

    for idx, rec in enumerate(all_source_records):
        try:
            _validate_manifest_record(rec, idx)
        except DatasetSampleError as exc:
            raise FederatedDataError(str(exc)) from exc

    held_out_test_records = [
        dict(r)
        for r in all_source_records
        if str(r.get("split", "")).strip().lower() == "test"
    ]
    official_valid_all = [
        dict(r)
        for r in all_source_records
        if str(r.get("split", "")).strip().lower() in {"valid", "val"}
    ]
    for r in official_valid_all:
        r["split"] = "valid"

    if validation_records is not None:
        explicit_val: list[dict[str, Any]] = []
        for idx, rec in enumerate(validation_records):
            try:
                _validate_manifest_record(rec, idx)
            except DatasetSampleError as exc:
                raise FederatedDataError(str(exc)) from exc
            sp = str(rec.get("split", "")).strip().lower()
            if sp == "test":
                raise FederatedDataError(
                    f"Forbidden 'test'-split record '{rec.get('sample_id')}' passed in validation_records."
                )
            v_rec = dict(rec)
            v_rec["split"] = "valid"
            explicit_val.append(v_rec)
        final_train = sorted(
            raw_client_train,
            key=lambda r: (str(r["dataset_name"]), str(r["sample_id"])),
        )
        final_val = sorted(
            explicit_val,
            key=lambda r: (str(r["dataset_name"]), str(r["sample_id"])),
        )
        val_source = "explicit"
    elif resolved_mode == "full_train_shared_val":
        if not official_valid_all:
            raise FederatedDataError(
                f"No official validation records available for val_split_mode='full_train_shared_val' "
                f"(client '{canonical_cid}')."
            )
        final_train = sorted(
            raw_client_train,
            key=lambda r: (str(r["dataset_name"]), str(r["sample_id"])),
        )
        final_val = sorted(
            official_valid_all,
            key=lambda r: (str(r["dataset_name"]), str(r["sample_id"])),
        )
        val_source = "official_shared"
    else:
        # "official_and_carved": per-dataset handling within the client
        by_ds_train: dict[str, list[dict[str, Any]]] = {}
        for rec in raw_client_train:
            by_ds_train.setdefault(str(rec["dataset_name"]), []).append(rec)

        by_ds_official_val: dict[str, list[dict[str, Any]]] = {}
        for rec in official_valid_all:
            by_ds_official_val.setdefault(str(rec["dataset_name"]), []).append(rec)

        final_train = []
        final_val = []
        val_sources_used: list[str] = []

        for ds_name in sorted(by_ds_train.keys()):
            ds_train_pool = by_ds_train[ds_name]
            ds_official_val = by_ds_official_val.get(ds_name, [])
            if ds_official_val:
                final_train.extend(ds_train_pool)
                final_val.extend(ds_official_val)
                val_sources_used.append("official")
            else:
                rem_train, carved_val = carve_client_validation_records(
                    ds_train_pool,
                    val_carve_ratio=resolved_carve_ratio,
                    seed=resolved_seed,
                )
                final_train.extend(rem_train)
                final_val.extend(carved_val)
                val_sources_used.append("carved_from_train")

        final_train.sort(key=lambda r: (str(r["dataset_name"]), str(r["sample_id"])))
        final_val.sort(key=lambda r: (str(r["dataset_name"]), str(r["sample_id"])))
        unique_sources = sorted(set(val_sources_used))
        val_source = unique_sources[0] if len(unique_sources) == 1 else "+".join(unique_sources)

    integrity = verify_client_split_integrity(
        client_id=canonical_cid,
        train_records=final_train,
        val_records=final_val,
        held_out_test_records=held_out_test_records,
        other_client_train_records=clients_map,
    )

    info = {
        "client_id": canonical_cid,
        "client_index": client_idx,
        "dataset_names": client_datasets,
        "val_split_mode": resolved_mode,
        "val_source": val_source,
        "val_carve_ratio": resolved_carve_ratio,
        "seed": resolved_seed,
        "integrity": integrity,
    }
    return final_train, final_val, info


def build_federated_client_dataloaders(
    client_id: int | str,
    partition_manifest: str | Path | Mapping[str, Any] | None = None,
    source_manifest: str | Path | Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    validation_records: Sequence[Mapping[str, Any]] | None = None,
    federated_config: Mapping[str, Any] | str | Path = "federated.yaml",
    dataset_config: Mapping[str, Any] | str | Path = "dataset.yaml",
    val_split_mode: str | None = None,
    val_carve_ratio: float | None = None,
    batch_size: int | None = None,
    val_batch_size: int | None = None,
    num_workers: int | None = None,
    pin_memory: bool | None = None,
    persistent_workers: bool | None = None,
    drop_last_train: bool | None = None,
    seed: int | None = None,
    image_size: tuple[int, int] | None = None,
    fov_crop: bool | None = None,
    return_metadata: bool = False,
    mask_dtype: torch.dtype = torch.float32,
    cache_in_memory: bool = False,
    verify_files_exist: bool = False,
) -> ClientDataBundle:
    """Construct local training and validation datasets and DataLoaders for a federated client.

    Args:
        client_id: Client identifier (``0``, ``1``, ``"client_0"``, ``"client_1"``,
            ``"DDR"``, or ``"IDRiD"``).
        partition_manifest: Optional path or dict for the partition manifest. Defaults
            to ``federated.yaml:partitioning.manifest_path``.
        source_manifest: Optional path, dict, or record list containing the full dataset
            manifest (used to obtain official ``valid`` and held-out ``test`` splits).
        validation_records: Optional explicit validation records override.
        federated_config: Path or mapping for ``federated.yaml``.
        dataset_config: Path or mapping for ``dataset.yaml``.
        val_split_mode: Validation mode (``"official_and_carved"`` or ``"full_train_shared_val"``).
        val_carve_ratio: Fraction of training records carved when a client's dataset has
            no official validation split (default ``0.2`` -> ``10`` of ``54`` for IDRiD).
        batch_size: Local training batch size override.
        val_batch_size: Local validation batch size override.
        num_workers: DataLoader worker count override.
        pin_memory: DataLoader ``pin_memory`` override.
        persistent_workers: DataLoader ``persistent_workers`` override.
        drop_last_train: Whether to drop the final incomplete training batch.
        seed: Base random seed override (default ``42``).
        image_size: Optional ``(height, width)`` override.
        fov_crop: Optional FOV cropping boolean override.
        return_metadata: Whether ``Dataset.__getitem__`` returns ``(image, mask, metadata)``.
        mask_dtype: PyTorch dtype for the 4-channel binary lesion mask tensor.
        cache_in_memory: Whether to cache FOV-cropped, resized samples in memory.
        verify_files_exist: When ``True``, immediately verifies that all image and
            non-null mask files exist on disk before returning.

    Returns:
        :class:`ClientDataBundle` containing the client's records, datasets, and DataLoaders.
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

    train_records, val_records, info = resolve_client_train_val_records(
        client_id=client_id,
        partition_manifest=partition_manifest,
        source_manifest=source_manifest,
        validation_records=validation_records,
        federated_config=fed_cfg,
        dataset_config=ds_cfg,
        val_split_mode=val_split_mode,
        val_carve_ratio=val_carve_ratio,
        seed=seed,
    )

    canonical_cid: str = info["client_id"]
    client_idx: int = info["client_index"]
    resolved_seed: int = info["seed"]

    if verify_files_exist:
        verify_client_record_files(train_records, client_id=f"{canonical_cid}:train")
        verify_client_record_files(val_records, client_id=f"{canonical_cid}:valid")

    # Build training transform (augmentation enabled per dataset_config) and
    # deterministic validation transform (augmentation strictly disabled)
    train_transform = build_segmentation_transforms(
        split="train",
        dataset_config=ds_cfg,
        image_size=image_size,
    )
    val_transform = build_segmentation_transforms(
        split="valid",
        dataset_config=ds_cfg,
        image_size=image_size,
    )

    train_dataset = RetinalLesionSegmentationDataset(
        manifest_or_records=train_records,
        split="train",
        transform=train_transform,
        dataset_config=ds_cfg,
        image_size=image_size,
        fov_crop=fov_crop,
        return_metadata=return_metadata,
        mask_dtype=mask_dtype,
        allow_empty=False,
        cache_in_memory=cache_in_memory,
    )
    val_dataset = RetinalLesionSegmentationDataset(
        manifest_or_records=val_records,
        split="valid",
        transform=val_transform,
        dataset_config=ds_cfg,
        image_size=image_size,
        fov_crop=fov_crop,
        return_metadata=return_metadata,
        mask_dtype=mask_dtype,
        allow_empty=False,
        cache_in_memory=cache_in_memory,
    )

    resolved_batch_size = int(
        batch_size
        if batch_size is not None
        else fed_cfg.get("local_batch_size", ds_cfg.get("batch_size", 8))
    )
    resolved_val_batch_size = int(
        val_batch_size if val_batch_size is not None else resolved_batch_size
    )
    resolved_workers = int(
        num_workers
        if num_workers is not None
        else fed_cfg.get("num_workers", ds_cfg.get("number_of_workers", 0))
    )

    # Offset training DataLoader generator seed deterministically by client_index
    train_loader_seed = int((resolved_seed + client_idx) % (MAX_UINT32_SEED + 1))

    train_loader = build_segmentation_dataloader(
        dataset=train_dataset,
        dataset_config=ds_cfg,
        batch_size=resolved_batch_size,
        shuffle=True,
        num_workers=resolved_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=drop_last_train,
        seed=train_loader_seed,
    )
    val_loader = build_segmentation_dataloader(
        dataset=val_dataset,
        dataset_config=ds_cfg,
        batch_size=resolved_val_batch_size,
        shuffle=False,
        num_workers=resolved_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=False,
        seed=resolved_seed,
    )

    # Final trainer/evaluator split-hygiene guardrails
    verify_no_test_split_leakage(train_loader, stage_name="train")
    verify_no_test_split_leakage(val_loader, stage_name="validation")
    verify_exact_split(val_loader, expected_split="valid")

    return ClientDataBundle(
        client_id=canonical_cid,
        client_index=client_idx,
        dataset_names=info["dataset_names"],
        train_records=train_records,
        val_records=val_records,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        train_loader=train_loader,
        val_loader=val_loader,
        val_split_mode=info["val_split_mode"],
        val_source=info["val_source"],
        seed=resolved_seed,
        batch_size=resolved_batch_size,
        val_batch_size=resolved_val_batch_size,
        num_workers=resolved_workers,
        channel_order=CANONICAL_LESION_CHANNELS,
    )


def build_all_federated_client_dataloaders(
    partition_manifest: str | Path | Mapping[str, Any] | None = None,
    source_manifest: str | Path | Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    federated_config: Mapping[str, Any] | str | Path = "federated.yaml",
    dataset_config: Mapping[str, Any] | str | Path = "dataset.yaml",
    val_split_mode: str | None = None,
    val_carve_ratio: float | None = None,
    batch_size: int | None = None,
    val_batch_size: int | None = None,
    num_workers: int | None = None,
    pin_memory: bool | None = None,
    persistent_workers: bool | None = None,
    drop_last_train: bool | None = None,
    seed: int | None = None,
    image_size: tuple[int, int] | None = None,
    fov_crop: bool | None = None,
    return_metadata: bool = False,
    mask_dtype: torch.dtype = torch.float32,
    cache_in_memory: bool = False,
    verify_files_exist: bool = False,
) -> dict[str, ClientDataBundle]:
    """Construct :class:`ClientDataBundle` instances for all clients in the partition manifest."""
    fed_cfg = (
        load_yaml_config(federated_config, validate=True)
        if isinstance(federated_config, (str, Path))
        else dict(federated_config)
    )
    part_cfg = fed_cfg.get("partitioning", {})
    resolved_seed = int(
        seed if seed is not None else fed_cfg.get("seed", part_cfg.get("seed", 42))
    )

    resolved_part_manifest: str | Path | Mapping[str, Any]
    if partition_manifest is None:
        default_part_rel = part_cfg.get(
            "manifest_path",
            "data/partitions/combined_cross_dataset_2clients_seed42.json",
        )
        resolved_part_manifest = resolve_project_path(default_part_rel)
    else:
        resolved_part_manifest = partition_manifest

    part_dict = _load_and_validate_partition_dict(resolved_part_manifest)
    client_keys = list(part_dict["clients"].keys())

    bundles: dict[str, ClientDataBundle] = {}
    for cid in client_keys:
        bundles[cid] = build_federated_client_dataloaders(
            client_id=cid,
            partition_manifest=resolved_part_manifest if partition_manifest is not None else None,
            source_manifest=source_manifest,
            federated_config=fed_cfg,
            dataset_config=dataset_config,
            val_split_mode=val_split_mode,
            val_carve_ratio=val_carve_ratio,
            batch_size=batch_size,
            val_batch_size=val_batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            drop_last_train=drop_last_train,
            seed=resolved_seed,
            image_size=image_size,
            fov_crop=fov_crop,
            return_metadata=return_metadata,
            mask_dtype=mask_dtype,
            cache_in_memory=cache_in_memory,
            verify_files_exist=verify_files_exist,
        )

    # Cross-verify that no two bundles share any local training record
    seen_train: dict[tuple[str, str], str] = {}
    for cid, bundle in bundles.items():
        for rec in bundle.train_records:
            key = (str(rec["dataset_name"]), str(rec["sample_id"]))
            if key in seen_train:
                raise FederatedDataError(
                    f"Cross-client training overlap on {key} between '{seen_train[key]}' and '{cid}'."
                )
            seen_train[key] = cid

    return bundles


__all__ = [
    "ClientDataBundle",
    "FederatedDataError",
    "build_all_federated_client_dataloaders",
    "build_federated_client_dataloaders",
    "carve_client_validation_records",
    "resolve_client_id",
    "resolve_client_train_val_records",
    "verify_client_record_files",
    "verify_client_split_integrity",
]

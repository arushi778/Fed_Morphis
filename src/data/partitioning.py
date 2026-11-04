"""Reproducible IID and non-IID federated client partitioning for DDR and IDRiD.

Design & Split Hygiene Guarantees:
- Partitions **only** the official ``train`` split records.
- Official ``test`` and ``valid`` split records are strictly excluded from client
  training partitions and verified for zero leakage.
- Every ``(dataset_name, sample_id)`` training sample is assigned to exactly one
  client; client partitions are pairwise disjoint and conserve 100% of training
  samples.
- Uses an isolated ``np.random.Generator`` seeded with ``seed`` so partitioning
  is 100% deterministic for a fixed seed and never mutates global RNG state.

Non-IID Dirichlet Algorithm (No Invented Severity Labels, Zero Raw File Re-Reads):
- Each training sample's discrete stratum key is derived directly from its manifest
  record metadata (``dataset_name``, ``lesion_pixel_counts``, and
  ``positive_lesion_masks`` across ``[MA, HE, EX, SE]``) without re-reading raw
  images or masks from disk and without inventing synthetic DR grading labels:
  ``stratum = f"{dataset_name}:{dominant_lesion_channel}"`` where
  ``dominant_lesion_channel`` is the lesion class in ``[MA, HE, EX, SE]`` with the
  highest positive foreground pixel count (or ``"NONE"`` if all 4 channels are zero).
- For each stratum $k$, a proportion vector $\\mathbf{p}_k \\sim \\text{Dir}(\\alpha \\cdot \\mathbf{1}_C)$
  is drawn across the $C$ clients and converted to exact integer counts via
  largest-remainder allocation.
- A deterministic post-allocation rebalancing pass guarantees every client receives
  at least ``min_samples_per_client`` (default ``1``) whenever
  ``total_train_samples >= num_clients * min_samples_per_client``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from src.data.dataset import _extract_records_from_input, _validate_manifest_record
from src.data.masks import CANONICAL_LESION_CHANNELS
from src.utils.config import SUPPORTED_PARTITION_STRATEGIES, load_yaml_config
from src.utils.paths import resolve_project_path
from src.utils.seed import MAX_UINT32_SEED

PARTITION_SCHEMA_VERSION: str = "1.0"


class PartitioningError(ValueError):
    """Raised when federated client partitioning configuration or invariants fail."""


def derive_sample_stratum_key(record: Mapping[str, Any]) -> str:
    """Derive a deterministic non-IID stratum key from a manifest record's lesion profile.

    Uses ``dataset_name`` and the dominant lesion channel in ``[MA, HE, EX, SE]``
    from precomputed ``lesion_pixel_counts`` (or ``positive_lesion_masks`` if pixel
    counts are absent), avoiding any raw image/mask disk I/O.
    """
    ds_name = str(record.get("dataset_name", "UNKNOWN")).strip()
    pixel_counts = record.get("lesion_pixel_counts")

    dominant = "NONE"
    if isinstance(pixel_counts, Mapping):
        best_count = 0
        for cls_name in CANONICAL_LESION_CHANNELS:
            cnt = int(pixel_counts.get(cls_name, 0))
            if cnt > best_count:
                best_count = cnt
                dominant = cls_name
    else:
        pos_masks = record.get("positive_lesion_masks")
        if isinstance(pos_masks, Sequence) and not isinstance(pos_masks, (str, bytes)):
            for cls_name in CANONICAL_LESION_CHANNELS:
                if cls_name in pos_masks:
                    dominant = cls_name
                    break

    return f"{ds_name}:{dominant}"


def _largest_remainder_counts(proportions: np.ndarray, total: int) -> np.ndarray:
    """Convert a probability vector ``proportions`` into integer counts summing to ``total``."""
    if total <= 0:
        return np.zeros(len(proportions), dtype=int)
    exact = proportions * float(total)
    floors = np.floor(exact).astype(int)
    remainder = int(total - floors.sum())
    if remainder > 0:
        fracs = exact - floors
        # Deterministic tie-breaking by client index
        order = np.lexsort((np.arange(len(proportions)), -fracs))
        for idx in order[:remainder]:
            floors[idx] += 1
    return floors


def _rebalance_minimum_client_sizes(
    client_bins: list[list[dict[str, Any]]],
    min_samples_per_client: int,
) -> None:
    """Ensure every client bin has at least ``min_samples_per_client`` records in-place."""
    num_clients = len(client_bins)
    total_samples = sum(len(b) for b in client_bins)
    if total_samples < num_clients * min_samples_per_client:
        raise PartitioningError(
            f"Cannot satisfy min_samples_per_client={min_samples_per_client} across "
            f"{num_clients} clients with only {total_samples} training samples."
        )

    while True:
        deficit_clients = [
            cid for cid, b in enumerate(client_bins) if len(b) < min_samples_per_client
        ]
        if not deficit_clients:
            break
        target_cid = deficit_clients[0]
        # Pick donor client with maximum size (ties broken by lowest client index)
        donor_cid = max(range(num_clients), key=lambda cid: (len(client_bins[cid]), -cid))
        if len(client_bins[donor_cid]) <= min_samples_per_client:
            raise PartitioningError(
                "Unable to rebalance client partitions without violating min_samples_per_client."
            )
        transferred = client_bins[donor_cid].pop()
        client_bins[target_cid].append(transferred)


def verify_client_partitions(
    client_records: Mapping[str, Sequence[Mapping[str, Any]]],
    expected_train_records: Sequence[Mapping[str, Any]],
    held_out_valid_records: Sequence[Mapping[str, Any]] = (),
    held_out_test_records: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Verify partition coverage, disjointness, and zero validation/test split leakage."""
    expected_keys = {
        (str(r["dataset_name"]), str(r["sample_id"])) for r in expected_train_records
    }
    if len(expected_keys) != len(expected_train_records):
        raise PartitioningError("Duplicate (dataset_name, sample_id) in expected_train_records.")

    valid_keys = {
        (str(r["dataset_name"]), str(r["sample_id"])) for r in held_out_valid_records
    }
    test_keys = {
        (str(r["dataset_name"]), str(r["sample_id"])) for r in held_out_test_records
    }

    seen_keys: set[tuple[str, str]] = set()
    duplicate_across_clients: list[tuple[str, str]] = []
    non_train_splits_found: list[str] = []

    for client_id, records in client_records.items():
        for rec in records:
            rec_split = str(rec.get("split", "")).strip().lower()
            if rec_split != "train":
                non_train_splits_found.append(
                    f"{client_id}:{rec.get('dataset_name')}:{rec.get('sample_id')}(split={rec_split})"
                )
            key = (str(rec["dataset_name"]), str(rec["sample_id"]))
            if key in seen_keys:
                duplicate_across_clients.append(key)
            seen_keys.add(key)

    missing_train_keys = sorted(expected_keys - seen_keys)
    unexpected_keys = sorted(seen_keys - expected_keys)
    val_leakage = sorted(seen_keys & valid_keys)
    test_leakage = sorted(seen_keys & test_keys)

    if non_train_splits_found:
        raise PartitioningError(
            f"Non-training split sample(s) leaked into client partitions: {non_train_splits_found[:5]}"
        )
    if duplicate_across_clients:
        raise PartitioningError(
            f"Client partitions are not disjoint; duplicate sample(s): {duplicate_across_clients[:5]}"
        )
    if missing_train_keys or unexpected_keys:
        raise PartitioningError(
            f"Training sample conservation failed: missing={len(missing_train_keys)}, "
            f"unexpected={len(unexpected_keys)}."
        )
    if val_leakage or test_leakage:
        raise PartitioningError(
            f"Split leakage detected: val_leakage={len(val_leakage)}, test_leakage={len(test_leakage)}."
        )

    return {
        "all_train_samples_assigned_once": True,
        "clients_disjoint": True,
        "total_assigned_train_samples": len(seen_keys),
        "expected_train_samples": len(expected_keys),
        "lost_train_sample_count": 0,
        "validation_leakage_count": 0,
        "test_leakage_count": 0,
    }


def partition_training_records(
    manifest_or_records: str | Path | Mapping[str, Any] | Sequence[Mapping[str, Any]],
    num_clients: int = 4,
    strategy: str = "dirichlet",
    seed: int = 42,
    dirichlet_alpha: float = 0.5,
    min_samples_per_client: int = 1,
) -> dict[str, Any]:
    """Partition official training records across simulated federated clients.

    Args:
        manifest_or_records: Manifest path, manifest dict, or sequence of manifest records.
        num_clients: Number of federated clients ($\\ge 1$).
        strategy: Partitioning strategy: ``"iid"``, ``"dirichlet"`` (or ``"non_iid"``),
            or ``"cross_dataset"`` (or ``"by_dataset"``).
        seed: Random seed in ``[0, 2**32 - 1]`` for deterministic assignment.
        dirichlet_alpha: Concentration parameter $\\alpha > 0$ for Dirichlet non-IID
            partitioning (smaller $\\alpha$ increases client heterogeneity).
        min_samples_per_client: Minimum training records guaranteed per client (default ``1``).

    Returns:
        Structured dictionary containing ``clients``, ``client_summaries``,
        ``verification``, and partition metadata.
    """
    if isinstance(num_clients, bool) or not isinstance(num_clients, int) or num_clients < 1:
        raise PartitioningError(
            f"num_clients must be an integer >= 1, got {num_clients!r}."
        )
    if isinstance(seed, bool) or not isinstance(seed, int) or not (0 <= seed <= MAX_UINT32_SEED):
        raise PartitioningError(
            f"Partition seed must be an integer in [0, {MAX_UINT32_SEED}], got {seed!r}."
        )
    if (
        isinstance(min_samples_per_client, bool)
        or not isinstance(min_samples_per_client, int)
        or min_samples_per_client < 1
    ):
        raise PartitioningError(
            f"min_samples_per_client must be an integer >= 1, got {min_samples_per_client!r}."
        )

    strategy_norm = str(strategy).strip().lower()
    if strategy_norm not in SUPPORTED_PARTITION_STRATEGIES:
        raise PartitioningError(
            f"Unsupported partition strategy {strategy!r}; supported strategies are "
            f"{SUPPORTED_PARTITION_STRATEGIES}."
        )
    if strategy_norm == "non_iid":
        strategy_norm = "dirichlet"
    elif strategy_norm == "by_dataset":
        strategy_norm = "cross_dataset"

    if strategy_norm == "dirichlet":
        if (
            isinstance(dirichlet_alpha, bool)
            or not isinstance(dirichlet_alpha, (int, float))
            or float(dirichlet_alpha) <= 0.0
            or not np.isfinite(float(dirichlet_alpha))
        ):
            raise PartitioningError(
                f"dirichlet_alpha must be a finite positive number > 0, got {dirichlet_alpha!r}."
            )

    all_records = _extract_records_from_input(manifest_or_records)
    for idx, rec in enumerate(all_records):
        _validate_manifest_record(rec, idx)

    train_records: list[dict[str, Any]] = []
    valid_records: list[dict[str, Any]] = []
    test_records: list[dict[str, Any]] = []

    for rec in all_records:
        sp = str(rec.get("split", "")).strip().lower()
        if sp == "train":
            train_records.append(dict(rec))
        elif sp in {"valid", "val"}:
            valid_records.append(dict(rec))
        elif sp == "test":
            test_records.append(dict(rec))
        else:
            raise PartitioningError(
                f"Unrecognized split {sp!r} for sample '{rec.get('sample_id')}'."
            )

    # Canonical sort of training records by (dataset_name, sample_id) before seeded RNG
    train_records.sort(key=lambda r: (str(r["dataset_name"]), str(r["sample_id"])))

    if len(train_records) < num_clients * min_samples_per_client:
        raise PartitioningError(
            f"Insufficient training samples ({len(train_records)}) to partition across "
            f"{num_clients} clients with min_samples_per_client={min_samples_per_client}."
        )

    # Check for duplicate (dataset_name, sample_id) in train_records
    seen_train_ids: set[tuple[str, str]] = set()
    for r in train_records:
        k = (str(r["dataset_name"]), str(r["sample_id"]))
        if k in seen_train_ids:
            raise PartitioningError(
                f"Duplicate training record detected before partitioning: {k}."
            )
        seen_train_ids.add(k)

    rng = np.random.default_rng(int(seed))
    client_bins: list[list[dict[str, Any]]] = [[] for _ in range(num_clients)]

    if strategy_norm == "iid":
        perm = rng.permutation(len(train_records)).tolist()
        shuffled = [train_records[i] for i in perm]
        splits = np.array_split(np.arange(len(shuffled)), num_clients)
        for cid, idx_arr in enumerate(splits):
            client_bins[cid] = [shuffled[int(i)] for i in idx_arr.tolist()]

    elif strategy_norm == "dirichlet":
        alpha = float(dirichlet_alpha)
        strata: dict[str, list[dict[str, Any]]] = {}
        for rec in train_records:
            s_key = derive_sample_stratum_key(rec)
            strata.setdefault(s_key, []).append(rec)

        for s_key in sorted(strata.keys()):
            group = strata[s_key]
            perm = rng.permutation(len(group)).tolist()
            shuffled_group = [group[i] for i in perm]

            proportions = rng.dirichlet(np.full(num_clients, alpha, dtype=np.float64))
            counts = _largest_remainder_counts(proportions, len(shuffled_group))

            cursor = 0
            for cid, cnt in enumerate(counts.tolist()):
                if cnt > 0:
                    client_bins[cid].extend(shuffled_group[cursor : cursor + cnt])
                    cursor += cnt

        _rebalance_minimum_client_sizes(client_bins, min_samples_per_client)

    elif strategy_norm == "cross_dataset":
        by_ds: dict[str, list[dict[str, Any]]] = {}
        for rec in train_records:
            by_ds.setdefault(str(rec["dataset_name"]), []).append(rec)

        ds_names = sorted(by_ds.keys())
        if len(ds_names) == 1 or num_clients == 1:
            perm = rng.permutation(len(train_records)).tolist()
            shuffled = [train_records[i] for i in perm]
            splits = np.array_split(np.arange(len(shuffled)), num_clients)
            for cid, idx_arr in enumerate(splits):
                client_bins[cid] = [shuffled[int(i)] for i in idx_arr.tolist()]
        else:
            # Allocate at least 1 client per dataset when num_clients >= len(ds_names),
            # and distribute remaining clients proportionally to dataset size.
            ds_Counts = np.array([len(by_ds[d]) for d in ds_names], dtype=np.float64)
            if num_clients >= len(ds_names):
                base_alloc = np.ones(len(ds_names), dtype=int)
                rem_clients = num_clients - len(ds_names)
                extra_alloc = _largest_remainder_counts(
                    ds_Counts / ds_Counts.sum(), rem_clients
                )
                client_alloc = base_alloc + extra_alloc
            else:
                client_alloc = _largest_remainder_counts(
                    ds_Counts / ds_Counts.sum(), num_clients
                )

            cid_cursor = 0
            for ds_idx, d_name in enumerate(ds_names):
                n_c = int(client_alloc[ds_idx])
                if n_c <= 0:
                    continue
                ds_group = by_ds[d_name]
                perm = rng.permutation(len(ds_group)).tolist()
                shuffled_ds = [ds_group[i] for i in perm]
                splits = np.array_split(np.arange(len(shuffled_ds)), n_c)
                for offset, idx_arr in enumerate(splits):
                    client_bins[cid_cursor + offset].extend(
                        [shuffled_ds[int(i)] for i in idx_arr.tolist()]
                    )
                cid_cursor += n_c

            _rebalance_minimum_client_sizes(client_bins, min_samples_per_client)

    # Sort each client's assigned records deterministically by (dataset_name, sample_id)
    clients_dict: dict[str, list[dict[str, Any]]] = {}
    client_summaries: dict[str, dict[str, Any]] = {}

    for cid in range(num_clients):
        client_id = f"client_{cid}"
        c_recs = sorted(
            client_bins[cid],
            key=lambda r: (str(r["dataset_name"]), str(r["sample_id"])),
        )
        clients_dict[client_id] = c_recs

        ds_counts: dict[str, int] = {}
        pos_class_counts = {c: 0 for c in CANONICAL_LESION_CHANNELS}
        stratum_counts: dict[str, int] = {}

        for r in c_recs:
            d_name = str(r["dataset_name"])
            ds_counts[d_name] = ds_counts.get(d_name, 0) + 1
            for cls_name in r.get("positive_lesion_masks", []):
                if cls_name in pos_class_counts:
                    pos_class_counts[cls_name] += 1
            st_key = derive_sample_stratum_key(r)
            stratum_counts[st_key] = stratum_counts.get(st_key, 0) + 1

        client_summaries[client_id] = {
            "num_samples": len(c_recs),
            "dataset_counts": dict(sorted(ds_counts.items())),
            "positive_lesion_counts": pos_class_counts,
            "stratum_counts": dict(sorted(stratum_counts.items())),
            "sample_ids": [f"{r['dataset_name']}:{r['sample_id']}" for r in c_recs],
        }

    verification = verify_client_partitions(
        client_records=clients_dict,
        expected_train_records=train_records,
        held_out_valid_records=valid_records,
        held_out_test_records=test_records,
    )

    observed_datasets = sorted({str(r["dataset_name"]) for r in train_records})

    return {
        "schema_version": PARTITION_SCHEMA_VERSION,
        "strategy": strategy_norm,
        "num_clients": num_clients,
        "seed": int(seed),
        "dirichlet_alpha": float(dirichlet_alpha) if strategy_norm == "dirichlet" else None,
        "min_samples_per_client": min_samples_per_client,
        "source_datasets": observed_datasets,
        "total_train_samples": len(train_records),
        "held_out_validation_samples": len(valid_records),
        "held_out_test_samples": len(test_records),
        "verification": verification,
        "client_summaries": client_summaries,
        "clients": clients_dict,
    }


def save_partition_manifest(partition_data: Mapping[str, Any], output_path: str | Path) -> Path:
    """Save a federated client partition dictionary deterministically to JSON."""
    out_p = resolve_project_path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(partition_data, indent=2)
    out_p.write_text(payload + "\n", encoding="utf-8")
    return out_p


def load_partition_manifest(partition_path: str | Path) -> dict[str, Any]:
    """Load a saved federated client partition manifest from JSON."""
    path = resolve_project_path(partition_path)
    if not path.exists():
        raise FileNotFoundError(f"Partition manifest not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def build_configured_partitions(
    federated_config_path: str | Path = "federated.yaml",
    dataset_config_path: str | Path = "dataset.yaml",
    output_dir: str | Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Build and save IID and non-IID client partitions from the generated dataset manifests."""
    fed_cfg = load_yaml_config(federated_config_path, validate=True)
    ds_cfg = load_yaml_config(dataset_config_path, validate=True)

    num_clients = int(fed_cfg.get("number_of_clients", 4))
    part_cfg = fed_cfg.get("partitioning", {})
    seed = int(part_cfg.get("seed", 42))
    alpha = float(part_cfg.get("dirichlet_alpha", 0.5))
    min_samples = int(part_cfg.get("min_samples_per_client", 1))

    paths_cfg = ds_cfg.get("paths", {})
    manifest_dir = resolve_project_path(
        paths_cfg.get("manifests", "data/processed/manifests")
    )
    if output_dir is None:
        rel_part_dir = part_cfg.get(
            "output_directory", paths_cfg.get("partitions", "data/partitions")
        )
        part_dir = resolve_project_path(rel_part_dir)
    else:
        part_dir = resolve_project_path(output_dir)

    part_dir.mkdir(parents=True, exist_ok=True)

    manifest_sources: dict[str, Path] = {
        "ddr": manifest_dir / "ddr_manifest.json",
        "idrid": manifest_dir / "idrid_manifest.json",
        "combined": manifest_dir / "combined_manifest.json",
    }

    generated: dict[str, dict[str, Any]] = {}

    for src_tag, m_path in manifest_sources.items():
        if not m_path.exists():
            continue

        # 1. IID Partition
        iid_part = partition_training_records(
            manifest_or_records=m_path,
            num_clients=num_clients,
            strategy="iid",
            seed=seed,
            min_samples_per_client=min_samples,
        )
        iid_file = part_dir / f"{src_tag}_iid_{num_clients}clients_seed{seed}.json"
        save_partition_manifest(iid_part, iid_file)
        generated[f"{src_tag}_iid"] = iid_part

        # 2. Non-IID Dirichlet Partition
        dir_part = partition_training_records(
            manifest_or_records=m_path,
            num_clients=num_clients,
            strategy="dirichlet",
            seed=seed,
            dirichlet_alpha=alpha,
            min_samples_per_client=min_samples,
        )
        dir_file = (
            part_dir
            / f"{src_tag}_dirichlet_a{alpha}_{num_clients}clients_seed{seed}.json"
        )
        save_partition_manifest(dir_part, dir_file)
        generated[f"{src_tag}_dirichlet"] = dir_part

        # 3. For combined multi-dataset manifest, also generate cross_dataset partition
        if src_tag == "combined":
            cross_part = partition_training_records(
                manifest_or_records=m_path,
                num_clients=num_clients,
                strategy="cross_dataset",
                seed=seed,
                min_samples_per_client=min_samples,
            )
            cross_file = (
                part_dir / f"{src_tag}_cross_dataset_{num_clients}clients_seed{seed}.json"
            )
            save_partition_manifest(cross_part, cross_file)
            generated[f"{src_tag}_cross_dataset"] = cross_part

    return generated


__all__ = [
    "PARTITION_SCHEMA_VERSION",
    "PartitioningError",
    "build_configured_partitions",
    "derive_sample_stratum_key",
    "load_partition_manifest",
    "partition_training_records",
    "save_partition_manifest",
    "verify_client_partitions",
]


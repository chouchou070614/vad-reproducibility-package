"""Deterministic, side-effect-free query sampling for Memory trajectory audit.

This module deliberately owns its own ``RandomState``.  It therefore never
advances NumPy's global RNG or PyTorch's RNG and cannot change the training or
MiniBatchKMeans trajectory that it observes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np


class DeterministicPrioritySampler:
    """Keep a deterministic uniform priority sample from a query stream."""

    def __init__(self, capacity: int, feature_dim: int, seed: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        self.capacity = int(capacity)
        self.feature_dim = int(feature_dim)
        self.seed = int(seed)
        self._rng = np.random.RandomState(self.seed)
        self._queries = np.empty((0, self.feature_dim), dtype=np.float32)
        self._priorities = np.empty((0,), dtype=np.float64)
        self._metadata: Dict[str, np.ndarray] = {}
        self.seen = 0

    def observe(self, queries: np.ndarray, metadata: Dict[str, np.ndarray]) -> None:
        queries = np.asarray(queries, dtype=np.float32)
        if queries.ndim != 2 or queries.shape[1] != self.feature_dim:
            raise ValueError("queries must have shape [N, feature_dim]")
        count = int(queries.shape[0])
        required = {
            "global_query_index",
            "dataset_index",
            "batch_index",
            "batch_item_index",
            "spatial_y",
            "spatial_x",
        }
        if set(metadata) != required:
            raise ValueError("metadata fields do not match the trace schema")
        normalized = {}
        for name, values in metadata.items():
            values = np.asarray(values, dtype=np.int64)
            if values.shape != (count,):
                raise ValueError("metadata field %s has the wrong length" % name)
            normalized[name] = values

        priorities = self._rng.random_sample(count)
        combined_queries = np.concatenate((self._queries, queries), axis=0)
        combined_priorities = np.concatenate((self._priorities, priorities), axis=0)
        combined_metadata = {}
        for name in sorted(required):
            old = self._metadata.get(name, np.empty((0,), dtype=np.int64))
            combined_metadata[name] = np.concatenate((old, normalized[name]), axis=0)

        if combined_priorities.size > self.capacity:
            chosen = np.argpartition(combined_priorities, self.capacity - 1)[: self.capacity]
        else:
            chosen = np.arange(combined_priorities.size)
        self._queries = combined_queries[chosen]
        self._priorities = combined_priorities[chosen]
        self._metadata = {name: values[chosen] for name, values in combined_metadata.items()}
        self.seen += count

    def finalize(self, centers: np.ndarray) -> Dict[str, np.ndarray]:
        centers = np.asarray(centers, dtype=np.float32)
        if centers.ndim != 2 or centers.shape[1] != self.feature_dim:
            raise ValueError("centers must have shape [K, feature_dim]")
        if self._queries.shape[0] == 0:
            raise ValueError("no queries were observed")

        # Stable provenance order makes the NPZ easy to diff and audit.  The
        # priority values remain saved, so the sampling decision is reproducible.
        order = np.argsort(self._metadata["global_query_index"], kind="mergesort")
        queries = self._queries[order]
        priorities = self._priorities[order]
        metadata = {name: values[order] for name, values in self._metadata.items()}

        similarities = np.matmul(queries, centers.T)
        nearest_slot = np.argmax(similarities, axis=1).astype(np.int64)
        row = np.arange(queries.shape[0])
        nearest_similarity = similarities[row, nearest_slot].astype(np.float32)
        nearest_cosine_distance = (1.0 - nearest_similarity).astype(np.float32)
        nearest_l2_distance = np.linalg.norm(
            queries - centers[nearest_slot], axis=1
        ).astype(np.float32)

        result = dict(metadata)
        result.update({
            "queries": queries.astype(np.float32, copy=False),
            "priority": priorities.astype(np.float64, copy=False),
            "nearest_slot": nearest_slot,
            "nearest_similarity": nearest_similarity,
            "nearest_cosine_distance": nearest_cosine_distance,
            "nearest_l2_distance": nearest_l2_distance,
        })
        return result


def save_query_trace(
    output: Path,
    trace: Dict[str, np.ndarray],
    query_count: int,
    memory_size: int,
    feature_dim: int,
    sample_seed: int,
    kmeans_seed: int,
    warmup_epoch: int,
) -> None:
    """Save a pickle-free NPZ that can be loaded with allow_pickle=False."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(trace)
    payload.update({
        "schema_version": np.asarray("memory-trajectory-v2"),
        "source_split": np.asarray("normal_training"),
        "model_epoch": np.asarray(warmup_epoch, dtype=np.int64),
        "query_count_full": np.asarray(query_count, dtype=np.int64),
        "query_sample_count": np.asarray(len(trace["queries"]), dtype=np.int64),
        "memory_size": np.asarray(memory_size, dtype=np.int64),
        "feature_dim": np.asarray(feature_dim, dtype=np.int64),
        "sample_seed": np.asarray(sample_seed, dtype=np.int64),
        "kmeans_seed": np.asarray(kmeans_seed, dtype=np.int64),
        "normalized_before_kmeans": np.asarray(1, dtype=np.int8),
        "loader_shuffle": np.asarray(0, dtype=np.int8),
    })
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("wb") as target:
        np.savez_compressed(target, **payload)
    temporary.replace(output)

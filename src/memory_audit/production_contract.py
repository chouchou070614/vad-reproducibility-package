"""Metadata and file-manifest helpers for the Ped2 memory-trajectory analysis."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np


TASK_ID = "COND01_ped2_P_kmemory_continue_trajectory_seed2020"
EVIDENCE_FAMILY = "KMemoryTrajectory"
SEED = 2020


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def report_passed(value: object) -> bool:
    """Read a completion flag or status from a report."""
    if not isinstance(value, dict):
        return False
    for key in ("overall_pass", "complete", "valid", "passed"):
        if key in value:
            return value[key] is True
    return str(value.get("state", "")).upper() in {
        "PASS", "PASSED", "COMPLETE", "COMPLETED"
    }


def tag_evidence_identity(path: Path, task_id: str) -> None:
    """Add trajectory metadata to the NPZ and its profile file."""
    if task_id != TASK_ID:
        raise ValueError("unexpected task id: %s" % task_id)
    with np.load(str(path), allow_pickle=False) as source:
        payload = {name: source[name] for name in source.files}
    if "family" in payload:
        existing = str(np.asarray(payload["family"]).reshape(-1)[0])
        if existing != EVIDENCE_FAMILY:
            raise ValueError("evidence family conflict in %s: %s" % (path, existing))
    payload.update({
        "family": np.asarray(EVIDENCE_FAMILY),
        "task_id": np.asarray(task_id),
        "seed": np.asarray(SEED, dtype=np.int64),
        "memory_update_policy": np.asarray("Continue"),
        "query_source_split": np.asarray("normal_training"),
        "kmeans_uses_test_labels": np.asarray(False, dtype=np.bool_),
    })
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as target:
        np.savez_compressed(target, **payload)
        target.flush()
        os.fsync(target.fileno())
    temporary.replace(path)

    profile_path = path.with_suffix(".profile.json")
    profile = (
        json.loads(profile_path.read_text(encoding="utf-8"))
        if profile_path.is_file()
        else {}
    )
    profile.update({
        "family": EVIDENCE_FAMILY,
        "source_family": "KMemory",
        "task_id": task_id,
        "seed": SEED,
        "memory_update_policy": "Continue",
        "query_source_split": "normal_training",
        "kmeans_uses_test_labels": False,
    })
    atomic_json(profile_path, profile)


def validate_outer_manifest_gate(task_dir: Path, manifest_path: Path) -> None:
    """Check required trajectory files against the manifest hashes."""
    state_path = task_dir / "artifacts/memory_trajectory/trajectory_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    names = state.get("files")
    if not isinstance(names, dict):
        raise ValueError("trajectory files map is missing")
    required = {
        "status.json",
        "completeness_report.json",
        "metrics/trajectory_validation.json",
        "artifacts/memory_trajectory/trajectory_state.json",
    }
    for key in (
        "warmup_pre_kmeans_memory",
        "actual_kmeans_centers",
        "post_replacement_memory",
        "final_memory",
        "warmup_query_trace",
        "final_checkpoint",
    ):
        raw = names.get(key)
        if not raw:
            raise ValueError("trajectory file missing from state: %s" % key)
        artifact = (state_path.parent / str(raw)).resolve()
        try:
            relative = artifact.relative_to(task_dir.resolve())
        except ValueError:
            raise ValueError("trajectory file escapes task directory: %s" % key)
        required.add(str(relative).replace("\\", "/"))

    with manifest_path.open("r", encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))
    by_relative = {}
    for row in rows:
        relative = str(row.get("relative_path") or row.get("path") or "").replace(
            "\\", "/"
        )
        if not relative or relative in by_relative:
            raise ValueError("manifest has a blank or duplicate relative path")
        by_relative[relative] = row
    for relative in sorted(required):
        row = by_relative.get(relative)
        if row is None:
            raise ValueError("outer-gate manifest row missing: %s" % relative)
        artifact = task_dir / relative
        declared = str(row.get("sha256") or row.get("hash") or "").lower()
        if not artifact.is_file() or artifact.stat().st_size <= 0:
            raise ValueError("outer-gate artifact missing or empty: %s" % relative)
        if not declared or sha256_file(artifact).lower() != declared:
            raise ValueError("outer-gate manifest hash mismatch: %s" % relative)

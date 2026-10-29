#!/usr/bin/env python3
"""Run the Ped2 prediction memory-trajectory analysis and save its outputs."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from production_contract import (
    EVIDENCE_FAMILY,
    TASK_ID,
    report_passed,
    tag_evidence_identity,
    validate_outer_manifest_gate,
)


EPOCHS = 60
WARMUP = 26
SEED = 2020
MEMORY_SIZE = 10
FEATURE_DIM = 512


def now() -> str:
    return dt.datetime.now().astimezone().isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def sample_gpu(stop: threading.Event, output: Path) -> None:
    query = (
        "timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,"
        "temperature.gpu,power.draw,power.limit"
    )
    with output.open("w", encoding="utf-8", newline="") as target:
        writer = csv.writer(target)
        writer.writerow(query.split(","))
        executable = shutil.which("nvidia-smi")
        if executable is None:
            target.flush()
            return
        while not stop.is_set():
            completed = subprocess.run(
                [executable, "--query-gpu=" + query, "--format=csv,noheader,nounits"],
                text=True,
                capture_output=True,
            )
            if completed.returncode == 0:
                for line in completed.stdout.strip().splitlines():
                    writer.writerow([part.strip() for part in line.split(",")])
                target.flush()
            stop.wait(5)


def run_logged(command: List[str], cwd: Path, log_path: Path, package_dir: Path) -> None:
    python_path = os.pathsep.join(
        value
        for value in (str(cwd), str(package_dir), os.environ.get("PYTHONPATH", ""))
        if value
    )
    environment = {
        **os.environ,
        "WANDB_MODE": "disabled",
        "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": python_path,
    }
    with log_path.open("w", encoding="utf-8") as log:
        log.write("COMMAND=" + json.dumps(command, ensure_ascii=False) + "\n")
        log.flush()
        completed = subprocess.run(
            command,
            cwd=cwd,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            env=environment,
        )
    if completed.returncode:
        raise RuntimeError("command failed (%d); see %s" % (completed.returncode, log_path))


def score_metrics(npz_path: Path) -> Dict[str, float]:
    with np.load(str(npz_path), allow_pickle=False) as archive:
        labels = np.asarray(archive["labels"], dtype=np.int8)
        key = "scores" if "scores" in archive.files else "official_per_video_scores"
        scores = np.asarray(archive[key], dtype=np.float64)
    if labels.shape != scores.shape or labels.ndim != 1:
        raise ValueError("unaligned labels/scores in %s" % npz_path)
    if not np.isfinite(scores).all():
        raise ValueError("non-finite scores in %s" % npz_path)
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1.0 - tpr
    index = int(np.nanargmin(np.abs(fpr - fnr)))
    return {
        "AUROC": float(roc_auc_score(labels, scores)),
        "AUPRC": float(average_precision_score(labels, scores)),
        "EER": float((fpr[index] + fnr[index]) / 2.0),
        "EER_threshold": float(thresholds[index]),
        "frames": int(labels.size),
    }


def parse_losses(log_path: Path, output: Path) -> List[Dict[str, object]]:
    epoch = None
    rows = []
    epoch_pattern = re.compile(r"^Epoch:\s*(\d+)")
    average_pattern = re.compile(r"^Epoch average total loss:\s*([0-9.eE+-]+)")
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = epoch_pattern.search(line)
        if match:
            epoch = int(match.group(1))
            continue
        match = average_pattern.search(line)
        if match and epoch is not None:
            rows.append({"epoch": epoch, "average_total_loss": float(match.group(1))})
            epoch = None
    if [row["epoch"] for row in rows] != list(range(1, EPOCHS + 1)):
        raise ValueError("training log does not contain exactly epochs 1..%d" % EPOCHS)
    if not all(math.isfinite(float(row["average_total_loss"])) for row in rows):
        raise ValueError("non-finite training loss")
    with output.open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=("epoch", "average_total_loss"))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def load_tensor(path: Path) -> Any:
    import torch

    value = torch.load(str(path), map_location="cpu")
    if not torch.is_tensor(value):
        raise TypeError("expected tensor in %s" % path)
    return value.detach().cpu()


def validate_trajectory(artifacts: Path, output: Path) -> Dict[str, object]:
    import torch

    trajectory_dir = artifacts / "memory_trajectory"
    state_path = trajectory_dir / "trajectory_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("schema_version") != "memory-trajectory-v2":
        raise ValueError("trajectory schema mismatch")
    if not state.get("same_training_trajectory") or not state.get("training_complete"):
        raise ValueError("trajectory did not finish in one training run")
    if state.get("task_id") != TASK_ID:
        raise ValueError("trajectory task id mismatch")
    if state.get("dataset") != "ped2" or state.get("branch") != "P":
        raise ValueError("trajectory dataset/branch mismatch")
    if int(state.get("seed", -1)) != SEED or int(state.get("epochs", -1)) != EPOCHS:
        raise ValueError("trajectory seed/epoch mismatch")
    if state.get("family") != "KMemory":
        raise ValueError("trajectory family mismatch")
    if state.get("memory_update_policy") != "Continue":
        raise ValueError("trajectory is not KMemory-Continue")
    if state.get("query_source_split") != "normal_training":
        raise ValueError("trajectory query source is not normal training data")
    if state.get("kmeans_uses_test_labels") is not False:
        raise ValueError("trajectory K-means provenance is not label-free")
    if int(state.get("warmup_epoch", -1)) != WARMUP:
        raise ValueError("trajectory warm-up mismatch")
    expected_stages = {
        "random_initial_memory",
        "warmup_pre_kmeans_memory",
        "actual_kmeans_centers",
        "post_replacement_memory",
        "warmup_query_trace",
        "final_memory",
        "final_checkpoint",
    }
    if set(state.get("completed_stages", [])) != expected_stages:
        raise ValueError("trajectory stage set is incomplete or duplicated")

    names = state["files"]
    declared_hashes = state.get("sha256")
    if not isinstance(declared_hashes, dict):
        raise ValueError("trajectory sha256 map missing")
    required_hashes = {
        "random_initial_memory",
        "warmup_pre_kmeans_memory",
        "actual_kmeans_centers",
        "raw_kmeans_centers_before_normalization",
        "post_replacement_memory",
        "final_memory",
        "warmup_query_trace",
        "final_checkpoint",
    }
    for stage in required_hashes:
        path = (trajectory_dir / names[stage]).resolve()
        if not path.is_file() or sha256(path) != declared_hashes.get(stage):
            raise ValueError("trajectory file/hash mismatch for %s" % stage)
    tensors = {}
    for stage in (
        "random_initial_memory",
        "warmup_pre_kmeans_memory",
        "actual_kmeans_centers",
        "post_replacement_memory",
        "final_memory",
    ):
        path = trajectory_dir / names[stage]
        tensors[stage] = load_tensor(path)
        if tuple(tensors[stage].shape) != (MEMORY_SIZE, FEATURE_DIM):
            raise ValueError("wrong Memory shape for %s" % stage)
        if not torch.isfinite(tensors[stage]).all().item():
            raise ValueError("non-finite Memory values for %s" % stage)

    centers = tensors["actual_kmeans_centers"]
    post = tensors["post_replacement_memory"]
    final_memory = tensors["final_memory"]
    raw_centers = load_tensor(
        trajectory_dir / names["raw_kmeans_centers_before_normalization"]
    )
    if tuple(raw_centers.shape) != (MEMORY_SIZE, FEATURE_DIM):
        raise ValueError("wrong raw K-means center shape")
    if not torch.isfinite(raw_centers).all().item():
        raise ValueError("non-finite raw K-means centers")
    normalized_raw_centers = torch.nn.functional.normalize(
        raw_centers.float(), dim=1
    )
    raw_centers = load_tensor(
        trajectory_dir / names["raw_kmeans_centers_before_normalization"]
    )
    if tuple(raw_centers.shape) != (MEMORY_SIZE, FEATURE_DIM):
        raise ValueError("wrong raw K-means center shape")
    if not torch.isfinite(raw_centers).all().item():
        raise ValueError("non-finite raw K-means centers")
    normalized_raw = torch.nn.functional.normalize(raw_centers.float(), dim=1)
    if not torch.allclose(
        normalized_raw, centers.float(), atol=2e-4, rtol=2e-4
    ):
        raise ValueError(
            "recorded actual K-means centers do not equal normalized raw sklearn centers"
        )
    final_keys = load_tensor(artifacts / "keys.pt")
    if not torch.equal(centers, post):
        raise ValueError(
            "post-replacement Memory is not the actual assigned K-means centers"
        )
    if not torch.allclose(
        normalized_raw_centers, centers.float(), atol=1e-7, rtol=1e-6
    ):
        raise ValueError("actual centers do not equal normalized raw sklearn centers")
    if not torch.equal(final_memory, final_keys):
        raise ValueError("final trajectory Memory and final keys.pt differ")
    for stage, tensor in tensors.items():
        norms = torch.linalg.norm(tensor.float(), dim=1)
        if not torch.allclose(norms, torch.ones_like(norms), atol=2e-4, rtol=2e-4):
            raise ValueError("Memory rows are not normalized for %s" % stage)

    trace_path = trajectory_dir / names["warmup_query_trace"]
    with np.load(str(trace_path), allow_pickle=False) as trace:
        required = {
            "queries",
            "priority",
            "global_query_index",
            "dataset_index",
            "batch_index",
            "batch_item_index",
            "spatial_y",
            "spatial_x",
            "nearest_slot",
            "nearest_similarity",
            "nearest_cosine_distance",
            "nearest_l2_distance",
            "video_name",
            "sequence_start_filename",
            "prediction_target_filename",
            "sequence_start_relative_path",
            "prediction_target_relative_path",
            "query_count_full",
            "query_sample_count",
            "model_epoch",
            "schema_version",
            "source_split",
            "sample_seed",
            "kmeans_seed",
            "memory_size",
            "feature_dim",
            "normalized_before_kmeans",
            "loader_shuffle",
        }
        if not required.issubset(set(trace.files)):
            raise ValueError("query trace fields are incomplete")
        queries = np.asarray(trace["queries"], dtype=np.float32)
        global_index = np.asarray(trace["global_query_index"], dtype=np.int64)
        nearest_slot = np.asarray(trace["nearest_slot"], dtype=np.int64)
        nearest_similarity = np.asarray(trace["nearest_similarity"], dtype=np.float32)
        cosine_distance = np.asarray(trace["nearest_cosine_distance"], dtype=np.float32)
        l2_distance = np.asarray(trace["nearest_l2_distance"], dtype=np.float32)
        full_count = int(np.asarray(trace["query_count_full"]).item())
        sample_count = int(np.asarray(trace["query_sample_count"]).item())
        model_epoch = int(np.asarray(trace["model_epoch"]).item())
        schema_version = str(np.asarray(trace["schema_version"]).item())
        source_split = str(np.asarray(trace["source_split"]).item())
        sample_seed = int(np.asarray(trace["sample_seed"]).item())
        trace_kmeans_seed = int(np.asarray(trace["kmeans_seed"]).item())
        trace_memory_size = int(np.asarray(trace["memory_size"]).item())
        trace_feature_dim = int(np.asarray(trace["feature_dim"]).item())
        normalized_before_kmeans = int(
            np.asarray(trace["normalized_before_kmeans"]).item()
        )
        loader_shuffle = int(np.asarray(trace["loader_shuffle"]).item())
        priority = np.asarray(trace["priority"], dtype=np.float64)
        provenance_lengths = {
            name: len(np.asarray(trace[name]))
            for name in (
                "video_name",
                "sequence_start_filename",
                "prediction_target_filename",
                "sequence_start_relative_path",
                "prediction_target_relative_path",
            )
        }

    if queries.shape != (sample_count, FEATURE_DIM) or sample_count <= 0:
        raise ValueError("query sample shape/count mismatch")
    if full_count != int(state["query_count_full"]) or full_count < sample_count:
        raise ValueError("full query count mismatch")
    if model_epoch != WARMUP:
        raise ValueError("query trace is not from the warm-up model")
    if schema_version != "memory-trajectory-v2" or source_split != "normal_training":
        raise ValueError("query trace schema/source mismatch")
    if trace_kmeans_seed != SEED or sample_seed != int(state["trajectory_sample_seed"]):
        raise ValueError("query trace seed mismatch")
    if trace_memory_size != MEMORY_SIZE or trace_feature_dim != FEATURE_DIM:
        raise ValueError("query trace Memory/feature dimension mismatch")
    if normalized_before_kmeans != 1 or loader_shuffle != 0:
        raise ValueError("query trace preprocessing/loader provenance mismatch")
    if any(length != sample_count for length in provenance_lengths.values()):
        raise ValueError("query source-path provenance length mismatch")
    if not np.all(global_index[1:] > global_index[:-1]):
        raise ValueError("query provenance indices are not unique and ordered")
    if global_index[0] < 0 or global_index[-1] >= full_count:
        raise ValueError("query provenance index is outside full traversal")
    all_priorities = np.random.RandomState(sample_seed).random_sample(full_count)
    expected_chosen = np.argpartition(
        all_priorities, sample_count - 1
    )[:sample_count]
    expected_chosen.sort()
    if not np.array_equal(global_index, expected_chosen):
        raise ValueError("stored queries are not the deterministic priority sample")
    if not np.array_equal(priority, all_priorities[global_index]):
        raise ValueError("stored sampling priorities do not reproduce from the seed")
    if not np.isfinite(queries).all():
        raise ValueError("query sample contains non-finite values")
    if not np.allclose(np.linalg.norm(queries, axis=1), 1.0, atol=2e-4, rtol=2e-4):
        raise ValueError("query sample is not normalized")

    centers_numpy = post.numpy().astype(np.float32, copy=False)
    similarities = np.matmul(queries, centers_numpy.T)
    expected_slot = np.argmax(similarities, axis=1).astype(np.int64)
    row = np.arange(sample_count)
    expected_cosine = (1.0 - similarities[row, expected_slot]).astype(np.float32)
    expected_l2 = np.linalg.norm(
        queries - centers_numpy[expected_slot], axis=1
    ).astype(np.float32)
    if not np.array_equal(expected_slot, nearest_slot):
        raise ValueError("stored nearest-slot assignments do not match actual centers")
    if not np.allclose(
        similarities[row, expected_slot], nearest_similarity, atol=1e-6, rtol=1e-5
    ):
        raise ValueError("stored nearest similarities do not match actual centers")
    if not np.allclose(expected_cosine, cosine_distance, atol=1e-6, rtol=1e-5):
        raise ValueError("stored cosine distances do not match actual centers")
    if not np.allclose(expected_l2, l2_distance, atol=1e-6, rtol=1e-5):
        raise ValueError("stored L2 distances do not match actual centers")

    report = {
        "state": "passed",
        "passed": True,
        "overall_pass": True,
        "schema_version": state["schema_version"],
        "task_id": TASK_ID,
        "dataset": "ped2",
        "family": "KMemory",
        "evidence_family": EVIDENCE_FAMILY,
        "branch": "P",
        "seed": SEED,
        "same_training_trajectory": True,
        "warmup_epoch": WARMUP,
        "memory_shape": [MEMORY_SIZE, FEATURE_DIM],
        "query_count_full": full_count,
        "query_sample_count": sample_count,
        "normalized_raw_centers_match_actual_and_post_replacement": True,
        "deterministic_priority_sample_recomputed": True,
        "final_memory_equal_final_keys": True,
        "nearest_slot_and_distance_recomputed": True,
        "checked_at": now(),
    }
    write_json(output, report)
    return report


def command_output(command: List[str]) -> str:
    completed = subprocess.run(command, text=True, capture_output=True)
    return (completed.stdout or completed.stderr).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_id", default=TASK_ID)
    parser.add_argument("--task_dir", required=True)
    parser.add_argument("--project_dir", required=True)
    parser.add_argument("--python_bin", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--evidence_batch_size", type=int, default=8)
    parser.add_argument("--trajectory_sample_count", type=int, default=4096)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.task_id != TASK_ID:
        raise ValueError("task_id=%s required" % TASK_ID)
    if args.trajectory_sample_count <= 0:
        raise ValueError("trajectory_sample_count must be positive")
    task_dir = Path(args.task_dir).resolve()
    project_dir = Path(args.project_dir).resolve()
    package_dir = Path(__file__).resolve().parent
    artifacts = task_dir / "artifacts"
    logs = task_dir / "logs"
    scores = task_dir / "scores"
    metrics_dir = task_dir / "metrics"
    for directory in (artifacts, logs, scores, metrics_dir):
        directory.mkdir(parents=True, exist_ok=True)

    exp_parent = project_dir / "exp" / "ped2"
    exp_parent.mkdir(parents=True, exist_ok=True)
    exp_link = exp_parent / args.task_id
    if exp_link.exists() or exp_link.is_symlink():
        if exp_link.resolve() != artifacts:
            raise FileExistsError(exp_link)
    else:
        exp_link.symlink_to(artifacts, target_is_directory=True)

    train_command = [
        args.python_bin,
        str(package_dir / "ped2_Train_Pred_trajectory.py"),
        "--dataset_path", args.dataset_path,
        "--dataset_type", "ped2",
        "--exp_dir", args.task_id,
        "--epochs", str(EPOCHS),
        "--scheduler_t_max", str(EPOCHS),
        "--batch_size", "4",
        "--num_workers", str(args.num_workers),
        "--device", "cuda",
        "--seed", str(SEED),
        "--save_every", "0",
        "--kmeans_init",
        "--kmeans_warmup_epochs", str(WARMUP),
        "--kmeans_seed", str(SEED),
        "--loss_compact", "0.1",
        "--loss_separate", "0.1",
        "--h", "256",
        "--w", "256",
        "--c", "3",
        "--lr", "0.0002",
        "--t_length", "5",
        "--fdim", str(FEATURE_DIM),
        "--mdim", str(FEATURE_DIM),
        "--msize", str(MEMORY_SIZE),
        "--alpha", "0.6",
        "--trajectory_sample_count", str(args.trajectory_sample_count),
    ]
    config = {
        "task_id": args.task_id,
        "task_dir": str(task_dir),
        "project_dir": str(project_dir),
        "dataset_path": args.dataset_path,
        "dataset_type": "ped2",
        "branch": "P",
        "family": "KMemory",
        "evidence_family": EVIDENCE_FAMILY,
        "memory_update_policy": "Continue",
        "query_source_split": "normal_training",
        "kmeans_uses_test_labels": False,
        "seed": SEED,
        "epochs": EPOCHS,
        "warmup": WARMUP,
        "memory_size": MEMORY_SIZE,
        "feature_dim": FEATURE_DIM,
        "batch_size": 4,
        "learning_rate": 2e-4,
        "loss_compact": 0.1,
        "loss_separate": 0.1,
        "kmeans_n_init": 10,
        "kmeans_batch_size": 4096,
        "kmeans_query_policy": "all normalized normal-training queries at epoch 26",
        "post_kmeans_memory_policy": "continue updating",
        "test_time_memory_update_for_raw_evidence": False,
        "trajectory_sample_count": args.trajectory_sample_count,
        "started_at": now(),
        "algorithm_change": "none",
    }
    write_json(task_dir / "config.json", config)
    write_json(task_dir / "commands.json", {"train": train_command})
    write_json(task_dir / "environment.json", {
        "captured_at": now(),
        "host": platform.node(),
        "platform": platform.platform(),
        "python": command_output([args.python_bin, "--version"]),
        "nvidia_smi": command_output(["nvidia-smi", "-L"]),
    })
    write_json(task_dir / "status.json", {
        "task_id": args.task_id,
        "state": "running",
        "stage": "Ped2-P KMemory-Continue same-trajectory training",
        "time": now(),
    })

    stop = threading.Event()
    monitor = threading.Thread(target=sample_gpu, args=(stop, logs / "gpu_usage.csv"), daemon=True)
    monitor.start()
    try:
        run_logged(train_command, project_dir, logs / "train.log", package_dir)
        loss_rows = parse_losses(logs / "train.log", metrics_dir / "loss_by_epoch.csv")
        model_path = artifacts / "model.pth"
        keys_path = artifacts / "keys.pt"
        if not model_path.is_file() or not keys_path.is_file():
            raise FileNotFoundError("final model.pth or keys.pt missing")
        trajectory_report = validate_trajectory(
            artifacts, metrics_dir / "trajectory_validation.json"
        )

        official_score = scores / "test_official_scores.npz"
        eval_command = [
            args.python_bin,
            "Evaluate_Pred.py",
            "--dataset_path", args.dataset_path,
            "--dataset_type", "ped2",
            "--model_dir", str(model_path),
            "--m_items_dir", str(keys_path),
            "--score_output", str(official_score),
            "--alpha", "0.6",
            "--th", "0.01",
            "--num_workers_test", str(args.num_workers),
            "--device", "cuda",
        ]
        run_logged(eval_command, project_dir, logs / "evaluate_official.log", package_dir)

        export_commands = {}
        for split in ("training", "testing"):
            raw_output = scores / (split + "_raw_evidence.npz")
            export_command = [
                args.python_bin,
                "export_split_evidence.py",
                "--branch", "P",
                "--split", split,
                "--dataset_type", "ped2",
                "--dataset_path", args.dataset_path,
                "--model_dir", str(model_path),
                "--m_items_dir", str(keys_path),
                "--output", str(raw_output),
                "--batch_size", str(args.evidence_batch_size),
                "--num_workers", str(args.num_workers),
                "--device", "cuda",
            ]
            export_commands[split] = export_command
            run_logged(
                export_command,
                project_dir,
                logs / ("export_%s.log" % split),
                package_dir,
            )
            tag_evidence_identity(raw_output, args.task_id)
        write_json(task_dir / "commands.json", {
            "train": train_command,
            "evaluate_official": eval_command,
            "export_raw_evidence": export_commands,
            "post_export_identity_annotation": {
                "implementation": "memory_trajectory_v2.production_contract.tag_evidence_identity",
                "family": EVIDENCE_FAMILY,
                "task_id": TASK_ID,
                "changes_numeric_evidence": False,
            },
        })

        results = {
            "official_online_update": score_metrics(official_score),
            "raw_no_update_official_style": score_metrics(
                scores / "testing_raw_evidence.npz"
            ),
        }
        write_json(metrics_dir / "metrics.json", results)
        with (metrics_dir / "metrics.csv").open(
            "w", encoding="utf-8-sig", newline=""
        ) as target:
            writer = csv.DictWriter(
                target,
                fieldnames=("protocol", "AUROC", "AUPRC", "EER", "EER_threshold", "frames"),
            )
            writer.writeheader()
            for protocol, values in results.items():
                writer.writerow({"protocol": protocol, **values})

        validator = project_dir / "validate_evidence_bundle.py"
        if not validator.is_file():
            raise FileNotFoundError("validate_evidence_bundle.py is required")
        run_logged([
            args.python_bin,
            str(validator),
            "--model", str(model_path),
            "--keys", str(keys_path),
            "--config", str(task_dir / "config.json"),
            "--training", str(scores / "training_raw_evidence.npz"),
            "--testing", str(scores / "testing_raw_evidence.npz"),
            "--candidate-dir", str(artifacts),
            "--output", str(task_dir / "completeness_report.json"),
        ], project_dir, logs / "validate_evidence.log", package_dir)
        completeness = json.loads((task_dir / "completeness_report.json").read_text(encoding="utf-8"))
        if not report_passed(completeness):
            raise RuntimeError("evidence completeness report did not pass")

        stop.set()
        monitor.join(timeout=10)
        write_json(task_dir / "status.json", {
            "task_id": args.task_id,
            "state": "completed",
            "stage": "training, trajectory, inference and evidence all validated",
            "time": now(),
            "loss_epochs": len(loss_rows),
            "trajectory": trajectory_report,
            "metrics": results,
        })
        manifest = []
        for path in sorted(
            p
            for p in task_dir.rglob("*")
            if p.is_file()
            and p.name not in {"manifest_sha256.csv", "manifest_sha256.csv.tmp"}
        ):
            manifest.append({
                "relative_path": str(path.relative_to(task_dir)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            })
        manifest_path = task_dir / "manifest_sha256.csv"
        temporary = manifest_path.with_suffix(".csv.tmp")
        with temporary.open("w", encoding="utf-8-sig", newline="") as target:
            writer = csv.DictWriter(
                target, fieldnames=("relative_path", "bytes", "sha256")
            )
            writer.writeheader()
            writer.writerows(manifest)
        temporary.replace(manifest_path)
        validate_outer_manifest_gate(task_dir, manifest_path)
    except Exception as error:
        write_json(task_dir / "status.json", {
            "task_id": args.task_id,
            "state": "failed",
            "time": now(),
            "error": repr(error),
        })
        raise
    finally:
        stop.set()
        monitor.join(timeout=10)


if __name__ == "__main__":
    main()

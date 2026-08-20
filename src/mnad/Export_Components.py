#!/usr/bin/env python3
"""Export label-free train-normal evidence and matching raw test evidence.

The official MNAD evaluators min-max normalise inside every test video.  That
representation cannot be compared directly with a normal training
distribution.  This exporter therefore preserves the raw per-frame pixel
error/PSNR and nearest-memory distance.  It also recreates the official
per-video score for audit purposes, but NCBAF calibration must use the raw
arrays only.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms

from utils import torch_load_compat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch", choices=("R", "P"), required=True)
    parser.add_argument("--split", choices=("training", "testing"), required=True)
    parser.add_argument("--dataset_type", choices=("ped2", "avenue", "shanghai"), required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--m_items_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument(
        "--feature_sample_count", type=int, default=50000,
        help="number of normal-training query features to retain; testing never samples features",
    )
    parser.add_argument("--feature_sample_seed", type=int, default=2020)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    if args.split == "testing":
        args.feature_sample_count = 0
    if args.alpha is None:
        args.alpha = 0.7 if args.branch == "R" or args.dataset_type == "avenue" else 0.6
    if args.batch_size < 1 or args.num_workers < 0 or args.feature_sample_count < 0:
        parser.error("batch_size must be positive; worker/sample counts must be non-negative")
    return args


def minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    span = float(values.max() - values.min())
    if not np.isfinite(span) or span <= np.finfo(np.float64).eps:
        return np.zeros_like(values, dtype=np.float64)
    return (values - values.min()) / span


def load_labels(dataset_type: str, project_dir: Path, video_names: list[str],
                full_lengths: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray]:
    labels = np.asarray(
        np.load(project_dir / "data" / f"frame_labels_{dataset_type}.npy"),
        dtype=np.int8,
    ).reshape(-1)
    if int(full_lengths.sum()) != labels.size:
        raise ValueError(
            f"label/frame mismatch: labels={labels.size}, frames={int(full_lengths.sum())}"
        )
    result: dict[str, np.ndarray] = {}
    offset = 0
    for name, length in zip(video_names, full_lengths.tolist()):
        result[name] = labels[offset:offset + length]
        offset += length
    return result, labels


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.feature_sample_seed)
    np.random.seed(args.feature_sample_seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    if args.branch == "P":
        from model.utils_pred import DataLoader
        time_step, num_pred = 4, 1
    else:
        from model.utils import DataLoader
        time_step, num_pred = 1, 0

    split_dir = Path(args.dataset_path) / args.dataset_type / args.split / "frames"
    dataset = DataLoader(
        str(split_dir), transforms.Compose([transforms.ToTensor()]),
        resize_height=256, resize_width=256, time_step=time_step, num_pred=num_pred,
    )
    loader = data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, drop_last=False, pin_memory=device.type == "cuda",
    )

    video_names = list(dataset.videos.keys())
    full_lengths = np.asarray(
        [dataset.videos[name]["length"] for name in video_names], dtype=np.int64
    )
    sample_video_ids: list[int] = []
    sample_frame_indices: list[int] = []
    score_lengths: list[int] = []
    target_offset = 4 if args.branch == "P" else 0
    for video_id, name in enumerate(video_names):
        sample_count = int(full_lengths[video_id]) - time_step - num_pred + 1
        score_lengths.append(sample_count)
        sample_video_ids.extend([video_id] * sample_count)
        sample_frame_indices.extend(range(target_offset, target_offset + sample_count))
    if len(sample_video_ids) != len(dataset):
        raise RuntimeError("sample metadata does not match loader length")

    model = torch_load_compat(args.model_dir, map_location=device)
    model.to(device).eval()
    keys = torch_load_compat(args.m_items_dir, map_location=device).to(device)
    model_params = sum(parameter.numel() for parameter in model.parameters())

    pixel_mse_parts: list[np.ndarray] = []
    memory_distance_parts: list[np.ndarray] = []
    assignment_counts = np.zeros(int(keys.shape[0]), dtype=np.int64)
    sampled_features: list[np.ndarray] = []
    sampled_assignments: list[np.ndarray] = []
    sampled_distances: list[np.ndarray] = []
    rng = np.random.default_rng(args.feature_sample_seed)
    per_frame_feature_samples = (
        max(1, math.ceil(args.feature_sample_count / max(1, len(dataset))))
        if args.feature_sample_count else 0
    )
    sampled_frame_indices: set[int] = set()
    if args.feature_sample_count:
        sampled_frame_indices = set(
            rng.choice(
                len(dataset), size=min(args.feature_sample_count, len(dataset)), replace=False
            ).tolist()
        )
    sampled_feature_count = 0

    start = time.perf_counter()
    seen = 0
    with torch.inference_mode():
        for imgs in loader:
            imgs = imgs.to(device, non_blocking=device.type == "cuda")
            if args.branch == "P":
                model_input, target = imgs[:, :12], imgs[:, 12:]
            else:
                model_input = target = imgs
            result = model.forward(model_input, keys, False)
            outputs, query, top1_keys, keys_ind = result[0], result[6], result[7], result[8]
            batch_size = int(outputs.shape[0])

            pixel_mse = torch.mean(
                (((outputs + 1.0) / 2.0) - ((target + 1.0) / 2.0)) ** 2,
                dim=(1, 2, 3),
            )
            queries_per_frame = int(query.shape[0] // batch_size)
            feature_sq = torch.mean((query - top1_keys) ** 2, dim=1)
            memory_distance = feature_sq.reshape(batch_size, queries_per_frame).mean(dim=1)
            assignments = keys_ind.reshape(batch_size, queries_per_frame)

            pixel_mse_parts.append(pixel_mse.cpu().numpy().astype(np.float64))
            memory_distance_parts.append(memory_distance.cpu().numpy().astype(np.float64))
            assignment_counts += np.bincount(
                assignments.cpu().numpy().reshape(-1), minlength=assignment_counts.size
            )[:assignment_counts.size]

            if per_frame_feature_samples:
                query_3d = query.reshape(batch_size, queries_per_frame, -1)
                feature_sq_2d = feature_sq.reshape(batch_size, queries_per_frame)
                for row in range(batch_size):
                    if seen + row not in sampled_frame_indices:
                        continue
                    remaining = args.feature_sample_count - sampled_feature_count
                    take = min(per_frame_feature_samples, queries_per_frame, remaining)
                    if take <= 0:
                        break
                    indices = rng.choice(queries_per_frame, size=take, replace=False)
                    sampled_features.append(
                        query_3d[row, indices].cpu().numpy().astype(np.float16)
                    )
                    sampled_assignments.append(
                        assignments[row, indices].cpu().numpy().astype(np.int16)
                    )
                    sampled_distances.append(
                        feature_sq_2d[row, indices].cpu().numpy().astype(np.float32)
                    )
                    sampled_feature_count += take
            seen += batch_size
            if seen % 10000 < batch_size:
                print(f"processed={seen}/{len(dataset)}", flush=True)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    pixel_mse = np.concatenate(pixel_mse_parts)
    memory_distance = np.concatenate(memory_distance_parts)
    pixel_psnr = 10.0 * np.log10(1.0 / np.maximum(pixel_mse, np.finfo(np.float64).tiny))

    official_parts: list[np.ndarray] = []
    cursor = 0
    for length in score_lengths:
        psnr_part = pixel_psnr[cursor:cursor + length]
        distance_part = memory_distance[cursor:cursor + length]
        pixel_anomaly = 1.0 - minmax(psnr_part)
        memory_anomaly = minmax(distance_part)
        official_parts.append(args.alpha * pixel_anomaly + (1.0 - args.alpha) * memory_anomaly)
        cursor += length
    official_scores = np.concatenate(official_parts).astype(np.float64)

    video_ids = np.asarray(sample_video_ids, dtype=np.int32)
    frame_indices = np.asarray(sample_frame_indices, dtype=np.int32)
    full_video_labels = np.zeros(int(full_lengths.sum()), dtype=np.int8)
    if args.split == "testing":
        labels_by_video, full_video_labels = load_labels(
            args.dataset_type, Path(__file__).resolve().parent, video_names, full_lengths
        )
        labels = np.asarray(
            [labels_by_video[video_names[vid]][frame]
             for vid, frame in zip(video_ids.tolist(), frame_indices.tolist())],
            dtype=np.int8,
        )
    else:
        labels = np.zeros(len(dataset), dtype=np.int8)

    full_video_offsets = np.concatenate((
        np.asarray([0], dtype=np.int64), np.cumsum(full_lengths, dtype=np.int64)
    ))
    video_labels = np.asarray([
        int(np.any(full_video_labels[full_video_offsets[i]:full_video_offsets[i + 1]] == 1))
        for i in range(len(video_names))
    ], dtype=np.int8)
    score_offsets = np.concatenate((
        np.asarray([0], dtype=np.int64),
        np.cumsum(np.asarray(score_lengths, dtype=np.int64), dtype=np.int64),
    ))

    feature_samples = np.empty((0, int(keys.shape[1])), dtype=np.float16)
    feature_assignments = np.empty((0,), dtype=np.int16)
    feature_distances = np.empty((0,), dtype=np.float32)
    if sampled_features:
        feature_samples = np.concatenate(sampled_features)[:args.feature_sample_count]
        feature_assignments = np.concatenate(sampled_assignments)[:args.feature_sample_count]
        feature_distances = np.concatenate(sampled_distances)[:args.feature_sample_count]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        schema_version=np.asarray("raw-evidence-v1"),
        dataset=np.asarray(args.dataset_type),
        branch=np.asarray(args.branch),
        split=np.asarray(args.split),
        video_names=np.asarray(video_names),
        full_video_lengths=full_lengths,
        full_video_offsets=full_video_offsets,
        full_video_labels=full_video_labels,
        video_labels=video_labels,
        lengths=np.asarray(score_lengths, dtype=np.int64),
        score_offsets=score_offsets,
        video_ids=video_ids,
        frame_indices=frame_indices,
        rp_alignment_video_ids=video_ids,
        rp_alignment_frame_indices=frame_indices,
        labels=labels,
        pixel_mse=pixel_mse,
        pixel_psnr=pixel_psnr,
        memory_distance=memory_distance,
        official_per_video_scores=official_scores,
        raw_branch_scores=official_scores,
        alpha=np.asarray(args.alpha, dtype=np.float64),
        memory_assignment_counts=assignment_counts,
        memory_items=keys.detach().cpu().numpy().astype(np.float32),
        feature_samples=feature_samples,
        feature_assignments=feature_assignments,
        feature_distances=feature_distances,
    )

    profile = {
        "schema_version": "raw-evidence-v1",
        "dataset": args.dataset_type,
        "branch": args.branch,
        "split": args.split,
        "frames_scored": int(len(dataset)),
        "elapsed_seconds": elapsed,
        "frames_per_second": float(len(dataset) / elapsed),
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "model_parameters": int(model_params),
        "new_ncbaf_parameters": 0,
        "peak_cuda_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "peak_cuda_reserved_bytes": (
            int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0
        ),
        "model_path": str(Path(args.model_dir).resolve()),
        "keys_path": str(Path(args.m_items_dir).resolve()),
        "output_path": str(output.resolve()),
        "feature_samples_saved": int(feature_samples.shape[0]),
        "test_time_memory_update": False,
    }
    output.with_suffix(".profile.json").write_text(
        json.dumps(profile, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(profile, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

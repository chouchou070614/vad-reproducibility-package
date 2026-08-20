#!/usr/bin/env python3
"""Run the frozen decision path from exported normal/test branch evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.decision.frozen_pipeline import (  # noqa: E402
    DATASET_CONFIGS,
    branch_score,
    ecdf_tail,
    exact_align,
    fixed_fuse,
    refine_per_video,
)


def load_branch(path: Path, *, require_labels: bool) -> dict:
    with np.load(path, allow_pickle=False) as bundle:
        required = {"video_names", "frame_indices", "pixel_mse", "memory_distance"}
        if require_labels:
            required.add("labels")
        missing = required.difference(bundle.files)
        if missing:
            raise ValueError(f"{path}: missing arrays {sorted(missing)}")
        return {name: np.asarray(bundle[name]) for name in required}


def aligned_labels(test_r: dict, video: np.ndarray, frame: np.ndarray) -> np.ndarray:
    lookup = {
        (str(v), int(f)): int(y)
        for v, f, y in zip(test_r["video_names"], test_r["frame_indices"], test_r["labels"])
    }
    return np.asarray([lookup[(str(v), int(f))] for v, f in zip(video, frame)], dtype=np.uint8)


def video_ids(video: np.ndarray) -> np.ndarray:
    mapping = {}
    output = []
    for value in video.astype(str).tolist():
        if value not in mapping:
            mapping[value] = len(mapping)
        output.append(mapping[value])
    return np.asarray(output, dtype=np.int64)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=tuple(DATASET_CONFIGS), required=True)
    parser.add_argument("--normal-r", type=Path, required=True)
    parser.add_argument("--normal-p", type=Path, required=True)
    parser.add_argument("--test-r", type=Path, required=True)
    parser.add_argument("--test-p", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = DATASET_CONFIGS[args.dataset]
    normal_r, normal_p = load_branch(args.normal_r, require_labels=False), load_branch(args.normal_p, require_labels=False)
    test_r, test_p = load_branch(args.test_r, require_labels=True), load_branch(args.test_p, require_labels=False)

    normal_r_score = branch_score(normal_r["pixel_mse"], normal_r["memory_distance"], video_ids(normal_r["video_names"]), config.alpha_r)
    normal_p_score = branch_score(normal_p["pixel_mse"], normal_p["memory_distance"], video_ids(normal_p["video_names"]), config.alpha_p)
    test_r_score = branch_score(test_r["pixel_mse"], test_r["memory_distance"], video_ids(test_r["video_names"]), config.alpha_r)
    test_p_score = branch_score(test_p["pixel_mse"], test_p["memory_distance"], video_ids(test_p["video_names"]), config.alpha_p)

    z_r = ecdf_tail(normal_r_score, test_r_score)
    z_p = ecdf_tail(normal_p_score, test_p_score)
    video, frame, aligned_r, aligned_p = exact_align(
        test_r["video_names"], test_r["frame_indices"], z_r,
        test_p["video_names"], test_p["frame_indices"], z_p,
    )
    ids = video_ids(video)
    fused = fixed_fuse(aligned_r, aligned_p, config.weight_r)
    final = refine_per_video(fused, ids, config.smoother, config.parameter)
    labels = aligned_labels(test_r, video, frame)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        dataset=args.dataset,
        video_names=video,
        video_ids=ids,
        frame_indices=frame,
        labels=labels,
        ECDF_R=aligned_r,
        ECDF_P=aligned_p,
        ECDF_Static_locked=fused,
        ECDF_Final_temporal=final,
        weight_R=config.weight_r,
        smoother=config.smoother,
        smoother_parameter=config.parameter,
    )
    print(json.dumps({"output": str(args.output), "frames": int(final.size), "config": config.__dict__}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

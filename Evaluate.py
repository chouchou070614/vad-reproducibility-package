#!/usr/bin/env python3
"""Export raw score components for one branch and data split."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def load_config() -> dict:
    return json.loads((ROOT / "configs" / "BRANCH_TRAINING.json").read_text(encoding="utf-8"))


def build_command(args: argparse.Namespace) -> list[str]:
    config = load_config()
    alpha = config[args.dataset][f"alpha_{args.branch}"]
    return [
        sys.executable,
        str(ROOT / "src" / "mnad" / "Export_Components.py"),
        "--branch", args.branch,
        "--split", args.split,
        "--dataset_type", args.dataset,
        "--dataset_path", str(Path(args.dataset_root)),
        "--model_dir", str(Path(args.model)),
        "--m_items_dir", str(Path(args.memory)),
        "--output", str(Path(args.output)),
        "--batch_size", str(args.batch_size),
        "--num_workers", str(args.num_workers),
        "--alpha", str(alpha),
        "--feature_sample_count", str(args.feature_sample_count),
        "--feature_sample_seed", str(args.seed),
        "--device", args.device,
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export KMTF-VAD branch components")
    parser.add_argument("--branch", choices=("R", "P"), required=True)
    parser.add_argument("--split", choices=("training", "testing"), required=True)
    parser.add_argument("--dataset", choices=("ped2", "avenue", "shanghai"), required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--memory", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--feature-sample-count", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    command = build_command(args)
    if args.dry_run:
        print(json.dumps(command, indent=2))
        return 0
    return subprocess.run(command, cwd=ROOT / "src" / "mnad", check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())

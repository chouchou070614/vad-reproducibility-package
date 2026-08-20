#!/usr/bin/env python3
"""Canonical KMTF-VAD training entry point built on the MNAD branch code."""

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
    dataset = config[args.dataset]
    branch = config["branches"][args.branch]
    alpha = dataset[f"alpha_{args.branch}"]
    script = "Train_Pred.py" if args.branch == "P" else "Train_recons.py"
    return [
        sys.executable,
        str(ROOT / "src" / "mnad" / script),
        "--gpus", str(args.gpu),
        "--dataset_type", args.dataset,
        "--dataset_path", str(Path(args.dataset_root)),
        "--exp_dir", str(Path(args.output)),
        "--epochs", str(dataset["epochs"]),
        "--h", str(config["common"]["input_height"]),
        "--w", str(config["common"]["input_width"]),
        "--msize", str(config["common"]["memory_slots"]),
        "--alpha", str(alpha),
        "--lr", str(branch["learning_rate"]),
        "--loss_compact", str(branch["loss_compact"]),
        "--loss_separate", str(branch["loss_separate"]),
        "--t_length", str(branch["t_length"]),
        "--kmeans_init",
        "--kmeans_warmup_epochs", str(dataset["warmup_epoch"]),
        "--kmeans_seed", str(args.seed),
        "--device", args.device,
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one frozen KMTF-VAD branch")
    parser.add_argument("--branch", choices=("R", "P"), required=True)
    parser.add_argument("--dataset", choices=("ped2", "avenue", "shanghai"), required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
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


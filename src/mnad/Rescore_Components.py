"""Evaluate pixel/memory score mixtures from saved components.

The alpha sweep reports AUROC using the stored evaluation labels.
"""
import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True, type=Path)
parser.add_argument("--alphas", nargs="+", type=float, default=[0.7])
parser.add_argument("--save-alpha", type=float, default=None)
parser.add_argument("--output", type=Path, default=None)
args = parser.parse_args()

data = np.load(args.input)
required = ("pixel_scores", "memory_scores", "labels")
missing = [key for key in required if key not in data.files]
if missing:
    raise KeyError(
        f"{args.input} is an older score file missing {missing}. "
        "Re-run Evaluate_Pred.py/Evaluate_recons.py once; retraining is not needed.")

pixel = np.asarray(data["pixel_scores"], dtype=np.float64)
memory = np.asarray(data["memory_scores"], dtype=np.float64)
labels = np.asarray(data["labels"], dtype=np.int8)
if not (len(pixel) == len(memory) == len(labels)):
    raise ValueError("pixel, memory and label lengths differ")
if not (np.isfinite(pixel).all() and np.isfinite(memory).all()):
    raise FloatingPointError("non-finite component scores found")

for alpha in args.alphas:
    if not 0.0 <= alpha <= 1.0:
        parser.error("all alphas must be in [0, 1]")
    scores = alpha * pixel + (1.0 - alpha) * memory
    print(f"alpha={alpha:.4f} AUC={roc_auc_score(labels, scores) * 100.0:.4f}%")

if args.save_alpha is not None:
    alpha = args.save_alpha
    if not 0.0 <= alpha <= 1.0:
        parser.error("--save-alpha must be in [0, 1]")
    if args.output is None:
        parser.error("--output is required with --save-alpha")
    scores = alpha * pixel + (1.0 - alpha) * memory
    payload = {key: data[key] for key in data.files}
    payload["scores"] = scores.astype(np.float32)
    payload["alpha"] = np.asarray(alpha, dtype=np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **payload)
    print("Saved:", args.output)

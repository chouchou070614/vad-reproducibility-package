"""Summarize saved checkpoint scores."""
import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


parser = argparse.ArgumentParser()
parser.add_argument("--audit-dir", required=True, type=Path)
parser.add_argument("--output", type=Path, default=None)
args = parser.parse_args()

rows = []
for path in sorted(args.audit_dir.glob("checkpoint_*.npz")):
    data = np.load(path)
    labels = data["labels"]
    scores = data["scores"]
    pixel_auc = roc_auc_score(labels, data["pixel_scores"]) * 100.0
    memory_auc = roc_auc_score(labels, data["memory_scores"]) * 100.0
    used = ""
    if "memory_assignment_counts" in data.files:
        used = int(np.count_nonzero(data["memory_assignment_counts"]))
    rows.append({
        "checkpoint": path.stem,
        "auc_percent": roc_auc_score(labels, scores) * 100.0,
        "pixel_auc_percent": pixel_auc,
        "memory_auc_percent": memory_auc,
        "alpha": float(data["alpha"]) if "alpha" in data.files else "",
        "memory_items_used": used,
    })

if not rows:
    raise FileNotFoundError(f"No checkpoint_*.npz files in {args.audit_dir}")

args.output = args.output or args.audit_dir / "checkpoint_audit.csv"
with args.output.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
    writer.writeheader()
    for row in rows:
        writer.writerow({
            **row,
            "auc_percent": f"{row['auc_percent']:.4f}",
            "pixel_auc_percent": f"{row['pixel_auc_percent']:.4f}",
            "memory_auc_percent": f"{row['memory_auc_percent']:.4f}",
        })

for row in rows:
    print(
        f"{row['checkpoint']:<28} AUC={row['auc_percent']:.4f}% "
        f"pixel={row['pixel_auc_percent']:.4f}% "
        f"memory={row['memory_auc_percent']:.4f}% used={row['memory_items_used']}")
print("Saved:", args.output)

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "evidence" / "freeze" / "ped2_seed2020_final_scores.npz"
EVIDENCE_DIR = ROOT / "evidence" / "score_curve"
FIGURE = ROOT / "figures" / "fig5_score_curve.png"
VIDEO = "02"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


with np.load(SOURCE, allow_pickle=False) as source:
    names = np.asarray(source["video_names"]).astype(str)
    ids = np.asarray(source["video_ids"], dtype=np.int64)
    selected_id = int(np.flatnonzero(names == VIDEO)[0])
    mask = ids == selected_id
    frames = np.asarray(source["frame_indices"], dtype=np.int64)[mask]
    labels = np.asarray(source["labels"], dtype=np.int8)[mask]
    score_r = np.asarray(source["ECDF_R"], dtype=np.float64)[mask]
    score_p = np.asarray(source["ECDF_P"], dtype=np.float64)[mask]
    score_final = np.asarray(source["ECDF_Final_temporal"], dtype=np.float64)[mask]

if not (frames.size == labels.size == score_r.size == score_p.size == score_final.size):
    raise RuntimeError("score-curve evidence arrays are not aligned")
if not np.all(np.diff(frames) == 1):
    raise RuntimeError("the selected video is not a continuous aligned sequence")

EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
csv_path = EVIDENCE_DIR / f"ped2_video_{VIDEO}_continuous_scores.csv"
with csv_path.open("w", encoding="utf-8", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(["frame", "ground_truth", "ecdf_reconstruction", "ecdf_prediction", "final_temporal_score"])
    writer.writerows(zip(frames, labels, score_r, score_p, score_final))

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9.2,
    "axes.labelsize": 9.5,
    "axes.titlesize": 9.8,
    "legend.fontsize": 8.2,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
})
fig = plt.figure(figsize=(7.25, 3.05), dpi=300)
grid = fig.add_gridspec(2, 1, height_ratios=[5.0, 0.62], hspace=0.08)
ax = fig.add_subplot(grid[0])
gt = fig.add_subplot(grid[1], sharex=ax)

positive = np.flatnonzero(labels == 1)
if positive.size:
    starts = positive[np.r_[True, np.diff(positive) > 1]]
    ends = positive[np.r_[np.diff(positive) > 1, True]]
    for start, end in zip(starts, ends):
        left, right = frames[start] - 0.5, frames[end] + 0.5
        ax.axvspan(left, right, color="#D94B4B", alpha=0.115, linewidth=0)
        gt.axvspan(left, right, color="#D94B4B", alpha=0.92, linewidth=0)

ax.plot(frames, score_r, color="#4C78A8", linewidth=1.05, alpha=0.68, label="Reconstruction branch")
ax.plot(frames, score_p, color="#E39C33", linewidth=1.05, alpha=0.68, label="Prediction branch")
ax.plot(frames, score_final, color="#087F5B", linewidth=2.15, label="Final fused score")
ax.set_ylabel("Anomaly score")
ax.set_xlim(frames[0], frames[-1])
upper = max(1.05, float(np.quantile(np.r_[score_r, score_p, score_final], 0.995)) * 1.06)
ax.set_ylim(0.0, upper)
ax.grid(axis="y", color="#D6DBE1", linewidth=0.65, alpha=0.7)
ax.legend(loc="upper left", frameon=True, framealpha=0.95, edgecolor="#D5DAE0", ncol=3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#7D8792")
ax.spines["bottom"].set_color("#7D8792")
ax.tick_params(axis="x", labelbottom=False)

gt.set_ylim(0, 1)
gt.set_yticks([0.5])
gt.set_yticklabels(["GT"])
gt.set_xlabel("Frame index")
gt.tick_params(axis="y", length=0, pad=5)
for spine in gt.spines.values():
    spine.set_visible(False)

fig.subplots_adjust(left=0.095, right=0.995, top=0.985, bottom=0.155)
FIGURE.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(FIGURE, dpi=300, facecolor="white", bbox_inches="tight")
plt.close(fig)

metadata = {
    "dataset": "UCSD Ped2",
    "video": VIDEO,
    "seed": 2020,
    "aligned_frame_start": int(frames[0]),
    "aligned_frame_end": int(frames[-1]),
    "aligned_frame_count": int(frames.size),
    "ground_truth_positive_frames": int(labels.sum()),
    "source_npz": str(SOURCE.relative_to(ROOT)),
    "source_npz_sha256": sha256(SOURCE),
    "csv": str(csv_path.relative_to(ROOT)),
    "csv_sha256": sha256(csv_path),
    "figure": str(FIGURE.relative_to(ROOT)),
    "figure_sha256": sha256(FIGURE),
    "score_fields": ["ECDF_R", "ECDF_P", "ECDF_Final_temporal"],
    "note": "Continuous aligned frames; shaded red interval and GT strip are derived directly from stored labels.",
}
(EVIDENCE_DIR / "score_curve_manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
print(FIGURE)

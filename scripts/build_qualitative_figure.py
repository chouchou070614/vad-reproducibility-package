from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "evidence" / "qualitative_expanded"
OUT = ROOT / "figures" / "fig4_qualitative.png"

columns = [
    ("target", "Target"),
    ("reconstruction_R", "Reconstruction"),
    ("prediction_P", "Prediction"),
    ("error_R", "Reconstruction error"),
    ("error_P", "Prediction error"),
]
rows = [
    ("video_02_frame_0090", "Normal\nv02 / f90"),
    ("video_03_frame_0075", "Cyclists\nv03 / f75"),
    ("video_12_frame_0120", "Skater\nv12 / f120"),
    ("video_04_frame_0100", "Vehicle\nv04 / f100"),
]

fig, axes = plt.subplots(4, 5, figsize=(11.2, 7.65), dpi=240)
for row_idx, (prefix, row_label) in enumerate(rows):
    for col_idx, (suffix, title) in enumerate(columns):
        ax = axes[row_idx, col_idx]
        image = Image.open(SRC / f"{prefix}_{suffix}.png").convert("RGB")
        ax.imshow(image)
        ax.set_xticks([])
        ax.set_yticks([])
        if row_idx == 0:
            ax.set_title(title, fontsize=10.5, fontweight="semibold", pad=5)
        for spine in ax.spines.values():
            spine.set_color("#C7CDD3")
            spine.set_linewidth(0.9)
    axes[row_idx, 0].set_ylabel(
        row_label,
        fontsize=10.5,
        fontweight="semibold",
        rotation=90,
        labelpad=7,
    )

fig.text(
    0.745,
    0.012,
    "Error maps share a fixed [0, 0.25] display scale.",
    ha="center",
    va="bottom",
    fontsize=8.8,
    color="#4B5563",
)
fig.subplots_adjust(left=0.072, right=0.995, top=0.905, bottom=0.065, wspace=0.055, hspace=0.095)
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(OUT)

# Task-Adapted Memory Reorganization and Complementary Dual-Branch Scoring for Video Anomaly Detection

PyTorch implementation.

## Installation

Python 3.9, PyTorch 2.0.1, and CUDA 11.8 are used in the experiments.

```bash
conda env create -f environments/nvidia-cu118.yml
conda activate kmtf-vad-cu118
```

## Data

Download UCSD Ped2, CUHK Avenue, and ShanghaiTech from their official sources.
Extract the video frames using the following layout:

```text
dataset/
  ped2/{training,testing}/frames/<video-id>/*
  avenue/{training,testing}/frames/<video-id>/*
  shanghai/{training,testing}/frames/<video-id>/*
```

See [docs/DATASETS.md](docs/DATASETS.md) for the Avenue conversion script.

## Training

Train the prediction and reconstruction branches separately.

```bash
python Train.py --branch P --dataset ped2 \
  --dataset-root ./dataset --output ./experiments/ped2_P

python Train.py --branch R --dataset ped2 \
  --dataset-root ./dataset --output ./experiments/ped2_R
```

Replace `ped2` with `avenue` or `shanghai`. Training settings are in
`configs/BRANCH_TRAINING.json`.

## Evaluation

Export training and testing scores for both branches.

```bash
python Evaluate.py --branch R --split training --dataset ped2 \
  --dataset-root ./dataset --model R_MODEL.pth --memory R_KEYS.pt \
  --output R_train.npz

python Evaluate.py --branch R --split testing --dataset ped2 \
  --dataset-root ./dataset --model R_MODEL.pth --memory R_KEYS.pt \
  --output R_test.npz

python Evaluate.py --branch P --split training --dataset ped2 \
  --dataset-root ./dataset --model P_MODEL.pth --memory P_KEYS.pt \
  --output P_train.npz

python Evaluate.py --branch P --split testing --dataset ped2 \
  --dataset-root ./dataset --model P_MODEL.pth --memory P_KEYS.pt \
  --output P_test.npz
```

Fuse the two branches with:

```bash
python Fuse.py --dataset ped2 \
  --normal-r R_train.npz --normal-p P_train.npz \
  --test-r R_test.npz --test-p P_test.npz \
  --output ped2_scores.npz
```

Fusion settings are in `configs/DATASET_CONFIG.json`. Additional commands are
in [docs/REPRODUCE.md](docs/REPRODUCE.md).

## Figures

```bash
python scripts/build_qualitative_figure.py
python scripts/build_score_curve.py
```

Figure inputs are in `evidence/`.

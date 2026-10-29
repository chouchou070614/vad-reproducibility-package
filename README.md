# Task-Adapted Memory Reorganization and Complementary Dual-Branch Scoring for Video Anomaly Detection

PyTorch implementation of KMTF-VAD.

## Setup

The NVIDIA environment uses Python 3.9, PyTorch 2.0.1, and CUDA 11.8.
Run the commands below from the repository root.

```bash
conda env create -f environments/nvidia-cu118.yml
conda activate kmtf-vad-cu118
```

Prepare Ped2, Avenue, or ShanghaiTech as described in
[Dataset preparation](docs/DATASETS.md). Full datasets and pretrained
checkpoints are not included.

## Training

Train the prediction and reconstruction branches separately:

```bash
python Train.py --branch P --dataset ped2 \
  --dataset-root "$PWD/dataset" --output "$PWD/experiments/ped2/P"

python Train.py --branch R --dataset ped2 \
  --dataset-root "$PWD/dataset" --output "$PWD/experiments/ped2/R"
```

Use `ped2`, `avenue`, or `shanghai` for `--dataset`. Training parameters are in
[`configs/BRANCH_TRAINING.json`](configs/BRANCH_TRAINING.json).

## Evaluation

Follow [Reproducing the experiments](docs/REPRODUCE.md) to export branch
components, fuse scores, and regenerate the paper figures.

This implementation builds on MNAD and MNADrc; see the
[third-party notice](licenses/THIRD_PARTY_NOTICE.md).

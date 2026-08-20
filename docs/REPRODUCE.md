# Reproduction guide

## 1. Create the environment

For the NVIDIA results reported in the paper:

```bash
conda env create -f environments/nvidia-cu118.yml
conda activate kmtf-vad-cu118
```

The Hygon/DTK file records the separate cluster environment used for selected
throughput runs.  Do not mix its PyTorch build with the NVIDIA environment.

## 2. Verify the release package

```bash
python scripts/verify_release.py
```

The command must finish with `"state": "PASS"` in `RELEASE_AUDIT.json`.

## 3. Train branch models

Run commands from `src/mnad/`.  Replace `<DATASET_ROOT>` and `<OUT>` with
absolute paths on the compute node.

Prediction branch example (Ped2):

```bash
python Train_Pred.py --gpus 0 --dataset_type ped2 \
  --dataset_path <DATASET_ROOT> --exp_dir <OUT>/ped2_P \
  --epochs 60 --msize 10 --alpha 0.6 --lr 2e-4 \
  --loss_compact 0.1 --loss_separate 0.1 --t_length 5 \
  --kmeans_init --kmeans_warmup_epochs 26 --kmeans_seed 2020
```

Reconstruction branch example (Ped2):

```bash
python Train_recons.py --gpus 0 --dataset_type ped2 \
  --dataset_path <DATASET_ROOT> --exp_dir <OUT>/ped2_R \
  --epochs 60 --msize 10 --alpha 0.7 --lr 2e-5 \
  --loss_compact 0.01 --loss_separate 0.01 --t_length 2 \
  --kmeans_init --kmeans_warmup_epochs 26 --kmeans_seed 2020
```

For Avenue, change the warm-up to 27 and use alpha 0.7 for both branches.  For
ShanghaiTech, use 10 epochs, warm-up 5, alpha 0.7 for R, and alpha 0.6 for P.
All other paper-level values are in `configs/FROZEN_PROTOCOL.json`.

Training writes the final model and memory keys to the experiment directory.
Keep the configuration, seed, logs, and warm-up replacement record together
with those files.

## 4. Export branch components

Evaluate R and P separately with the root `Evaluate.py` entry point.  It calls
`src/mnad/Export_Components.py`, which performs the same MNAD forward pass
without test-time memory updates and preserves the unnormalized components.
For the frozen fusion runner, each branch must be exported as an NPZ containing:

```text
video_names        string array, one item per aligned frame
frame_indices      integer array, index within each video
pixel_mse          raw per-frame image error
memory_distance    raw per-frame nearest-memory distance
labels             test labels (required in the R test export)
```

Training-normal exports use the same keys without `labels`.  The prediction
branch naturally begins after its context window; alignment is performed by
the exact `(video_name, frame_index)` intersection, never by blind truncation.

## 5. Run the frozen decision path

```bash
python scripts/run_frozen_pipeline.py \
  --dataset ped2 \
  --normal-r <R_TRAIN.npz> --normal-p <P_TRAIN.npz> \
  --test-r <R_TEST.npz> --test-p <P_TEST.npz> \
  --output <PED2_FUSED.npz>
```

Repeat with `--dataset avenue` and `--dataset shanghai`.  The script performs
normal-reference ECDF calibration, fixed fusion, and the dataset-specific
per-video Gaussian/EMA refinement.  It never uses test labels to choose a
weight or smoothing parameter.

## 6. Rebuild paper figures

```bash
python scripts/build_score_curve.py
python scripts/build_qualitative_figure.py
```

The scripts only read the compact evidence records shipped in this package.

## Reproducibility checklist

- record Python/PyTorch/CUDA versions and GPU type;
- preserve seeds, warm-up epoch, memory size, and branch alpha;
- retain final checkpoints and memory keys;
- retain per-video frame names, frame counts, labels, raw R/P image errors,
  memory distances, and alignment indices;
- reset smoothing at video boundaries;
- report failures and missing artifacts rather than fabricating results.

# Reproducing the experiments

## Environment

```bash
conda env create -f environments/nvidia-cu118.yml
conda activate kmtf-vad-cu118
```

## Branch training

The root entry point reads the training settings from
`configs/BRANCH_TRAINING.json`:

```bash
python Train.py --branch P --dataset ped2 \
  --dataset-root <DATASET_ROOT> --output <OUTPUT>/ped2_P --seed 2020

python Train.py --branch R --dataset ped2 \
  --dataset-root <DATASET_ROOT> --output <OUTPUT>/ped2_R --seed 2020
```

The available dataset names are `ped2`, `avenue`, and `shanghai`. Add
`--dry-run` to print the underlying branch command.

The equivalent Ped2 commands from `src/mnad/` are:

```bash
python Train_Pred.py --gpus 0 --dataset_type ped2 \
  --dataset_path <DATASET_ROOT> --exp_dir <OUTPUT>/ped2_P \
  --epochs 60 --msize 10 --alpha 0.6 --lr 2e-4 \
  --loss_compact 0.1 --loss_separate 0.1 --t_length 5 \
  --kmeans_init --kmeans_warmup_epochs 26 --kmeans_seed 2020

python Train_recons.py --gpus 0 --dataset_type ped2 \
  --dataset_path <DATASET_ROOT> --exp_dir <OUTPUT>/ped2_R \
  --epochs 60 --msize 10 --alpha 0.7 --lr 2e-5 \
  --loss_compact 0.01 --loss_separate 0.01 --t_length 2 \
  --kmeans_init --kmeans_warmup_epochs 26 --kmeans_seed 2020
```

## Component export

Run `Evaluate.py` for the R/P branches and for both data splits. Each output
NPZ contains the video names, frame indices, image errors, memory distances,
and labels when the testing split is used.

```bash
python Evaluate.py --branch P --split testing --dataset ped2 \
  --dataset-root <DATASET_ROOT> --model <MODEL.pth> --memory <KEYS.pt> \
  --output <P_TEST.npz>
```

## Score integration

```bash
python Fuse.py --dataset ped2 \
  --normal-r <R_TRAIN.npz> --normal-p <P_TRAIN.npz> \
  --test-r <R_TEST.npz> --test-p <P_TEST.npz> \
  --output <PED2_SCORES.npz>
```

The same command supports `avenue` and `shanghai`. It performs exact frame-key
alignment, branch-wise ECDF calibration, weighted fusion, and per-video
temporal refinement using `configs/DATASET_CONFIG.json`.

## Figures

```bash
python scripts/build_qualitative_figure.py
python scripts/build_score_curve.py
```

The scripts read the data in `evidence/` and write the rebuilt figures to
`figures/`.

## Tests

```bash
python -m unittest discover -s tests -v
```

# Reproducing the experiments

Install the environment in the [README](../README.md#setup) and prepare the
[datasets and test labels](DATASETS.md). Run these commands from the repository
root. Use absolute paths for data, checkpoints, and outputs.

## Branch training

The example below uses Ped2; change `DATASET` to `avenue` or `shanghai` as needed.
Training parameters are read from `configs/BRANCH_TRAINING.json`.

```bash
DATASET=ped2
DATASET_ROOT="$PWD/dataset"
RUN_ROOT="$PWD/experiments/$DATASET"

python Train.py --branch P --dataset "$DATASET" \
  --dataset-root "$DATASET_ROOT" --output "$RUN_ROOT/P" --seed 2020

python Train.py --branch R --dataset "$DATASET" \
  --dataset-root "$DATASET_ROOT" --output "$RUN_ROOT/R" --seed 2020
```

Each branch saves `model.pth` and `keys.pt` in its output directory.
Add `--dry-run` to `Train.py` or `Evaluate.py` to inspect the command.

## Component export

Export both branches on normal training frames and test frames. Each NPZ
contains frame identifiers, pixel errors, and memory distances; test exports
also contain the frame labels.

```bash
for BRANCH in R P; do
  for SPLIT in training testing; do
    python Evaluate.py --branch "$BRANCH" --split "$SPLIT" --dataset "$DATASET" \
      --dataset-root "$DATASET_ROOT" \
      --model "$RUN_ROOT/$BRANCH/model.pth" --memory "$RUN_ROOT/$BRANCH/keys.pt" \
      --output "$RUN_ROOT/${BRANCH}_${SPLIT}.npz"
  done
done
```

## Score integration

```bash
python Fuse.py --dataset "$DATASET" \
  --normal-r "$RUN_ROOT/R_training.npz" --normal-p "$RUN_ROOT/P_training.npz" \
  --test-r "$RUN_ROOT/R_testing.npz" --test-p "$RUN_ROOT/P_testing.npz" \
  --output "$RUN_ROOT/final_scores.npz"
```

Fusion aligns R/P frame keys, applies normal-training ECDF calibration,
combines the branches, and refines scores within each video. Runtime settings
are defined in `DATASET_CONFIGS` in `src/decision/frozen_pipeline.py` and
recorded in `configs/DATASET_CONFIG.json`.

The pipeline uses complete-video normalization and is intended for offline
evaluation.

## Figures and checks

Rebuild the figures from the supplied data in `evidence/`:

```bash
python scripts/build_qualitative_figure.py
python scripts/build_score_curve.py
```

Outputs are written to `figures/`. Run the entry-point and score-processing tests with:

```bash
python -m unittest discover -s tests -v
```

`src/memory_audit/` contains additional development scripts that depend on
runner modules and templates not included in this repository.

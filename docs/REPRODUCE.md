# Training and evaluation

Install the environment in the [README](../README.md#setup) and prepare the
[datasets and test labels](DATASETS.md). Run these commands from the repository
root. Use absolute paths for data, checkpoints, and outputs.

## Branch training

Datasets: `ped2`, `avenue`, `shanghai`. Parameters: `configs/BRANCH_TRAINING.json`.

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

Export R/P components for both splits. Test exports include frame labels.

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

Runtime settings: `DATASET_CONFIGS` in `src/decision/frozen_pipeline.py`.
The pipeline uses offline, complete-video normalization.

## Figures and checks

Figure inputs: `evidence/`. Outputs: `figures/`.

```bash
python scripts/build_qualitative_figure.py
python scripts/build_score_curve.py
```

Run the entry-point and score-processing tests:

```bash
python -m unittest discover -s tests -v
```

`src/memory_audit/` requires the original experiment runner and task templates.

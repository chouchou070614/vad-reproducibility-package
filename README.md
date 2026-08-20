# Task-Adapted Memory Reorganization and Complementary Dual-Branch Scoring for Video Anomaly Detection

This repository accompanies the anonymous submission **Task-Adapted Memory
Reorganization and Complementary Dual-Branch Scoring for Video Anomaly
Detection** (KMTF-VAD).  It contains the training and evaluation code,
the frozen decision protocol, environment specifications, figure-generation
scripts, and the compact numerical evidence used to audit the manuscript.

This package was assembled from the authors' working copy of
[MNADrc](https://github.com/alchemi5t/MNADrc), an independent ML
Reproducibility Challenge 2020 reimplementation of MNAD, together with the
experiment-specific changes reported in the submission.  The original
[MNAD project](https://cvlab.yonsei.ac.kr/projects/MNAD/),
[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Park_Learning_Memory-Guided_Normality_for_Anomaly_Detection_CVPR_2020_paper.pdf),
and [official repository](https://github.com/cvlab-yonsei/MNAD) are cited as
methodological and interface references; this archive does not claim that its
working tree was packaged directly from the official repository.  The frozen
paper pipeline contains three submission-specific parts:

1. K-means initialization of the memory slots after a dataset-specific
   normal-only warm-up;
2. frame-key-aligned integration of reconstruction (R) and prediction (P)
   branch outputs;
3. normal-reference ECDF calibration, fixed R/P fusion, and per-video
   temporal smoothing.

The dynamic NCBAF-v2 and V3 gate prototypes are intentionally **not** part of
the reported method.  This separation prevents experimental prototypes from
silently changing the frozen paper protocol.

## Repository map

```text
configs/                 Frozen paper configuration
environments/            NVIDIA/CUDA and Hygon/DTK environment records
src/mnad/                MNADrc-derived branch training and evaluation code
src/decision/            Frozen calibration, fusion, and smoothing pipeline
src/memory_audit/        Same-trajectory memory initialization audit
scripts/                 Reproduction, figure, and package-verification tools
tests/                   CPU unit tests for the frozen score path
evidence/                Compact paper-result and qualitative audit records
figures/                 Rebuilt paper figures
docs/                    Dataset, provenance, and reproduction notes
licenses/                Upstream-license status and third-party notices
```

Datasets, trained checkpoints, and large frame-level archives are not bundled
because of size and redistribution restrictions.  Their expected layout and
the exact commands are documented in [docs/DATASETS.md](docs/DATASETS.md) and
[docs/REPRODUCE.md](docs/REPRODUCE.md).

The repository is intentionally a submission artifact rather than a dump of
the authors' experiment workspace. Development-only prototypes, scheduler
files, private paths, raw logs, duplicate protocol copies, and unpublished
checkpoints are excluded. The inclusion and exclusion rules are recorded in
[docs/RELEASE_SCOPE.md](docs/RELEASE_SCOPE.md).

## Frozen configurations

| Dataset | Epochs | K-means warm-up | R weight | Temporal refinement |
|---|---:|---:|---:|---|
| UCSD Ped2 | 60 | 26 | 0.40 | Gaussian, sigma=3 |
| CUHK Avenue | 60 | 27 | 0.10 | Gaussian, sigma=3 |
| ShanghaiTech | 10 | 5 | 0.55 | EMA, rho=0.725 |

The authoritative machine-readable record is
[`configs/FROZEN_PROTOCOL.json`](configs/FROZEN_PROTOCOL.json).  If a command,
comment, or historical script disagrees with that file, the frozen record
takes precedence.

## Quick verification (CPU)

From the repository root:

```bash
python scripts/verify_release.py
```

The verifier checks the required structure, compiles Python sources, runs the
frozen-pipeline unit tests, scans text and packaged binary evidence for private
paths and credentials, audits the visible Git history when available, and
writes `MANIFEST.sha256` plus `RELEASE_AUDIT.json`.  Those two generated files
are deliberately excluded from the recursive manifest to avoid a
self-referential checksum. The verifier does not download data or claim to
reproduce GPU training on a CPU-only host.

## Reproduction levels

- **Level A — package audit:** run `scripts/verify_release.py`.
- **Level B — decision path:** export R/P frame components and run
  `scripts/run_frozen_pipeline.py`; no network or GPU is required.
- **Level C — branch evaluation:** provide checkpoints and datasets, then run
  `Evaluate_recons.py` and `Evaluate_Pred.py`.
- **Level D — full training:** train both branches with the dataset-specific
  warm-up and seed recorded in the frozen configuration.

Exact commands and output contracts are in [docs/REPRODUCE.md](docs/REPRODUCE.md).

## Release entry points

For reproducibility, the package exposes compact `Train`, `Evaluate`, and
`Fuse` entry points while retaining the branch conventions of the working
MNADrc code:

```bash
# Train one branch using the frozen dataset-specific settings.
python Train.py --branch P --dataset ped2 --dataset-root ./dataset --output ./exp/ped2_P
python Train.py --branch R --dataset ped2 --dataset-root ./dataset --output ./exp/ped2_R

# Export raw normal/test components from a trained branch.
python Evaluate.py --branch P --split testing --dataset ped2 \
  --dataset-root ./dataset --model MODEL.pth --memory KEYS.pt --output P_test.npz

# Calibrate, align, fuse, and temporally refine R/P scores.
python Fuse.py --dataset ped2 --normal-r R_train.npz --normal-p P_train.npz \
  --test-r R_test.npz --test-p P_test.npz --output ped2_final.npz
```

Use `--dry-run` with `Train.py` or `Evaluate.py` to inspect an exact command
without launching a GPU job.

## Anonymity and release status

The package contains no author names, institutional paths, server addresses,
or credentials.  Its verified read-only review mirror is available at
<https://anonymous.4open.science/r/m8q2v7k4p9>.  The identity-bearing source
repository remains private; see
[`PUBLIC_URL_STATUS.md`](PUBLIC_URL_STATUS.md) for the frozen review status.

## Licensing note

Part of `src/mnad/` follows the public MNADrc reimplementation lineage, which
in turn reimplements MNAD.  The local MNADrc snapshot did not contain an
explicit license file when this package was assembled.  Consequently, this
archive is provided for anonymous review and reproducibility inspection;
permanent public redistribution requires the licensing question to be
resolved.  See
[`licenses/LICENSE_STATUS.md`](licenses/LICENSE_STATUS.md) and
[`licenses/THIRD_PARTY_NOTICE.md`](licenses/THIRD_PARTY_NOTICE.md).

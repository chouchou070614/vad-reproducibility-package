# Implementation provenance and scope

## Included implementation

The implementation tree used for the experiments started from the authors'
working copy of `MNADrc-master`, corresponding to the independent
[MNADrc](https://github.com/alchemi5t/MNADrc) reimplementation by Menon and
Stephen.  It was not packaged by freshly cloning the official MNAD
repository.  The original MNAD paper, project page, and official repository
serve as method and interface references.  The working implementation was
then adapted to support:

- dataset-explicit command-line interfaces;
- normal-only K-means memory replacement after a configurable warm-up;
- deterministic K-means seeding;
- checkpoint, memory-key, and frame-score export;
- reconstruction/prediction branch execution on the three paper datasets;
- offline fixed fusion and temporal refinement.

The frozen score implementation in `src/decision/frozen_pipeline.py` is the
authoritative implementation of the paper's final decision path.  It uses
only training-normal reference distributions for ECDF calibration and resets
temporal state for every video.

## Deliberately excluded from the reported method

The following explored mechanisms are not the paper's frozen main method and
must not be used to reproduce its tables:

- NCBAF-v2 dynamic sample-dependent weighting;
- the V3 gate prototype;
- the historical 99.1513% exploratory result;
- label-selected post-hoc parameter variants.

## Evidence boundary

Files under `evidence/` are compact audit records and paper-figure inputs.
They are not substitutes for raw datasets or full checkpoints.  The manifest
and release audit establish package integrity, not independent reproduction
of every reported GPU run.

All source references stored in released CSV/JSON records use repository
relative paths or the non-resolving `artifact://` scheme. Absolute workstation,
cluster, account, and scheduler paths are deliberately removed from the
review package while the corresponding content hashes are preserved.

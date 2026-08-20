# Third-party notices

## MNADrc implementation lineage

- Project: [Re] Learning Memory Guided Normality for Anomaly Detection
- Authors: Varun Menon and Kevin Stephen
- Public repository: https://github.com/alchemi5t/MNADrc
- Reproducibility report: https://openreview.net/forum?id=vvLWTXkJ2Zv
- Scope in this archive: the starting implementation lineage for prediction
  and reconstruction models, memory modules, data loading, and branch
  training/evaluation code under `src/mnad/`
- License observed in the local source snapshot at freeze time: not specified

## Original MNAD method and reference implementation

- Project: Learning Memory-guided Normality for Anomaly Detection
- Public repository: https://github.com/cvlab-yonsei/MNAD
- Scope in this archive: methodological, architectural, and interface
  reference.  This notice does not claim that the release tree was packaged
  directly from the official repository.

## Python dependencies

PyTorch, torchvision, NumPy, SciPy, scikit-learn, Pillow, OpenCV, Matplotlib,
and their transitive dependencies are governed by their own licenses.  The
environment files identify versions but do not redistribute those packages.

## Datasets

UCSD Ped2, CUHK Avenue, and ShanghaiTech are not included.  Users must obtain
them from authorized sources and comply with the providers' terms.

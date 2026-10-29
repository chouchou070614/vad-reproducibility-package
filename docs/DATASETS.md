# Dataset preparation

Download UCSD Ped2, CUHK Avenue, and ShanghaiTech from their official sources.

## Directory layout

```text
dataset/
  ped2/{training,testing}/frames/<video-id>/*
  avenue/{training,testing}/frames/<video-id>/*
  shanghai/{training,testing}/frames/<video-id>/*
```

Each video is stored in a separate directory. Frame names should be
zero-padded and chronologically ordered.

For Avenue, `src/mnad/convert_avenue_cv32_exact_png.py` converts frames using
Python 3.5 and OpenCV 3.2. The branch loaders are `src/mnad/model/utils.py`
and `src/mnad/model/utils_pred.py`.

## Test labels

Place the frame-level label arrays under `src/mnad/data/`:

```text
frame_labels_ped2.npy
frame_labels_avenue.npy
frame_labels_shanghai.npy
```

Use one label per original test frame (`0` normal, `1` anomalous), concatenated
in sorted video and frame order. Include every test frame, including the first
four frames of each video.

# Dataset preparation

Download UCSD Ped2, CUHK Avenue, and ShanghaiTech from their official sources.
Full datasets and the label arrays required for evaluation must be prepared
separately. The files in `evidence/` are saved results and figure inputs.

## Directory layout

```text
dataset/
  ped2/
    training/frames/<video-id>/*
    testing/frames/<video-id>/*
  avenue/
    training/frames/<video-id>/*
    testing/frames/<video-id>/*
  shanghai/
    training/frames/<video-id>/*
    testing/frames/<video-id>/*
```

Each video is stored in a separate directory. Frame names should be
zero-padded and chronologically ordered.

The prediction branch uses four context frames and one target frame. The
reconstruction branch processes one target frame. R/P outputs are matched by
`(video_name, frame_index)` before fusion, and temporal smoothing is applied
independently to each video.

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
in sorted video and frame order. Include the first four frames of every video;
the exporter handles the prediction offset. `Export_Components.py` checks that
the label count matches the test frames.

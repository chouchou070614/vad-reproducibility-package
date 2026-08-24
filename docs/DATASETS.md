# Dataset preparation

Download UCSD Ped2, CUHK Avenue, and ShanghaiTech from their official sources.
The repository does not include dataset frames or annotations.

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

For Avenue, `src/mnad/convert_avenue_cv32_exact_png.py` contains the OpenCV
frame-conversion utility used in our experiments. The dataset loader and label
handling are implemented in `src/mnad/data/data_seqkey_all.py`.

Before training, the dataset can be checked with:

```bash
python src/mnad/preflight.py --help
```

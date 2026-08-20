# Dataset preparation

Obtain UCSD Ped2, CUHK Avenue, and ShanghaiTech from their official or
authorized distribution channels.  This package does not redistribute video
frames or annotations.

Use the following layout, where every video is a directory of zero-padded,
chronologically ordered image frames:

```text
dataset/
  ped2/
    training/frames/<video-id>/*.jpg
    testing/frames/<video-id>/*.jpg
  avenue/
    training/frames/<video-id>/*.jpg
    testing/frames/<video-id>/*.jpg
  shanghai/
    training/frames/<video-id>/*.jpg
    testing/frames/<video-id>/*.jpg
```

The loader resolves labels through the dataset logic in
`src/mnad/data/data_seqkey_all.py`.  Preserve the official video ordering and
frame counts.  Do not concatenate videos before temporal smoothing: the
Gaussian/EMA state is reset at every video boundary.

For Avenue, `src/mnad/convert_avenue_cv32_exact_png.py` records the historical
OpenCV-based conversion path used during reproduction.  Before a long run,
execute `src/mnad/preflight.py` and verify:

- all training and testing videos are present;
- frame names sort chronologically;
- no frame is unreadable;
- label length equals the test-frame count after branch-specific alignment;
- the prediction branch uses four context frames and one target frame;
- the reconstruction branch uses the single target frame.

Dataset access terms remain those of the original dataset providers.


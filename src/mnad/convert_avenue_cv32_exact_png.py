#!/usr/bin/env python
"""Materialize paper-era Avenue tensors as lossless 256x256 PNG frames.

Run this with the isolated Python 3.5 / OpenCV 3.2 environment.  The source is
the JPEG frame set decoded and encoded by that same environment, matching the
two image operations used by the original MNAD input loader: JPEG decode and
bilinear resize.  PNG is used only as a lossless cache of the resulting pixels.
"""
from __future__ import print_function

import argparse
import hashlib
import json
import os
import shutil
import sys

import cv2
import numpy as np


EXPECTED = {"training": (16, 15328), "testing": (21, 15324)}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dataset-root", required=True)
    parser.add_argument("--output-dataset-root", required=True)
    args = parser.parse_args()

    source = os.path.join(os.path.abspath(args.source_dataset_root), "avenue")
    target = os.path.join(os.path.abspath(args.output_dataset_root), "avenue")
    if not os.path.isfile(os.path.join(source, "READY")):
        raise RuntimeError("Source OpenCV 3.2 extraction is not verified: {0}".format(source))
    if os.path.exists(target):
        raise RuntimeError("Refusing to overwrite existing target: {0}".format(target))

    manifest = {
        "purpose": "lossless cache of OpenCV 3.2 JPEG decode plus 256x256 INTER_LINEAR resize",
        "source": source,
        "python": sys.version,
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "resize": [256, 256],
        "interpolation": "INTER_LINEAR",
        "splits": {},
        "sample_raw_pixel_sha256": [],
    }
    try:
        for split in ("training", "testing"):
            source_split = os.path.join(source, split, "frames")
            videos = sorted(name for name in os.listdir(source_split)
                            if os.path.isdir(os.path.join(source_split, name)))
            expected_videos, expected_frames = EXPECTED[split]
            if len(videos) != expected_videos:
                raise RuntimeError("{0}: expected {1} videos, found {2}".format(
                    split, expected_videos, len(videos)))
            counts = []
            for video in videos:
                source_video = os.path.join(source_split, video)
                output_video = os.path.join(target, split, "frames", video)
                os.makedirs(output_video)
                frames = sorted(name for name in os.listdir(source_video)
                                if name.lower().endswith(".jpg"))
                for filename in frames:
                    image = cv2.imread(os.path.join(source_video, filename), cv2.IMREAD_COLOR)
                    if image is None:
                        raise RuntimeError("Cannot decode {0}".format(filename))
                    resized = cv2.resize(image, (256, 256), interpolation=cv2.INTER_LINEAR)
                    output = os.path.join(output_video, os.path.splitext(filename)[0] + ".png")
                    if not cv2.imwrite(output, resized, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
                        raise RuntimeError("Cannot write {0}".format(output))
                    if len(manifest["sample_raw_pixel_sha256"]) < 8:
                        manifest["sample_raw_pixel_sha256"].append({
                            "path": os.path.relpath(output, target).replace("\\", "/"),
                            "sha256": hashlib.sha256(resized.tobytes()).hexdigest(),
                        })
                counts.append(len(frames))
                print("{0} {1}: {2} frames".format(split, video, len(frames)), flush=True)
            if sum(counts) != expected_frames:
                raise RuntimeError("{0}: expected {1} frames, found {2}".format(
                    split, expected_frames, sum(counts)))
            manifest["splits"][split] = {
                "video_count": len(videos),
                "frame_count": sum(counts),
                "per_video_counts": counts,
            }

        source_manifest = os.path.join(source, "extraction_manifest.json")
        manifest["source_manifest_sha256"] = sha256_file(source_manifest)
        with open(os.path.join(target, "exact_input_manifest.json"), "w") as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True)
        with open(os.path.join(target, "READY"), "w") as stream:
            stream.write("verified OpenCV 3.2 exact pixels\n")
        print("Exact-pixel dataset verified: {0}".format(target))
    except Exception:
        if os.path.isdir(target):
            shutil.rmtree(target)
        raise


if __name__ == "__main__":
    main()

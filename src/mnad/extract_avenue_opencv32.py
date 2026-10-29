#!/usr/bin/env python
"""Extract Avenue video frames using Python 3.5 and OpenCV 3.2."""
from __future__ import print_function

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import zipfile

import cv2
import numpy as np


EXPECTED = {
    "training": {"videos": 16, "frames": 15328, "archive_dir": "training_videos"},
    "testing": {"videos": 21, "frames": 15324, "archive_dir": "testing_videos"},
}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def archive_videos(archive, directory_name):
    suffix = "/{0}/".format(directory_name)
    names = []
    for name in archive.namelist():
        normalized = name.replace("\\", "/")
        if suffix in normalized and normalized.lower().endswith(".avi"):
            names.append(name)
    return sorted(names, key=lambda item: int(os.path.splitext(os.path.basename(item))[0]))


def extract_video(archive, member, temporary_dir):
    filename = os.path.basename(member)
    destination = os.path.join(temporary_dir, filename)
    with archive.open(member) as source, open(destination, "wb") as target:
        shutil.copyfileobj(source, target, 1024 * 1024)
    return destination


def write_frames(video_path, destination):
    os.makedirs(destination)
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise RuntimeError("Cannot open video: {0}".format(video_path))
    count = 0
    first_shape = None
    while True:
        ok, image = capture.read()
        if not ok:
            break
        if first_shape is None:
            first_shape = list(image.shape)
        output = os.path.join(destination, "{0:06d}.jpg".format(count))
        if not cv2.imwrite(output, image):
            raise RuntimeError("Cannot write frame: {0}".format(output))
        count += 1
    capture.release()
    if count == 0:
        raise RuntimeError("No frames decoded from: {0}".format(video_path))
    return count, first_shape


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--work-root", required=True)
    args = parser.parse_args()

    archive_path = os.path.abspath(args.archive)
    dataset_root = os.path.abspath(args.dataset_root)
    target = os.path.join(dataset_root, "avenue")
    if os.path.exists(target):
        raise RuntimeError("Refusing to overwrite existing target: {0}".format(target))
    os.makedirs(dataset_root) if not os.path.isdir(dataset_root) else None
    os.makedirs(args.work_root) if not os.path.isdir(args.work_root) else None
    temporary_dir = tempfile.mkdtemp(prefix="avenue_opencv32_", dir=args.work_root)

    manifest = {
        "python": sys.version,
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "archive": archive_path,
        "archive_sha256": sha256_file(archive_path),
        "splits": {},
    }
    try:
        with zipfile.ZipFile(archive_path) as archive:
            for split in ("training", "testing"):
                specification = EXPECTED[split]
                members = archive_videos(archive, specification["archive_dir"])
                if len(members) != specification["videos"]:
                    raise RuntimeError("{0}: expected {1} videos, found {2}".format(
                        split, specification["videos"], len(members)))
                counts = []
                shapes = []
                for index, member in enumerate(members, 1):
                    video_path = extract_video(archive, member, temporary_dir)
                    destination = os.path.join(target, split, "frames", "{0:02d}".format(index))
                    count, shape = write_frames(video_path, destination)
                    counts.append(count)
                    shapes.append(shape)
                    os.remove(video_path)
                    print("{0} {1:02d}: {2} frames".format(split, index, count), flush=True)
                if sum(counts) != specification["frames"]:
                    raise RuntimeError("{0}: expected {1} frames, found {2}".format(
                        split, specification["frames"], sum(counts)))
                manifest["splits"][split] = {
                    "video_count": len(counts),
                    "frame_count": sum(counts),
                    "per_video_counts": counts,
                    "decoded_shapes": shapes,
                }

        samples = []
        for relative in (
                os.path.join("training", "frames", "01", "000000.jpg"),
                os.path.join("training", "frames", "16", "000000.jpg"),
                os.path.join("testing", "frames", "01", "000000.jpg"),
                os.path.join("testing", "frames", "21", "000000.jpg")):
            path = os.path.join(target, relative)
            samples.append({"path": relative.replace("\\", "/"), "sha256": sha256_file(path)})
        manifest["sample_sha256"] = samples
        with open(os.path.join(target, "extraction_manifest.json"), "w") as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True)
        with open(os.path.join(target, "READY"), "w") as stream:
            stream.write("verified\n")
        print("Extraction verified: {0}".format(target))
    except Exception:
        if os.path.isdir(target):
            shutil.rmtree(target)
        raise
    finally:
        shutil.rmtree(temporary_dir, ignore_errors=True)


if __name__ == "__main__":
    main()

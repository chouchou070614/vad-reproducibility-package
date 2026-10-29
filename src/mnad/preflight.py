"""Check the environment, dataset layout, and model forward passes."""
import argparse
from pathlib import Path
import sys

import cv2
import numpy as np
import scipy
import sklearn
import torch
import torchvision

from model.Prediction import convAE as PredictionModel
from model.reconstruction_model import convAE as ReconstructionModel


def fail(message):
    print(f"ERROR: {message}", file=sys.stderr)
    return 1


def main():
    parser = argparse.ArgumentParser(description="MNAD server preflight")
    parser.add_argument("--dataset-type", default="ped2", choices=("ped2", "avenue", "shanghai", "shanghaitech"))
    parser.add_argument("--dataset-path", default="./dataset")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--skip-model-smoke", action="store_true", help="skip the MNAD forward-pass check")
    args = parser.parse_args()

    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Torchvision: {torchvision.__version__}")
    print(f"OpenCV: {cv2.__version__}")
    print(f"NumPy: {np.__version__}")
    print(f"SciPy: {scipy.__version__}")
    print(f"Scikit-learn: {sklearn.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    elif args.require_cuda:
        return fail("CUDA is not available. Install the matching NVIDIA driver and PyTorch CUDA wheel.")

    root = Path(args.dataset_path) / args.dataset_type
    train = root / "training" / "frames"
    test = root / "testing" / "frames"
    missing = [str(path) for path in (train, test) if not path.is_dir()]
    if missing:
        return fail("dataset directories are missing: " + ", ".join(missing))

    train_images = list(train.glob("*/*.jpg"))
    test_images = list(test.glob("*/*.jpg"))
    if not train_images or not test_images:
        return fail("no .jpg frames found under the training/testing directories")
    if cv2.imread(str(train_images[0])) is None:
        return fail(f"cannot decode sample image: {train_images[0]}")

    for split_dir in (train, test):
        for video_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
            frames = sorted(video_dir.glob("*.jpg"))
            if len(frames) < 5:
                return fail(f"video has fewer than 5 frames: {video_dir}")
            try:
                frame_numbers = [int(frame.stem) for frame in frames]
            except ValueError:
                return fail(f"frame names must be numeric: {video_dir}")
            if frame_numbers != list(range(len(frame_numbers))):
                return fail(f"frame names must run from 000... without gaps: {video_dir}")

    label_name = "frame_labels_shanghai.npy" if args.dataset_type == "shanghaitech" else f"frame_labels_{args.dataset_type}.npy"
    labels = Path("data") / label_name
    if not labels.is_file():
        return fail(f"label file is missing: {labels}")
    label_values = np.load(labels).reshape(-1)
    if len(label_values) != len(test_images):
        return fail(
            f"label count ({len(label_values)}) does not match testing frames ({len(test_images)})")
    print(f"Dataset: {len(train_images)} training frames, {len(test_images)} testing frames")
    if not args.skip_model_smoke:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        keys = torch.nn.functional.normalize(torch.rand(10, 512, device=device), dim=1)

        prediction_model = PredictionModel().to(device).eval()
        prediction_frames = torch.randn(1, 12, 64, 64, device=device)
        reconstruction_model = ReconstructionModel(t_length=2).to(device).eval()
        reconstruction_frames = torch.randn(1, 3, 64, 64, device=device)

        with torch.no_grad():
            prediction_output = prediction_model(prediction_frames, keys, train=False)[0]
            reconstruction_output = reconstruction_model(reconstruction_frames, keys, train=False)[0]
        expected = (1, 3, 64, 64)
        if tuple(prediction_output.shape) != expected:
            return fail(f"prediction model returned {tuple(prediction_output.shape)}, expected {expected}")
        if tuple(reconstruction_output.shape) != expected:
            return fail(f"reconstruction model returned {tuple(reconstruction_output.shape)}, expected {expected}")
        if device.type == "cuda":
            torch.cuda.synchronize()
        print(f"R/P model forward passes: passed on {device}")
    print("Preflight passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

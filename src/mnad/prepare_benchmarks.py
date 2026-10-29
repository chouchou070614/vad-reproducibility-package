"""Extract benchmark frames into dataset/<name>/<split>/frames/<video>/."""
import argparse
import shutil
import tarfile
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

import cv2
import numpy as np


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def safe_extract_tar(archive: Path, destination: Path) -> None:
    with tarfile.open(archive) as tar:
        root = destination.resolve()
        for member in tar.getmembers():
            target = (destination / member.name).resolve()
            if root not in target.parents and target != root:
                raise RuntimeError(f"Unsafe tar entry: {member.name}")
        tar.extractall(destination)


def safe_extract_zip(archive: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive) as zip_file:
        root = destination.resolve()
        for member in zip_file.infolist():
            target = (destination / member.filename).resolve()
            if root not in target.parents and target != root:
                raise RuntimeError(f"Unsafe zip entry: {member.filename}")
        zip_file.extractall(destination)


def images_in(directory: Path):
    return sorted(path for path in directory.iterdir()
                  if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def write_images(images, destination: Path) -> int:
    destination.mkdir(parents=True, exist_ok=False)
    for index, image_path in enumerate(images):
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Cannot decode image: {image_path}")
        if not cv2.imwrite(str(destination / f"{index:06d}.jpg"), image):
            raise RuntimeError(f"Cannot write frame {index} to {destination}")
    return len(images)


def write_video(video_path: Path, destination: Path) -> int:
    destination.mkdir(parents=True, exist_ok=False)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    index = 0
    while True:
        ok, image = capture.read()
        if not ok:
            break
        if not cv2.imwrite(str(destination / f"{index:06d}.jpg"), image):
            raise RuntimeError(f"Cannot write frame {index} from {video_path}")
        index += 1
    capture.release()
    if index == 0:
        raise RuntimeError(f"Video contains no frames: {video_path}")
    return index


def choose_directory(root: Path, wanted_name: str) -> Path:
    matches = [path for path in root.rglob("*")
               if path.is_dir() and path.name.lower() == wanted_name.lower()]
    if not matches:
        raise FileNotFoundError(f"Cannot find {wanted_name!r} below {root}")
    return sorted(matches, key=lambda path: (len(path.parts), str(path)))[0]


def ensure_empty_target(target: Path) -> None:
    if target.exists():
        raise FileExistsError(
            f"Dataset directory exists: {target}.")


def prepare_ped2(archive: Path, target: Path) -> None:
    ensure_empty_target(target)
    with tempfile.TemporaryDirectory(prefix="mnad_ped2_") as temporary:
        root = Path(temporary)
        safe_extract_tar(archive, root)
        ped2_root = choose_directory(root, "UCSDped2")
        train_root = choose_directory(ped2_root, "Train")
        test_root = choose_directory(ped2_root, "Test")
        for split, source_root in (("training", train_root), ("testing", test_root)):
            output_root = target / split / "frames"
            sequence_dirs = [path for path in sorted(source_root.iterdir())
                             if path.is_dir() and not path.name.lower().endswith("_gt")
                             and images_in(path)]
            if not sequence_dirs:
                raise RuntimeError(f"No image sequences found in {source_root}")
            for index, sequence in enumerate(sequence_dirs, start=1):
                write_images(images_in(sequence), output_root / f"{index:02d}")


def prepare_avenue(archive: Path, target: Path) -> None:
    ensure_empty_target(target)
    with tempfile.TemporaryDirectory(prefix="mnad_avenue_") as temporary:
        root = Path(temporary)
        safe_extract_zip(archive, root)
        train_root = choose_directory(root, "training_videos")
        test_root = choose_directory(root, "testing_videos")
        for split, source_root in (("training", train_root), ("testing", test_root)):
            output_root = target / split / "frames"
            videos = sorted(path for path in source_root.iterdir()
                            if path.is_file() and path.suffix.lower() in {".avi", ".mp4", ".mov"})
            if not videos:
                raise RuntimeError(f"No videos found in {source_root}")
            for index, video in enumerate(videos, start=1):
                write_video(video, output_root / f"{index:02d}")


def prepare_shanghai(archives, target: Path) -> None:
    """Build the ShanghaiTech split from sequence-ID suffixes."""
    ensure_empty_target(target)
    sequence_names = {"training": set(), "testing": set()}
    frame_counts = {"training": 0, "testing": 0}

    try:
        for archive in archives:
            with zipfile.ZipFile(archive) as zip_file:
                for member in zip_file.infolist():
                    if member.is_dir():
                        continue
                    source_path = PurePosixPath(member.filename)
                    parts = source_path.parts
                    try:
                        frames_index = parts.index("frames")
                    except ValueError:
                        continue
                    if len(parts) != frames_index + 3:
                        continue
                    sequence_name = parts[frames_index + 1]
                    frame_name = parts[frames_index + 2]
                    if Path(frame_name).suffix.lower() not in IMAGE_SUFFIXES:
                        continue

                    suffix = sequence_name.rsplit("_", 1)[-1]
                    if len(suffix) == 3 and suffix.isdigit():
                        split = "training"
                    elif len(suffix) == 4 and suffix.isdigit():
                        split = "testing"
                    else:
                        raise RuntimeError(f"Unexpected ShanghaiTech sequence ID: {sequence_name}")

                    frame_path = Path(frame_name)
                    try:
                        frame_number = int(frame_path.stem)
                    except ValueError:
                        raise RuntimeError(f"Non-numeric ShanghaiTech frame name: {frame_name}")
                    normalized_name = f"{frame_number:06d}{frame_path.suffix.lower()}"
                    destination = target / split / "frames" / sequence_name / normalized_name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    if destination.exists():
                        raise RuntimeError(f"Duplicate ShanghaiTech frame: {destination}")
                    with zip_file.open(member) as source, destination.open("xb") as output:
                        shutil.copyfileobj(source, output)
                    sequence_names[split].add(sequence_name)
                    frame_counts[split] += 1
    except Exception:
        if target.exists():
            shutil.rmtree(target)
        raise

    expected_sequences = {"training": 330, "testing": 107}
    for split, expected in expected_sequences.items():
        actual = len(sequence_names[split])
        if actual != expected:
            raise RuntimeError(
                f"ShanghaiTech {split}: expected {expected} sequences, found {actual}")
    if frame_counts["testing"] != 40791:
        raise RuntimeError(
            f"ShanghaiTech testing: expected 40791 frames, found {frame_counts['testing']}")
    print(
        "shanghai: restored original split "
        f"({len(sequence_names['training'])} training sequences / "
        f"{len(sequence_names['testing'])} testing sequences)")


def verify(dataset_root: Path, dataset_type: str) -> None:
    target = dataset_root / dataset_type
    test_images = sorted((target / "testing" / "frames").glob("*/*.jpg"))
    label_file = Path(__file__).resolve().parent / "data" / f"frame_labels_{dataset_type}.npy"
    labels = np.load(label_file).reshape(-1)
    if len(test_images) != len(labels):
        raise RuntimeError(
            f"{dataset_type}: {len(test_images)} testing frames but {len(labels)} built-in labels. "
            "Frame/label count mismatch.")
    print(f"{dataset_type}: prepared and verified ({len(test_images)} testing frames)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Ped2, Avenue and ShanghaiTech for MNAD")
    parser.add_argument("--archive-dir", type=Path, default=Path("dataset_archives"))
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset"))
    parser.add_argument("--only", choices=("ped2", "avenue", "shanghai"), required=True)
    args = parser.parse_args()

    archives = {
        "ped2": args.archive_dir / "UCSD_Anomaly_Dataset.tar.gz",
        "avenue": args.archive_dir / "Avenue_Dataset.zip",
        "shanghai": (
            args.archive_dir / "ShanghaiTech_train.zip",
            args.archive_dir / "ShanghaiTech_test.zip",
        ),
    }
    archive = archives[args.only]
    archive_files = archive if isinstance(archive, tuple) else (archive,)
    for archive_file in archive_files:
        if not archive_file.is_file():
            raise FileNotFoundError(f"Archive is missing: {archive_file}")
    target = args.dataset_root / args.only
    {"ped2": prepare_ped2, "avenue": prepare_avenue, "shanghai": prepare_shanghai}[args.only](archive, target)
    verify(args.dataset_root, args.only)


if __name__ == "__main__":
    main()

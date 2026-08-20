#!/usr/bin/env python3
"""Fail-closed release audit: structure, secrets, syntax, tests, and manifest."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = (
    "README.md",
    "CITATION.cff",
    "configs/FROZEN_PROTOCOL.json",
    "docs/DATASETS.md",
    "docs/REPRODUCE.md",
    "docs/RELEASE_SCOPE.md",
    "environments/nvidia-cu118.yml",
    "environments/hygon-dtk2204.yml",
    "src/mnad/Train_Pred.py",
    "src/mnad/Train_recons.py",
    "src/mnad/Evaluate_Pred.py",
    "src/mnad/Evaluate_recons.py",
    "src/mnad/Export_Components.py",
    "src/mnad/kmeans_memory.py",
    "src/decision/frozen_pipeline.py",
    "scripts/run_frozen_pipeline.py",
    "tests/test_frozen_pipeline.py",
    "Train.py",
    "Evaluate.py",
    "Fuse.py",
)
TEXT_SUFFIXES = {
    ".py", ".sh", ".slurm", ".md", ".txt", ".json", ".csv", ".yml",
    ".yaml", ".cff", ".ps1",
}
FORBIDDEN_RELEASE_SUFFIXES = {".7z", ".ckpt", ".gz", ".pdf", ".pth", ".pt", ".tar", ".zip"}
FORBIDDEN = (
    re.compile(r"[A-Za-z]:\\Users\\[^\\\s]+", re.I),
    re.compile(r"(?:ssh|connect)\.[A-Za-z0-9.-]+\.(?:com|cn)", re.I),
    re.compile(r"/root/(?:autodl|mmm|workspace|home)[^\s'\"]*", re.I),
    re.compile(
        r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\."
        r"(?:com|org|edu|net|cn|io|ai|co|uk|de|jp|kr|fr|ru|xyz|tech)",
        re.I,
    ),
    re.compile(r"(?:password|passwd|token|secret)\s*[=:]\s*['\"][^'\"]+", re.I),
)
ALLOWED_GIT_IDENTITIES = {
    ("Anonymous Authors", "anonymous@anonymous.org"),
    ("Anonymous Authors", "anonymous@invalid.example"),
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def scan_payload(label: str, payload: bytes, errors: list[str], *, binary: bool = False) -> None:
    if binary:
        # Restrict binary scans to human-readable strings. Removing NUL bytes
        # also exposes UTF-16/UTF-32 path fragments embedded in NPY members.
        readable = []
        for candidate in (payload, payload.replace(b"\x00", b"")):
            readable.extend(re.findall(rb"[\x20-\x7e]{4,}", candidate))
        text = "\n".join(item.decode("ascii", errors="ignore") for item in readable)
    else:
        text = payload.decode("utf-8", errors="ignore")
    text_variants = (text, text.replace("\\\\", "\\"))
    for pattern in FORBIDDEN:
        if any(pattern.search(candidate) for candidate in text_variants):
            errors.append(f"private credential/path pattern in {label}: {pattern.pattern}")


def audit_git_history(errors: list[str]) -> str:
    if not (ROOT / ".git").exists():
        return "NOT_PRESENT_IN_EXTRACTED_PACKAGE"
    result = subprocess.run(
        ["git", "log", "--format=%an%x00%ae%x00%cn%x00%ce"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    if result.returncode:
        errors.append("unable to audit Git commit identities")
        return "FAIL"
    for line in result.stdout.splitlines():
        fields = line.split("\x00")
        if len(fields) != 4:
            errors.append("unparseable Git identity record")
            continue
        author = (fields[0], fields[1])
        committer = (fields[2], fields[3])
        if author not in ALLOWED_GIT_IDENTITIES or committer not in ALLOWED_GIT_IDENTITIES:
            errors.append("non-anonymous identity exists in visible Git history")
            break
    return "PASS" if not any("Git" in item or "identity" in item for item in errors) else "FAIL"


def main() -> int:
    errors = []
    for relative in REQUIRED:
        if not (ROOT / relative).is_file():
            errors.append(f"missing: {relative}")

    excluded_generated = {"MANIFEST.sha256", "RELEASE_AUDIT.json"}
    files = [
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and path.name not in excluded_generated
        and ".git" not in path.relative_to(ROOT).parts
    ]
    for path in files:
        relative = path.relative_to(ROOT).as_posix()
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            errors.append(f"generated cache included: {relative}")
        if path.suffix.lower() in FORBIDDEN_RELEASE_SUFFIXES:
            errors.append(f"out-of-scope archive/checkpoint/manuscript included: {relative}")
        if relative != "scripts/verify_release.py":
            is_text = path.suffix.lower() in TEXT_SUFFIXES
            scan_payload(relative, path.read_bytes(), errors, binary=not is_text)
            # NPZ and similar ZIP containers can hide identity-bearing strings
            # in compressed members, so inspect each decompressed payload too.
            if zipfile.is_zipfile(path):
                with zipfile.ZipFile(path) as archive:
                    for member in archive.infolist():
                        if not member.is_dir():
                            scan_payload(
                                f"{relative}!{member.filename}",
                                archive.read(member),
                                errors,
                                binary=True,
                            )

    python_files = [
        path
        for path in ROOT.rglob("*.py")
        if "__pycache__" not in path.parts
        and ".git" not in path.relative_to(ROOT).parts
    ]
    for path in python_files:
        try:
            compile(path.read_text(encoding="utf-8"), str(path), "exec")
        except (SyntaxError, UnicodeError) as exc:
            errors.append(f"source compilation failed: {path.relative_to(ROOT).as_posix()}: {exc}")
    test_env = dict(os.environ)
    test_env["PYTHONDONTWRITEBYTECODE"] = "1"
    test_result = subprocess.run(
        [sys.executable, "-B", "-m", "unittest", "discover", "-s", "tests", "-v"],
        cwd=ROOT,
        env=test_env,
    )
    if test_result.returncode:
        errors.append("unit tests failed")

    git_history = audit_git_history(errors)

    manifest_lines = [f"{digest(path)}  {path.relative_to(ROOT).as_posix()}" for path in sorted(files)]
    with (ROOT / "MANIFEST.sha256").open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(manifest_lines) + "\n")
    report = {
        "state": "PASS" if not errors else "FAIL",
        "files": len(files),
        "manifest_entries": len(manifest_lines),
        "generated_files_excluded": ["MANIFEST.sha256", "RELEASE_AUDIT.json"],
        "manifest_sha256": digest(ROOT / "MANIFEST.sha256"),
        "git_history": git_history,
        "errors": errors,
    }
    with (ROOT / "RELEASE_AUDIT.json").open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())

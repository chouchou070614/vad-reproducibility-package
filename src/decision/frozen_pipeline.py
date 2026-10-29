"""KMTF-VAD branch scoring, ECDF calibration, fusion, and temporal refinement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np


EPS = 1e-8


@dataclass(frozen=True)
class DatasetConfig:
    alpha_r: float
    alpha_p: float
    weight_r: float
    smoother: str
    parameter: float
    warmup_epoch: int
    epochs: int


DATASET_CONFIGS: Mapping[str, DatasetConfig] = {
    "ped2": DatasetConfig(0.7, 0.6, 0.40, "gaussian", 3.0, 26, 60),
    "avenue": DatasetConfig(0.7, 0.7, 0.10, "gaussian", 3.0, 27, 60),
    "shanghai": DatasetConfig(0.7, 0.6, 0.55, "ema", 0.725, 5, 10),
}


def _one_dimensional(name: str, values: Sequence[float]) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or result.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains NaN or infinity")
    return result


def _contiguous_video_slices(video_ids: Sequence[int], expected: int) -> Iterable[np.ndarray]:
    ids = np.asarray(video_ids)
    if ids.ndim != 1 or ids.size != expected:
        raise ValueError("video_ids must be one-dimensional and match score length")
    starts = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1]
    stops = np.r_[starts[1:], ids.size]
    seen = set()
    for start, stop in zip(starts.tolist(), stops.tolist()):
        key = ids[start].item() if hasattr(ids[start], "item") else ids[start]
        if key in seen:
            raise ValueError(f"video id {key!r} occurs in disjoint blocks")
        seen.add(key)
        yield np.arange(start, stop, dtype=np.int64)


def _minmax_per_video(values: np.ndarray, video_ids: Sequence[int]) -> np.ndarray:
    output = np.empty_like(values)
    for index in _contiguous_video_slices(video_ids, values.size):
        current = values[index]
        span = float(np.max(current) - np.min(current))
        if span <= 0.0:
            raise ValueError("each per-video component stream must have non-zero range")
        output[index] = (current - float(np.min(current))) / span
    return output


def branch_score(
    pixel_mse: Sequence[float],
    memory_distance: Sequence[float],
    video_ids: Sequence[int],
    alpha: float,
) -> np.ndarray:
    """Equation (4) in the manuscript, with complete-video normalization."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0,1]")
    pixel = _one_dimensional("pixel_mse", pixel_mse)
    memory = _one_dimensional("memory_distance", memory_distance)
    if pixel.size != memory.size or np.any(pixel <= 0.0):
        raise ValueError("component arrays must match and pixel MSE must be positive")
    psnr = 10.0 * np.log10(1.0 / pixel)
    psnr_normality = _minmax_per_video(psnr, video_ids)
    memory_anomaly = _minmax_per_video(memory, video_ids)
    return alpha * (1.0 - psnr_normality) + (1.0 - alpha) * memory_anomaly


def ecdf_tail(normal_scores: Sequence[float], values: Sequence[float]) -> np.ndarray:
    """Normal-training high-tail ECDF in Eq. (5)--(6)."""
    normal = np.sort(_one_dimensional("normal_scores", normal_scores))
    target = _one_dimensional("values", values)
    count = np.searchsorted(normal, target, side="right").astype(np.float64)
    n = float(normal.size)
    u = (count + 0.5) / (n + 1.0)
    limit = 1.0 / (2.0 * (n + 1.0))
    u = np.clip(u, limit, 1.0 - limit)
    return -np.log(1.0 - u + EPS)


def exact_align(
    r_video: Sequence[str],
    r_frame: Sequence[int],
    r_score: Sequence[float],
    p_video: Sequence[str],
    p_frame: Sequence[int],
    p_score: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Align R/P scores on shared video/frame keys."""
    rv, rf, rs = np.asarray(r_video).astype(str), np.asarray(r_frame), _one_dimensional("r_score", r_score)
    pv, pf, ps = np.asarray(p_video).astype(str), np.asarray(p_frame), _one_dimensional("p_score", p_score)
    if rv.size != rf.size or rv.size != rs.size or pv.size != pf.size or pv.size != ps.size:
        raise ValueError("key and score lengths do not match")
    p_map: Dict[Tuple[str, int], float] = {}
    for video, frame, score in zip(pv.tolist(), pf.tolist(), ps.tolist()):
        key = (str(video), int(frame))
        if key in p_map:
            raise ValueError(f"duplicate prediction key: {key}")
        p_map[key] = float(score)
    keys, aligned_r, aligned_p = [], [], []
    seen_r = set()
    for video, frame, score in zip(rv.tolist(), rf.tolist(), rs.tolist()):
        key = (str(video), int(frame))
        if key in seen_r:
            raise ValueError(f"duplicate reconstruction key: {key}")
        seen_r.add(key)
        if key in p_map:
            keys.append(key)
            aligned_r.append(float(score))
            aligned_p.append(p_map[key])
    if not keys:
        raise ValueError("R/P exact-key intersection is empty")
    return (
        np.asarray([key[0] for key in keys]),
        np.asarray([key[1] for key in keys], dtype=np.int64),
        np.asarray(aligned_r, dtype=np.float64),
        np.asarray(aligned_p, dtype=np.float64),
    )


def fixed_fuse(z_r: Sequence[float], z_p: Sequence[float], weight_r: float) -> np.ndarray:
    r, p = _one_dimensional("z_r", z_r), _one_dimensional("z_p", z_p)
    if r.size != p.size or not 0.0 <= weight_r <= 1.0:
        raise ValueError("aligned score lengths and weight must be valid")
    return weight_r * r + (1.0 - weight_r) * p


def _gaussian(values: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    radius = int(np.ceil(3.0 * sigma))
    offset = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-(offset * offset) / (2.0 * sigma * sigma))
    kernel /= np.sum(kernel)
    padded = np.pad(values, (radius, radius), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _ema(values: np.ndarray, rho: float) -> np.ndarray:
    if not 0.0 <= rho < 1.0:
        raise ValueError("rho must be in [0,1)")
    output = np.empty_like(values)
    output[0] = values[0]
    for index in range(1, values.size):
        output[index] = rho * output[index - 1] + (1.0 - rho) * values[index]
    return output


def refine_per_video(
    fused: Sequence[float],
    video_ids: Sequence[int],
    smoother: str,
    parameter: float,
) -> np.ndarray:
    values = _one_dimensional("fused", fused)
    output = np.empty_like(values)
    for index in _contiguous_video_slices(video_ids, values.size):
        if smoother == "gaussian":
            output[index] = _gaussian(values[index], parameter)
        elif smoother == "ema":
            output[index] = _ema(values[index], parameter)
        else:
            raise ValueError("smoother must be 'gaussian' or 'ema'")
    return output

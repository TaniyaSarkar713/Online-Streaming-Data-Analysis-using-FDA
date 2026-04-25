from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def denoise_curves(X: np.ndarray, window: int = 7) -> np.ndarray:
    """
    Simple reflection-padded moving-average denoising for streaming curves.
    """
    win = int(max(3, window))
    if win % 2 == 0:
        win += 1
    kernel = np.ones(win, dtype=float) / win
    pad = win // 2

    out = np.zeros_like(X)
    for i in range(X.shape[0]):
        xp = np.pad(X[i], (pad, pad), mode="reflect")
        out[i] = np.convolve(xp, kernel, mode="valid")
    return np.clip(out, 0.0, None)


def missingness_features(X: np.ndarray, grid: np.ndarray, eps: float = 1e-8) -> pd.DataFrame:
    """
    Missingness-aware stream observability descriptors.
    """
    n, m = X.shape
    obs = X > eps

    visible_fraction = obs.mean(axis=1)
    first_idx = np.argmax(obs, axis=1)
    has_obs = obs.any(axis=1)
    first_idx = np.where(has_obs, first_idx, m - 1)

    last_idx = m - 1 - np.argmax(obs[:, ::-1], axis=1)
    last_idx = np.where(has_obs, last_idx, 0)

    first_time = grid[first_idx]
    last_time = grid[last_idx]
    observed_span = np.maximum(0.0, last_time - first_time)
    start_cutoff = (first_idx > 0).astype(float)
    end_cutoff = (last_idx < (m - 1)).astype(float)

    num_segments = np.zeros(n, dtype=float)
    num_gaps = np.zeros(n, dtype=float)
    longest_gap = np.zeros(n, dtype=float)
    internal_missing_fraction = np.zeros(n, dtype=float)

    for i in range(n):
        oi = obs[i].astype(int)
        d = np.diff(np.r_[0, oi, 0])
        starts = np.where(d == 1)[0]
        ends = np.where(d == -1)[0]
        seg_lens = ends - starts
        num_segments[i] = len(seg_lens)
        num_gaps[i] = max(0, len(seg_lens) - 1)

        zero = 1 - oi
        dz = np.diff(np.r_[0, zero, 0])
        zstarts = np.where(dz == 1)[0]
        zends = np.where(dz == -1)[0]
        z_lens = zends - zstarts
        longest_gap[i] = (z_lens.max() / m) if len(z_lens) else 0.0

        if has_obs[i] and last_idx[i] > first_idx[i]:
            mid = oi[first_idx[i] : last_idx[i] + 1]
            internal_missing_fraction[i] = float((mid == 0).mean())
        else:
            internal_missing_fraction[i] = 1.0

    return pd.DataFrame(
        {
            "visible_fraction": visible_fraction,
            "first_seen_time": first_time,
            "last_seen_time": last_time,
            "observed_span": observed_span,
            "start_cutoff": start_cutoff,
            "end_cutoff": end_cutoff,
            "num_observed_segments": num_segments,
            "num_internal_gaps": num_gaps,
            "longest_gap_fraction": longest_gap,
            "internal_missing_fraction": internal_missing_fraction,
        }
    )


def raw_summary_features(X: np.ndarray) -> pd.DataFrame:
    """Tabular baseline summaries that discard timing shape."""
    activity_count = (X > 0).sum(axis=1)
    return pd.DataFrame(
        {
            "mean": X.mean(axis=1),
            "std": X.std(axis=1),
            "max": X.max(axis=1),
            "count": activity_count,
            "total_points": X.sum(axis=1),
        }
    ).fillna(0.0)


def functional_shape_features(X: np.ndarray, grid: np.ndarray) -> pd.DataFrame:
    """Timing-aware functional features."""
    early = grid <= 0.30
    mid = (grid > 0.30) & (grid <= 0.70)
    late = grid > 0.70

    total = np.trapezoid(X, grid, axis=1)
    peak = X.max(axis=1)
    peak_time = grid[np.argmax(X, axis=1)]
    early_mass = np.trapezoid(X[:, early], grid[early], axis=1)
    mid_mass = np.trapezoid(X[:, mid], grid[mid], axis=1)
    late_mass = np.trapezoid(X[:, late], grid[late], axis=1)

    sustainedness = total / (peak + 1e-8)
    early_ratio = early_mass / (total + 1e-8)
    late_ratio = late_mass / (total + 1e-8)
    roughness = np.mean(np.abs(np.gradient(X, axis=1)), axis=1)

    return pd.DataFrame(
        {
            "total_activity": total,
            "peak_intensity": peak,
            "peak_time": peak_time,
            "early_mass": early_mass,
            "mid_mass": mid_mass,
            "late_mass": late_mass,
            "early_ratio": early_ratio,
            "late_ratio": late_ratio,
            "sustainedness": sustainedness,
            "roughness": roughness,
        }
    )


def fpca_scores(
    X: np.ndarray,
    n_components: int = 6,
    prefix: str = "fpca",
) -> Tuple[pd.DataFrame, PCA]:
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(X)
    columns = [f"{prefix}_{j + 1}" for j in range(n_components)]
    return pd.DataFrame(scores, columns=columns), pca

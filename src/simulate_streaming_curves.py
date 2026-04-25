from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SimulationConfig:
    n_users: int = 1200
    grid_size: int = 120
    noise_sd: float = 0.03
    shift_sd: float = 0.13
    cutoff_probability: float = 0.70
    min_visible_fraction: float = 0.35
    max_visible_fraction: float = 0.95
    dropout_probability: float = 0.40
    min_dropout_fraction: float = 0.05
    max_dropout_fraction: float = 0.22
    burst_noise_probability: float = 0.55
    seed: int = 713


def gaussian_bump(t: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return np.exp(-0.5 * ((t - mu) / sigma) ** 2)


def _apply_shift(curve: np.ndarray, grid: np.ndarray, shift: float) -> np.ndarray:
    return np.interp(grid + shift, grid, curve, left=0.0, right=0.0)


def _stream_observation_mask(grid_size: int, cfg: SimulationConfig, rng: np.random.Generator) -> np.ndarray:
    """
    Realistic stream visibility mask:
    - Main observed window (stream starts/ends at random places)
    - Optional random dropout inside the window (disconnects/pauses)
    """
    mask = np.ones(grid_size, dtype=float)

    if rng.random() < cfg.cutoff_probability:
        visible_fraction = rng.uniform(cfg.min_visible_fraction, cfg.max_visible_fraction)
        visible_len = max(2, int(round(visible_fraction * grid_size)))
        start = int(rng.integers(0, grid_size - visible_len + 1))
        end = start + visible_len
        mask[:] = 0.0
        mask[start:end] = 1.0

    if rng.random() < cfg.dropout_probability:
        drop_fraction = rng.uniform(cfg.min_dropout_fraction, cfg.max_dropout_fraction)
        drop_len = max(1, int(round(drop_fraction * grid_size)))
        drop_start = int(rng.integers(0, grid_size - drop_len + 1))
        drop_end = drop_start + drop_len
        mask[drop_start:drop_end] = 0.0

    return mask


def simulate_streaming_monetization(
    config: SimulationConfig | None = None,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Controlled simulation where timing and shape matter beyond activity totals.
    Returns:
        X: (n_users, grid_size) nonnegative transaction intensity curves.
        grid: normalized time grid in [0, 1].
        truth: latent drivers and generated outcomes.
    """
    cfg = config or SimulationConfig()
    rng = np.random.default_rng(cfg.seed)

    grid = np.linspace(0.0, 1.0, cfg.grid_size)
    X = np.zeros((cfg.n_users, cfg.grid_size), dtype=float)
    X_canonical = np.zeros((cfg.n_users, cfg.grid_size), dtype=float)

    user_type = rng.choice(
        ["early_converter", "late_gifter", "sustained_supporter", "two_peak_social"],
        size=cfg.n_users,
        p=[0.30, 0.25, 0.25, 0.20],
    )
    true_shift = rng.normal(0.0, cfg.shift_sd, size=cfg.n_users)
    amplitude = rng.lognormal(mean=0.0, sigma=0.08, size=cfg.n_users)
    streamer_quality = rng.normal(0.0, 1.0, size=cfg.n_users)

    for i in range(cfg.n_users):
        amp = amplitude[i]

        if user_type[i] == "early_converter":
            base = (
                1.20 * gaussian_bump(grid, 0.24, 0.055)
                + 0.24 * gaussian_bump(grid, 0.57, 0.11)
            )
        elif user_type[i] == "late_gifter":
            base = (
                0.26 * gaussian_bump(grid, 0.30, 0.10)
                + 1.15 * gaussian_bump(grid, 0.72, 0.060)
            )
        elif user_type[i] == "sustained_supporter":
            base = (
                0.62 * gaussian_bump(grid, 0.32, 0.11)
                + 0.62 * gaussian_bump(grid, 0.64, 0.11)
                + 0.28
            )
        else:
            base = (
                0.82 * gaussian_bump(grid, 0.36, 0.055)
                + 0.82 * gaussian_bump(grid, 0.68, 0.055)
            )

        # Normalize shapes so outcome cannot be trivially decoded from totals alone.
        base = base / (np.trapezoid(base, grid) + 1e-8)
        canonical = amp * base
        shifted = _apply_shift(canonical, grid, true_shift[i])
        noisy = shifted + rng.normal(0.0, cfg.noise_sd, size=cfg.grid_size)

        # Add localized burst noise to emulate sudden chat/connection volatility.
        if rng.random() < cfg.burst_noise_probability:
            burst_center = float(rng.uniform(0.08, 0.92))
            burst_width = float(rng.uniform(0.03, 0.11))
            burst_amp = float(rng.uniform(0.6, 1.4) * cfg.noise_sd)
            burst = burst_amp * np.exp(-0.5 * ((grid - burst_center) / burst_width) ** 2)
            noisy = noisy + rng.normal(0.0, 1.0, cfg.grid_size) * burst

        mask = _stream_observation_mask(cfg.grid_size, cfg, rng)
        observed = noisy * mask

        X[i] = np.clip(observed, 0.0, None)
        X_canonical[i] = canonical

    early_mask = grid <= 0.30
    mid_mask = (grid > 0.30) & (grid <= 0.70)
    late_mask = grid > 0.70

    total_mass = np.trapezoid(X_canonical, grid, axis=1)
    early_mass = np.trapezoid(X_canonical[:, early_mask], grid[early_mask], axis=1)
    mid_mass = np.trapezoid(X_canonical[:, mid_mask], grid[mid_mask], axis=1)
    late_mass = np.trapezoid(X_canonical[:, late_mask], grid[late_mask], axis=1)
    peak_time = grid[np.argmax(X_canonical, axis=1)]
    sustainedness = total_mass / (X_canonical.max(axis=1) + 1e-8)
    early_ratio = early_mass / (total_mass + 1e-8)
    late_ratio = late_mass / (total_mass + 1e-8)

    revenue_score = (
        2.95 * np.tanh(5.0 * early_ratio)
        + 1.90 * np.tanh(2.8 * sustainedness)
        - 1.95 * np.tanh(4.2 * late_ratio)
        - 1.70 * np.abs(true_shift)
        + 0.50 * streamer_quality
        + 0.30 * np.sin(4 * np.pi * peak_time)
    )
    y_revenue = 1000 + 480 * revenue_score + rng.normal(0.0, 65.0, size=cfg.n_users)

    logits = (
        -0.25
        + 3.10 * early_ratio
        + 1.35 * sustainedness
        - 2.50 * late_ratio
        - 1.65 * np.abs(true_shift)
        + 0.35 * streamer_quality
    )
    subscribe_prob = 1.0 / (1.0 + np.exp(-logits))
    y_subscribe = rng.binomial(1, subscribe_prob, size=cfg.n_users)

    truth = pd.DataFrame(
        {
            "user_type": user_type,
            "true_shift": true_shift,
            "streamer_quality": streamer_quality,
            "revenue": y_revenue,
            "subscribe": y_subscribe,
            "total_mass_true": total_mass,
            "early_mass_true": early_mass,
            "mid_mass_true": mid_mass,
            "late_mass_true": late_mass,
            "early_ratio_true": early_ratio,
            "late_ratio_true": late_ratio,
            "sustainedness_true": sustainedness,
            "peak_time_true": peak_time,
        }
    )
    return X, grid, truth

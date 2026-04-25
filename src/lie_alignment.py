from __future__ import annotations

from typing import Tuple

import numpy as np


def estimate_lie_shift_identity(
    x: np.ndarray,
    ref: np.ndarray,
    grid: np.ndarray,
    eta: float = 0.10,
    n_steps: int = 8,
    initial_shift: float = 0.0,
    max_shift: float = 0.30,
) -> float:
    """
    Iterative Lie-gradient descent for gamma_a(t)=t+a.
    L(a) = integral (x(t+a)-ref(t))^2 dt
    dL/da   = 2 integral (x(t+a)-ref(t)) x'(t+a) dt
    """
    dx = np.gradient(x, grid)
    shift = float(initial_shift)
    for _ in range(n_steps):
        shifted_x = np.interp(grid + shift, grid, x, left=0.0, right=0.0)
        shifted_dx = np.interp(grid + shift, grid, dx, left=0.0, right=0.0)
        grad = 2.0 * np.trapezoid((shifted_x - ref) * shifted_dx, grid)
        shift = float(np.clip(shift - eta * grad, -max_shift, max_shift))
    return shift


def shift_curve(x: np.ndarray, grid: np.ndarray, shift: float) -> np.ndarray:
    """Evaluate x(t + shift) on grid, zero-filling outside [0, 1]."""
    return np.interp(grid + shift, grid, x, left=0.0, right=0.0)


def align_curves(
    X: np.ndarray,
    grid: np.ndarray,
    eta: float = 0.10,
    max_shift: float = 0.25,
    n_iter: int = 2,
    inner_steps: int = 8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Iterative local alignment around identity.
    Returns aligned curves, cumulative shifts, and reference curve.
    """
    X_work = X.copy()
    total_shifts = np.zeros(X.shape[0], dtype=float)

    for _ in range(n_iter):
        ref = X_work.mean(axis=0)
        total_shifts = np.array(
            [
                estimate_lie_shift_identity(
                    X[i],
                    ref,
                    grid,
                    eta=eta,
                    n_steps=inner_steps,
                    initial_shift=total_shifts[i],
                    max_shift=max_shift,
                )
                for i in range(X.shape[0])
            ]
        )
        # Apply inverse shift to register observed curves to the reference frame.
        X_work = np.vstack(
            [shift_curve(X[i], grid, -total_shifts[i]) for i in range(X.shape[0])]
        )

    return X_work, total_shifts, X_work.mean(axis=0)


def _mass_landmarks(
    x: np.ndarray,
    grid: np.ndarray,
    quantiles: tuple[float, ...] = (0.25, 0.50, 0.75),
) -> np.ndarray:
    """Landmarks by cumulative mass quantiles, robust to noisy peak locations."""
    mass = np.clip(x, 0.0, None)
    total = np.trapezoid(mass, grid)
    if total <= 1e-10:
        return np.array(quantiles, dtype=float)

    dt = np.gradient(grid)
    weighted = mass * dt
    cdf = np.cumsum(weighted)
    cdf = cdf / (cdf[-1] + 1e-12)
    return np.interp(np.array(quantiles, dtype=float), cdf, grid)


def _strictly_increasing_knots(
    knots: np.ndarray,
    low: float = 0.0,
    high: float = 1.0,
    eps: float = 1e-4,
) -> np.ndarray:
    out = np.clip(knots.astype(float), low, high)
    out = np.maximum.accumulate(out)
    for i in range(1, len(out)):
        if out[i] <= out[i - 1]:
            out[i] = min(high, out[i - 1] + eps)
    out[-1] = min(high, out[-1])
    return out


def piecewise_warp_curve(
    x: np.ndarray,
    grid: np.ndarray,
    source_knots: np.ndarray,
    target_knots: np.ndarray,
) -> np.ndarray:
    """
    Piecewise-linear warping y(t)=x(gamma(t)),
    where gamma maps target timeline -> source timeline.
    """
    gamma = np.interp(grid, target_knots, source_knots)
    return np.interp(gamma, grid, x, left=0.0, right=0.0)


def piecewise_landmark_alignment(
    X: np.ndarray,
    grid: np.ndarray,
    reference: np.ndarray | None = None,
    quantiles: tuple[float, ...] = (0.25, 0.50, 0.75),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Align curves with piecewise-linear warping based on mass landmarks.
    Returns:
      - warped curves
      - per-curve source landmarks (without 0/1 anchors)
      - target landmarks (without 0/1 anchors)
    """
    ref = X.mean(axis=0) if reference is None else reference
    target_internal = _mass_landmarks(ref, grid, quantiles=quantiles)
    target_knots = _strictly_increasing_knots(
        np.concatenate(([0.0], target_internal, [1.0]))
    )

    warped = np.zeros_like(X)
    src_internal_all = np.zeros((X.shape[0], len(quantiles)))

    for i in range(X.shape[0]):
        src_internal = _mass_landmarks(X[i], grid, quantiles=quantiles)
        src_internal_all[i] = src_internal
        src_knots = _strictly_increasing_knots(np.concatenate(([0.0], src_internal, [1.0])))
        warped[i] = piecewise_warp_curve(X[i], grid, src_knots, target_knots)

    return warped, src_internal_all, target_internal

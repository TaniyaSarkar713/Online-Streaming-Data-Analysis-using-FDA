"""
Notebook-friendly driver for:
Lie-Gradient Functional Data Analysis for Monetization Prediction
in Online Streaming Platforms.

This script is intentionally linear so it can be copied into Colab cells.
Core reusable logic lives in src/.
"""

from __future__ import annotations

from pathlib import Path
import sys
import os

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
FIG_DIR = ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.benchmark_models import run_classification_benchmarks, run_regression_benchmarks
from src.evaluation import calculate_robust_metrics
from src.functional_features import (
    denoise_curves,
    fpca_scores,
    functional_shape_features,
    missingness_features,
    raw_summary_features,
)
from src.lie_alignment import align_curves, piecewise_landmark_alignment
from src.simulate_streaming_curves import SimulationConfig, simulate_streaming_monetization


# ------------------------------------------------------------
# 1) Controlled simulation where timing/shape matter
# ------------------------------------------------------------
config = SimulationConfig(
    n_users=1200,
    grid_size=120,
    noise_sd=0.03,
    shift_sd=0.13,
    seed=713,
)

X_sim, grid, truth = simulate_streaming_monetization(config)
X_denoised = denoise_curves(X_sim, window=7)
missing_df = missingness_features(X_sim, grid)
print("Simulated curve matrix:", X_sim.shape)
print(truth[["user_type", "revenue", "subscribe", "true_shift"]].head())

# Quick visual
plt.figure(figsize=(10, 5))
for i in range(20):
    plt.plot(grid, X_sim[i], alpha=0.35)
plt.title("Simulated Online Streaming Transaction Curves")
plt.xlabel("Normalized stream time")
plt.ylabel("Points/gifts intensity")
plt.grid(True)
plt.tight_layout()
plt.savefig(FIG_DIR / "simulation_curves_overview.png", dpi=180)
plt.close()


# ------------------------------------------------------------
# 2) Lie-gradient alignment
# ------------------------------------------------------------
X_aligned, estimated_shift, aligned_ref = align_curves(
    X_denoised,
    grid,
    eta=0.08,
    max_shift=0.30,
    n_iter=2,
    inner_steps=8,
)
X_piecewise, source_landmarks, target_landmarks = piecewise_landmark_alignment(
    X_aligned,
    grid,
    reference=aligned_ref,
    quantiles=(0.25, 0.50, 0.75),
)

shift_corr = np.corrcoef(truth["true_shift"], estimated_shift)[0, 1]
print(f"Correlation(true shift, estimated Lie shift): {shift_corr:.4f}")

plt.figure(figsize=(10, 5))
for i in range(20):
    plt.plot(grid, X_aligned[i], alpha=0.35)
plt.plot(grid, aligned_ref, linewidth=3, color="black", label="Aligned mean")
plt.title("Lie-Aligned Curves")
plt.xlabel("Normalized stream time")
plt.ylabel("Aligned intensity")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(FIG_DIR / "lie_aligned_curves_overview.png", dpi=180)
plt.close()


# ------------------------------------------------------------
# 3) Feature construction
# ------------------------------------------------------------
raw_features = raw_summary_features(X_sim)
raw_with_missing = pd.concat([raw_features, missing_df], axis=1)
fda_unaligned = functional_shape_features(X_denoised, grid)
fda_aligned = functional_shape_features(X_aligned, grid)
fda_aligned["lie_shift"] = estimated_shift
fda_aligned["abs_lie_shift"] = np.abs(estimated_shift)
fda_piecewise = functional_shape_features(X_piecewise, grid)
fda_piecewise["lie_shift"] = estimated_shift
fda_piecewise["abs_lie_shift"] = np.abs(estimated_shift)
fda_piecewise["landmark_q25"] = source_landmarks[:, 0]
fda_piecewise["landmark_q50"] = source_landmarks[:, 1]
fda_piecewise["landmark_q75"] = source_landmarks[:, 2]
fda_piecewise["target_q25"] = target_landmarks[0]
fda_piecewise["target_q50"] = target_landmarks[1]
fda_piecewise["target_q75"] = target_landmarks[2]
fda_piecewise_with_missing = pd.concat([fda_piecewise, missing_df], axis=1)

fpca_raw_scores, _ = fpca_scores(X_denoised, n_components=6, prefix="raw_fpca")
fpca_piecewise_scores, _ = fpca_scores(X_piecewise, n_components=6, prefix="piecewise_fpca")


# ------------------------------------------------------------
# 4) Revenue benchmarks
# ------------------------------------------------------------
feature_sets_revenue = {
    "Raw summaries + Ridge": raw_features,
    "Raw summaries + missingness + Random Forest": raw_with_missing,
    "Raw summaries + missingness + Gradient Boosting": raw_with_missing,
    "Raw summaries + missingness + MLP": raw_with_missing,
    "Raw FPCA + Ridge": fpca_raw_scores,
    "Unaligned FDA + Ridge": fda_unaligned,
    "Lie-aligned FDA + Ridge": fda_aligned,
    "Lie-aligned FDA + Random Forest": fda_aligned,
    "Lie-aligned FDA + Gradient Boosting": fda_aligned,
    "Lie-piecewise FPCA + Ridge": fpca_piecewise_scores,
    "Lie-piecewise FDA + missingness + Ridge": fda_piecewise_with_missing,
    "Lie-piecewise FDA + missingness + Random Forest": fda_piecewise_with_missing,
    "Lie-piecewise FDA + missingness + Gradient Boosting": fda_piecewise_with_missing,
}

regression_table = run_regression_benchmarks(
    feature_sets_revenue,
    truth["revenue"].to_numpy(),
    test_size=0.30,
    random_state=713,
)

print("\nRevenue benchmark table (sorted by RMSE):")
print(regression_table.to_string(index=False))


# ------------------------------------------------------------
# 5) Subscription conversion benchmarks
# ------------------------------------------------------------
feature_sets_subscribe = {
    "Raw summaries + Logistic": raw_features,
    "Raw summaries + missingness + Random Forest": raw_with_missing,
    "Raw summaries + missingness + Gradient Boosting": raw_with_missing,
    "Raw FPCA + Logistic": fpca_raw_scores,
    "Unaligned FDA + Logistic": fda_unaligned,
    "Lie-aligned FDA + Logistic": fda_aligned,
    "Lie-aligned FDA + Gradient Boosting": fda_aligned,
    "Lie-piecewise FDA + missingness + Logistic": fda_piecewise_with_missing,
    "Lie-piecewise FDA + missingness + Gradient Boosting": fda_piecewise_with_missing,
}

classification_table = run_classification_benchmarks(
    feature_sets_subscribe,
    truth["subscribe"].to_numpy(),
    test_size=0.30,
    random_state=713,
)

print("\nSubscription benchmark table (sorted by AUC):")
print(classification_table.to_string(index=False))


# ------------------------------------------------------------
# 6) Explicit robust metrics fix demo
# ------------------------------------------------------------
# This addresses the previous undefined function issue.
best_row = regression_table.iloc[0]
print("\nBest revenue model:", best_row["Model"])

# Example usage of robust metrics function remains available here:
example_metrics = calculate_robust_metrics(
    y_true=np.array([1.0, 2.0, 3.0]),
    y_pred=np.array([1.1, 2.2, 2.9]),
    model_name="sanity_check",
)
print("Sanity-check robust metrics:", example_metrics)

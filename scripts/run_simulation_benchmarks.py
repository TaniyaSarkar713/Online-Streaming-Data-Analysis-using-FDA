from __future__ import annotations

import argparse
from pathlib import Path
import sys
import os

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.benchmark_models import run_classification_benchmarks, run_regression_benchmarks
from src.functional_features import (
    denoise_curves,
    fpca_scores,
    functional_shape_features,
    missingness_features,
    raw_summary_features,
)
from src.lie_alignment import align_curves, piecewise_landmark_alignment
from src.simulate_streaming_curves import SimulationConfig, simulate_streaming_monetization


def run_experiment(
    n_users: int = 1200,
    grid_size: int = 120,
    seed: int = 713,
    out_dir: Path = Path("results"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    X, grid, truth = simulate_streaming_monetization(
        SimulationConfig(n_users=n_users, grid_size=grid_size, seed=seed)
    )
    X_denoised = denoise_curves(X, window=7)
    miss = missingness_features(X, grid)

    X_aligned, est_shift, ref_curve = align_curves(
        X_denoised,
        grid,
        eta=0.08,
        max_shift=0.30,
        n_iter=2,
        inner_steps=8,
    )
    X_piecewise, source_landmarks, target_landmarks = piecewise_landmark_alignment(
        X_aligned, grid, reference=ref_curve, quantiles=(0.25, 0.50, 0.75)
    )

    raw = raw_summary_features(X)
    raw_with_missing = pd.concat([raw, miss], axis=1)
    fda_unaligned = functional_shape_features(X_denoised, grid)
    fda_aligned = functional_shape_features(X_aligned, grid)
    fda_aligned["lie_shift"] = est_shift
    fda_aligned["abs_lie_shift"] = np.abs(est_shift)
    fda_piecewise = functional_shape_features(X_piecewise, grid)
    fda_piecewise["lie_shift"] = est_shift
    fda_piecewise["abs_lie_shift"] = np.abs(est_shift)
    fda_piecewise["landmark_q25"] = source_landmarks[:, 0]
    fda_piecewise["landmark_q50"] = source_landmarks[:, 1]
    fda_piecewise["landmark_q75"] = source_landmarks[:, 2]
    fda_piecewise["target_q25"] = target_landmarks[0]
    fda_piecewise["target_q50"] = target_landmarks[1]
    fda_piecewise["target_q75"] = target_landmarks[2]
    fda_piecewise_with_missing = pd.concat([fda_piecewise, miss], axis=1)

    fpca_raw, _ = fpca_scores(X_denoised, n_components=6, prefix="raw_fpca")
    fpca_piecewise, _ = fpca_scores(X_piecewise, n_components=6, prefix="piecewise_fpca")

    feature_sets_reg = {
        "Raw summaries + Ridge": raw,
        "Raw summaries + missingness + Random Forest": raw_with_missing,
        "Raw summaries + missingness + Gradient Boosting": raw_with_missing,
        "Raw summaries + missingness + MLP": raw_with_missing,
        "Raw FPCA + Ridge": fpca_raw,
        "Unaligned FDA + Ridge": fda_unaligned,
        "Lie-aligned FDA + Ridge": fda_aligned,
        "Lie-aligned FDA + Random Forest": fda_aligned,
        "Lie-aligned FDA + Gradient Boosting": fda_aligned,
        "Lie-piecewise FDA + missingness + Ridge": fda_piecewise_with_missing,
        "Lie-piecewise FPCA + Ridge": fpca_piecewise,
        "Lie-piecewise FDA + missingness + Random Forest": fda_piecewise_with_missing,
        "Lie-piecewise FDA + missingness + Gradient Boosting": fda_piecewise_with_missing,
    }
    reg_results = run_regression_benchmarks(feature_sets_reg, truth["revenue"].to_numpy())
    reg_results.to_csv(out_dir / "tables" / "revenue_benchmark.csv", index=False)

    feature_sets_cls = {
        "Raw summaries + Logistic": raw,
        "Raw summaries + missingness + Random Forest": raw_with_missing,
        "Raw summaries + missingness + Gradient Boosting": raw_with_missing,
        "Raw FPCA + Logistic": fpca_raw,
        "Unaligned FDA + Logistic": fda_unaligned,
        "Lie-aligned FDA + Logistic": fda_aligned,
        "Lie-aligned FDA + Gradient Boosting": fda_aligned,
        "Lie-piecewise FDA + missingness + Logistic": fda_piecewise_with_missing,
        "Lie-piecewise FDA + missingness + Gradient Boosting": fda_piecewise_with_missing,
    }
    cls_results = run_classification_benchmarks(
        feature_sets_cls, truth["subscribe"].to_numpy()
    )
    cls_results.to_csv(out_dir / "tables" / "subscription_benchmark.csv", index=False)

    _plot_benchmarks(reg_results, out_dir / "figures")
    truth.to_csv(out_dir / "tables" / "simulation_truth.csv", index=False)
    np.save(out_dir / "tables" / "simulated_curves.npy", X)
    np.save(out_dir / "tables" / "aligned_curves.npy", X_aligned)
    np.save(out_dir / "tables" / "piecewise_aligned_curves.npy", X_piecewise)

    return reg_results, cls_results


def _plot_benchmarks(reg_results: pd.DataFrame, figure_dir: Path) -> None:
    plt.figure(figsize=(11, 6))
    plt.barh(reg_results["Model"], reg_results["R2"])
    plt.title("Revenue Prediction Performance (Higher is Better)")
    plt.xlabel("Test R2")
    plt.gca().invert_yaxis()
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(figure_dir / "revenue_r2_comparison.png", dpi=180)
    plt.close()

    plt.figure(figsize=(11, 6))
    plt.barh(reg_results["Model"], reg_results["RMSE"])
    plt.title("Revenue Prediction Error (Lower is Better)")
    plt.xlabel("Test RMSE")
    plt.gca().invert_yaxis()
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(figure_dir / "revenue_rmse_comparison.png", dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Lie-FDA simulation benchmarks.")
    parser.add_argument("--n-users", type=int, default=1200)
    parser.add_argument("--grid-size", type=int, default=120)
    parser.add_argument("--seed", type=int, default=713)
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    reg_results, cls_results = run_experiment(
        n_users=args.n_users,
        grid_size=args.grid_size,
        seed=args.seed,
        out_dir=args.out_dir,
    )
    print("\nRegression benchmarks (top 5 by RMSE):")
    print(reg_results.head(5).to_string(index=False))
    print("\nClassification benchmarks (top 5 by AUC):")
    print(cls_results.head(5).to_string(index=False))


if __name__ == "__main__":
    main()

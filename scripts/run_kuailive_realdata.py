from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.benchmark_models import run_classification_benchmarks, run_regression_benchmarks
from src.kuailive_realdata import (
    KuaiLiveConfig,
    build_kuailive_curve_dataset,
    build_kuailive_feature_sets,
)


def run_kuailive_pipeline(
    data_dir: Path,
    out_dir: Path = Path("results"),
    grid_size: int = 120,
    min_interactions_per_curve: int = 3,
    max_samples: int | None = 150000,
    random_state: int = 713,
    light_mode: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    print("[KuaiLive] Building curve dataset...", flush=True)

    cfg = KuaiLiveConfig(
        data_dir=data_dir,
        grid_size=grid_size,
        min_interactions_per_curve=min_interactions_per_curve,
        max_samples=max_samples,
        random_state=random_state,
    )

    X_all, X_no_gift, grid, meta = build_kuailive_curve_dataset(cfg)
    print("[KuaiLive] Building feature sets (denoise + Lie + piecewise)...", flush=True)
    watch_reg_sets, gift_reg_sets, gift_cls_sets, diagnostics = build_kuailive_feature_sets(
        X_all, X_no_gift, grid, meta
    )
    if light_mode:
        watch_reg_sets = {
            k: v
            for k, v in watch_reg_sets.items()
            if ("Ridge" in k) or ("Gradient Boosting" in k and "Lie-piecewise" in k)
        }
        gift_reg_sets = {
            k: v
            for k, v in gift_reg_sets.items()
            if ("Ridge" in k) or ("Gradient Boosting" in k and "Lie-piecewise" in k)
        }
        gift_cls_sets = {
            k: v
            for k, v in gift_cls_sets.items()
            if ("Logistic" in k) or ("Gradient Boosting" in k and "Lie-piecewise" in k)
        }
        print(
            f"[KuaiLive] Light mode enabled: watch-reg={len(watch_reg_sets)}, "
            f"gift-reg={len(gift_reg_sets)}, gift-cls={len(gift_cls_sets)} models.",
            flush=True,
        )

    watch_y = meta["watch_time_target"].to_numpy()
    gift_y = meta["gift_price_target"].to_numpy()
    gift_any_y = meta["gift_any_target"].to_numpy()

    watch_results = run_regression_benchmarks(
        watch_reg_sets,
        watch_y,
        test_size=0.30,
        random_state=random_state,
    )
    print("[KuaiLive] Finished watch-time regression benchmarks.", flush=True)
    gift_results = run_regression_benchmarks(
        gift_reg_sets,
        gift_y,
        test_size=0.30,
        random_state=random_state,
    )
    print("[KuaiLive] Finished gift-price regression benchmarks.", flush=True)
    gift_cls_results = run_classification_benchmarks(
        gift_cls_sets,
        gift_any_y,
        test_size=0.30,
        random_state=random_state,
    )
    print("[KuaiLive] Finished gift occurrence classification benchmarks.", flush=True)

    watch_results.to_csv(out_dir / "tables" / "kuailive_watchtime_benchmark.csv", index=False)
    gift_results.to_csv(out_dir / "tables" / "kuailive_giftprice_benchmark.csv", index=False)
    gift_cls_results.to_csv(out_dir / "tables" / "kuailive_gift_binary_benchmark.csv", index=False)
    meta.to_csv(out_dir / "tables" / "kuailive_curve_metadata.csv", index=False)
    _plot_lie_alignment_quality(diagnostics, out_dir / "figures")

    return watch_results, gift_results, gift_cls_results


def _plot_lie_alignment_quality(diagnostics: dict[str, np.ndarray], figure_dir: Path) -> None:
    grid = diagnostics["grid"]
    ref = diagnostics["watch_ref"]
    pre = diagnostics["watch_pre_l2"]
    post = diagnostics["watch_post_l2"]
    gain = diagnostics["watch_gain"]
    before_examples = diagnostics["watch_before_examples"]
    after_examples = diagnostics["watch_after_examples"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    for i in range(before_examples.shape[0]):
        axes[0].plot(grid, before_examples[i], color="#999999", alpha=0.22, linewidth=1)
    for i in range(after_examples.shape[0]):
        axes[0].plot(grid, after_examples[i], color="#1f77b4", alpha=0.22, linewidth=1)
    axes[0].plot(grid, ref, color="black", linewidth=2.0, label="Aligned reference")
    axes[0].set_title("Sample Curves: Before (gray) vs After Lie (blue)")
    axes[0].set_xlabel("Normalized room time")
    axes[0].set_ylabel("Interaction intensity")
    axes[0].grid(alpha=0.25)

    axes[1].scatter(pre, post, alpha=0.14, s=8)
    lim_low = float(min(pre.min(), post.min()))
    lim_high = float(max(pre.max(), post.max()))
    axes[1].plot([lim_low, lim_high], [lim_low, lim_high], "r--", linewidth=1.5)
    axes[1].set_title("Curve-to-Reference Distance")
    axes[1].set_xlabel("Pre-alignment L2 distance")
    axes[1].set_ylabel("Post-alignment L2 distance")
    axes[1].grid(alpha=0.25)

    axes[2].hist(gain, bins=50, alpha=0.85, color="#2ca02c")
    axes[2].axvline(np.median(gain), color="black", linestyle="--", linewidth=1.6)
    axes[2].set_title(
        "Alignment Gain Distribution\n"
        f"median={(np.median(gain) * 100):.2f}%, mean={(np.mean(gain) * 100):.2f}%"
    )
    axes[2].set_xlabel("(pre - post) / pre")
    axes[2].set_ylabel("Count")
    axes[2].grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(figure_dir / "kuailive_lie_alignment_quality.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run KuaiLive real-data FDA/Lie-alignment benchmarks."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/KuaiLive"),
        help="Directory containing KuaiLive csv files (click/comment/like/gift/room/user/streamer).",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    parser.add_argument("--grid-size", type=int, default=120)
    parser.add_argument("--min-interactions-per-curve", type=int, default=3)
    parser.add_argument("--max-samples", type=int, default=150000)
    parser.add_argument("--seed", type=int, default=713)
    parser.add_argument(
        "--light-mode",
        action="store_true",
        help="Run complete-data pipeline with fewer benchmark models for faster turnaround.",
    )
    args = parser.parse_args()

    max_samples = None if args.max_samples <= 0 else args.max_samples
    watch, gift, gift_cls = run_kuailive_pipeline(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        grid_size=args.grid_size,
        min_interactions_per_curve=args.min_interactions_per_curve,
        max_samples=max_samples,
        random_state=args.seed,
        light_mode=args.light_mode,
    )
    print("\nKuaiLive watch-time regression (top 5 by RMSE):")
    print(watch.head(5).to_string(index=False))
    print("\nKuaiLive gift-price regression (top 5 by RMSE):")
    print(gift.head(5).to_string(index=False))
    print("\nKuaiLive gift occurrence classification (top 5 by AUC):")
    print(gift_cls.head(5).to_string(index=False))


if __name__ == "__main__":
    main()

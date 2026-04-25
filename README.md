# Lie-Gradient FDA for Online Streaming Monetization

Controlled simulation + real-data POC for online streaming monetization where outcome depends on engagement trajectory shape and timing, not only static summaries.

## Project Story

This project models transaction intensity as a function:

- `X_i(t), t in [0,1]`: points/gift curve over normalized stream time.

Monetization is generated from timing and shape-sensitive effects:

- early gift concentration,
- sustained support,
- timing misalignment penalties,
- and latent streamer quality.

The core hypothesis is that Lie-aligned FDA features outperform traditional tabular summaries because they preserve temporal structure.

## Repository Structure

```text
online-streaming-lie-fda/
├── README.md
├── data/
│   └── README.md
├── notebooks/
│   ├── 01_real_data_exploration.ipynb
│   ├── 02_simulation_design.ipynb
│   ├── 03_lie_gradient_fda.ipynb
│   └── 04_prediction_benchmarks.ipynb
├── src/
│   ├── benchmark_models.py
│   ├── evaluation.py
│   ├── functional_features.py
│   ├── lie_alignment.py
│   └── simulate_streaming_curves.py
├── scripts/
│   └── run_simulation_benchmarks.py
└── results/
    ├── figures/
    └── tables/
```

## Quick Start

1. Install dependencies:

```bash
pip install numpy pandas scipy scikit-learn matplotlib
```

2. Run the simulation benchmarks:

```bash
python scripts/run_simulation_benchmarks.py --n-users 1200 --grid-size 120 --seed 713
```

3. Inspect outputs:

- `results/tables/revenue_benchmark.csv`
- `results/tables/subscription_benchmark.csv`
- `results/figures/revenue_r2_comparison.png`
- `results/figures/revenue_rmse_comparison.png`

## Benchmark Design

- Baseline 1: raw summary features (`mean/std/max/quantiles/skew`)
- Baseline 2: nonlinear models on raw summaries (RF, GBM, MLP)
- Baseline 3: FPCA on unaligned curves
- Baseline 4: FDA features without registration
- Proposed: Lie-aligned FDA + model benchmarks

## Current Notebook-Compatible Script

Use `online_streaming_data_analysis.py` for a linear script version that:

- runs the controlled simulation,
- performs Lie alignment,
- computes FDA vs baseline features,
- and prints benchmark tables.

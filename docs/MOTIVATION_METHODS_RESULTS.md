# Motivation, Methods, and Result Summary

## Motivation

Online streaming monetization is fundamentally temporal.

Two users can produce similar total points but differ in when and how those points arrive:
- early concentrated engagement (often stronger conversion signal),
- late engagement after social proof,
- sustained support across the stream,
- fragmented behavior caused by real platform conditions (dropouts, reconnects, and partial stream visibility).

Traditional tabular summaries (`mean`, `std`, `max`, `count`, `total`) collapse this temporal structure and can miss key behavioral patterns. This project uses Functional Data Analysis (FDA) with alignment to preserve trajectory shape and timing as predictive signals.

## Methods

### 1. Controlled Simulation

The simulation generates user transaction intensity curves `X_i(t)` on normalized stream time `t in [0,1]` with:
- multiple behavioral archetypes (early, late, sustained, two-peak),
- random phase shifts,
- realistic observation artifacts: cut-offs, internal dropouts, and bursty noise.

Targets are:
- `revenue` (continuous),
- `subscribe` (binary).

Both outcomes depend on trajectory timing/shape, not just totals.

### 2. Preprocessing

- Denoising: reflection-padded moving-average smoothing.
- Missingness-aware features:
  - visible fraction,
  - first/last seen time,
  - observed span,
  - cutoff indicators,
  - number of observed segments,
  - internal gap counts and longest gap fraction.

### 3. Registration and Functional Representation

- Stage A: Lie-gradient shift alignment (global phase correction).
- Stage B: piecewise landmark warping using cumulative-mass quantile landmarks (`q25`, `q50`, `q75`).
- Feature construction from aligned curves:
  - early/mid/late mass,
  - peak time/intensity,
  - sustainedness,
  - roughness,
  - alignment descriptors (shift + landmarks),
  - optional FPCA scores.

### 4. Benchmarks

Compared models include:
- raw summaries (with/without missingness-aware additions),
- unaligned FDA,
- Lie-aligned FDA,
- Lie + piecewise FDA + missingness-aware features.

Evaluation metrics:
- Regression: `RMSE`, `MAE`, `R2`, `Spearman`.
- Classification: `Accuracy`, `AUC`, `F1`.

### 5. Real-Data Path with KuaiLive

For real-world validation, the project uses KuaiLive live-stream logs:
- room lifecycle (`start_timestamp`, `end_timestamp`) from `room.csv`,
- real-time interactions from `click.csv`, `comment.csv`, `like.csv`, `gift.csv`,
- user/streamer side information from `user.csv` and `streamer.csv`.

Pipeline:
1. Build user-streamer-room interaction curves over normalized room time.
2. Denoise and extract missingness-aware observability features.
3. Apply Lie alignment and piecewise landmark warping.
4. Predict:
   - watch-time target (`watch_live_time` aggregated per curve),
   - gift-price target (`gift_price` aggregated per curve),
   - optional gift occurrence classification (`gift_price > 0`).

Correctness note:
- For gift-related tasks, feature curves are built from `click+comment+like` interactions (excluding gift events) to avoid target leakage from gift signals.
- For watch-time tasks, full interaction curves (`click+comment+like+gift`) are retained.

Script:

```bash
python scripts/run_kuailive_realdata.py --data-dir data/KuaiLive --grid-size 120
```

### 6. KuaiLive Result Snapshot (Leakage-Safe Setup)

Run setting used for this snapshot:

```bash
python scripts/run_kuailive_realdata.py --data-dir KuaiLive --max-samples 50000 --grid-size 120 --seed 713 --light-mode
```

Lie-gradient alignment quality graph:
- `results/figures/kuailive_lie_alignment_quality.png`
- This plot shows:
  - sample trajectories before vs after Lie alignment,
  - pre/post curve-to-reference L2 distances,
  - distribution of relative alignment gain `(pre - post) / pre`.

On a KuaiLive sample run (`max-samples=20000`), alignment diagnostics were:
- median alignment gain: `3.08%`
- mean alignment gain: `3.28%`
- percentage of curves improved (`post < pre`): `96.80%`

Watch-time regression (lower RMSE is better):

| Rank | Model | RMSE | R2 |
|---|---|---:|---:|
| 1 | Lie-piecewise FDA + missingness + side + Gradient Boosting | 1.099e6 | 0.189 |
| 2 | Raw + side + Ridge | 1.108e6 | 0.176 |
| 3 | Lie-piecewise FDA + missingness + side + Ridge | 1.115e6 | 0.165 |

Gift-price regression (lower RMSE is better):

| Rank | Model | RMSE | R2 |
|---|---|---:|---:|
| 1 | Raw + side + Ridge | 244.02 | 0.019 |
| 2 | Lie-piecewise FDA + missingness + side + Ridge | 244.31 | 0.017 |
| 3 | Unaligned FDA + side + Ridge | 244.59 | 0.015 |

Gift occurrence classification (higher AUC is better):

| Rank | Model | AUC | Accuracy |
|---|---|---:|---:|
| 1 | Lie-piecewise FDA + missingness + side + Gradient Boosting | 0.877 | 0.859 |
| 2 | Raw + side + Logistic | 0.838 | 0.844 |
| 3 | Lie-piecewise FDA + missingness + side + Logistic | 0.825 | 0.834 |

Interpretation:
- The FDA + Lie + piecewise + missingness stack provides the strongest classification quality for gift occurrence.
- Watch-time prediction also benefits from the registered FDA stack (best RMSE/R2 in this run).
- Gift-price regression remains difficult and near-noise for all current feature/model choices, suggesting the need for richer covariates or alternative target transforms.
- The Lie-gradient quality plot confirms that alignment consistently reduces curve-to-reference mismatch on real KuaiLive interaction trajectories.
- Note: `R2` is not the strongest comparison metric in this setting because much of the predictive relationship is nonlinear, so RMSE/MAE and rank-based metrics are more informative.

## Research Contributions

- Proposed a timing-sensitive FDA formulation for live-stream monetization where curve phase and shape are central predictive signals.
- Implemented Lie-gradient registration with piecewise landmark warping to improve temporal comparability across users and rooms.
- Built missingness-aware functional features to address realistic stream truncation and interaction sparsity.
- Developed and validated a leakage-safe KuaiLive benchmark setup for watch-time, gift-price, and gift-occurrence tasks.
- Added quantitative alignment diagnostics demonstrating robust reduction in curve-to-reference mismatch after Lie alignment.

## Result Summary (Current Repo Artifacts)

Source files:
- `results/tables/revenue_benchmark.csv`
- `results/tables/subscription_benchmark.csv`

Top revenue models (lower RMSE is better):

| Rank | Model | RMSE | R2 |
|---|---|---:|---:|
| 1 | Lie-piecewise FDA + missingness + Gradient Boosting | 456.76 | 0.502 |
| 2 | Lie-piecewise FDA + missingness + Random Forest | 464.43 | 0.485 |
| 3 | Raw summaries + missingness + Random Forest | 493.52 | 0.419 |

Top subscription models (higher AUC is better):

| Rank | Model | AUC | Accuracy |
|---|---|---:|---:|
| 1 | Lie-piecewise FDA + missingness + Gradient Boosting | 0.628 | 0.600 |
| 2 | Lie-aligned FDA + Gradient Boosting | 0.568 | 0.556 |
| 3 | Lie-aligned FDA + Logistic | 0.568 | 0.567 |

Interpretation:
- For revenue prediction, piecewise-registered FDA with missingness-aware features currently leads.
- For subscription classification, performance is modest overall, but the best model is still the FDA + registration + missingness stack.
- The project’s practical takeaway is robustness: timing-aware functional pipelines retain predictive signal better than purely static summaries under noisy, cut-off stream observations.

## Reproducibility

Run:

```bash
python scripts/run_simulation_benchmarks.py --n-users 1200 --grid-size 120 --seed 713
```

Then review:
- `results/tables/revenue_benchmark.csv`
- `results/tables/subscription_benchmark.csv`

## References

1. Qu, C., Dai, S., Guo, K., Zhao, L., Niu, Y., Zhang, X., & Xu, J. (2025). *KuaiLive: A Real-time Interactive Dataset for Live Streaming Recommendation*. arXiv:2508.05633. https://arxiv.org/abs/2508.05633
2. KuaiLive official dataset page and schema documentation: https://imgkkk574.github.io/KuaiLive/ and https://imgkkk574.github.io/KuaiLive/detailed_statistics.html
3. KuaiLive Zenodo release (dataset archive): https://zenodo.org/records/16565801
4. Ramsay, J. O., & Silverman, B. W. (2005). *Functional Data Analysis* (2nd ed.). Springer. https://doi.org/10.1007/b98888
5. Srivastava, A., & Klassen, E. P. (2016). *Functional and Shape Data Analysis*. Springer. https://doi.org/10.1007/978-1-4939-4020-2
6. Tang, R., & Müller, H.-G. (2008). Pairwise curve synchronization for functional data. *Biometrika, 95*(4), 875-889. https://doi.org/10.1093/biomet/asn047
7. Breiman, L. (2001). Random forests. *Machine Learning, 45*, 5-32. https://doi.org/10.1023/A:1010933404324
8. Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. *Annals of Statistics, 29*(5), 1189-1232. https://doi.org/10.1214/aos/1013203451

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.functional_features import (
    denoise_curves,
    functional_shape_features,
    missingness_features,
    raw_summary_features,
)
from src.lie_alignment import align_curves, piecewise_landmark_alignment


@dataclass(frozen=True)
class KuaiLiveConfig:
    data_dir: Path
    grid_size: int = 120
    min_interactions_per_curve: int = 3
    max_samples: int | None = 150000
    random_state: int = 713

    # Column mappings (defaults follow the official KuaiLive docs).
    user_id_col: str = "user_id"
    streamer_id_col: str = "streamer_id"
    live_id_col: str = "live_id"
    timestamp_col: str = "timestamp"
    room_start_col: str = "start_timestamp"
    room_end_col: str = "end_timestamp"
    watch_time_col: str = "watch_live_time"
    gift_price_col: str = "gift_price"
    verbose: bool = True


def _log(cfg: KuaiLiveConfig, msg: str) -> None:
    if cfg.verbose:
        print(f"[KuaiLive] {msg}", flush=True)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)


def load_kuailive_tables(cfg: KuaiLiveConfig) -> dict[str, pd.DataFrame]:
    base = cfg.data_dir
    _log(cfg, f"Loading CSV tables from {base}")
    tables = {
        "room": _read_csv(base / "room.csv"),
        "user": _read_csv(base / "user.csv"),
        "streamer": _read_csv(base / "streamer.csv"),
        "click": _read_csv(base / "click.csv"),
        "comment": _read_csv(base / "comment.csv"),
        "like": _read_csv(base / "like.csv"),
        "gift": _read_csv(base / "gift.csv"),
    }
    for k, v in tables.items():
        _log(cfg, f"{k}.csv rows={len(v):,} cols={len(v.columns)}")
    return tables


def _to_numeric_like(series: pd.Series) -> pd.Series:
    """
    Convert scalars/ranges like '10-100' or '5000+' to numeric proxies.
    """
    s = series.astype(str).str.strip()
    out = pd.to_numeric(s, errors="coerce")

    # midpoint for ranges "a-b"
    mask_range = out.isna() & s.str.contains("-", regex=False, na=False)
    if mask_range.any():
        parts = s[mask_range].str.split("-", n=1, expand=True)
        lo = pd.to_numeric(parts[0], errors="coerce")
        hi = pd.to_numeric(parts[1], errors="coerce")
        out.loc[mask_range] = (lo + hi) / 2.0

    # lower bound for "a+"
    mask_plus = out.isna() & s.str.endswith("+", na=False)
    if mask_plus.any():
        out.loc[mask_plus] = pd.to_numeric(s[mask_plus].str.rstrip("+"), errors="coerce")

    return out


def _encode_side_info(df: pd.DataFrame, id_col: str, prefix: str) -> pd.DataFrame:
    side = df.copy()
    if id_col not in side.columns:
        raise KeyError(f"Expected ID column '{id_col}' in side info table.")
    side = side.drop_duplicates(subset=[id_col]).set_index(id_col)

    encoded = {}
    for col in side.columns:
        val = side[col]
        if np.issubdtype(val.dtype, np.number):
            encoded[f"{prefix}_{col}"] = val.astype(float)
            continue

        num = _to_numeric_like(val)
        if num.notna().mean() >= 0.8:
            encoded[f"{prefix}_{col}"] = num
        else:
            encoded[f"{prefix}_{col}"] = val.astype("category").cat.codes.astype(float)

    out = pd.DataFrame(encoded).replace(-1.0, np.nan)
    return out.fillna(out.median(numeric_only=True)).fillna(0.0)


def _prepare_interactions_with_room_time(
    events: pd.DataFrame,
    room: pd.DataFrame,
    cfg: KuaiLiveConfig,
    event_name: str,
) -> pd.DataFrame:
    required = [cfg.live_id_col, cfg.timestamp_col]
    for c in required:
        if c not in events.columns:
            raise KeyError(f"{event_name}.csv missing required column '{c}'.")

    room_cols = [cfg.live_id_col, cfg.room_start_col, cfg.room_end_col]
    for c in room_cols:
        if c not in room.columns:
            raise KeyError(f"room.csv missing required column '{c}'.")

    merged = events.merge(room[room_cols], on=cfg.live_id_col, how="left")
    merged = merged.dropna(subset=[cfg.room_start_col, cfg.room_end_col, cfg.timestamp_col])

    duration = (merged[cfg.room_end_col] - merged[cfg.room_start_col]).astype(float)
    valid = duration > 0
    merged = merged.loc[valid].copy()
    duration = duration.loc[valid]

    merged["t_norm"] = (merged[cfg.timestamp_col] - merged[cfg.room_start_col]) / duration
    merged["t_norm"] = merged["t_norm"].clip(0.0, 1.0)
    return merged


def _event_weighted_counts(
    df: pd.DataFrame,
    grid_size: int,
    weight_col: str,
    group_cols: list[str],
) -> pd.DataFrame:
    idx = (df["t_norm"] * (grid_size - 1)).round().astype(int).clip(0, grid_size - 1)
    work = df[group_cols + [weight_col]].copy()
    work["bin_idx"] = idx
    out = (
        work.groupby(group_cols + ["bin_idx"], observed=True)[weight_col]
        .sum()
        .reset_index()
    )
    return out


def build_kuailive_curve_dataset(
    cfg: KuaiLiveConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    tables = load_kuailive_tables(cfg)
    room = tables["room"]
    _log(cfg, "Preparing interactions with normalized room-time")
    click = _prepare_interactions_with_room_time(tables["click"], room, cfg, "click")
    comment = _prepare_interactions_with_room_time(tables["comment"], room, cfg, "comment")
    like = _prepare_interactions_with_room_time(tables["like"], room, cfg, "like")
    gift = _prepare_interactions_with_room_time(tables["gift"], room, cfg, "gift")
    _log(
        cfg,
        "Normalized events rows: "
        f"click={len(click):,}, comment={len(comment):,}, like={len(like):,}, gift={len(gift):,}",
    )

    for frame, name in [(click, "click"), (comment, "comment"), (like, "like"), (gift, "gift")]:
        for col in [cfg.user_id_col, cfg.streamer_id_col, cfg.live_id_col]:
            if col not in frame.columns:
                raise KeyError(f"{name}.csv missing required column '{col}'.")

    click = click.copy()
    comment = comment.copy()
    like = like.copy()
    gift = gift.copy()

    click["weight"] = 1.0
    comment["weight"] = 1.2
    like["weight"] = 1.0
    gift["weight"] = 1.0 + np.log1p(pd.to_numeric(gift.get(cfg.gift_price_col, 0), errors="coerce").fillna(0.0))

    group_cols = [cfg.user_id_col, cfg.streamer_id_col, cfg.live_id_col]
    b_click = _event_weighted_counts(click, cfg.grid_size, "weight", group_cols)
    b_comment = _event_weighted_counts(comment, cfg.grid_size, "weight", group_cols)
    b_like = _event_weighted_counts(like, cfg.grid_size, "weight", group_cols)
    b_gift = _event_weighted_counts(gift, cfg.grid_size, "weight", group_cols)
    _log(cfg, "Built weighted binned event counts")

    all_bins = pd.concat(
        [b_click, b_comment, b_like, b_gift],
        axis=0,
        ignore_index=True,
    )
    all_bins = (
        all_bins.groupby(group_cols + ["bin_idx"], observed=True)["weight"]
        .sum()
        .reset_index()
    )

    all_bins_no_gift = pd.concat(
        [b_click, b_comment, b_like],
        axis=0,
        ignore_index=True,
    )
    all_bins_no_gift = (
        all_bins_no_gift.groupby(group_cols + ["bin_idx"], observed=True)["weight"]
        .sum()
        .reset_index()
    )

    key_df = all_bins[group_cols].drop_duplicates().reset_index(drop=True)
    _log(cfg, f"Constructed {len(key_df):,} user-streamer-room curves")
    key_df["curve_id"] = np.arange(len(key_df))
    merged_bins = all_bins.merge(key_df, on=group_cols, how="left")
    merged_bins_no_gift = all_bins_no_gift.merge(key_df, on=group_cols, how="left")

    X_all = np.zeros((len(key_df), cfg.grid_size), dtype=float)
    X_all[merged_bins["curve_id"].to_numpy(), merged_bins["bin_idx"].to_numpy()] = merged_bins[
        "weight"
    ].to_numpy()
    X_no_gift = np.zeros((len(key_df), cfg.grid_size), dtype=float)
    X_no_gift[
        merged_bins_no_gift["curve_id"].to_numpy(), merged_bins_no_gift["bin_idx"].to_numpy()
    ] = merged_bins_no_gift["weight"].to_numpy()

    # Targets by user-streamer-room tuple.
    watch_agg = (
        click.groupby(group_cols, observed=True)[cfg.watch_time_col]
        .sum()
        .rename("watch_time_target")
        if cfg.watch_time_col in click.columns
        else pd.Series(0.0, index=pd.MultiIndex.from_frame(key_df[group_cols]), name="watch_time_target")
    )
    gift_agg = (
        gift.groupby(group_cols, observed=True)[cfg.gift_price_col]
        .sum()
        .rename("gift_price_target")
        if cfg.gift_price_col in gift.columns
        else pd.Series(0.0, index=pd.MultiIndex.from_frame(key_df[group_cols]), name="gift_price_target")
    )
    event_count = (
        pd.concat(
            [
                click[group_cols].assign(evt=1),
                comment[group_cols].assign(evt=1),
                like[group_cols].assign(evt=1),
                gift[group_cols].assign(evt=1),
            ],
            axis=0,
            ignore_index=True,
        )
        .groupby(group_cols, observed=True)["evt"]
        .sum()
        .rename("event_count")
    )

    meta = key_df.set_index(group_cols).join([watch_agg, gift_agg, event_count], how="left").reset_index()
    meta["watch_time_target"] = meta["watch_time_target"].fillna(0.0)
    meta["gift_price_target"] = meta["gift_price_target"].fillna(0.0)
    meta["event_count"] = meta["event_count"].fillna(0).astype(int)
    meta["gift_any_target"] = (meta["gift_price_target"] > 0).astype(int)

    keep = meta["event_count"] >= cfg.min_interactions_per_curve
    meta = meta.loc[keep].reset_index(drop=True)
    X_all = X_all[keep.to_numpy()]
    X_no_gift = X_no_gift[keep.to_numpy()]
    _log(cfg, f"After min_interactions filter: {len(meta):,} curves")

    # Optional downsampling for very large datasets.
    if cfg.max_samples is not None and len(meta) > cfg.max_samples:
        rng = np.random.default_rng(cfg.random_state)
        take = rng.choice(len(meta), size=cfg.max_samples, replace=False)
        meta = meta.iloc[take].reset_index(drop=True)
        X_all = X_all[take]
        X_no_gift = X_no_gift[take]
        _log(cfg, f"Downsampled to max_samples={cfg.max_samples:,} curves")

    # Attach side information.
    u_side = _encode_side_info(tables["user"], cfg.user_id_col, "user")
    s_side = _encode_side_info(tables["streamer"], cfg.streamer_id_col, "streamer")
    meta = meta.merge(u_side, left_on=cfg.user_id_col, right_index=True, how="left")
    meta = meta.merge(s_side, left_on=cfg.streamer_id_col, right_index=True, how="left")
    meta = meta.fillna(0.0)
    _log(cfg, f"Final curve matrix shape: all={X_all.shape}, no_gift={X_no_gift.shape}")

    grid = np.linspace(0.0, 1.0, cfg.grid_size)
    return X_all, X_no_gift, grid, meta


def build_kuailive_feature_sets(
    X_all: np.ndarray,
    X_no_gift: np.ndarray,
    grid: np.ndarray,
    meta: pd.DataFrame,
) -> tuple[
    dict[str, pd.DataFrame],
    dict[str, pd.DataFrame],
    dict[str, pd.DataFrame],
    dict[str, np.ndarray],
]:
    """
    Build feature sets for watch-time (regression) and gift prediction (regression/classification).
    """
    X_watch_denoised = denoise_curves(X_all, window=7)
    miss_watch = missingness_features(X_all, grid)

    X_gift_denoised = denoise_curves(X_no_gift, window=7)
    miss_gift = missingness_features(X_no_gift, grid)

    X_watch_lie, watch_shifts, watch_ref = align_curves(
        X_watch_denoised,
        grid,
        eta=0.08,
        max_shift=0.30,
        n_iter=2,
        inner_steps=8,
    )
    X_watch_pw, watch_src_landmarks, watch_tgt_landmarks = piecewise_landmark_alignment(
        X_watch_lie, grid, reference=watch_ref, quantiles=(0.25, 0.50, 0.75)
    )

    X_gift_lie, gift_shifts, gift_ref = align_curves(
        X_gift_denoised,
        grid,
        eta=0.08,
        max_shift=0.30,
        n_iter=2,
        inner_steps=8,
    )
    X_gift_pw, gift_src_landmarks, gift_tgt_landmarks = piecewise_landmark_alignment(
        X_gift_lie, grid, reference=gift_ref, quantiles=(0.25, 0.50, 0.75)
    )

    raw_watch = raw_summary_features(X_all)
    raw_gift = raw_summary_features(X_no_gift)
    side_cols = [c for c in meta.columns if c.startswith("user_") or c.startswith("streamer_")]
    side = meta[side_cols].reset_index(drop=True) if side_cols else pd.DataFrame(index=np.arange(len(meta)))

    fda_watch_unaligned = functional_shape_features(X_watch_denoised, grid)
    fda_watch_lie = functional_shape_features(X_watch_lie, grid)
    fda_watch_lie["lie_shift"] = watch_shifts
    fda_watch_lie["abs_lie_shift"] = np.abs(watch_shifts)
    fda_watch_pw = functional_shape_features(X_watch_pw, grid)
    fda_watch_pw["lie_shift"] = watch_shifts
    fda_watch_pw["abs_lie_shift"] = np.abs(watch_shifts)
    fda_watch_pw["landmark_q25"] = watch_src_landmarks[:, 0]
    fda_watch_pw["landmark_q50"] = watch_src_landmarks[:, 1]
    fda_watch_pw["landmark_q75"] = watch_src_landmarks[:, 2]
    fda_watch_pw["target_q25"] = watch_tgt_landmarks[0]
    fda_watch_pw["target_q50"] = watch_tgt_landmarks[1]
    fda_watch_pw["target_q75"] = watch_tgt_landmarks[2]

    fda_gift_unaligned = functional_shape_features(X_gift_denoised, grid)
    fda_gift_lie = functional_shape_features(X_gift_lie, grid)
    fda_gift_lie["lie_shift"] = gift_shifts
    fda_gift_lie["abs_lie_shift"] = np.abs(gift_shifts)
    fda_gift_pw = functional_shape_features(X_gift_pw, grid)
    fda_gift_pw["lie_shift"] = gift_shifts
    fda_gift_pw["abs_lie_shift"] = np.abs(gift_shifts)
    fda_gift_pw["landmark_q25"] = gift_src_landmarks[:, 0]
    fda_gift_pw["landmark_q50"] = gift_src_landmarks[:, 1]
    fda_gift_pw["landmark_q75"] = gift_src_landmarks[:, 2]
    fda_gift_pw["target_q25"] = gift_tgt_landmarks[0]
    fda_gift_pw["target_q50"] = gift_tgt_landmarks[1]
    fda_gift_pw["target_q75"] = gift_tgt_landmarks[2]

    raw_watch_miss_side = pd.concat([raw_watch, miss_watch, side], axis=1)
    fda_watch_pw_miss_side = pd.concat([fda_watch_pw, miss_watch, side], axis=1)
    fda_watch_lie_side = pd.concat([fda_watch_lie, side], axis=1)
    fda_watch_unaligned_side = pd.concat([fda_watch_unaligned, side], axis=1)

    raw_gift_miss_side = pd.concat([raw_gift, miss_gift, side], axis=1)
    fda_gift_pw_miss_side = pd.concat([fda_gift_pw, miss_gift, side], axis=1)
    fda_gift_lie_side = pd.concat([fda_gift_lie, side], axis=1)
    fda_gift_unaligned_side = pd.concat([fda_gift_unaligned, side], axis=1)

    watch_reg_feature_sets = {
        "Raw + side + Ridge": raw_watch_miss_side,
        "Raw + side + Random Forest": raw_watch_miss_side,
        "Raw + side + Gradient Boosting": raw_watch_miss_side,
        "Unaligned FDA + side + Ridge": fda_watch_unaligned_side,
        "Lie-aligned FDA + side + Ridge": fda_watch_lie_side,
        "Lie-aligned FDA + side + Random Forest": fda_watch_lie_side,
        "Lie-piecewise FDA + missingness + side + Ridge": fda_watch_pw_miss_side,
        "Lie-piecewise FDA + missingness + side + Random Forest": fda_watch_pw_miss_side,
        "Lie-piecewise FDA + missingness + side + Gradient Boosting": fda_watch_pw_miss_side,
    }

    gift_reg_feature_sets = {
        "Raw + side + Ridge": raw_gift_miss_side,
        "Raw + side + Random Forest": raw_gift_miss_side,
        "Raw + side + Gradient Boosting": raw_gift_miss_side,
        "Unaligned FDA + side + Ridge": fda_gift_unaligned_side,
        "Lie-aligned FDA + side + Ridge": fda_gift_lie_side,
        "Lie-aligned FDA + side + Random Forest": fda_gift_lie_side,
        "Lie-piecewise FDA + missingness + side + Ridge": fda_gift_pw_miss_side,
        "Lie-piecewise FDA + missingness + side + Random Forest": fda_gift_pw_miss_side,
        "Lie-piecewise FDA + missingness + side + Gradient Boosting": fda_gift_pw_miss_side,
    }

    gift_cls_feature_sets = {
        "Raw + side + Logistic": raw_gift_miss_side,
        "Raw + side + Random Forest": raw_gift_miss_side,
        "Raw + side + Gradient Boosting": raw_gift_miss_side,
        "Unaligned FDA + side + Logistic": fda_gift_unaligned_side,
        "Lie-aligned FDA + side + Logistic": fda_gift_lie_side,
        "Lie-piecewise FDA + missingness + side + Logistic": fda_gift_pw_miss_side,
        "Lie-piecewise FDA + missingness + side + Gradient Boosting": fda_gift_pw_miss_side,
    }
    # Lie alignment diagnostics on watch-time pathway.
    watch_pre_l2 = np.sqrt(np.trapezoid((X_watch_denoised - watch_ref) ** 2, grid, axis=1))
    watch_post_l2 = np.sqrt(np.trapezoid((X_watch_lie - watch_ref) ** 2, grid, axis=1))
    watch_gain = (watch_pre_l2 - watch_post_l2) / (watch_pre_l2 + 1e-12)

    rng = np.random.default_rng(713)
    n_show = min(30, len(X_watch_denoised))
    show_idx = rng.choice(len(X_watch_denoised), size=n_show, replace=False)

    diagnostics = {
        "grid": grid,
        "watch_ref": watch_ref,
        "watch_pre_l2": watch_pre_l2,
        "watch_post_l2": watch_post_l2,
        "watch_gain": watch_gain,
        "watch_before_examples": X_watch_denoised[show_idx],
        "watch_after_examples": X_watch_lie[show_idx],
    }

    return watch_reg_feature_sets, gift_reg_feature_sets, gift_cls_feature_sets, diagnostics

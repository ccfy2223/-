"""
train_autogluon.py
==================
Multi-horizon wave-height forecasting with AutoGluon TimeSeries.

Split   : 70 % train / 20 % validation / 10 % test  (per station, chronological)
Horizons: default = 1 3 6 12 24 48 72 120 hours (customizable via --prediction-hours)
Metrics : MAE  MASE  RMSE  RMSLE  SMAPE
GPU     : set --gpu-id 0  to pin the RTX 4090 on AutoDL

After training, plots are saved to <output-root>/plots/
Training curves are preserved when curve logging is enabled.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore", category=FutureWarning)
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))


# ── GPU pinning must happen before any torch / autogluon import ──────────────
def _read_cli_value(flag: str, default: str) -> str:
    try:
        idx = sys.argv.index(flag)
    except ValueError:
        return default
    if idx + 1 >= len(sys.argv):
        return default
    return sys.argv[idx + 1]


os.environ["CUDA_VISIBLE_DEVICES"] = _read_cli_value("--gpu-id", "0")
# Use HuggingFace mirror (required on AutoDL — hf.co is blocked)
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# ── third-party ──────────────────────────────────────────────────────────────
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import autogluon.timeseries as agts
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
from lightning.pytorch.loggers import CSVLogger

matplotlib.use("Agg")
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.unicode_minus": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
    }
)

# ── constants ─────────────────────────────────────────────────────────────────
HORIZON_HOURS = [1, 3, 6, 12, 24, 48, 72, 120]
DEEP_MODEL_TYPES = [
    "TemporalFusionTransformer",
    "DeepAR",
    "PatchTST",
    "DLinear",
]
TABULAR_MODEL_TYPES = ["DirectTabular", "RecursiveTabular"]

KNOWN_COVARIATES = [
    "time_sin_hour", "time_cos_hour",
    "time_sin_doy",  "time_cos_doy",
    "month", "day_of_week",
]

DYNAMIC_COLUMNS = [
    "WDIR", "WSPD", "GST", "WVHT",
    "DPD", "APD", "MWD",
    "PRES", "ATMP", "WTMP", "DEWP", "VIS", "TIDE",
]

EXTRA_METRICS = ["MAE", "MASE", "RMSE", "RMSLE", "SMAPE"]

METRIC_LABEL = {
    "MAE":   "MAE (m)",
    "RMSE":  "RMSE (m)",
    "RMSLE": "RMSLE",
    "MASE":  "MASE",
    "SMAPE": "SMAPE",
}

# Colour cycle consistent across all figures
_COLOURS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#17becf",
]
_MARKERS = ["o", "s", "^", "D", "v", "P", "*", "X"]


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StationSummary:
    item_id: str
    rows_original: int
    rows_after_dedup: int
    rows_after_window: int
    duplicate_rows_removed: int
    missing_before_fill: int
    missing_after_fill: int
    region_key: str | None
    station_name: str | None
    first_timestamp: str
    last_timestamp: str
    train_rows: int
    val_rows: int
    test_rows: int


@dataclass
class HorizonResult:
    horizon_hours: int
    prediction_length: int
    best_model: str
    val_mase: float
    metrics: dict[str, float]           # test-set metrics for best model
    val_leaderboard_csv: str
    test_leaderboard_csv: str
    model_dir: str
    training_curve_dir: str
    elapsed_seconds: float


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="AutoGluon multi-horizon wave-height forecasting (70/20/10 split)"
    )
    p.add_argument("--input-dir",   type=Path,
                   default=Path("processed_csv/aligned_stations"))
    p.add_argument("--metadata-path", type=Path,
                   default=Path("processed_csv/shared_timeline_metadata.json"))
    p.add_argument("--output-root", type=Path, default=Path("autogluon_runs"))
    p.add_argument("--log-path",    type=Path, default=Path("train_log.json"))
    p.add_argument("--target",      default="WVHT")
    p.add_argument("--freq",        default="10min")
    p.add_argument("--gpu-id",      default="0")
    p.add_argument("--prediction-hours",
                   default=",".join(str(h) for h in HORIZON_HOURS))
    p.add_argument("--train-ratio", type=float, default=0.70)
    p.add_argument("--val-ratio",   type=float, default=0.20)
    p.add_argument("--test-ratio",  type=float, default=0.10)
    p.add_argument("--presets",     default="best_quality")
    p.add_argument("--eval-metric", default="MASE")
    p.add_argument("--time-limit",  type=int, default=7200,
                   help="Training time-limit per horizon (seconds). Use 0 to disable.")
    p.add_argument("--random-seed", type=int, default=42)
    p.add_argument("--excluded-models", default="",
                   help="Comma-separated model types to exclude. Empty = include all (Chronos included).")
    p.add_argument("--verbosity",   type=int, default=2)
    p.add_argument("--limit-stations", type=int, default=0)
    p.add_argument("--max-steps",   type=int, default=0,
                   help="Cap rows per station (0 = all).")
    p.add_argument("--no-shared-window", action="store_true")
    p.add_argument("--plot-format", default="png",
                   choices=["png", "pdf", "svg"])
    p.add_argument("--deep-epochs", type=int, default=240,
                   help="Max epochs for AutoGluon deep models (TFT / DeepAR / PatchTST / DLinear).")
    p.add_argument("--deep-batch-size", type=int, default=64,
                   help="Batch size for AutoGluon deep models.")
    p.add_argument("--deep-num-batches-per-epoch", type=int, default=128,
                   help="Number of batches per epoch for deep models. Use 0 for model defaults.")
    p.add_argument("--deep-early-stopping-patience", type=int, default=60,
                   help="Validation patience for deep models. Use -1 to disable early stopping.")
    p.add_argument("--deep-log-every-n-steps", type=int, default=20,
                   help="Lightning logging frequency for deep models.")
    p.add_argument("--no-train-progress-bar", action="store_true",
                   help="Disable per-model training progress bars.")
    p.add_argument("--no-curve-logs", action="store_true",
                   help="Disable epoch/step curve logs. Leave this off if you need thesis plots.")
    p.add_argument("--chronos-model-path", default="amazon/chronos-t5-large",
                   help="Chronos model path used by AutoGluon.")
    p.add_argument("--chronos-device", default="cuda",
                   help="Chronos device. Typically 'cuda' on AutoDL.")
    p.add_argument("--chronos-batch-size", type=int, default=8,
                   help="Chronos inference / zero-shot batch size.")
    p.add_argument("--chronos-context-length", type=int, default=2048,
                   help="Chronos context length. 2048 is the practical ceiling for Chronos-T5-Large.")
    p.add_argument("--chronos-fine-tune", action="store_true",
                   help="Fine-tune Chronos during AutoGluon training and keep step-wise logs.")
    p.add_argument("--chronos-fine-tune-steps", type=int, default=3000,
                   help="Fine-tuning steps for Chronos when --chronos-fine-tune is enabled.")
    p.add_argument("--chronos-fine-tune-batch-size", type=int, default=8,
                   help="Per-device fine-tuning batch size for Chronos.")
    p.add_argument("--chronos-fine-tune-lr", type=float, default=1e-5,
                   help="Learning rate for Chronos fine-tuning.")
    p.add_argument("--chronos-logging-steps", type=int, default=20,
                   help="Transformers logging interval during Chronos fine-tuning.")
    p.add_argument("--chronos-eval-steps", type=int, default=100,
                   help="Transformers eval / checkpoint interval during Chronos fine-tuning.")
    p.add_argument("--chronos-gradient-accumulation", type=int, default=4,
                   help="Gradient accumulation steps during Chronos fine-tuning.")
    # ── soft-label distillation ───────────────────────────────────────────────
    p.add_argument("--soft-label-path", type=Path, default=None,
                   help="CSV with columns item_id,timestamp,chronos_pred. "
                        "If provided, blends target with Chronos predictions.")
    p.add_argument("--teacher-weight", type=float, default=0.1,
                   help="Weight w for soft labels: target = (1-w)*true + w*teacher. "
                        "Only used when --soft-label-path is set.")
    return p


# ─────────────────────────────────────────────────────────────────────────────
# Data loading & preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def load_shared_window(
    meta: Path, disabled: bool
) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    if disabled or not meta.exists():
        return None, None
    d = json.loads(meta.read_text(encoding="utf-8"))
    s, e = d.get("shared_start"), d.get("shared_end")
    return (pd.Timestamp(s) if s else None, pd.Timestamp(e) if e else None)


def load_station(
    path: Path,
    target: str,
    shared_start: pd.Timestamp | None,
    shared_end: pd.Timestamp | None,
    max_steps: int,
) -> tuple[pd.DataFrame, StationSummary]:
    item_id = path.stem.split("_")[0]
    cols_all = pd.read_csv(path, nrows=0).columns.tolist()
    dyn = [c for c in DYNAMIC_COLUMNS if c in cols_all]
    kno = [c for c in KNOWN_COVARIATES if c in cols_all]

    df = pd.read_csv(
        path,
        usecols=["datetime"] + dyn + kno,
        parse_dates=["datetime"],
        low_memory=False,
    )
    rows_orig = len(df)

    dup = int(df.duplicated("datetime").sum())
    df = df.sort_values("datetime").drop_duplicates("datetime", keep="last").reset_index(drop=True)
    rows_dedup = len(df)

    if shared_start is not None:
        df = df[(df["datetime"] >= shared_start) & (df["datetime"] <= shared_end)].reset_index(drop=True)

    if max_steps > 0 and len(df) > max_steps:
        df = df.iloc[-max_steps:].reset_index(drop=True)
    rows_win = len(df)

    if rows_win == 0:
        raise ValueError(f"{path.name}: no rows after window filter")

    num_cols = [c for c in df.columns if c != "datetime"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")

    miss_before = int(df[num_cols].isna().sum().sum())
    df[num_cols] = df[num_cols].ffill().bfill()
    miss_after = int(df[num_cols].isna().sum().sum())

    if target not in df.columns:
        raise KeyError(f"Target '{target}' not in {path.name}")
    if df[target].isna().any():
        raise ValueError(f"Target '{target}' still has NaN after fill in {path.name}")

    df.insert(0, "item_id", item_id)
    df = df.rename(columns={"datetime": "timestamp"})

    n = len(df)
    train_end = max(1, int(n * 0.70))       # temporary; overwritten later with actual args
    val_end   = max(train_end + 1, int(n * 0.90))
    val_end   = min(val_end, n - 1)

    summary = StationSummary(
        item_id=item_id,
        rows_original=rows_orig,
        rows_after_dedup=rows_dedup,
        rows_after_window=rows_win,
        duplicate_rows_removed=dup,
        missing_before_fill=miss_before,
        missing_after_fill=miss_after,
        region_key=None,
        station_name=None,
        first_timestamp=str(df["timestamp"].iloc[0]),
        last_timestamp=str(df["timestamp"].iloc[-1]),
        train_rows=train_end,
        val_rows=val_end - train_end,
        test_rows=n - val_end,
    )
    return df, summary


def build_all_stations(
    files: list[Path],
    target: str,
    shared_start: pd.Timestamp | None,
    shared_end: pd.Timestamp | None,
    limit: int,
    max_steps: int,
    train_ratio: float,
    val_ratio: float,
) -> tuple[dict[str, pd.DataFrame], list[StationSummary]]:
    selected = files[:limit] if limit > 0 else files
    frames: dict[str, pd.DataFrame] = {}
    summaries: list[StationSummary] = []

    for p in selected:
        df, sm = load_station(p, target, shared_start, shared_end, max_steps)
        n = len(df)
        train_end = max(1, int(n * train_ratio))
        val_end   = max(train_end + 1, int(n * (train_ratio + val_ratio)))
        val_end   = min(val_end, n - 1)
        sm.train_rows = train_end
        sm.val_rows   = val_end - train_end
        sm.test_rows  = n - val_end
        frames[sm.item_id] = df
        summaries.append(sm)
        print(f"  {sm.item_id}: {n} rows | train {sm.train_rows} | val {sm.val_rows} | test {sm.test_rows}")

    if not frames:
        raise ValueError("No station data loaded.")

    # Drop past-covariate columns that have any remaining NaN across all stations
    combined = pd.concat(frames.values(), ignore_index=True)
    past_candidates = [
        c for c in combined.columns
        if c not in {"item_id", "timestamp", target, *KNOWN_COVARIATES}
    ]
    bad = {c for c in past_candidates if combined[c].isna().any()}
    if bad:
        keep = ["item_id", "timestamp", target] + [c for c in past_candidates if c not in bad] + KNOWN_COVARIATES
        for k in frames:
            frames[k] = frames[k][[c for c in keep if c in frames[k].columns]].copy()
        print(f"  Dropped past covariates with residual NaN: {sorted(bad)}")

    return frames, summaries


def apply_soft_labels(
    frames: dict[str, pd.DataFrame],
    soft_label_path: Path,
    target: str,
    teacher_weight: float,
) -> None:
    """
    In-place: blend target column with Chronos soft labels.
    target_blended = (1 - w) * target_true + w * chronos_pred
    Only train split rows are blended (split=='train' in the CSV).
    """
    print(f"\nApplying soft labels (teacher_weight={teacher_weight}) from {soft_label_path}")
    labels = pd.read_csv(soft_label_path, parse_dates=["timestamp"])
    labels_train = labels[labels["split"] == "train"][["item_id", "timestamp", "chronos_pred"]]

    total_blended = 0
    for iid, df in frames.items():
        sub = labels_train[labels_train["item_id"] == iid]
        if sub.empty:
            print(f"  {iid}: no soft labels found, skipping")
            continue
        sub = sub.set_index("timestamp")["chronos_pred"]
        df["_teacher"] = df["timestamp"].map(sub)
        mask = df["_teacher"].notna()
        n_blend = int(mask.sum())
        df.loc[mask, target] = (
            (1 - teacher_weight) * df.loc[mask, target]
            + teacher_weight * df.loc[mask, "_teacher"]
        ).astype("float32")
        df.drop(columns=["_teacher"], inplace=True)
        total_blended += n_blend
        coverage = n_blend / len(df) * 100
        print(f"  {iid}: blended {n_blend}/{len(df)} rows ({coverage:.1f}%)")

    print(f"  Total blended rows: {total_blended}")


def _build_split_df(
    frames: dict[str, pd.DataFrame],
    summaries: list[StationSummary],
    split: str,
) -> pd.DataFrame:
    parts = []
    sm_map = {s.item_id: s for s in summaries}
    for iid, df in frames.items():
        sm = sm_map[iid]
        te = sm.train_rows
        ve = te + sm.val_rows
        if split == "train":
            parts.append(df.iloc[:te].copy())
        elif split == "val":
            parts.append(df.iloc[:ve].copy())       # cumulative (for AutoGluon tuning_data)
        else:                                         # test: full series
            parts.append(df.copy())
    return pd.concat(parts, ignore_index=True)


def to_tsdf(df: pd.DataFrame) -> TimeSeriesDataFrame:
    return TimeSeriesDataFrame.from_data_frame(
        df=df, id_column="item_id", timestamp_column="timestamp"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def _flip_sign(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = out[c].abs()
    if "score_val" in out.columns:
        out["validation_MASE"] = out["score_val"].abs()
    if "score_test" in out.columns:
        out["test_MASE"] = out["score_test"].abs()
    return out


def _build_deep_model_hyperparameters(
    model_name: str,
    curve_root: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    trainer_kwargs: dict[str, Any] = {
        "enable_progress_bar": not args.no_train_progress_bar,
        "log_every_n_steps": max(1, args.deep_log_every_n_steps),
        "num_sanity_val_steps": 0,
    }
    if not args.no_curve_logs:
        trainer_kwargs["logger"] = CSVLogger(save_dir=str(curve_root), name=model_name)

    params: dict[str, Any] = {
        "max_epochs": args.deep_epochs,
        "batch_size": args.deep_batch_size,
        "keep_lightning_logs": not args.no_curve_logs,
        "trainer_kwargs": trainer_kwargs,
        "early_stopping_patience": (
            None if args.deep_early_stopping_patience < 0 else args.deep_early_stopping_patience
        ),
    }
    if args.deep_num_batches_per_epoch > 0:
        params["num_batches_per_epoch"] = args.deep_num_batches_per_epoch
    return params


def _build_chronos_hyperparameters(args: argparse.Namespace) -> list[dict[str, Any]]:
    chronos_device = args.chronos_device
    if chronos_device == "cuda" and not torch.cuda.is_available():
        chronos_device = "cpu"

    chronos_cfg: dict[str, Any] = {
        "model_path": args.chronos_model_path,
        "device": chronos_device,
        "batch_size": args.chronos_batch_size,
        "context_length": args.chronos_context_length,
    }

    if args.chronos_fine_tune:
        chronos_cfg.update(
            {
                "fine_tune": True,
                "fine_tune_steps": args.chronos_fine_tune_steps,
                "fine_tune_batch_size": args.chronos_fine_tune_batch_size,
                "fine_tune_lr": args.chronos_fine_tune_lr,
                "eval_during_fine_tune": True,
                "keep_transformers_logs": not args.no_curve_logs,
                "fine_tune_trainer_kwargs": {
                    "logging_steps": max(1, args.chronos_logging_steps),
                    "save_steps": max(1, args.chronos_eval_steps),
                    "eval_steps": max(1, args.chronos_eval_steps),
                    "gradient_accumulation_steps": max(1, args.chronos_gradient_accumulation),
                    "disable_tqdm": args.no_train_progress_bar,
                    "overwrite_output_dir": True,
                },
            }
        )

    return [chronos_cfg]


def build_fit_hyperparameters(
    horizon_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    curve_root = horizon_dir / "training_curves"
    if curve_root.exists():
        shutil.rmtree(curve_root, ignore_errors=True)
    if not args.no_curve_logs:
        curve_root.mkdir(parents=True, exist_ok=True)

    fit_hyperparameters: dict[str, Any] = {
        "Chronos": _build_chronos_hyperparameters(args),
    }
    for model_name in DEEP_MODEL_TYPES:
        fit_hyperparameters[model_name] = _build_deep_model_hyperparameters(
            model_name=model_name,
            curve_root=curve_root,
            args=args,
        )
    for model_name in TABULAR_MODEL_TYPES:
        fit_hyperparameters[model_name] = {}
    return fit_hyperparameters


def train_one_horizon(
    hours: int,
    pred_len: int,
    train_ts: TimeSeriesDataFrame,
    val_ts: TimeSeriesDataFrame,
    test_ts: TimeSeriesDataFrame,
    args: argparse.Namespace,
) -> HorizonResult:
    h_dir = args.output_root / f"horizon_{hours:03d}h"
    m_dir = h_dir / "model"
    h_dir.mkdir(parents=True, exist_ok=True)

    excluded = [x.strip() for x in args.excluded_models.split(",") if x.strip()]
    fit_hyperparameters = build_fit_hyperparameters(h_dir, args)
    time_limit = None if args.time_limit <= 0 else args.time_limit

    predictor = TimeSeriesPredictor(
        target=args.target,
        prediction_length=pred_len,
        freq=args.freq,
        eval_metric=args.eval_metric,
        known_covariates_names=KNOWN_COVARIATES,
        path=str(m_dir),
        verbosity=args.verbosity,
    )

    t0 = time.perf_counter()
    predictor.fit(
        train_data=train_ts,
        tuning_data=val_ts,
        presets=args.presets,
        hyperparameters=fit_hyperparameters,
        excluded_model_types=excluded,
        time_limit=time_limit,
        random_seed=args.random_seed,
    )
    elapsed = time.perf_counter() - t0

    best_model = str(predictor.leaderboard(display=False).iloc[0]["model"])

    val_lb  = _flip_sign(predictor.leaderboard(data=val_ts,  extra_metrics=EXTRA_METRICS, display=False), EXTRA_METRICS)
    test_lb = _flip_sign(predictor.leaderboard(data=test_ts, extra_metrics=EXTRA_METRICS, display=False), EXTRA_METRICS)

    val_csv  = h_dir / "leaderboard_validation.csv"
    test_csv = h_dir / "leaderboard_test.csv"
    val_lb.to_csv(val_csv,   index=False, encoding="utf-8-sig")
    test_lb.to_csv(test_csv, index=False, encoding="utf-8-sig")

    best_row = test_lb.loc[test_lb["model"] == best_model]
    if best_row.empty:
        raise ValueError(f"Best model '{best_model}' missing from test leaderboard at {hours}h")

    row = best_row.iloc[0]
    metrics = {m: float(row[m]) for m in EXTRA_METRICS}
    val_mase = float(
        val_lb.loc[val_lb["model"] == best_model, "validation_MASE"].iloc[0]
    )

    return HorizonResult(
        horizon_hours=hours,
        prediction_length=pred_len,
        best_model=best_model,
        val_mase=val_mase,
        metrics=metrics,
        val_leaderboard_csv=str(val_csv.resolve()),
        test_leaderboard_csv=str(test_csv.resolve()),
        model_dir=str(m_dir.resolve()),
        training_curve_dir=str((h_dir / "training_curves").resolve()),
        elapsed_seconds=elapsed,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def _savefig(fig: plt.Figure, path: Path, fmt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = path.with_suffix(f".{fmt}")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"    saved → {out.name}")


def plot_dataset_overview(
    summaries: list[StationSummary],
    out_dir: Path,
    fmt: str,
) -> None:
    """Figure 1 – training data statistics (4 sub-panels)."""
    ids     = [s.item_id for s in summaries]
    n_sta   = len(ids)
    xs      = np.arange(n_sta)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle("Dataset Overview", fontsize=14, fontweight="bold", y=1.01)

    # (a) row counts per station
    ax = axes[0, 0]
    total = [s.rows_after_window for s in summaries]
    train = [s.train_rows        for s in summaries]
    val   = [s.val_rows          for s in summaries]
    test  = [s.test_rows         for s in summaries]
    w = 0.2
    ax.bar(xs - w,   train, w, label="Train (70 %)", color="#1f77b4")
    ax.bar(xs,       val,   w, label="Val (20 %)",   color="#ff7f0e")
    ax.bar(xs + w,   test,  w, label="Test (10 %)",  color="#2ca02c")
    ax.set_xticks(xs); ax.set_xticklabels(ids, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Rows"); ax.set_title("(a) Row counts per station")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v/1e3:.0f}k"))
    ax.legend(fontsize=8); ax.grid(axis="y", linestyle="--", alpha=0.4)

    # (b) missing values before/after fill
    ax = axes[0, 1]
    mb = [s.missing_before_fill for s in summaries]
    ma = [s.missing_after_fill  for s in summaries]
    ax.bar(xs - 0.18, mb, 0.36, label="Before fill", color="#d62728", alpha=0.8)
    ax.bar(xs + 0.18, ma, 0.36, label="After fill",  color="#2ca02c", alpha=0.8)
    ax.set_xticks(xs); ax.set_xticklabels(ids, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Missing values"); ax.set_title("(b) Missing values before / after fill")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v/1e3:.0f}k"))
    ax.legend(fontsize=8); ax.grid(axis="y", linestyle="--", alpha=0.4)

    # (c) duplicate rows removed
    ax = axes[1, 0]
    dup = [s.duplicate_rows_removed for s in summaries]
    ax.bar(ids, dup, color="#9467bd", alpha=0.85)
    ax.set_xticklabels(ids, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Duplicate rows removed"); ax.set_title("(c) Duplicate rows removed")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # (d) total rows original vs after window
    ax = axes[1, 1]
    orig = [s.rows_original     for s in summaries]
    win  = [s.rows_after_window for s in summaries]
    ax.bar(xs - 0.18, orig, 0.36, label="Original",      color="#8c564b", alpha=0.8)
    ax.bar(xs + 0.18, win,  0.36, label="After window",  color="#1f77b4", alpha=0.8)
    ax.set_xticks(xs); ax.set_xticklabels(ids, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Rows"); ax.set_title("(d) Rows original vs after shared window")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v/1e3:.0f}k"))
    ax.legend(fontsize=8); ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    _savefig(fig, out_dir / "fig1_dataset_overview", fmt)


def plot_model_comparison(
    all_test_lb: pd.DataFrame,
    horizons: list[int],
    out_dir: Path,
    fmt: str,
) -> None:
    """Figure 2 – one sub-plot per metric, lines = models."""
    models = sorted(all_test_lb["model"].unique())
    cmap   = {m: _COLOURS[i % len(_COLOURS)] for i, m in enumerate(models)}
    mmap   = {m: _MARKERS[i % len(_MARKERS)] for i, m in enumerate(models)}

    for metric in EXTRA_METRICS:
        if metric not in all_test_lb.columns:
            continue
        fig, ax = plt.subplots(figsize=(10, 5))
        for model in models:
            sub = all_test_lb[all_test_lb["model"] == model].sort_values("horizon_hours")
            if sub.empty:
                continue
            ax.plot(
                sub["horizon_hours"], sub[metric].abs(),
                marker=mmap[model], color=cmap[model],
                label=model, linewidth=1.8, markersize=6,
            )
        ax.set_xlabel("Prediction Horizon (hours)", fontsize=12)
        ax.set_ylabel(METRIC_LABEL.get(metric, metric), fontsize=12)
        ax.set_title(f"Model Comparison — {metric}", fontsize=13, fontweight="bold")
        ax.set_xticks(horizons)
        ax.tick_params(axis="x", labelsize=9)
        ax.legend(fontsize=9, framealpha=0.85, loc="upper left")
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        fig.tight_layout()
        _savefig(fig, out_dir / f"fig2_comparison_{metric.lower()}", fmt)


def plot_best_model_curves(
    metrics_df: pd.DataFrame,
    horizons: list[int],
    out_dir: Path,
    fmt: str,
) -> None:
    """Figure 3 – best-model error curves, dual y-axis."""
    df = metrics_df[metrics_df["horizon_hours"].isin(horizons)].sort_values("horizon_hours")
    xs = df["horizon_hours"].tolist()

    fig, ax1 = plt.subplots(figsize=(11, 5))
    ax2 = ax1.twinx()

    lines = []
    for col, ax, ls in [
        ("MAE",   ax1, "o-"),
        ("RMSE",  ax1, "s--"),
        ("RMSLE", ax1, "^:"),
        ("MASE",  ax2, "D-."),
        ("SMAPE", ax2, "v--"),
    ]:
        if col not in df.columns:
            continue
        target_ax = ax
        colour = _COLOURS[["MAE","RMSE","RMSLE","MASE","SMAPE"].index(col) % len(_COLOURS)]
        l, = target_ax.plot(
            xs, df[col].abs(), ls,
            color=colour, linewidth=2, markersize=6, label=col,
        )
        lines.append(l)

    ax1.set_xlabel("Prediction Horizon (hours)", fontsize=12)
    ax1.set_ylabel("MAE / RMSE / RMSLE  (m)", fontsize=11, color="#1f77b4")
    ax2.set_ylabel("MASE / SMAPE",            fontsize=11, color="#d62728")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    ax1.set_xticks(horizons); ax1.tick_params(axis="x", labelsize=9)
    ax1.grid(axis="y", linestyle="--", alpha=0.3)
    ax1.legend(lines, [l.get_label() for l in lines], fontsize=9,
               loc="upper left", framealpha=0.85)
    ax1.set_title("Best Model — All Metrics across Horizons",
                  fontsize=13, fontweight="bold")
    fig.tight_layout()
    _savefig(fig, out_dir / "fig3_best_model_all_metrics", fmt)


def plot_mase_heatmap(
    all_test_lb: pd.DataFrame,
    horizons: list[int],
    out_dir: Path,
    fmt: str,
) -> None:
    """Figure 4 – MASE heatmap: rows = models, cols = horizons."""
    if "MASE" not in all_test_lb.columns:
        return
    pivot = (
        all_test_lb.pivot_table(index="model", columns="horizon_hours",
                                values="MASE", aggfunc="mean")
        .abs()
        .reindex(columns=[h for h in horizons if h in all_test_lb["horizon_hours"].unique()])
    )
    pivot = pivot.sort_values(by=pivot.columns.tolist(), na_position="last")
    nrow, ncol = pivot.shape

    fig, ax = plt.subplots(figsize=(max(10, ncol * 0.85), max(3, nrow * 0.65)))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
    plt.colorbar(im, ax=ax, label="MASE (lower = better)", shrink=0.8)

    ax.set_xticks(range(ncol))
    ax.set_xticklabels([f"{h}h" for h in pivot.columns], fontsize=9)
    ax.set_yticks(range(nrow))
    ax.set_yticklabels(pivot.index.tolist(), fontsize=9)
    ax.set_xlabel("Prediction Horizon", fontsize=11)
    ax.set_title("MASE Heatmap — All Models × All Horizons",
                 fontsize=13, fontweight="bold")

    vmax = np.nanmax(pivot.values)
    for r in range(nrow):
        for c in range(ncol):
            v = pivot.values[r, c]
            if not np.isnan(v):
                txt_col = "white" if v > vmax * 0.6 else "black"
                ax.text(c, r, f"{v:.3f}", ha="center", va="center",
                        fontsize=7, color=txt_col)

    fig.tight_layout()
    _savefig(fig, out_dir / "fig4_mase_heatmap", fmt)


def plot_val_vs_test(
    all_val_lb: pd.DataFrame,
    all_test_lb: pd.DataFrame,
    horizons: list[int],
    out_dir: Path,
    fmt: str,
) -> None:
    """Figure 5 – per-horizon grouped bar: validation MASE vs test MASE."""
    avail = sorted(all_test_lb["horizon_hours"].unique())
    n = len(avail)
    ncols = min(4, n)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 4, nrows * 3.5),
                             squeeze=False)
    fig.suptitle("Validation MASE vs Test MASE per Horizon",
                 fontsize=13, fontweight="bold", y=1.01)

    for idx, h in enumerate(avail):
        ax = axes[idx // ncols][idx % ncols]
        vl = all_val_lb[all_val_lb["horizon_hours"] == h]
        tl = all_test_lb[all_test_lb["horizon_hours"] == h]
        models = tl["model"].tolist()
        if not models:
            ax.set_visible(False); continue

        val_col = "validation_MASE" if "validation_MASE" in vl.columns else "MASE"
        test_col = "MASE"
        val_vals  = [float(vl.loc[vl["model"]==m, val_col].iloc[0])
                     if m in vl["model"].values else np.nan for m in models]
        test_vals = tl[test_col].abs().tolist()

        x = np.arange(len(models))
        w = 0.35
        ax.bar(x - w/2, val_vals,  w, label="Val MASE",  color="#5b9bd5", alpha=0.85)
        ax.bar(x + w/2, test_vals, w, label="Test MASE", color="#ed7d31", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=25, ha="right", fontsize=7)
        ax.set_title(f"{h}h", fontsize=10, fontweight="bold")
        ax.set_ylabel("MASE", fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(axis="y", linestyle="--", alpha=0.35)

    for idx in range(len(avail), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.tight_layout()
    _savefig(fig, out_dir / "fig5_val_vs_test_mase", fmt)


def plot_summary_grid(
    metrics_df: pd.DataFrame,
    horizons: list[int],
    out_dir: Path,
    fmt: str,
) -> None:
    """Figure 6 – 5-panel grid: one sub-plot per metric."""
    df   = metrics_df[metrics_df["horizon_hours"].isin(horizons)].sort_values("horizon_hours")
    avail = [m for m in EXTRA_METRICS if m in df.columns]
    ncols = 3
    nrows = math.ceil(len(avail) / ncols)

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 5, nrows * 3.8),
                             squeeze=False)
    fig.suptitle("Best Model — Forecast Error across All Horizons",
                 fontsize=14, fontweight="bold", y=1.01)

    for idx, metric in enumerate(avail):
        ax   = axes[idx // ncols][idx % ncols]
        col  = _COLOURS[idx % len(_COLOURS)]
        xs   = df["horizon_hours"].tolist()
        ys   = df[metric].abs().tolist()
        ax.plot(xs, ys, "o-", color=col, linewidth=2, markersize=6)
        ax.fill_between(xs, ys, alpha=0.10, color=col)
        for x_, y_ in zip(xs, ys):
            ax.annotate(f"{y_:.4f}", (x_, y_),
                        textcoords="offset points", xytext=(0, 5),
                        ha="center", fontsize=6, color=col)
        ax.set_xticks(horizons[::2])          # every-other tick to avoid crowding
        ax.tick_params(axis="x", labelsize=8, rotation=30)
        ax.set_xlabel("Horizon (h)", fontsize=9)
        ax.set_ylabel(METRIC_LABEL.get(metric, metric), fontsize=9)
        ax.set_title(metric, fontsize=11, fontweight="bold")
        ax.grid(linestyle="--", alpha=0.35)

    for idx in range(len(avail), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.tight_layout()
    _savefig(fig, out_dir / "fig6_summary_grid", fmt)


def plot_elapsed_time(
    metrics_df: pd.DataFrame,
    horizons: list[int],
    out_dir: Path,
    fmt: str,
) -> None:
    """Figure 7 – training time per horizon."""
    if "elapsed_seconds" not in metrics_df.columns:
        return
    df = metrics_df[metrics_df["horizon_hours"].isin(horizons)].sort_values("horizon_hours")
    labels = [f"{h}h" for h in df["horizon_hours"]]
    mins   = (df["elapsed_seconds"] / 60).tolist()

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(labels, mins, color="#5b9bd5", alpha=0.85, edgecolor="white")
    for b, v in zip(bars, mins):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.2,
                f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("Prediction Horizon", fontsize=11)
    ax.set_ylabel("Training Time (min)", fontsize=11)
    ax.set_title("Training Time per Horizon", fontsize=12, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    _savefig(fig, out_dir / "fig7_training_time", fmt)


def generate_all_plots(
    station_summaries: list[StationSummary],
    horizon_results: list[HorizonResult],
    run_dir: Path,
    fmt: str,
) -> None:
    out_dir = run_dir / "plots"
    print(f"\n[Plotting] saving to {out_dir}/")

    metrics_df = pd.DataFrame([
        {
            "horizon_hours":    r.horizon_hours,
            "prediction_length": r.prediction_length,
            "best_model":        r.best_model,
            "validation_MASE":   r.val_mase,
            **r.metrics,
            "training_curve_dir": r.training_curve_dir,
            "elapsed_seconds":   r.elapsed_seconds,
        }
        for r in horizon_results
    ])
    horizons = sorted(metrics_df["horizon_hours"].tolist())

    # load per-horizon leaderboards
    test_frames, val_frames = [], []
    for r in horizon_results:
        t = pd.read_csv(r.test_leaderboard_csv)
        t["horizon_hours"] = r.horizon_hours
        t = t.apply(lambda col: col.abs() if col.name in EXTRA_METRICS else col)
        test_frames.append(t)

        v = pd.read_csv(r.val_leaderboard_csv)
        v["horizon_hours"] = r.horizon_hours
        v = v.apply(lambda col: col.abs() if col.name in EXTRA_METRICS else col)
        val_frames.append(v)

    all_test_lb = pd.concat(test_frames, ignore_index=True)
    all_val_lb  = pd.concat(val_frames,  ignore_index=True)

    print("  [1/7] Dataset overview ...")
    plot_dataset_overview(station_summaries, out_dir, fmt)

    print("  [2/7] Model comparison (per metric) ...")
    plot_model_comparison(all_test_lb, horizons, out_dir, fmt)

    print("  [3/7] Best-model dual-axis curves ...")
    plot_best_model_curves(metrics_df, horizons, out_dir, fmt)

    print("  [4/7] MASE heatmap ...")
    plot_mase_heatmap(all_test_lb, horizons, out_dir, fmt)

    print("  [5/7] Validation vs Test MASE grid ...")
    plot_val_vs_test(all_val_lb, all_test_lb, horizons, out_dir, fmt)

    print("  [6/7] Summary grid (5 metrics) ...")
    plot_summary_grid(metrics_df, horizons, out_dir, fmt)

    print("  [7/7] Training time ...")
    plot_elapsed_time(metrics_df, horizons, out_dir, fmt)

    print(f"  All plots saved to {out_dir.resolve()}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def collect_env() -> dict[str, Any]:
    return {
        "python_version":            sys.version,
        "torch_version":             torch.__version__,
        "cuda_version":              torch.version.cuda,
        "cuda_available":            bool(torch.cuda.is_available()),
        "device_count":              int(torch.cuda.device_count()),
        "device_name":               torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda_visible_devices":      os.environ.get("CUDA_VISIBLE_DEVICES"),
        "autogluon_timeseries_version": agts.__version__,
    }


def steps_per_hour(freq: str) -> int:
    delta   = pd.Timedelta(freq)
    one_hr  = pd.Timedelta(hours=1)
    if one_hr % delta != pd.Timedelta(0):
        raise ValueError(f"Frequency '{freq}' does not evenly divide 1 hour")
    return int(one_hr / delta)


def main() -> None:
    args = build_parser().parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    effective_time_limit = None if args.time_limit <= 0 else args.time_limit

    env = collect_env()
    print("=" * 60)
    print(f"AutoGluon wave-height forecasting")
    print(f"  CUDA : {env['cuda_available']} | device: {env['device_name']}")
    print(f"  Split: {args.train_ratio:.0%} / {args.val_ratio:.0%} / {args.test_ratio:.0%}")
    print(
        f"  Preset: {args.presets} | time-limit: "
        f"{'unlimited' if effective_time_limit is None else f'{effective_time_limit}s/horizon'}"
    )
    print(
        f"  Deep  : epochs={args.deep_epochs} | batch={args.deep_batch_size} | "
        f"batches/epoch={args.deep_num_batches_per_epoch if args.deep_num_batches_per_epoch > 0 else 'default'} | "
        f"patience={'off' if args.deep_early_stopping_patience < 0 else args.deep_early_stopping_patience}"
    )
    print(
        f"  Curves: {'enabled' if not args.no_curve_logs else 'disabled'} | "
        f"progress-bar: {'on' if not args.no_train_progress_bar else 'off'}"
    )
    if args.chronos_fine_tune:
        print(
            f"  Chronos fine-tune: steps={args.chronos_fine_tune_steps} | "
            f"batch={args.chronos_fine_tune_batch_size} | "
            f"grad-accum={args.chronos_gradient_accumulation}"
        )
    elif args.verbosity < 3 and not args.no_train_progress_bar:
        print("  Tip   : use --verbosity 3 if you want Chronos fine-tune losses streamed to console.")
    print("=" * 60)

    total = args.train_ratio + args.val_ratio + args.test_ratio
    if not math.isclose(total, 1.0, abs_tol=1e-6):
        raise ValueError(f"Ratios must sum to 1.0, got {total}")

    horizon_list = [int(h) for h in args.prediction_hours.split(",") if h.strip()]
    sph          = steps_per_hour(args.freq)

    shared_start, shared_end = load_shared_window(args.metadata_path, args.no_shared_window)

    station_files = sorted(args.input_dir.glob("*_aligned_10min.csv"))
    if not station_files:
        raise FileNotFoundError(f"No *_aligned_10min.csv in {args.input_dir}")
    print(f"\nFound {len(station_files)} station files")

    print("\nLoading stations ...")
    frames, station_summaries = build_all_stations(
        files        = station_files,
        target       = args.target,
        shared_start = shared_start,
        shared_end   = shared_end,
        limit        = args.limit_stations,
        max_steps    = args.max_steps,
        train_ratio  = args.train_ratio,
        val_ratio    = args.val_ratio,
    )

    if args.soft_label_path is not None:
        apply_soft_labels(frames, args.soft_label_path, args.target, args.teacher_weight)

    train_df = make_split(frames, station_summaries, "train")
    val_df   = make_split(frames, station_summaries, "val")
    test_df  = make_split(frames, station_summaries, "test")

    # Save split summary
    split_csv = args.output_root / "station_split_summary.csv"
    pd.DataFrame([asdict(s) for s in station_summaries]).to_csv(
        split_csv, index=False, encoding="utf-8-sig"
    )

    train_ts = to_tsdf(train_df)
    val_ts   = to_tsdf(val_df)
    test_ts  = to_tsdf(test_df)

    # ── Train each horizon ───────────────────────────────────────────────────
    horizon_results: list[HorizonResult] = []
    for hours in horizon_list:
        pred_len = hours * sph
        print(f"\n{'='*60}")
        print(f" Horizon {hours}h  ({pred_len} steps)")
        print(f"{'='*60}")
        result = train_one_horizon(
            hours    = hours,
            pred_len = pred_len,
            train_ts = train_ts,
            val_ts   = val_ts,
            test_ts  = test_ts,
            args     = args,
        )
        horizon_results.append(result)
        print(
            f"  Best model : {result.best_model}\n"
            f"  Val  MASE  : {result.val_mase:.6f}\n"
            f"  Test MAE   : {result.metrics['MAE']:.6f}\n"
            f"  Test RMSE  : {result.metrics['RMSE']:.6f}\n"
            f"  Test MASE  : {result.metrics['MASE']:.6f}\n"
            f"  Test RMSLE : {result.metrics['RMSLE']:.6f}\n"
            f"  Test SMAPE : {result.metrics['SMAPE']:.6f}\n"
            f"  Elapsed    : {result.elapsed_seconds/60:.1f} min"
        )

    # ── Save metrics summary ─────────────────────────────────────────────────
    metrics_rows = [
        {
            "horizon_hours":    r.horizon_hours,
            "prediction_length": r.prediction_length,
            "best_model":        r.best_model,
            "validation_MASE":   r.val_mase,
            "MAE":               r.metrics["MAE"],
            "MASE":              r.metrics["MASE"],
            "RMSE":              r.metrics["RMSE"],
            "RMSLE":             r.metrics["RMSLE"],
            "SMAPE":             r.metrics["SMAPE"],
            "training_curve_dir": r.training_curve_dir,
            "elapsed_seconds":   r.elapsed_seconds,
        }
        for r in horizon_results
    ]
    metrics_csv = args.output_root / "metrics_summary.csv"
    pd.DataFrame(metrics_rows).to_csv(metrics_csv, index=False, encoding="utf-8-sig")

    # ── Write JSON log ───────────────────────────────────────────────────────
    log = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config": vars(args) | {
            "input_dir":  str(args.input_dir.resolve()),
            "output_root": str(args.output_root.resolve()),
        },
        "environment":  env,
        "stations":     [asdict(s) for s in station_summaries],
        "horizons":     [asdict(r) for r in horizon_results],
    }
    args.log_path.parent.mkdir(parents=True, exist_ok=True)
    args.log_path.write_text(
        json.dumps(log, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    # ── Generate plots ───────────────────────────────────────────────────────
    generate_all_plots(
        station_summaries = station_summaries,
        horizon_results   = horizon_results,
        run_dir           = args.output_root,
        fmt               = args.plot_format,
    )

    print(f"\nDone.")
    print(f"  Metrics : {metrics_csv.resolve()}")
    print(f"  Log     : {args.log_path.resolve()}")
    print(f"  Plots   : {(args.output_root / 'plots').resolve()}")


if __name__ == "__main__":
    main()

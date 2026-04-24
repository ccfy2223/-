"""
train_tft_chronos2_cascade.py
=============================

Two-stage cascade training for wave-height forecasting:

1. Train a short-horizon TFT model.
2. Convert each long-horizon forecast origin into an independent window item.
3. Use TFT short-term forecasts as origin-specific future covariates.
4. Train Chronos-2 on the window dataset for long-horizon forecasting.

This script is intentionally standalone and does not modify train_autogluon.py.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


def _extract_cli_value(flag: str, default: str) -> str:
    if flag in sys.argv:
        idx = sys.argv.index(flag)
        if idx + 1 < len(sys.argv):
            return sys.argv[idx + 1]
    return default


warnings.filterwarnings("ignore", category=FutureWarning)
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
os.environ["CUDA_VISIBLE_DEVICES"] = _extract_cli_value("--gpu-id", "0")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np
import pandas as pd
import torch
import autogluon.timeseries as agts
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor


BASE_KNOWN_COVARIATES = [
    "time_sin_hour",
    "time_cos_hour",
    "time_sin_doy",
    "time_cos_doy",
    "month",
    "day_of_week",
]

DYNAMIC_COLUMNS = [
    "WDIR",
    "WSPD",
    "GST",
    "WVHT",
    "DPD",
    "APD",
    "MWD",
    "PRES",
    "ATMP",
    "WTMP",
    "DEWP",
    "VIS",
    "TIDE",
]

GUIDANCE_COLUMNS = [
    "tft_short_pred",
    "tft_short_mask",
    "tft_short_mean",
    "tft_short_last",
    "tft_short_slope",
]

EXTRA_METRICS = ["MAE", "MASE", "RMSE", "RMSLE", "SMAPE"]
DEFAULT_LONG_HORIZONS = [12, 24, 48, 72, 120]


def build_patchtst_configs(
    prediction_length: int,
) -> list[tuple[str, dict]]:
    base_context = max(72, min(512, prediction_length * 4))
    return [
        (
            "balanced",
            {
                "context_length": base_context,
                "patch_len": 16,
                "stride": 8,
                "d_model": 64,
                "nhead": 4,
                "num_encoder_layers": 3,
                "dropout": 0.1,
                "lr": 1e-4,
                "batch_size": 32,
                "max_epochs": 30,
                "num_batches_per_epoch": 100,
                "early_stopping_patience": 10,
                "trainer_kwargs": {"gradient_clip_val": 1.0},
            },
        ),
    ]


def build_tft_retry_configs(
    prediction_length: int,
    allow_no_past_covariates_fallback: bool = True,
) -> list[tuple[str, dict[str, Any]]]:
    base_context = max(72, min(168, prediction_length * 3))
    configs: list[tuple[str, dict[str, Any]]] = [
        (
            "balanced",
            {
                "context_length": base_context,
                "hidden_dim": 32,
                "variable_dim": 32,
                "num_heads": 4,
                "dropout_rate": 0.1,
                "lr": 1e-4,
                "batch_size": 32,
                "max_epochs": 30,
                "num_batches_per_epoch": 50,
                "early_stopping_patience": 10,
                "disable_static_features": True,
                "trainer_kwargs": {"gradient_clip_val": 1.0},
            },
        ),
        (
            "stable_past_covariates",
            {
                "context_length": base_context,
                "hidden_dim": 24,
                "variable_dim": 24,
                "num_heads": 4,
                "dropout_rate": 0.15,
                "lr": 5e-5,
                "batch_size": 24,
                "max_epochs": 30,
                "num_batches_per_epoch": 45,
                "early_stopping_patience": 10,
                "disable_static_features": True,
                "trainer_kwargs": {"gradient_clip_val": 0.7},
            },
        ),
        (
            "robust_past_covariates",
            {
                "context_length": base_context,
                "hidden_dim": 16,
                "variable_dim": 16,
                "num_heads": 2,
                "dropout_rate": 0.2,
                "lr": 3e-5,
                "batch_size": 16,
                "max_epochs": 25,
                "num_batches_per_epoch": 30,
                "early_stopping_patience": 8,
                "disable_static_features": True,
                "trainer_kwargs": {"gradient_clip_val": 0.5},
            },
        ),
    ]
    if allow_no_past_covariates_fallback:
        configs.append(
            (
                "no_past_covariates_last_resort",
                {
                    "context_length": max(72, prediction_length * 2),
                    "hidden_dim": 16,
                    "variable_dim": 16,
                    "num_heads": 2,
                    "dropout_rate": 0.2,
                    "lr": 5e-5,
                    "batch_size": 16,
                    "max_epochs": 20,
                    "num_batches_per_epoch": 30,
                    "early_stopping_patience": 6,
                    "disable_static_features": True,
                    "disable_past_covariates": True,
                    "trainer_kwargs": {"gradient_clip_val": 0.5},
                },
            )
        )
    return configs


@dataclass
class StationSummary:
    item_id: str
    rows_original: int
    rows_after_dedup: int
    rows_after_window: int
    duplicate_rows_removed: int
    missing_before_fill: int
    missing_after_fill: int
    first_timestamp: str
    last_timestamp: str
    train_rows: int
    val_rows: int
    test_rows: int


@dataclass
class ShortModelResult:
    horizon_hours: int
    prediction_length: int
    best_model: str
    val_mase: float
    metrics: dict[str, float]
    val_leaderboard_csv: str
    test_leaderboard_csv: str
    model_dir: str
    elapsed_seconds: float


@dataclass
class WindowSpec:
    item_id: str
    source_item_id: str
    split_name: str
    horizon_hours: int
    origin_index: int
    origin_timestamp: str


@dataclass
class HorizonResult:
    horizon_hours: int
    prediction_length: int
    train_windows: int
    val_windows: int
    test_windows: int
    best_model: str
    val_mase: float
    metrics: dict[str, float]
    val_leaderboard_csv: str
    test_leaderboard_csv: str
    model_dir: str
    elapsed_seconds: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Two-stage TFT + Chronos-2 cascade training for wave-height forecasting."
    )
    parser.add_argument("--input-dir", type=Path, default=Path("processed_csv/aligned_stations"))
    parser.add_argument("--metadata-path", type=Path, default=Path("processed_csv/shared_timeline_metadata.json"))
    parser.add_argument("--output-root", type=Path, default=Path("cascade_runs"))
    parser.add_argument("--log-path", type=Path, default=Path("cascade_train_log.json"))
    parser.add_argument("--target", default="WVHT")
    parser.add_argument("--freq", default="10min")
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.20)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    parser.add_argument("--limit-stations", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--no-shared-window", action="store_true")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--verbosity", type=int, default=2)
    parser.add_argument("--presets", default="best_quality")
    parser.add_argument("--eval-metric", default="MASE")
    parser.add_argument("--short-horizon-hours", type=int, default=6)
    parser.add_argument(
        "--long-horizons",
        default=",".join(str(h) for h in DEFAULT_LONG_HORIZONS),
        help="Comma-separated long-horizon hours, e.g. 12,24,48,72,120",
    )
    parser.add_argument("--context-hours", type=int, default=168, help="History context for each window.")
    parser.add_argument(
        "--window-stride-hours",
        type=int,
        default=6,
        help="Stride between forecast origins when building long-horizon windows.",
    )
    parser.add_argument("--train-max-windows-per-station", type=int, default=256)
    parser.add_argument("--val-max-windows-per-station", type=int, default=64)
    parser.add_argument("--test-max-windows-per-station", type=int, default=64)
    parser.add_argument("--tft-guidance-batch-size", type=int, default=64)
    parser.add_argument("--tft-time-limit", type=int, default=3600)
    parser.add_argument("--chronos2-time-limit", type=int, default=3600)
    parser.add_argument("--chronos2-model-path", default="autogluon/chronos-2")
    parser.add_argument("--chronos2-batch-size", type=int, default=64)
    parser.add_argument("--chronos2-fine-tune", action="store_true")
    parser.add_argument("--chronos2-fine-tune-steps", type=int, default=300)
    parser.add_argument("--chronos2-fine-tune-batch-size", type=int, default=16)
    parser.add_argument("--chronos2-cross-learning", action="store_true", default=False)
    parser.add_argument(
        "--pretrained-tft-path",
        type=Path,
        default=None,
        help="Path to an existing AutoGluon TimeSeriesPredictor directory for the short-horizon TFT stage.",
    )
    parser.add_argument(
        "--pretrained-tft-model-name",
        default="TemporalFusionTransformer",
        help="Model name inside the loaded predictor to use for short-horizon guidance.",
    )
    parser.add_argument(
        "--allow-no-past-covariates-fallback",
        action="store_true",
        default=False,
        help="Allow disabling past covariates as the final fallback if all robust TFT attempts fail.",
    )
    return parser


def collect_env() -> dict[str, Any]:
    return {
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "autogluon_timeseries_version": agts.__version__,
    }


def steps_per_hour(freq: str) -> int:
    delta = pd.Timedelta(freq)
    one_hour = pd.Timedelta(hours=1)
    if one_hour % delta != pd.Timedelta(0):
        raise ValueError(f"Frequency '{freq}' does not evenly divide one hour.")
    return int(one_hour / delta)


def parse_horizon_hours(raw: str) -> list[int]:
    values = [int(x) for x in raw.split(",") if x.strip()]
    if not values:
        raise ValueError("At least one long horizon must be provided.")
    return sorted(set(values))


def load_shared_window(meta_path: Path, disabled: bool) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    if disabled or not meta_path.exists():
        return None, None
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    start = payload.get("shared_start")
    end = payload.get("shared_end")
    return (pd.Timestamp(start) if start else None, pd.Timestamp(end) if end else None)


def load_station(
    path: Path,
    target: str,
    shared_start: pd.Timestamp | None,
    shared_end: pd.Timestamp | None,
    max_steps: int,
) -> tuple[pd.DataFrame, StationSummary]:
    item_id = path.stem.split("_")[0]
    available_columns = pd.read_csv(path, nrows=0).columns.tolist()
    dynamic = [c for c in DYNAMIC_COLUMNS if c in available_columns]
    known = [c for c in BASE_KNOWN_COVARIATES if c in available_columns]

    df = pd.read_csv(
        path,
        usecols=["datetime"] + dynamic + known,
        parse_dates=["datetime"],
        low_memory=False,
    )
    rows_original = len(df)

    duplicates = int(df.duplicated("datetime").sum())
    df = df.sort_values("datetime").drop_duplicates("datetime", keep="last").reset_index(drop=True)
    rows_after_dedup = len(df)

    if shared_start is not None and shared_end is not None:
        df = df[(df["datetime"] >= shared_start) & (df["datetime"] <= shared_end)].reset_index(drop=True)

    if max_steps > 0 and len(df) > max_steps:
        df = df.iloc[-max_steps:].reset_index(drop=True)
    rows_after_window = len(df)

    if rows_after_window == 0:
        raise ValueError(f"{path.name}: no rows left after filtering.")

    numeric_columns = [c for c in df.columns if c != "datetime"]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce").astype("float32")
    # Convert +/-inf to NaN before fill; some raw files may contain non-finite values.
    df[numeric_columns] = df[numeric_columns].replace([np.inf, -np.inf], np.nan)

    missing_before_fill = int(df[numeric_columns].isna().sum().sum())
    df[numeric_columns] = df[numeric_columns].ffill().bfill()
    missing_after_fill = int(df[numeric_columns].isna().sum().sum())

    if target not in df.columns:
        raise KeyError(f"Target '{target}' not found in {path.name}.")
    if df[target].isna().any():
        raise ValueError(f"Target '{target}' still contains NaN in {path.name}.")

    df.insert(0, "item_id", item_id)
    df = df.rename(columns={"datetime": "timestamp"})

    summary = StationSummary(
        item_id=item_id,
        rows_original=rows_original,
        rows_after_dedup=rows_after_dedup,
        rows_after_window=rows_after_window,
        duplicate_rows_removed=duplicates,
        missing_before_fill=missing_before_fill,
        missing_after_fill=missing_after_fill,
        first_timestamp=str(df["timestamp"].iloc[0]),
        last_timestamp=str(df["timestamp"].iloc[-1]),
        train_rows=0,
        val_rows=0,
        test_rows=0,
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
) -> tuple[dict[str, pd.DataFrame], list[StationSummary], list[str], list[str]]:
    selected = files[:limit] if limit > 0 else files
    frames: dict[str, pd.DataFrame] = {}
    summaries: list[StationSummary] = []

    for path in selected:
        df, summary = load_station(path, target, shared_start, shared_end, max_steps)
        n_rows = len(df)
        train_end = max(1, int(n_rows * train_ratio))
        val_end = max(train_end + 1, int(n_rows * (train_ratio + val_ratio)))
        val_end = min(val_end, n_rows - 1)
        summary.train_rows = train_end
        summary.val_rows = val_end - train_end
        summary.test_rows = n_rows - val_end
        frames[summary.item_id] = df
        summaries.append(summary)
        print(
            f"  {summary.item_id}: {n_rows} rows | "
            f"train {summary.train_rows} | val {summary.val_rows} | test {summary.test_rows}"
        )

    if not frames:
        raise ValueError("No station data was loaded.")

    common_columns = set.intersection(*(set(df.columns) for df in frames.values()))
    ordered_common = [c for c in frames[next(iter(frames))].columns if c in common_columns]
    for item_id in frames:
        frames[item_id] = frames[item_id][ordered_common].copy()

    base_known = [c for c in BASE_KNOWN_COVARIATES if c in ordered_common]
    past_covariates = [c for c in ordered_common if c not in {"item_id", "timestamp", target, *base_known}]

    # Some columns (e.g. VIS/TIDE) can remain entirely NaN even after ffill/bfill.
    # These unstable covariates often cause NaN predictions in TFT, so drop them globally.
    residual_nan_covariates: list[str] = []
    for covariate in past_covariates:
        if any(frames[item_id][covariate].isna().any() for item_id in frames):
            residual_nan_covariates.append(covariate)

    if residual_nan_covariates:
        for item_id in frames:
            frames[item_id] = frames[item_id].drop(columns=residual_nan_covariates)
        past_covariates = [c for c in past_covariates if c not in residual_nan_covariates]
        print(f"  Dropped past covariates with residual NaN: {residual_nan_covariates}")

    return frames, summaries, base_known, past_covariates


def make_base_split(
    frames: dict[str, pd.DataFrame],
    summaries: list[StationSummary],
    split_name: str,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    summary_map = {summary.item_id: summary for summary in summaries}
    for item_id, df in frames.items():
        summary = summary_map[item_id]
        train_end = summary.train_rows
        val_end = train_end + summary.val_rows
        if split_name == "train":
            parts.append(df.iloc[:train_end].copy())
        elif split_name == "val":
            parts.append(df.iloc[:val_end].copy())
        elif split_name == "test":
            parts.append(df.copy())
        else:
            raise ValueError(f"Unknown split '{split_name}'.")
    return pd.concat(parts, ignore_index=True)


def to_tsdf(df: pd.DataFrame) -> TimeSeriesDataFrame:
    return TimeSeriesDataFrame.from_data_frame(df=df, id_column="item_id", timestamp_column="timestamp")


def _flip_sign(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for metric in EXTRA_METRICS:
        if metric in out.columns:
            out[metric] = out[metric].abs()
    if "score_val" in out.columns:
        out["validation_MASE"] = out["score_val"].abs()
    if "score_test" in out.columns:
        out["test_MASE"] = out["score_test"].abs()
    return out


def train_short_tft(
    train_ts: TimeSeriesDataFrame,
    val_ts: TimeSeriesDataFrame,
    test_ts: TimeSeriesDataFrame,
    args: argparse.Namespace,
    base_known_covariates: list[str],
    prediction_length: int,
    model_name: str = "TemporalFusionTransformer",
) -> tuple[TimeSeriesPredictor, ShortModelResult]:
    run_dir = args.output_root / f"short_tft_{args.short_horizon_hours:03d}h"
    run_dir.mkdir(parents=True, exist_ok=True)
    attempt_records: list[dict[str, Any]] = []
    last_error: Exception | None = None

    if model_name == "PatchTST":
        retry_configs = build_patchtst_configs(prediction_length=prediction_length)
    else:
        retry_configs = build_tft_retry_configs(
            prediction_length=prediction_length,
            allow_no_past_covariates_fallback=bool(
                getattr(args, "allow_no_past_covariates_fallback", True)
            ),
        )
    for attempt_idx, (attempt_name, tft_params) in enumerate(retry_configs, start=1):
        attempt_dir = run_dir / f"attempt_{attempt_idx:02d}_{attempt_name}"
        model_dir = attempt_dir / "model"
        attempt_dir.mkdir(parents=True, exist_ok=True)

        predictor = TimeSeriesPredictor(
            target=args.target,
            prediction_length=prediction_length,
            freq=args.freq,
            eval_metric=args.eval_metric,
            known_covariates_names=base_known_covariates,
            path=str(model_dir),
            verbosity=args.verbosity,
        )

        print(f"  TFT attempt {attempt_idx}: {attempt_name}")
        print(f"    params: {tft_params}")

        try:
            t0 = time.perf_counter()
            predictor.fit(
                train_data=train_ts,
                tuning_data=val_ts,
                presets=args.presets,
                hyperparameters={model_name: tft_params},
                time_limit=args.tft_time_limit,
                random_seed=args.random_seed,
            )
            elapsed = time.perf_counter() - t0

            leaderboard = predictor.leaderboard(display=False)
            if leaderboard.empty:
                raise RuntimeError("No models were successfully trained in this TFT attempt.")

            val_lb = _flip_sign(predictor.leaderboard(data=val_ts, extra_metrics=EXTRA_METRICS, display=False))
            test_lb = _flip_sign(predictor.leaderboard(data=test_ts, extra_metrics=EXTRA_METRICS, display=False))

            val_csv = run_dir / "leaderboard_validation.csv"
            test_csv = run_dir / "leaderboard_test.csv"
            val_lb.to_csv(val_csv, index=False, encoding="utf-8-sig")
            test_lb.to_csv(test_csv, index=False, encoding="utf-8-sig")

            best_model = str(leaderboard.iloc[0]["model"])
            test_row = test_lb.loc[test_lb["model"] == best_model]
            if test_row.empty:
                raise ValueError(f"TFT best model '{best_model}' missing from test leaderboard.")
            val_row = val_lb.loc[val_lb["model"] == best_model]
            if val_row.empty:
                raise ValueError(f"TFT best model '{best_model}' missing from validation leaderboard.")

            attempt_records.append(
                {
                    "attempt": attempt_idx,
                    "name": attempt_name,
                    "status": "success",
                    "model_dir": str(model_dir.resolve()),
                    "params": tft_params,
                    "elapsed_seconds": elapsed,
                }
            )
            (run_dir / "tft_attempts.json").write_text(
                json.dumps(attempt_records, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )

            result = ShortModelResult(
                horizon_hours=args.short_horizon_hours,
                prediction_length=prediction_length,
                best_model=best_model,
                val_mase=float(val_row.iloc[0]["validation_MASE"]),
                metrics={metric: float(test_row.iloc[0][metric]) for metric in EXTRA_METRICS},
                val_leaderboard_csv=str(val_csv.resolve()),
                test_leaderboard_csv=str(test_csv.resolve()),
                model_dir=str(model_dir.resolve()),
                elapsed_seconds=elapsed,
            )
            return predictor, result
        except Exception as exc:
            last_error = exc
            attempt_records.append(
                {
                    "attempt": attempt_idx,
                    "name": attempt_name,
                    "status": "failed",
                    "model_dir": str(model_dir.resolve()),
                    "params": tft_params,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            (run_dir / "tft_attempts.json").write_text(
                json.dumps(attempt_records, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
            print(f"    attempt failed: {type(exc).__name__}: {exc}")

    raise RuntimeError(f"All TFT attempts failed. Last error: {last_error}")


def load_pretrained_short_tft(
    val_ts: TimeSeriesDataFrame,
    test_ts: TimeSeriesDataFrame,
    args: argparse.Namespace,
    prediction_length: int,
) -> tuple[TimeSeriesPredictor, ShortModelResult]:
    if args.pretrained_tft_path is None:
        raise ValueError("pretrained_tft_path must be provided.")

    run_dir = args.output_root / f"short_tft_{args.short_horizon_hours:03d}h"
    run_dir.mkdir(parents=True, exist_ok=True)

    predictor = TimeSeriesPredictor.load(str(args.pretrained_tft_path))
    if predictor.prediction_length != prediction_length:
        raise ValueError(
            f"Loaded TFT predictor prediction_length={predictor.prediction_length}, "
            f"but current short horizon requires {prediction_length}."
        )

    model_names = predictor.model_names()
    if args.pretrained_tft_model_name not in model_names:
        raise ValueError(
            f"Model '{args.pretrained_tft_model_name}' not found in loaded predictor. "
            f"Available models: {model_names}"
        )

    val_lb = _flip_sign(
        predictor.leaderboard(data=val_ts, extra_metrics=EXTRA_METRICS, display=False, use_cache=False)
    )
    test_lb = _flip_sign(
        predictor.leaderboard(data=test_ts, extra_metrics=EXTRA_METRICS, display=False, use_cache=False)
    )

    val_csv = run_dir / "leaderboard_validation.csv"
    test_csv = run_dir / "leaderboard_test.csv"
    val_lb.to_csv(val_csv, index=False, encoding="utf-8-sig")
    test_lb.to_csv(test_csv, index=False, encoding="utf-8-sig")

    best_model = args.pretrained_tft_model_name
    test_row = test_lb.loc[test_lb["model"] == best_model]
    if test_row.empty:
        raise ValueError(f"Loaded TFT model '{best_model}' missing from test leaderboard.")
    val_row = val_lb.loc[val_lb["model"] == best_model]
    if val_row.empty:
        raise ValueError(f"Loaded TFT model '{best_model}' missing from validation leaderboard.")

    result = ShortModelResult(
        horizon_hours=args.short_horizon_hours,
        prediction_length=prediction_length,
        best_model=best_model,
        val_mase=float(val_row.iloc[0]["validation_MASE"]),
        metrics={metric: float(test_row.iloc[0][metric]) for metric in EXTRA_METRICS},
        val_leaderboard_csv=str(val_csv.resolve()),
        test_leaderboard_csv=str(test_csv.resolve()),
        model_dir=str(Path(args.pretrained_tft_path).resolve()),
        elapsed_seconds=0.0,
    )
    return predictor, result


def chunked(values: list[WindowSpec], size: int) -> Iterable[list[WindowSpec]]:
    for start in range(0, len(values), max(1, size)):
        yield values[start : start + max(1, size)]


def select_origin_indices(origin_min: int, origin_max: int, stride: int, max_windows: int) -> list[int]:
    if origin_max < origin_min:
        return []
    origins = list(range(origin_min, origin_max + 1, max(1, stride)))
    if max_windows > 0 and len(origins) > max_windows:
        positions = np.linspace(0, len(origins) - 1, num=max_windows, dtype=int)
        origins = [origins[pos] for pos in positions]
    return origins


def build_window_specs(
    frames: dict[str, pd.DataFrame],
    summaries: list[StationSummary],
    split_name: str,
    horizon_hours: int,
    prediction_length: int,
    context_steps: int,
    stride_steps: int,
    max_windows_per_station: int,
) -> list[WindowSpec]:
    summary_map = {summary.item_id: summary for summary in summaries}
    specs: list[WindowSpec] = []

    for source_item_id, df in frames.items():
        summary = summary_map[source_item_id]
        train_end = summary.train_rows
        val_end = summary.train_rows + summary.val_rows
        n_rows = len(df)

        if split_name == "train":
            origin_min = max(context_steps - 1, 0)
            origin_max = train_end - prediction_length - 1
        elif split_name == "val":
            origin_min = max(context_steps - 1, train_end - 1)
            origin_max = val_end - prediction_length - 1
        elif split_name == "test":
            origin_min = max(context_steps - 1, val_end - 1)
            origin_max = n_rows - prediction_length - 1
        else:
            raise ValueError(f"Unknown split '{split_name}'.")

        origins = select_origin_indices(origin_min, origin_max, stride_steps, max_windows_per_station)
        for origin_index in origins:
            origin_timestamp = str(df.iloc[origin_index]["timestamp"])
            specs.append(
                WindowSpec(
                    item_id=f"{source_item_id}__{split_name}__{origin_index:07d}__h{horizon_hours:03d}",
                    source_item_id=source_item_id,
                    split_name=split_name,
                    horizon_hours=horizon_hours,
                    origin_index=origin_index,
                    origin_timestamp=origin_timestamp,
                )
            )
    return specs


def compute_guidance_stats(values: np.ndarray) -> dict[str, float]:
    if len(values) == 0:
        return {"mean": 0.0, "last": 0.0, "slope": 0.0}
    first = float(values[0])
    last = float(values[-1])
    denom = max(1, len(values) - 1)
    return {
        "mean": float(np.mean(values)),
        "last": last,
        "slope": float((last - first) / denom),
    }


def build_tft_guidance_predictions(
    specs: list[WindowSpec],
    frames: dict[str, pd.DataFrame],
    tft_predictor: TimeSeriesPredictor,
    short_model_name: str,
    short_pred_len: int,
    context_steps: int,
    base_known_covariates: list[str],
    batch_size: int,
) -> dict[str, np.ndarray]:
    guidance_map: dict[str, np.ndarray] = {}
    if not specs:
        return guidance_map

    for chunk in chunked(specs, batch_size):
        past_frames: list[pd.DataFrame] = []
        future_frames: list[pd.DataFrame] = []
        for spec in chunk:
            source = frames[spec.source_item_id]
            context_start = spec.origin_index - context_steps + 1
            past_window = source.iloc[context_start : spec.origin_index + 1].copy()
            past_window["item_id"] = spec.item_id
            past_frames.append(past_window)

            future_window = source.iloc[spec.origin_index + 1 : spec.origin_index + short_pred_len + 1].copy()
            future_window["item_id"] = spec.item_id
            future_frames.append(future_window[["item_id", "timestamp", *base_known_covariates]])

        past_ts = to_tsdf(pd.concat(past_frames, ignore_index=True))
        if base_known_covariates:
            future_known_ts = to_tsdf(pd.concat(future_frames, ignore_index=True))
            preds = tft_predictor.predict(past_ts, known_covariates=future_known_ts, model=short_model_name)
        else:
            preds = tft_predictor.predict(past_ts, model=short_model_name)
        pred_df = preds.reset_index()
        for item_id, group in pred_df.groupby("item_id"):
            guidance_map[str(item_id)] = group["mean"].to_numpy(dtype=np.float32)

    missing = [spec.item_id for spec in specs if spec.item_id not in guidance_map]
    if missing:
        raise ValueError(f"Missing TFT guidance predictions for {len(missing)} windows.")
    return guidance_map


def build_window_dataset(
    specs: list[WindowSpec],
    frames: dict[str, pd.DataFrame],
    target: str,
    base_known_covariates: list[str],
    past_covariates: list[str],
    tft_predictor: TimeSeriesPredictor,
    short_model_name: str,
    short_pred_len: int,
    long_pred_len: int,
    context_steps: int,
    guidance_batch_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not specs:
        raise ValueError("No window specs were created.")

    guidance_map = build_tft_guidance_predictions(
        specs=specs,
        frames=frames,
        tft_predictor=tft_predictor,
        short_model_name=short_model_name,
        short_pred_len=short_pred_len,
        context_steps=context_steps,
        base_known_covariates=base_known_covariates,
        batch_size=guidance_batch_size,
    )

    required_columns = ["item_id", "timestamp", target, *past_covariates, *base_known_covariates, *GUIDANCE_COLUMNS]
    window_frames: list[pd.DataFrame] = []
    meta_rows: list[dict[str, Any]] = []

    for spec in specs:
        source = frames[spec.source_item_id]
        context_start = spec.origin_index - context_steps + 1
        window_stop = spec.origin_index + long_pred_len + 1
        window_df = source.iloc[context_start:window_stop].copy().reset_index(drop=True)
        expected_length = context_steps + long_pred_len
        if len(window_df) != expected_length:
            raise ValueError(
                f"Window length mismatch for {spec.item_id}: got {len(window_df)}, expected {expected_length}."
            )

        window_df["item_id"] = spec.item_id
        for column in GUIDANCE_COLUMNS:
            window_df[column] = np.float32(0.0)

        future_start = context_steps
        future_index = window_df.index[future_start:]
        short_guidance = guidance_map[spec.item_id][: min(short_pred_len, len(future_index))]
        short_index = future_index[: len(short_guidance)]
        stats = compute_guidance_stats(short_guidance)

        window_df.loc[short_index, "tft_short_pred"] = short_guidance
        window_df.loc[short_index, "tft_short_mask"] = np.float32(1.0)
        window_df.loc[future_index, "tft_short_mean"] = np.float32(stats["mean"])
        window_df.loc[future_index, "tft_short_last"] = np.float32(stats["last"])
        window_df.loc[future_index, "tft_short_slope"] = np.float32(stats["slope"])

        window_frames.append(window_df[required_columns])
        meta_rows.append(
            {
                "item_id": spec.item_id,
                "source_item_id": spec.source_item_id,
                "split_name": spec.split_name,
                "horizon_hours": spec.horizon_hours,
                "origin_index": spec.origin_index,
                "origin_timestamp": spec.origin_timestamp,
                "context_start_timestamp": str(window_df["timestamp"].iloc[0]),
                "context_end_timestamp": str(window_df["timestamp"].iloc[context_steps - 1]),
                "forecast_start_timestamp": str(window_df["timestamp"].iloc[context_steps]),
                "forecast_end_timestamp": str(window_df["timestamp"].iloc[-1]),
                "tft_guidance_mean": stats["mean"],
                "tft_guidance_last": stats["last"],
                "tft_guidance_slope": stats["slope"],
            }
        )

    return pd.concat(window_frames, ignore_index=True), pd.DataFrame(meta_rows)


def train_long_horizon(
    args: argparse.Namespace,
    frames: dict[str, pd.DataFrame],
    summaries: list[StationSummary],
    target: str,
    base_known_covariates: list[str],
    past_covariates: list[str],
    tft_predictor: TimeSeriesPredictor,
    short_model_name: str,
    horizon_hours: int,
    short_pred_len: int,
    long_pred_len: int,
    context_steps: int,
    stride_steps: int,
) -> HorizonResult:
    run_dir = args.output_root / f"horizon_{horizon_hours:03d}h"
    model_dir = run_dir / "model"
    run_dir.mkdir(parents=True, exist_ok=True)

    train_specs = build_window_specs(
        frames=frames,
        summaries=summaries,
        split_name="train",
        horizon_hours=horizon_hours,
        prediction_length=long_pred_len,
        context_steps=context_steps,
        stride_steps=stride_steps,
        max_windows_per_station=args.train_max_windows_per_station,
    )
    val_specs = build_window_specs(
        frames=frames,
        summaries=summaries,
        split_name="val",
        horizon_hours=horizon_hours,
        prediction_length=long_pred_len,
        context_steps=context_steps,
        stride_steps=stride_steps,
        max_windows_per_station=args.val_max_windows_per_station,
    )
    test_specs = build_window_specs(
        frames=frames,
        summaries=summaries,
        split_name="test",
        horizon_hours=horizon_hours,
        prediction_length=long_pred_len,
        context_steps=context_steps,
        stride_steps=stride_steps,
        max_windows_per_station=args.test_max_windows_per_station,
    )

    if not train_specs:
        raise ValueError(f"No train windows available for {horizon_hours}h.")
    if not val_specs:
        raise ValueError(f"No validation windows available for {horizon_hours}h.")
    if not test_specs:
        raise ValueError(f"No test windows available for {horizon_hours}h.")

    print(f"  Window counts | train {len(train_specs)} | val {len(val_specs)} | test {len(test_specs)}")

    train_df, train_meta = build_window_dataset(
        specs=train_specs,
        frames=frames,
        target=target,
        base_known_covariates=base_known_covariates,
        past_covariates=past_covariates,
        tft_predictor=tft_predictor,
        short_model_name=short_model_name,
        short_pred_len=short_pred_len,
        long_pred_len=long_pred_len,
        context_steps=context_steps,
        guidance_batch_size=args.tft_guidance_batch_size,
    )
    val_df, val_meta = build_window_dataset(
        specs=val_specs,
        frames=frames,
        target=target,
        base_known_covariates=base_known_covariates,
        past_covariates=past_covariates,
        tft_predictor=tft_predictor,
        short_model_name=short_model_name,
        short_pred_len=short_pred_len,
        long_pred_len=long_pred_len,
        context_steps=context_steps,
        guidance_batch_size=args.tft_guidance_batch_size,
    )
    test_df, test_meta = build_window_dataset(
        specs=test_specs,
        frames=frames,
        target=target,
        base_known_covariates=base_known_covariates,
        past_covariates=past_covariates,
        tft_predictor=tft_predictor,
        short_model_name=short_model_name,
        short_pred_len=short_pred_len,
        long_pred_len=long_pred_len,
        context_steps=context_steps,
        guidance_batch_size=args.tft_guidance_batch_size,
    )

    train_meta.to_csv(run_dir / "window_index_train.csv", index=False, encoding="utf-8-sig")
    val_meta.to_csv(run_dir / "window_index_validation.csv", index=False, encoding="utf-8-sig")
    test_meta.to_csv(run_dir / "window_index_test.csv", index=False, encoding="utf-8-sig")

    train_ts = to_tsdf(train_df)
    val_ts = to_tsdf(val_df)
    test_ts = to_tsdf(test_df)

    predictor = TimeSeriesPredictor(
        target=target,
        prediction_length=long_pred_len,
        freq=args.freq,
        eval_metric=args.eval_metric,
        known_covariates_names=base_known_covariates + GUIDANCE_COLUMNS,
        path=str(model_dir),
        verbosity=args.verbosity,
    )

    chronos2_hparams = {
        "Chronos2": {
            "model_path": args.chronos2_model_path,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "batch_size": args.chronos2_batch_size,
            "cross_learning": args.chronos2_cross_learning,
            "fine_tune": args.chronos2_fine_tune,
            "fine_tune_steps": args.chronos2_fine_tune_steps,
            "fine_tune_batch_size": args.chronos2_fine_tune_batch_size,
        }
    }

    t0 = time.perf_counter()
    predictor.fit(
        train_data=train_ts,
        tuning_data=val_ts,
        presets=args.presets,
        hyperparameters=chronos2_hparams,
        time_limit=args.chronos2_time_limit,
        random_seed=args.random_seed,
    )
    elapsed = time.perf_counter() - t0

    val_lb = _flip_sign(predictor.leaderboard(data=val_ts, extra_metrics=EXTRA_METRICS, display=False))
    test_lb = _flip_sign(predictor.leaderboard(data=test_ts, extra_metrics=EXTRA_METRICS, display=False))

    val_csv = run_dir / "leaderboard_validation.csv"
    test_csv = run_dir / "leaderboard_test.csv"
    val_lb.to_csv(val_csv, index=False, encoding="utf-8-sig")
    test_lb.to_csv(test_csv, index=False, encoding="utf-8-sig")

    best_model = str(predictor.leaderboard(display=False).iloc[0]["model"])
    best_test_row = test_lb.loc[test_lb["model"] == best_model]
    if best_test_row.empty:
        raise ValueError(f"Best model '{best_model}' missing from test leaderboard for {horizon_hours}h.")
    best_val_row = val_lb.loc[val_lb["model"] == best_model]
    if best_val_row.empty:
        raise ValueError(f"Best model '{best_model}' missing from validation leaderboard for {horizon_hours}h.")

    return HorizonResult(
        horizon_hours=horizon_hours,
        prediction_length=long_pred_len,
        train_windows=len(train_specs),
        val_windows=len(val_specs),
        test_windows=len(test_specs),
        best_model=best_model,
        val_mase=float(best_val_row.iloc[0]["validation_MASE"]),
        metrics={metric: float(best_test_row.iloc[0][metric]) for metric in EXTRA_METRICS},
        val_leaderboard_csv=str(val_csv.resolve()),
        test_leaderboard_csv=str(test_csv.resolve()),
        model_dir=str(model_dir.resolve()),
        elapsed_seconds=elapsed,
    )


def main() -> None:
    args = build_parser().parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.log_path.parent.mkdir(parents=True, exist_ok=True)

    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if not math.isclose(total_ratio, 1.0, abs_tol=1e-6):
        raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}.")

    env = collect_env()
    steps_per_h = steps_per_hour(args.freq)
    short_pred_len = args.short_horizon_hours * steps_per_h
    long_horizons = parse_horizon_hours(args.long_horizons)
    long_pred_lengths = {hours: hours * steps_per_h for hours in long_horizons}
    context_steps = args.context_hours * steps_per_h
    stride_steps = args.window_stride_hours * steps_per_h

    print("=" * 72)
    print("TFT + Chronos-2 cascade training")
    print(f"  CUDA         : {env['cuda_available']} | device: {env['device_name']}")
    print(f"  Split        : {args.train_ratio:.0%} / {args.val_ratio:.0%} / {args.test_ratio:.0%}")
    print(f"  Short horizon: {args.short_horizon_hours}h ({short_pred_len} steps)")
    print(f"  Long horizons: {long_horizons}")
    print(f"  Context      : {args.context_hours}h ({context_steps} steps)")
    print("=" * 72)

    shared_start, shared_end = load_shared_window(args.metadata_path, args.no_shared_window)
    station_files = sorted(args.input_dir.glob("*_aligned_10min.csv"))
    if not station_files:
        raise FileNotFoundError(f"No *_aligned_10min.csv files found in {args.input_dir}")
    print(f"\nFound {len(station_files)} station files")

    print("\nLoading stations ...")
    frames, station_summaries, base_known_covariates, past_covariates = build_all_stations(
        files=station_files,
        target=args.target,
        shared_start=shared_start,
        shared_end=shared_end,
        limit=args.limit_stations,
        max_steps=args.max_steps,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )

    split_csv = args.output_root / "station_split_summary.csv"
    pd.DataFrame([asdict(summary) for summary in station_summaries]).to_csv(
        split_csv, index=False, encoding="utf-8-sig"
    )

    train_ts = to_tsdf(make_base_split(frames, station_summaries, "train"))
    val_ts = to_tsdf(make_base_split(frames, station_summaries, "val"))
    test_ts = to_tsdf(make_base_split(frames, station_summaries, "test"))

    if args.pretrained_tft_path is not None:
        print("\n[Stage 1] Loading pretrained TFT predictor ...")
        tft_predictor, short_result = load_pretrained_short_tft(
            val_ts=val_ts,
            test_ts=test_ts,
            args=args,
            prediction_length=short_pred_len,
        )
    else:
        print("\n[Stage 1] Training short-horizon TFT ...")
        tft_predictor, short_result = train_short_tft(
            train_ts=train_ts,
            val_ts=val_ts,
            test_ts=test_ts,
            args=args,
            base_known_covariates=base_known_covariates,
            prediction_length=short_pred_len,
        )
    print(
        f"  TFT best model : {short_result.best_model}\n"
        f"  Val MASE       : {short_result.val_mase:.6f}\n"
        f"  Test MASE      : {short_result.metrics['MASE']:.6f}\n"
        f"  Test RMSE      : {short_result.metrics['RMSE']:.6f}\n"
        f"  Elapsed        : {short_result.elapsed_seconds / 60:.1f} min"
    )

    horizon_results: list[HorizonResult] = []
    for horizon_hours in long_horizons:
        long_pred_len = long_pred_lengths[horizon_hours]
        print(f"\n[Stage 2] Training Chronos-2 for {horizon_hours}h ({long_pred_len} steps) ...")
        result = train_long_horizon(
            args=args,
            frames=frames,
            summaries=station_summaries,
            target=args.target,
            base_known_covariates=base_known_covariates,
            past_covariates=past_covariates,
            tft_predictor=tft_predictor,
            short_model_name=short_result.best_model,
            horizon_hours=horizon_hours,
            short_pred_len=short_pred_len,
            long_pred_len=long_pred_len,
            context_steps=context_steps,
            stride_steps=stride_steps,
        )
        horizon_results.append(result)
        print(
            f"  Best model : {result.best_model}\n"
            f"  Windows    : train {result.train_windows} | val {result.val_windows} | test {result.test_windows}\n"
            f"  Val MASE   : {result.val_mase:.6f}\n"
            f"  Test MASE  : {result.metrics['MASE']:.6f}\n"
            f"  Test RMSE  : {result.metrics['RMSE']:.6f}\n"
            f"  Elapsed    : {result.elapsed_seconds / 60:.1f} min"
        )

    metrics_rows = [
        {
            "horizon_hours": result.horizon_hours,
            "prediction_length": result.prediction_length,
            "train_windows": result.train_windows,
            "val_windows": result.val_windows,
            "test_windows": result.test_windows,
            "best_model": result.best_model,
            "validation_MASE": result.val_mase,
            "MAE": result.metrics["MAE"],
            "MASE": result.metrics["MASE"],
            "RMSE": result.metrics["RMSE"],
            "RMSLE": result.metrics["RMSLE"],
            "SMAPE": result.metrics["SMAPE"],
            "elapsed_seconds": result.elapsed_seconds,
        }
        for result in horizon_results
    ]
    metrics_csv = args.output_root / "cascade_metrics_summary.csv"
    pd.DataFrame(metrics_rows).to_csv(metrics_csv, index=False, encoding="utf-8-sig")

    log_payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config": vars(args)
        | {
            "input_dir": str(args.input_dir.resolve()),
            "output_root": str(args.output_root.resolve()),
            "base_known_covariates": base_known_covariates,
            "past_covariates": past_covariates,
            "guidance_covariates": GUIDANCE_COLUMNS,
        },
        "environment": env,
        "stations": [asdict(summary) for summary in station_summaries],
        "short_tft": asdict(short_result),
        "long_horizons": [asdict(result) for result in horizon_results],
    }
    args.log_path.write_text(
        json.dumps(log_payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    print("\nDone.")
    print(f"  Short TFT log   : {short_result.test_leaderboard_csv}")
    print(f"  Cascade metrics : {metrics_csv.resolve()}")
    print(f"  Train log       : {args.log_path.resolve()}")


if __name__ == "__main__":
    main()

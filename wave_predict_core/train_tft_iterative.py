"""
train_tft_iterative.py

Short-step TFT training / loading plus long-horizon iterative rollout evaluation
for wave-height forecasting.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import sys
import time
import warnings
from dataclasses import asdict, dataclass, field
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
from autogluon.timeseries import TimeSeriesPredictor

from train_tft_chronos2_cascade import (
    EXTRA_METRICS,
    ShortModelResult,
    StationSummary,
    WindowSpec,
    build_all_stations,
    build_window_specs,
    collect_env,
    load_shared_window,
    make_base_split,
    parse_horizon_hours,
    steps_per_hour,
    to_tsdf,
    train_short_tft,
)


DEFAULT_LONG_HORIZONS = [12, 24, 48, 72, 120]
ROLL_PAST_COVARIATE_STRATEGIES = ["last", "actual", "zero"]


@dataclass
class MetricAccumulator:
    count: int = 0
    sum_abs: float = 0.0
    sum_sq: float = 0.0
    sum_log_sq: float = 0.0
    sum_smape: float = 0.0
    sum_scaled_abs: float = 0.0

    def update(self, y_true: np.ndarray, y_pred: np.ndarray, scale: float) -> None:
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)
        abs_err = np.abs(y_pred - y_true)
        safe_scale = max(float(scale), 1e-8)
        self.count += int(len(y_true))
        self.sum_abs += float(abs_err.sum())
        self.sum_sq += float(np.square(y_pred - y_true).sum())
        self.sum_log_sq += float(
            np.square(np.log1p(np.clip(y_pred, 0.0, None)) - np.log1p(np.clip(y_true, 0.0, None))).sum()
        )
        self.sum_smape += float((2.0 * abs_err / (np.abs(y_true) + np.abs(y_pred) + 1e-8)).sum())
        self.sum_scaled_abs += float((abs_err / safe_scale).sum())

    def to_metrics(self) -> dict[str, float]:
        if self.count <= 0:
            return {metric: float("nan") for metric in EXTRA_METRICS}
        return {
            "MAE": self.sum_abs / self.count,
            "MASE": self.sum_scaled_abs / self.count,
            "RMSE": math.sqrt(self.sum_sq / self.count),
            "RMSLE": math.sqrt(self.sum_log_sq / self.count),
            "SMAPE": self.sum_smape / self.count,
        }


@dataclass
class RolloutWindowState:
    spec: WindowSpec
    source_df: pd.DataFrame
    history_df: pd.DataFrame
    generated_steps: int = 0
    metrics: MetricAccumulator = field(default_factory=MetricAccumulator)
    sample_rows: list[dict[str, Any]] | None = None


@dataclass
class HorizonRolloutResult:
    horizon_hours: int
    prediction_length: int
    val_windows: int
    test_windows: int
    val_metrics: dict[str, float]
    test_metrics: dict[str, float]
    val_window_metrics_csv: str
    test_window_metrics_csv: str
    val_sample_forecasts_csv: str | None
    test_sample_forecasts_csv: str | None
    elapsed_seconds: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train or load a short-horizon TFT and evaluate iterative long-horizon rollout."
    )
    parser.add_argument("--input-dir", type=Path, default=Path("processed_csv/aligned_stations"))
    parser.add_argument("--metadata-path", type=Path, default=Path("processed_csv/shared_timeline_metadata.json"))
    parser.add_argument("--output-root", type=Path, default=Path("iterative_tft_runs"))
    parser.add_argument("--log-path", type=Path, default=Path("iterative_tft_train_log.json"))
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
    parser.add_argument(
        "--short-horizon-hours",
        type=int,
        default=6,
        help="Chunk size used by the short-horizon TFT during iterative rollout.",
    )
    parser.add_argument(
        "--long-horizons",
        default=",".join(str(h) for h in DEFAULT_LONG_HORIZONS),
        help="Comma-separated long-horizon hours, e.g. 12,24,48,72,120",
    )
    parser.add_argument("--context-hours", type=int, default=168, help="History context used for each rollout.")
    parser.add_argument(
        "--window-stride-hours",
        type=int,
        default=6,
        help="Stride between forecast origins when building validation and test windows.",
    )
    parser.add_argument("--val-max-windows-per-station", type=int, default=32)
    parser.add_argument("--test-max-windows-per-station", type=int, default=32)
    parser.add_argument("--rollout-batch-size", type=int, default=32)
    parser.add_argument(
        "--rollout-past-covariates",
        default="last",
        choices=ROLL_PAST_COVARIATE_STRATEGIES,
        help="How to fill past covariates for generated steps during iterative rollout.",
    )
    parser.add_argument(
        "--seasonal-period",
        type=int,
        default=0,
        help="MASE seasonal period. Use 0 to infer from frequency.",
    )
    parser.add_argument(
        "--sample-windows-per-split",
        type=int,
        default=5,
        help="How many validation/test windows per horizon to save full forecast traces for.",
    )
    parser.add_argument("--tft-time-limit", type=int, default=3600)
    parser.add_argument(
        "--allow-no-past-covariates-fallback",
        action="store_true",
        default=False,
        help="Allow disabling past covariates as final fallback if robust TFT attempts all fail.",
    )
    parser.add_argument(
        "--pretrained-tft-path",
        type=Path,
        default=None,
        help="Existing AutoGluon predictor directory for the short-horizon TFT stage.",
    )
    parser.add_argument(
        "--pretrained-tft-model-name",
        default="TemporalFusionTransformer",
        help="Model name inside the loaded predictor to use for rollout.",
    )
    return parser


def enable_cross_platform_pickle_paths() -> None:
    if os.name == "nt":
        pathlib.PosixPath = pathlib.WindowsPath  # type: ignore[assignment]


def load_pretrained_short_tft_for_rollout(
    predictor_path: Path,
    model_name: str,
    prediction_length: int,
    horizon_hours: int,
) -> tuple[TimeSeriesPredictor, ShortModelResult]:
    enable_cross_platform_pickle_paths()
    try:
        predictor = TimeSeriesPredictor.load(str(predictor_path), reset_paths=True)
    except TypeError:
        # AutoGluon 1.5.x does not support `reset_paths` in some builds.
        predictor = TimeSeriesPredictor.load(str(predictor_path))
    if predictor.prediction_length != prediction_length:
        raise ValueError(
            f"Loaded TFT predictor prediction_length={predictor.prediction_length}, "
            f"but current short horizon requires {prediction_length}."
        )

    available_models = predictor.model_names()
    if model_name not in available_models:
        raise ValueError(
            f"Model '{model_name}' not found in loaded predictor. Available models: {available_models}"
        )

    # Support both:
    # 1) .../short_tft_006h/attempt_xx/model (leaderboards in .../short_tft_006h/)
    # 2) .../short_tft_006h/model (leaderboards in same dir)
    candidate_dirs = [
        predictor_path.parent,
        predictor_path.parent.parent,
        predictor_path,
    ]
    val_csv = None
    test_csv = None
    for directory in candidate_dirs:
        candidate_val = directory / "leaderboard_validation.csv"
        candidate_test = directory / "leaderboard_test.csv"
        if candidate_val.exists() and candidate_test.exists():
            val_csv = candidate_val
            test_csv = candidate_test
            break

    val_mase = float("nan")
    metrics = {metric: float("nan") for metric in EXTRA_METRICS}

    if val_csv is not None and val_csv.exists():
        val_df = pd.read_csv(val_csv)
        if "model" in val_df.columns:
            row = val_df.loc[val_df["model"] == model_name]
            if not row.empty and "validation_MASE" in row.columns:
                val_mase = float(row.iloc[0]["validation_MASE"])

    if test_csv is not None and test_csv.exists():
        test_df = pd.read_csv(test_csv)
        if "model" in test_df.columns:
            row = test_df.loc[test_df["model"] == model_name]
            if not row.empty:
                for metric in EXTRA_METRICS:
                    if metric in row.columns:
                        metrics[metric] = float(row.iloc[0][metric])

    result = ShortModelResult(
        horizon_hours=horizon_hours,
        prediction_length=prediction_length,
        best_model=model_name,
        val_mase=val_mase,
        metrics=metrics,
        val_leaderboard_csv=str(val_csv.resolve()) if (val_csv is not None and val_csv.exists()) else "",
        test_leaderboard_csv=str(test_csv.resolve()) if (test_csv is not None and test_csv.exists()) else "",
        model_dir=str(predictor_path.resolve()),
        elapsed_seconds=0.0,
    )
    return predictor, result


def infer_seasonal_period(freq: str, override: int) -> int:
    if override > 0:
        return override
    delta = pd.Timedelta(freq)
    daily = pd.Timedelta(days=1)
    weekly = pd.Timedelta(days=7)
    if daily % delta == pd.Timedelta(0):
        return max(1, int(daily / delta))
    if weekly % delta == pd.Timedelta(0):
        return max(1, int(weekly / delta))
    return 1


def compute_series_scale(values: np.ndarray, seasonal_period: int) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) <= 1:
        return 1.0
    if seasonal_period > 0 and len(arr) > seasonal_period:
        diffs = np.abs(arr[seasonal_period:] - arr[:-seasonal_period])
    else:
        diffs = np.abs(np.diff(arr))
    if len(diffs) == 0:
        return 1.0
    scale = float(np.mean(diffs))
    return max(scale, 1e-6)


def build_scale_map(
    frames: dict[str, pd.DataFrame],
    summaries: list[StationSummary],
    target: str,
    seasonal_period: int,
) -> dict[str, float]:
    summary_map = {summary.item_id: summary for summary in summaries}
    scale_map: dict[str, float] = {}
    for item_id, df in frames.items():
        train_end = summary_map[item_id].train_rows
        scale_map[item_id] = compute_series_scale(df[target].iloc[:train_end].to_numpy(dtype=np.float64), seasonal_period)
    return scale_map


def chunked(values: list[RolloutWindowState], size: int) -> Iterable[list[RolloutWindowState]]:
    chunk_size = max(1, size)
    for start in range(0, len(values), chunk_size):
        yield values[start : start + chunk_size]


def specs_to_frame(specs: list[WindowSpec]) -> pd.DataFrame:
    return pd.DataFrame([asdict(spec) for spec in specs])


def update_history_frame(
    state: RolloutWindowState,
    appended_df: pd.DataFrame,
    past_covariates: list[str],
    target: str,
    strategy: str,
) -> None:
    if strategy == "last" and past_covariates:
        last_row = state.history_df.iloc[-1]
        for column in past_covariates:
            appended_df[column] = np.float32(last_row[column])
    elif strategy == "zero" and past_covariates:
        for column in past_covariates:
            appended_df[column] = np.float32(0.0)
    elif strategy == "actual":
        pass
    else:
        if strategy not in ROLL_PAST_COVARIATE_STRATEGIES:
            raise ValueError(f"Unknown rollout past covariate strategy: {strategy}")

    state.history_df = pd.concat([state.history_df, appended_df[state.history_df.columns]], ignore_index=True)


def rollout_states_once(
    states: list[RolloutWindowState],
    predictor: TimeSeriesPredictor,
    model_name: str,
    target: str,
    base_known_covariates: list[str],
    past_covariates: list[str],
    short_pred_len: int,
    long_pred_len: int,
    scale_map: dict[str, float],
    strategy: str,
) -> None:
    if not states:
        return

    past_frames: list[pd.DataFrame] = []
    future_frames: list[pd.DataFrame] = []
    chunk_lengths: dict[str, int] = {}
    actual_chunks: dict[str, pd.DataFrame] = {}

    for state in states:
        spec = state.spec
        chunk_len = min(short_pred_len, long_pred_len - state.generated_steps)
        if chunk_len <= 0:
            continue
        start = spec.origin_index + state.generated_steps + 1
        stop = start + chunk_len
        future_slice = state.source_df.iloc[start:stop].copy().reset_index(drop=True)
        future_slice["item_id"] = spec.item_id
        actual_chunks[spec.item_id] = future_slice
        chunk_lengths[spec.item_id] = chunk_len
        past_frames.append(state.history_df)
        if base_known_covariates:
            future_frames.append(future_slice[["item_id", "timestamp", *base_known_covariates]])

    if not past_frames:
        return

    past_ts = to_tsdf(pd.concat(past_frames, ignore_index=True))
    if base_known_covariates:
        future_known_ts = to_tsdf(pd.concat(future_frames, ignore_index=True))
        preds = predictor.predict(past_ts, known_covariates=future_known_ts, model=model_name)
    else:
        preds = predictor.predict(past_ts, model=model_name)

    pred_map: dict[str, np.ndarray] = {}
    pred_df = preds.reset_index()
    for item_id, group in pred_df.groupby("item_id"):
        sorted_group = group.sort_values("timestamp")
        pred_map[str(item_id)] = sorted_group["mean"].to_numpy(dtype=np.float64)

    for state in states:
        spec = state.spec
        if spec.item_id not in chunk_lengths:
            continue
        if spec.item_id not in pred_map:
            raise ValueError(f"Missing forecast for rollout window {spec.item_id}.")

        chunk_len = chunk_lengths[spec.item_id]
        pred_values = pred_map[spec.item_id][:chunk_len]
        actual_df = actual_chunks[spec.item_id]
        y_true = actual_df[target].to_numpy(dtype=np.float64)
        scale = scale_map[spec.source_item_id]

        state.metrics.update(y_true, pred_values, scale)

        if state.sample_rows is not None:
            for offset, (timestamp, actual, pred) in enumerate(
                zip(actual_df["timestamp"], y_true, pred_values),
                start=state.generated_steps + 1,
            ):
                state.sample_rows.append(
                    {
                        "window_item_id": spec.item_id,
                        "source_item_id": spec.source_item_id,
                        "split_name": spec.split_name,
                        "horizon_hours": spec.horizon_hours,
                        "forecast_step": offset,
                        "timestamp": str(timestamp),
                        "y_true": float(actual),
                        "y_pred": float(pred),
                        "abs_error": float(abs(pred - actual)),
                    }
                )

        appended_df = actual_df.copy()
        appended_df[target] = pred_values.astype(np.float32)
        update_history_frame(
            state=state,
            appended_df=appended_df,
            past_covariates=past_covariates,
            target=target,
            strategy=strategy,
        )
        state.generated_steps += chunk_len


def evaluate_rollout_split(
    specs: list[WindowSpec],
    frames: dict[str, pd.DataFrame],
    predictor: TimeSeriesPredictor,
    model_name: str,
    target: str,
    base_known_covariates: list[str],
    past_covariates: list[str],
    short_pred_len: int,
    long_pred_len: int,
    context_steps: int,
    scale_map: dict[str, float],
    rollout_batch_size: int,
    rollout_past_covariates: str,
    sample_windows_per_split: int,
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame | None]:
    states: list[RolloutWindowState] = []
    for idx, spec in enumerate(specs):
        source_df = frames[spec.source_item_id]
        context_start = spec.origin_index - context_steps + 1
        history_df = source_df.iloc[context_start : spec.origin_index + 1].copy().reset_index(drop=True)
        history_df["item_id"] = spec.item_id
        states.append(
            RolloutWindowState(
                spec=spec,
                source_df=source_df,
                history_df=history_df,
                sample_rows=[] if idx < sample_windows_per_split else None,
            )
        )

    rollout_steps = math.ceil(long_pred_len / short_pred_len)
    for rollout_idx in range(rollout_steps):
        active_states = [state for state in states if state.generated_steps < long_pred_len]
        if not active_states:
            break
        for batch in chunked(active_states, rollout_batch_size):
            rollout_states_once(
                states=batch,
                predictor=predictor,
                model_name=model_name,
                target=target,
                base_known_covariates=base_known_covariates,
                past_covariates=past_covariates,
                short_pred_len=short_pred_len,
                long_pred_len=long_pred_len,
                scale_map=scale_map,
                strategy=rollout_past_covariates,
            )

    window_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    total_metrics = MetricAccumulator()
    for state in states:
        metrics = state.metrics.to_metrics()
        total_metrics.count += state.metrics.count
        total_metrics.sum_abs += state.metrics.sum_abs
        total_metrics.sum_sq += state.metrics.sum_sq
        total_metrics.sum_log_sq += state.metrics.sum_log_sq
        total_metrics.sum_smape += state.metrics.sum_smape
        total_metrics.sum_scaled_abs += state.metrics.sum_scaled_abs
        window_rows.append(
            {
                "window_item_id": state.spec.item_id,
                "source_item_id": state.spec.source_item_id,
                "split_name": state.spec.split_name,
                "horizon_hours": state.spec.horizon_hours,
                "origin_index": state.spec.origin_index,
                "origin_timestamp": state.spec.origin_timestamp,
                "forecast_points": state.metrics.count,
                **metrics,
            }
        )
        if state.sample_rows:
            sample_rows.extend(state.sample_rows)

    sample_df = pd.DataFrame(sample_rows) if sample_rows else None
    return total_metrics.to_metrics(), pd.DataFrame(window_rows), sample_df


def evaluate_horizon(
    args: argparse.Namespace,
    frames: dict[str, pd.DataFrame],
    summaries: list[StationSummary],
    target: str,
    base_known_covariates: list[str],
    past_covariates: list[str],
    predictor: TimeSeriesPredictor,
    model_name: str,
    short_pred_len: int,
    horizon_hours: int,
    long_pred_len: int,
    context_steps: int,
    stride_steps: int,
    scale_map: dict[str, float],
) -> HorizonRolloutResult:
    run_dir = args.output_root / f"horizon_{horizon_hours:03d}h"
    run_dir.mkdir(parents=True, exist_ok=True)

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

    if not val_specs:
        raise ValueError(f"No validation windows available for {horizon_hours}h.")
    if not test_specs:
        raise ValueError(f"No test windows available for {horizon_hours}h.")

    specs_to_frame(val_specs).to_csv(run_dir / "window_index_validation.csv", index=False, encoding="utf-8-sig")
    specs_to_frame(test_specs).to_csv(run_dir / "window_index_test.csv", index=False, encoding="utf-8-sig")

    t0 = time.perf_counter()
    val_metrics, val_window_df, val_sample_df = evaluate_rollout_split(
        specs=val_specs,
        frames=frames,
        predictor=predictor,
        model_name=model_name,
        target=target,
        base_known_covariates=base_known_covariates,
        past_covariates=past_covariates,
        short_pred_len=short_pred_len,
        long_pred_len=long_pred_len,
        context_steps=context_steps,
        scale_map=scale_map,
        rollout_batch_size=args.rollout_batch_size,
        rollout_past_covariates=args.rollout_past_covariates,
        sample_windows_per_split=args.sample_windows_per_split,
    )
    test_metrics, test_window_df, test_sample_df = evaluate_rollout_split(
        specs=test_specs,
        frames=frames,
        predictor=predictor,
        model_name=model_name,
        target=target,
        base_known_covariates=base_known_covariates,
        past_covariates=past_covariates,
        short_pred_len=short_pred_len,
        long_pred_len=long_pred_len,
        context_steps=context_steps,
        scale_map=scale_map,
        rollout_batch_size=args.rollout_batch_size,
        rollout_past_covariates=args.rollout_past_covariates,
        sample_windows_per_split=args.sample_windows_per_split,
    )
    elapsed = time.perf_counter() - t0

    val_window_csv = run_dir / "window_metrics_validation.csv"
    test_window_csv = run_dir / "window_metrics_test.csv"
    val_window_df.to_csv(val_window_csv, index=False, encoding="utf-8-sig")
    test_window_df.to_csv(test_window_csv, index=False, encoding="utf-8-sig")

    val_sample_csv: str | None = None
    test_sample_csv: str | None = None
    if val_sample_df is not None and not val_sample_df.empty:
        path = run_dir / "sample_forecasts_validation.csv"
        val_sample_df.to_csv(path, index=False, encoding="utf-8-sig")
        val_sample_csv = str(path.resolve())
    if test_sample_df is not None and not test_sample_df.empty:
        path = run_dir / "sample_forecasts_test.csv"
        test_sample_df.to_csv(path, index=False, encoding="utf-8-sig")
        test_sample_csv = str(path.resolve())

    return HorizonRolloutResult(
        horizon_hours=horizon_hours,
        prediction_length=long_pred_len,
        val_windows=len(val_specs),
        test_windows=len(test_specs),
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        val_window_metrics_csv=str(val_window_csv.resolve()),
        test_window_metrics_csv=str(test_window_csv.resolve()),
        val_sample_forecasts_csv=val_sample_csv,
        test_sample_forecasts_csv=test_sample_csv,
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
    seasonal_period = infer_seasonal_period(args.freq, args.seasonal_period)

    print("=" * 72)
    print("TFT iterative rollout training")
    print(f"  CUDA         : {env['cuda_available']} | device: {env['device_name']}")
    print(f"  Split        : {args.train_ratio:.0%} / {args.val_ratio:.0%} / {args.test_ratio:.0%}")
    print(f"  Short horizon: {args.short_horizon_hours}h ({short_pred_len} steps)")
    print(f"  Long horizons: {long_horizons}")
    print(f"  Context      : {args.context_hours}h ({context_steps} steps)")
    print(f"  Rollout fill : {args.rollout_past_covariates}")
    print(f"  MASE season  : {seasonal_period}")
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

    scale_map = build_scale_map(frames=frames, summaries=station_summaries, target=args.target, seasonal_period=seasonal_period)

    train_ts = to_tsdf(make_base_split(frames, station_summaries, "train"))
    val_ts = to_tsdf(make_base_split(frames, station_summaries, "val"))
    test_ts = to_tsdf(make_base_split(frames, station_summaries, "test"))

    if args.pretrained_tft_path is not None:
        print("\n[Stage 1] Loading pretrained short-horizon TFT ...")
        tft_predictor, short_result = load_pretrained_short_tft_for_rollout(
            predictor_path=args.pretrained_tft_path,
            model_name=args.pretrained_tft_model_name,
            prediction_length=short_pred_len,
            horizon_hours=args.short_horizon_hours,
        )
    else:
        print("\n[Stage 1] Training short-horizon TFT ...")
        args.tft_time_limit = None if args.tft_time_limit <= 0 else args.tft_time_limit
        tft_predictor, short_result = train_short_tft(
            train_ts=train_ts,
            val_ts=val_ts,
            test_ts=test_ts,
            args=args,
            base_known_covariates=base_known_covariates,
            prediction_length=short_pred_len,
        )

    print(
        f"  TFT model      : {short_result.best_model}\n"
        f"  Val MASE       : {short_result.val_mase:.6f}\n"
        f"  Test MASE      : {short_result.metrics['MASE']:.6f}\n"
        f"  Test RMSE      : {short_result.metrics['RMSE']:.6f}\n"
        f"  Source path    : {short_result.model_dir}"
    )

    horizon_results: list[HorizonRolloutResult] = []
    for horizon_hours in long_horizons:
        long_pred_len = long_pred_lengths[horizon_hours]
        print(f"\n[Stage 2] Iterative rollout for {horizon_hours}h ({long_pred_len} steps) ...")
        result = evaluate_horizon(
            args=args,
            frames=frames,
            summaries=station_summaries,
            target=args.target,
            base_known_covariates=base_known_covariates,
            past_covariates=past_covariates,
            predictor=tft_predictor,
            model_name=short_result.best_model,
            short_pred_len=short_pred_len,
            horizon_hours=horizon_hours,
            long_pred_len=long_pred_len,
            context_steps=context_steps,
            stride_steps=stride_steps,
            scale_map=scale_map,
        )
        horizon_results.append(result)
        print(
            f"  Val windows : {result.val_windows}\n"
            f"  Test windows: {result.test_windows}\n"
            f"  Val MASE    : {result.val_metrics['MASE']:.6f}\n"
            f"  Test MASE   : {result.test_metrics['MASE']:.6f}\n"
            f"  Test RMSE   : {result.test_metrics['RMSE']:.6f}\n"
            f"  Elapsed     : {result.elapsed_seconds / 60:.1f} min"
        )

    metrics_rows = [
        {
            "horizon_hours": result.horizon_hours,
            "prediction_length": result.prediction_length,
            "short_horizon_hours": args.short_horizon_hours,
            "short_model": short_result.best_model,
            "val_windows": result.val_windows,
            "test_windows": result.test_windows,
            "val_MAE": result.val_metrics["MAE"],
            "val_MASE": result.val_metrics["MASE"],
            "val_RMSE": result.val_metrics["RMSE"],
            "val_RMSLE": result.val_metrics["RMSLE"],
            "val_SMAPE": result.val_metrics["SMAPE"],
            "test_MAE": result.test_metrics["MAE"],
            "test_MASE": result.test_metrics["MASE"],
            "test_RMSE": result.test_metrics["RMSE"],
            "test_RMSLE": result.test_metrics["RMSLE"],
            "test_SMAPE": result.test_metrics["SMAPE"],
            "elapsed_seconds": result.elapsed_seconds,
        }
        for result in horizon_results
    ]
    metrics_csv = args.output_root / "iterative_metrics_summary.csv"
    pd.DataFrame(metrics_rows).to_csv(metrics_csv, index=False, encoding="utf-8-sig")

    log_payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config": vars(args)
        | {
            "input_dir": str(args.input_dir.resolve()),
            "output_root": str(args.output_root.resolve()),
            "base_known_covariates": base_known_covariates,
            "past_covariates": past_covariates,
            "seasonal_period": seasonal_period,
        },
        "environment": env,
        "stations": [asdict(summary) for summary in station_summaries],
        "short_tft": asdict(short_result),
        "rollout_horizons": [asdict(result) for result in horizon_results],
        "scale_map": scale_map,
    }
    args.log_path.write_text(
        json.dumps(log_payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    print("\nDone.")
    print(f"  Short TFT log   : {short_result.test_leaderboard_csv}")
    print(f"  Iterative table : {metrics_csv.resolve()}")
    print(f"  Train log       : {args.log_path.resolve()}")


if __name__ == "__main__":
    main()

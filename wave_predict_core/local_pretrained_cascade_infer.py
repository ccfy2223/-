"""
local_pretrained_cascade_infer.py
=================================

Batch local inference using pretrained AutoGluon TimeSeries predictors.

Outputs:
- shared history input
- shared 6h TFT forecast
- per-horizon Chronos forecast
- per-horizon stitched cascade forecast

The cascade rule is:
- first short horizon from TFT
- remaining steps from Chronos

This script performs inference only and does not retrain models.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
import warnings
from datetime import datetime
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
warnings.filterwarnings("ignore", message="Could not find the number of physical cores")
warnings.filterwarnings("ignore", message="`torch_dtype` is deprecated! Use `dtype` instead!")

try:
    # Some Windows Python environments do not expose powershell.exe / wmic.
    # Pre-filling the cache avoids loky emitting a noisy pseudo-traceback.
    from joblib.externals.loky.backend import context as loky_context

    loky_context.physical_cores_cache = os.cpu_count() or 1
except Exception:
    pass

import numpy as np
import pandas as pd
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

DEFAULT_LONG_HORIZONS = [24, 48, 72, 120]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch local inference using pretrained TFT + Chronos AutoGluon predictors."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("processed_csv/aligned_stations"),
        help="Folder containing aligned station CSV files.",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=Path("processed_csv/shared_timeline_metadata.json"),
        help="Shared timeline metadata JSON.",
    )
    parser.add_argument(
        "--predictor-roots",
        default="autogluon_runs,export_models/autogluon_runs",
        help="Comma-separated predictor roots searched for horizon_XXXh/model.",
    )
    parser.add_argument(
        "--tft-path",
        type=Path,
        default=None,
        help="Optional explicit path to pretrained 6h TFT AutoGluon predictor directory.",
    )
    parser.add_argument(
        "--chronos-path",
        type=Path,
        default=None,
        help="Optional explicit path to a single long-horizon Chronos predictor directory.",
    )
    parser.add_argument(
        "--horizons",
        default=None,
        help="Comma-separated long horizons in hours, e.g. 24,48,72,120. "
        "If omitted and --chronos-path is set, the horizon is inferred from the path.",
    )
    parser.add_argument("--tft-model-name", default="TemporalFusionTransformer")
    parser.add_argument("--chronos-model-name", default="Chronos[amazon__chronos-t5-large]")
    parser.add_argument("--target", default="WVHT")
    parser.add_argument("--freq", default="10min")
    parser.add_argument("--output-dir", type=Path, default=Path("local_cascade_batch_output"))
    parser.add_argument("--limit-stations", type=int, default=0)
    parser.add_argument("--max-history-steps", type=int, default=0, help="0 means use all available history.")
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Recompute a horizon even if its output CSV files already exist.",
    )
    parser.add_argument(
        "--progress-interval-seconds",
        type=int,
        default=30,
        help="Heartbeat interval in seconds for long-running prediction stages. Use 0 to disable.",
    )
    parser.add_argument(
        "--strict-horizons",
        action="store_true",
        help="Fail instead of skipping when a requested horizon model is missing.",
    )
    parser.add_argument(
        "--use-shared-window",
        dest="use_shared_window",
        action="store_true",
        help="Apply shared timeline window from metadata.",
    )
    parser.add_argument(
        "--no-shared-window",
        dest="use_shared_window",
        action="store_false",
        help="Disable shared timeline window from metadata.",
    )
    parser.set_defaults(use_shared_window=True)
    return parser


def patch_pathlib_for_windows_pickles() -> None:
    """Allow Windows to unpickle AutoGluon predictors serialized on Linux."""
    if os.name != "nt":
        return

    import pathlib

    if pathlib.PosixPath is not pathlib.WindowsPath:
        pathlib.PosixPath = pathlib.WindowsPath  # type: ignore[assignment]


def parse_path_list(raw: str) -> list[Path]:
    return [Path(chunk.strip()) for chunk in raw.split(",") if chunk.strip()]


def parse_horizon_list(raw: str) -> list[int]:
    values = sorted({int(chunk.strip()) for chunk in raw.split(",") if chunk.strip()})
    if not values:
        raise ValueError("No valid horizons were provided.")
    return values


def infer_horizon_from_path(path: Path) -> int:
    match = re.search(r"horizon_(\d+)h", str(path))
    if not match:
        raise ValueError(f"Unable to infer horizon from path: {path}")
    return int(match.group(1))


def resolve_requested_horizons(args: argparse.Namespace) -> list[int]:
    if args.horizons:
        return parse_horizon_list(args.horizons)
    if args.chronos_path is not None:
        return [infer_horizon_from_path(args.chronos_path)]
    return DEFAULT_LONG_HORIZONS.copy()


def resolve_predictor_path(horizon_hours: int, explicit_path: Path | None, roots: list[Path]) -> Path | None:
    if explicit_path is not None:
        return explicit_path.resolve()

    for root in roots:
        direct_model_dir = root / f"horizon_{horizon_hours:03d}h" / "model"
        if direct_model_dir.exists():
            return direct_model_dir.resolve()

        if root.name == f"horizon_{horizon_hours:03d}h" and (root / "model").exists():
            return (root / "model").resolve()

    return None


def resolve_model_name(predictor: TimeSeriesPredictor, preferred: str, keyword: str) -> str:
    model_names = list(predictor.model_names())
    if preferred in model_names:
        return preferred

    matches = [name for name in model_names if keyword.lower() in name.lower()]
    matches = sorted(matches, key=lambda name: (0 if name.startswith(keyword) else 1, len(name), name))
    if matches:
        return matches[0]

    raise ValueError(
        f"Model '{preferred}' not found and no model containing '{keyword}' exists. Available: {model_names}"
    )


def load_shared_window(meta_path: Path, enabled: bool) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    if not enabled or not meta_path.exists():
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
    max_history_steps: int,
) -> pd.DataFrame:
    available_columns = pd.read_csv(path, nrows=0).columns.tolist()
    dynamic = [c for c in DYNAMIC_COLUMNS if c in available_columns]
    known = [c for c in BASE_KNOWN_COVARIATES if c in available_columns]

    df = pd.read_csv(
        path,
        usecols=["datetime"] + dynamic + known,
        parse_dates=["datetime"],
        low_memory=False,
    )
    df = df.sort_values("datetime").drop_duplicates("datetime", keep="last").reset_index(drop=True)

    if shared_start is not None and shared_end is not None:
        df = df[(df["datetime"] >= shared_start) & (df["datetime"] <= shared_end)].reset_index(drop=True)

    if max_history_steps > 0 and len(df) > max_history_steps:
        df = df.iloc[-max_history_steps:].reset_index(drop=True)

    numeric_columns = [c for c in df.columns if c != "datetime"]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce").astype("float32")
    df[numeric_columns] = df[numeric_columns].ffill().bfill()

    if target not in df.columns:
        raise KeyError(f"Target '{target}' not found in {path.name}.")

    item_id = path.stem.split("_")[0]
    df.insert(0, "item_id", item_id)
    df = df.rename(columns={"datetime": "timestamp"})
    return df


def to_tsdf(df: pd.DataFrame) -> TimeSeriesDataFrame:
    return TimeSeriesDataFrame.from_data_frame(df=df, id_column="item_id", timestamp_column="timestamp")


def build_history_dataset(args: argparse.Namespace) -> pd.DataFrame:
    shared_start, shared_end = load_shared_window(args.metadata_path, args.use_shared_window)
    station_files = sorted(args.input_dir.glob("*_aligned_10min.csv"))
    if args.limit_stations > 0:
        station_files = station_files[: args.limit_stations]
    if not station_files:
        raise FileNotFoundError(f"No *_aligned_10min.csv files found in {args.input_dir}")

    frames = [
        load_station(
            path=path,
            target=args.target,
            shared_start=shared_start,
            shared_end=shared_end,
            max_history_steps=args.max_history_steps,
        )
        for path in station_files
    ]
    return pd.concat(frames, ignore_index=True)


def add_future_time_features(future_df: pd.DataFrame) -> pd.DataFrame:
    future_df = future_df.copy()
    ts = pd.to_datetime(future_df["timestamp"])
    hour = ts.dt.hour + ts.dt.minute / 60.0
    day_of_year = ts.dt.dayofyear.astype(float)

    future_df["time_sin_hour"] = np.sin(2 * np.pi * hour / 24.0).astype(np.float32)
    future_df["time_cos_hour"] = np.cos(2 * np.pi * hour / 24.0).astype(np.float32)
    future_df["time_sin_doy"] = np.sin(2 * np.pi * day_of_year / 365.25).astype(np.float32)
    future_df["time_cos_doy"] = np.cos(2 * np.pi * day_of_year / 365.25).astype(np.float32)
    future_df["month"] = ts.dt.month.astype(np.float32)
    future_df["day_of_week"] = ts.dt.dayofweek.astype(np.float32)
    return future_df


def stitch_cascade_forecast(
    tft_forecast: TimeSeriesDataFrame,
    chronos_forecast: TimeSeriesDataFrame,
    short_prediction_length: int,
) -> TimeSeriesDataFrame:
    tft_df = tft_forecast.reset_index()
    chronos_df = chronos_forecast.reset_index()
    stitched_parts: list[pd.DataFrame] = []

    forecast_columns = [c for c in chronos_df.columns if c not in {"item_id", "timestamp"}]
    for item_id, chronos_group in chronos_df.groupby("item_id"):
        tft_group = tft_df[tft_df["item_id"] == item_id].sort_values("timestamp")
        chronos_group = chronos_group.sort_values("timestamp")

        cutoff = min(short_prediction_length, len(tft_group), len(chronos_group))
        first_part = tft_group.iloc[:cutoff].copy()
        second_part = chronos_group.iloc[cutoff:].copy()

        for column in forecast_columns:
            if column not in first_part.columns:
                first_part[column] = np.nan
        first_part = first_part[["item_id", "timestamp", *forecast_columns]]
        second_part = second_part[["item_id", "timestamp", *forecast_columns]]
        stitched_parts.append(pd.concat([first_part, second_part], ignore_index=True))

    stitched = pd.concat(stitched_parts, ignore_index=True)
    return TimeSeriesDataFrame.from_data_frame(stitched, id_column="item_id", timestamp_column="timestamp")


def save_forecast_csv(tsdf: TimeSeriesDataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tsdf.reset_index().to_csv(path, index=False, encoding="utf-8-sig")


def save_plot_history_csv(history_df: pd.DataFrame, target: str, path: Path) -> None:
    columns = ["item_id", "timestamp", target]
    plot_df = history_df[columns].copy()
    path.parent.mkdir(parents=True, exist_ok=True)
    plot_df.to_csv(path, index=False, encoding="utf-8-sig")


def format_duration(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def run_with_heartbeat(label: str, func, interval_seconds: int) -> tuple[object, float]:
    started = time.perf_counter()
    stop_event = threading.Event()

    def _heartbeat_worker() -> None:
        while not stop_event.wait(interval_seconds):
            elapsed = time.perf_counter() - started
            print(f"  [{label}] running... elapsed {format_duration(elapsed)}", flush=True)

    worker: threading.Thread | None = None
    if interval_seconds > 0:
        worker = threading.Thread(target=_heartbeat_worker, name=f"heartbeat_{label}", daemon=True)
        worker.start()

    try:
        result = func()
    except Exception:
        elapsed = time.perf_counter() - started
        print(f"  [{label}] failed after {format_duration(elapsed)}", flush=True)
        raise
    else:
        elapsed = time.perf_counter() - started
        print(f"  [{label}] finished in {format_duration(elapsed)}", flush=True)
        return result, elapsed
    finally:
        stop_event.set()
        if worker is not None:
            worker.join(timeout=0.2)


def build_horizon_paths(output_dir: Path, horizon_hours: int) -> dict[str, Path]:
    horizon_dir = output_dir / f"horizon_{horizon_hours:03d}h"
    return {
        "dir": horizon_dir,
        "chronos_csv": horizon_dir / f"chronos_{horizon_hours:03d}h_forecast.csv",
        "cascade_csv": horizon_dir / f"cascade_{horizon_hours:03d}h_forecast.csv",
        "meta_json": horizon_dir / "horizon_meta.json",
    }


def horizon_outputs_complete(output_dir: Path, horizon_hours: int) -> bool:
    paths = build_horizon_paths(output_dir, horizon_hours)
    return paths["chronos_csv"].exists() and paths["cascade_csv"].exists()


def count_csv_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def load_existing_horizon_summary(output_dir: Path, horizon_hours: int) -> dict[str, object]:
    paths = build_horizon_paths(output_dir, horizon_hours)
    meta: dict[str, object] = {}
    if paths["meta_json"].exists():
        meta = json.loads(paths["meta_json"].read_text(encoding="utf-8"))

    return {
        "horizon_hours": horizon_hours,
        "status": "skipped_existing",
        "chronos_predictor_path": meta.get("chronos_predictor_path"),
        "chronos_model_name": meta.get("chronos_model_name"),
        "prediction_length": meta.get("chronos_prediction_length"),
        "chronos_rows": count_csv_rows(paths["chronos_csv"]),
        "cascade_rows": count_csv_rows(paths["cascade_csv"]),
        "elapsed_seconds": float(meta.get("elapsed_seconds", 0.0) or 0.0),
    }


def sync_single_horizon_legacy_files(output_dir: Path, horizon_hours: int) -> None:
    paths = build_horizon_paths(output_dir, horizon_hours)
    legacy_chronos_csv = output_dir / f"chronos_{horizon_hours}h_forecast.csv"
    legacy_cascade_csv = output_dir / "cascade_stitched_forecast.csv"
    pd.read_csv(paths["chronos_csv"]).to_csv(legacy_chronos_csv, index=False, encoding="utf-8-sig")
    pd.read_csv(paths["cascade_csv"]).to_csv(legacy_cascade_csv, index=False, encoding="utf-8-sig")


def main() -> None:
    args = build_parser().parse_args()
    run_started_at = datetime.now()
    run_started_perf = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    patch_pathlib_for_windows_pickles()

    predictor_roots = parse_path_list(args.predictor_roots)
    horizons = resolve_requested_horizons(args)
    if args.chronos_path is not None and len(horizons) > 1:
        raise ValueError("--chronos-path only supports a single horizon. Use --predictor-roots for batch mode.")

    history_df = build_history_dataset(args)
    history_ts = to_tsdf(history_df)
    history_csv = args.output_dir / "history_input.csv"
    plot_history_csv = args.output_dir / "history_plot_input.csv"
    history_df.to_csv(history_csv, index=False, encoding="utf-8-sig")
    save_plot_history_csv(history_df, target=args.target, path=plot_history_csv)

    station_count = history_df["item_id"].nunique()
    print("Local cascade batch inference", flush=True)
    print(f"  Output dir : {args.output_dir.resolve()}", flush=True)
    print(f"  Stations   : {station_count}", flush=True)
    print(f"  Horizons   : {horizons}", flush=True)
    print(f"  Overwrite  : {args.overwrite_existing}", flush=True)
    print(f"  Heartbeat  : {args.progress_interval_seconds}s", flush=True)

    tft_path = resolve_predictor_path(horizon_hours=6, explicit_path=args.tft_path, roots=predictor_roots)
    if tft_path is None:
        raise FileNotFoundError("Unable to locate 6h TFT predictor directory.")

    print("Loading local TFT predictor ...", flush=True)
    tft_predictor = TimeSeriesPredictor.load(str(tft_path))
    tft_model_name = resolve_model_name(tft_predictor, args.tft_model_name, "TemporalFusionTransformer")
    print(f"  TFT predictor : {tft_path}", flush=True)
    print(f"  TFT model     : {tft_model_name}", flush=True)

    print("Running shared 6h TFT forecast ...", flush=True)
    tft_csv = args.output_dir / "tft_006h_forecast.csv"
    legacy_tft_csv = args.output_dir / "tft_6h_forecast.csv"
    if not args.overwrite_existing and tft_csv.exists():
        print("  Reusing existing TFT forecast CSV", flush=True)
        tft_forecast_df = pd.read_csv(tft_csv, parse_dates=["timestamp"], low_memory=False)
        tft_forecast = TimeSeriesDataFrame.from_data_frame(
            df=tft_forecast_df,
            id_column="item_id",
            timestamp_column="timestamp",
        )
        tft_elapsed_seconds = 0.0
        if not legacy_tft_csv.exists():
            tft_forecast_df.to_csv(legacy_tft_csv, index=False, encoding="utf-8-sig")
    else:
        tft_forecast, tft_elapsed_seconds = run_with_heartbeat(
            "TFT 6h",
            lambda: tft_predictor.predict(
                history_ts,
                known_covariates=add_future_time_features(tft_predictor.make_future_data_frame(history_ts)),
                model=tft_model_name,
                use_cache=False,
            ),
            args.progress_interval_seconds,
        )
        save_forecast_csv(tft_forecast, tft_csv)
        save_forecast_csv(tft_forecast, legacy_tft_csv)

    summary_rows: list[dict[str, object]] = []
    completed_horizons: list[int] = []
    skipped_existing_horizons: list[int] = []
    skipped_horizons: list[dict[str, object]] = []

    for index, horizon_hours in enumerate(horizons, start=1):
        print(f"[{index}/{len(horizons)}] Horizon {horizon_hours}h", flush=True)
        horizon_path = resolve_predictor_path(
            horizon_hours=horizon_hours,
            explicit_path=args.chronos_path,
            roots=predictor_roots,
        )
        if horizon_path is None:
            message = f"Missing predictor for horizon {horizon_hours}h."
            if args.strict_horizons:
                raise FileNotFoundError(message)
            print(f"Skipping {horizon_hours}h: {message}", flush=True)
            summary_rows.append(
                {
                    "horizon_hours": horizon_hours,
                    "status": "missing_model",
                    "chronos_predictor_path": None,
                    "chronos_model_name": None,
                    "prediction_length": None,
                    "chronos_rows": 0,
                    "cascade_rows": 0,
                    "elapsed_seconds": 0.0,
                }
            )
            skipped_horizons.append({"horizon_hours": horizon_hours, "reason": message})
            continue

        if not args.overwrite_existing and horizon_outputs_complete(args.output_dir, horizon_hours):
            existing_summary = load_existing_horizon_summary(args.output_dir, horizon_hours)
            summary_rows.append(existing_summary)
            skipped_existing_horizons.append(horizon_hours)
            existing_elapsed = float(existing_summary.get("elapsed_seconds", 0.0) or 0.0)
            elapsed_text = format_duration(existing_elapsed) if existing_elapsed > 0 else "unknown"
            print(f"  Skipping {horizon_hours}h: existing output detected ({elapsed_text})", flush=True)
            continue

        horizon_started_perf = time.perf_counter()
        print(f"Running Chronos forecast for {horizon_hours}h ...", flush=True)
        horizon_result, _prepare_elapsed = run_with_heartbeat(
            f"{horizon_hours}h prepare",
            lambda: (
                (lambda predictor: (
                    predictor,
                    resolve_model_name(predictor, args.chronos_model_name, "Chronos"),
                    add_future_time_features(predictor.make_future_data_frame(history_ts)),
                ))(TimeSeriesPredictor.load(str(horizon_path)))
            ),
            args.progress_interval_seconds,
        )
        chronos_predictor, chronos_model_name, chronos_future_covariates = horizon_result
        chronos_forecast, chronos_predict_elapsed = run_with_heartbeat(
            f"{horizon_hours}h predict",
            lambda: chronos_predictor.predict(
                history_ts,
                known_covariates=chronos_future_covariates,
                model=chronos_model_name,
                use_cache=False,
            ),
            args.progress_interval_seconds,
        )
        cascade_forecast = stitch_cascade_forecast(
            tft_forecast=tft_forecast,
            chronos_forecast=chronos_forecast,
            short_prediction_length=tft_predictor.prediction_length,
        )

        horizon_paths = build_horizon_paths(args.output_dir, horizon_hours)
        horizon_dir = horizon_paths["dir"]
        horizon_dir.mkdir(parents=True, exist_ok=True)

        chronos_csv = horizon_paths["chronos_csv"]
        cascade_csv = horizon_paths["cascade_csv"]
        save_forecast_csv(chronos_forecast, chronos_csv)
        save_forecast_csv(cascade_forecast, cascade_csv)
        horizon_elapsed_seconds = time.perf_counter() - horizon_started_perf

        horizon_meta = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "status": "ok",
            "horizon_hours": horizon_hours,
            "target": args.target,
            "freq": args.freq,
            "num_items": int(history_ts.num_items),
            "tft_predictor_path": str(tft_path.resolve()),
            "tft_model_name": tft_model_name,
            "tft_prediction_length": int(tft_predictor.prediction_length),
            "tft_elapsed_seconds": tft_elapsed_seconds,
            "chronos_predictor_path": str(horizon_path.resolve()),
            "chronos_model_name": chronos_model_name,
            "chronos_prediction_length": int(chronos_predictor.prediction_length),
            "chronos_predict_elapsed_seconds": chronos_predict_elapsed,
            "elapsed_seconds": horizon_elapsed_seconds,
            "output_files": {
                "history_input": str(history_csv.resolve()),
                "history_plot_input": str(plot_history_csv.resolve()),
                "tft_forecast": str(tft_csv.resolve()),
                "chronos_forecast": str(chronos_csv.resolve()),
                "cascade_forecast": str(cascade_csv.resolve()),
            },
        }
        (horizon_dir / "horizon_meta.json").write_text(
            json.dumps(horizon_meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        summary_rows.append(
            {
                "horizon_hours": horizon_hours,
                "status": "ok",
                "chronos_predictor_path": str(horizon_path.resolve()),
                "chronos_model_name": chronos_model_name,
                "prediction_length": int(chronos_predictor.prediction_length),
                "chronos_rows": int(len(chronos_forecast.reset_index())),
                "cascade_rows": int(len(cascade_forecast.reset_index())),
                "elapsed_seconds": horizon_elapsed_seconds,
            }
        )
        completed_horizons.append(horizon_hours)
        print(
            f"  Finished {horizon_hours}h | elapsed {format_duration(horizon_elapsed_seconds)}",
            flush=True,
        )

    ready_horizons = sorted(set(completed_horizons + skipped_existing_horizons))
    if not ready_horizons:
        raise RuntimeError("No horizons were completed successfully.")

    summary_csv = args.output_dir / "batch_run_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False, encoding="utf-8-sig")

    if len(ready_horizons) == 1:
        sync_single_horizon_legacy_files(args.output_dir, ready_horizons[0])

    total_elapsed_seconds = time.perf_counter() - run_started_perf
    batch_meta = {
        "created_at": run_started_at.isoformat(timespec="seconds"),
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "target": args.target,
        "freq": args.freq,
        "requested_horizons": horizons,
        "completed_horizons": ready_horizons,
        "completed_horizons_this_run": completed_horizons,
        "skipped_existing_horizons": skipped_existing_horizons,
        "skipped_horizons": skipped_horizons,
        "num_items": int(history_ts.num_items),
        "history_rows": int(len(history_df)),
        "tft_elapsed_seconds": tft_elapsed_seconds,
        "elapsed_seconds_total": total_elapsed_seconds,
        "predictor_roots": [str(path.resolve()) for path in predictor_roots],
        "tft_predictor_path": str(tft_path.resolve()),
        "tft_model_name": tft_model_name,
        "tft_prediction_length": int(tft_predictor.prediction_length),
        "output_files": {
            "history_input": str(history_csv.resolve()),
            "history_plot_input": str(plot_history_csv.resolve()),
            "tft_forecast": str(tft_csv.resolve()),
            "summary_csv": str(summary_csv.resolve()),
        },
    }
    (args.output_dir / "batch_infer_meta.json").write_text(
        json.dumps(batch_meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Done.", flush=True)
    print(f"  History : {history_csv.resolve()}", flush=True)
    print(f"  TFT     : {tft_csv.resolve()}", flush=True)
    print(f"  Summary : {summary_csv.resolve()}", flush=True)
    print(f"  Horizons: {ready_horizons}", flush=True)
    print(f"  Total   : {format_duration(total_elapsed_seconds)}", flush=True)


if __name__ == "__main__":
    main()

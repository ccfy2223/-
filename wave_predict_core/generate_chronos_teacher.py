"""
generate_chronos_teacher.py

Fit or load a Chronos teacher on the base station splits, then cache
short-horizon teacher forecasts for distillation.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import pathlib
import sys
import time
import warnings
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

# 配置日志确保实时输出
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


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
os.environ.setdefault("HF_HUB_DISABLE_XET", _extract_cli_value("--hf-disable-xet", "1"))
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", _extract_cli_value("--hf-download-timeout", "600"))
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", _extract_cli_value("--hf-etag-timeout", "600"))

import pandas as pd
import torch
from autogluon.timeseries import TimeSeriesPredictor

from train_tft_chronos2_cascade import (
    WindowSpec,
    build_all_stations,
    build_window_specs,
    collect_env,
    load_shared_window,
    make_base_split,
    steps_per_hour,
    to_tsdf,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate Chronos teacher forecasts for short-horizon distillation."
    )
    parser.add_argument("--input-dir", type=Path, default=Path("processed_csv/aligned_stations"))
    parser.add_argument("--metadata-path", type=Path, default=Path("processed_csv/shared_timeline_metadata.json"))
    parser.add_argument("--output-root", type=Path, default=Path("chronos_teacher_cache"))
    parser.add_argument("--log-path", type=Path, default=Path("chronos_teacher_cache/teacher_log.json"))
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
    parser.add_argument("--context-hours", type=int, default=168)
    parser.add_argument("--window-stride-hours", type=int, default=6)
    parser.add_argument("--train-max-windows-per-station", type=int, default=256)
    parser.add_argument("--val-max-windows-per-station", type=int, default=64)
    parser.add_argument("--test-max-windows-per-station", type=int, default=64)
    parser.add_argument("--teacher-batch-size", type=int, default=16)
    parser.add_argument("--teacher-time-limit", type=int, default=3600, 
                       help="Time limit in seconds for AutoGluon fit (0 = no limit)")
    parser.add_argument("--teacher-model-path", default="amazon/chronos-t5-large")
    parser.add_argument("--teacher-context-length", type=int, default=2048)
    parser.add_argument("--model-cache-dir", type=Path, default=None)
    parser.add_argument("--skip-model-precache", action="store_true")
    parser.add_argument("--hf-disable-xet", default="1")
    parser.add_argument("--hf-download-timeout", type=int, default=600)
    parser.add_argument("--hf-etag-timeout", type=int, default=600)
    parser.add_argument("--teacher-fine-tune", action="store_true")
    parser.add_argument("--teacher-fine-tune-steps", type=int, default=1000)
    parser.add_argument("--teacher-fine-tune-batch-size", type=int, default=8)
    parser.add_argument("--teacher-fine-tune-lr", type=float, default=1e-5)
    parser.add_argument("--teacher-logging-steps", type=int, default=20)
    parser.add_argument("--teacher-eval-steps", type=int, default=100)
    parser.add_argument("--teacher-gradient-accumulation", type=int, default=4)
    parser.add_argument(
        "--pretrained-teacher-path",
        type=Path,
        default=None,
        help="Existing AutoGluon predictor directory that contains a Chronos model.",
    )
    parser.add_argument(
        "--pretrained-teacher-model-name",
        default="",
        help="Optional explicit Chronos model name inside the loaded predictor.",
    )
    return parser


def build_teacher_hyperparameters(args: argparse.Namespace, teacher_model_path: str) -> dict[str, list[dict[str, Any]]]:
    chronos_entry: dict[str, Any] = {
        "model_path": teacher_model_path,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "batch_size": args.teacher_batch_size,
        "context_length": args.teacher_context_length,
    }
    if args.teacher_fine_tune:
        chronos_entry.update(
            {
                "fine_tune": True,
                "fine_tune_steps": args.teacher_fine_tune_steps,
                "fine_tune_batch_size": args.teacher_fine_tune_batch_size,
                "fine_tune_lr": args.teacher_fine_tune_lr,
                "eval_during_fine_tune": True,
                "keep_transformers_logs": True,
                "fine_tune_trainer_kwargs": {
                    "disable_tqdm": False,
                    "logging_steps": args.teacher_logging_steps,
                    "eval_steps": args.teacher_eval_steps,
                    "gradient_accumulation_steps": args.teacher_gradient_accumulation,
                    "overwrite_output_dir": True,
                    "save_steps": args.teacher_eval_steps,
                },
            }
        )
    return {"Chronos": [chronos_entry]}


def enable_cross_platform_pickle_paths() -> None:
    if os.name == "nt":
        pathlib.PosixPath = pathlib.WindowsPath  # type: ignore[assignment]


def resolve_teacher_model_path(args: argparse.Namespace) -> str:
    raw_path = str(args.teacher_model_path)
    if args.skip_model_precache:
        return raw_path

    candidate = Path(raw_path)
    if candidate.exists():
        return str(candidate)

    if "/" not in raw_path:
        return raw_path

    from huggingface_hub import snapshot_download

    cache_dir = args.model_cache_dir
    if cache_dir is None:
        cache_dir = args.output_root / "hf_model_cache" / raw_path.replace("/", "--")
    cache_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Pre-caching teacher model to {cache_dir}")
    snapshot_download(
        repo_id=raw_path,
        local_dir=str(cache_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
        max_workers=1,
    )
    return str(cache_dir)


def select_teacher_model_name(predictor: TimeSeriesPredictor, preferred: str) -> str:
    model_names = predictor.model_names()
    if preferred:
        if preferred not in model_names:
            raise ValueError(f"Teacher model '{preferred}' not found. Available models: {model_names}")
        return preferred
    chronos_candidates = [name for name in model_names if "Chronos" in name]
    if not chronos_candidates:
        raise ValueError(f"No Chronos model found in predictor. Available models: {model_names}")
    return chronos_candidates[0]


def fit_or_load_teacher(
    train_ts,
    val_ts,
    args: argparse.Namespace,
    base_known_covariates: list[str],
    prediction_length: int,
) -> tuple[TimeSeriesPredictor, str, str]:
    if args.pretrained_teacher_path is not None:
        logger.info(f"Loading pretrained teacher from {args.pretrained_teacher_path}")
        enable_cross_platform_pickle_paths()
        predictor = TimeSeriesPredictor.load(str(args.pretrained_teacher_path))
        model_name = select_teacher_model_name(predictor, args.pretrained_teacher_model_name)
        logger.info(f"Loaded teacher model: {model_name}")
        return predictor, model_name, str(args.pretrained_teacher_path.resolve())

    effective_time_limit = None if args.teacher_time_limit <= 0 else args.teacher_time_limit
    resolved_teacher_model_path = resolve_teacher_model_path(args)
    model_dir = args.output_root / "teacher_model"
    
    logger.info(f"Creating TimeSeriesPredictor with target={args.target}, prediction_length={prediction_length}")
    predictor = TimeSeriesPredictor(
        target=args.target,
        prediction_length=prediction_length,
        freq=args.freq,
        eval_metric=args.eval_metric,
        known_covariates_names=base_known_covariates,
        path=str(model_dir),
        verbosity=args.verbosity,
    )
    
    logger.info("=" * 80)
    logger.info("Starting AutoGluon fit() with Chronos teacher model")
    logger.info(f"  Fine-tune: {args.teacher_fine_tune}")
    logger.info(f"  Fine-tune steps: {args.teacher_fine_tune_steps}")
    logger.info(f"  Batch size: {args.teacher_fine_tune_batch_size}")
    logger.info(f"  Time limit: {effective_time_limit}")
    logger.info(f"  Train data shape: {train_ts.shape}")
    logger.info(f"  Val data shape: {val_ts.shape}")
    logger.info("=" * 80)
    sys.stdout.flush()
    
    fit_start = time.perf_counter()
    try:
        predictor.fit(
            train_data=train_ts,
            tuning_data=val_ts,
            presets=args.presets,
            hyperparameters=build_teacher_hyperparameters(args, resolved_teacher_model_path),
            time_limit=effective_time_limit,
            random_seed=args.random_seed,
        )
        fit_elapsed = time.perf_counter() - fit_start
        logger.info(f"AutoGluon fit completed in {fit_elapsed:.1f} seconds")
    except Exception as e:
        logger.error(f"Error during fit: {type(e).__name__}: {e}", exc_info=True)
        raise
    
    sys.stdout.flush()
    model_name = select_teacher_model_name(predictor, "")
    logger.info(f"Selected model: {model_name}")
    return predictor, model_name, str(model_dir.resolve())


def specs_to_frame(specs: list[WindowSpec]) -> pd.DataFrame:
    return pd.DataFrame([asdict(spec) for spec in specs])


def build_teacher_predictions(
    specs: list[WindowSpec],
    frames: dict[str, pd.DataFrame],
    predictor: TimeSeriesPredictor,
    model_name: str,
    prediction_length: int,
    context_steps: int,
    base_known_covariates: list[str],
    batch_size: int,
    target: str = "WVHT",
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    chunk_size = max(1, batch_size)
    total_batches = (len(specs) + chunk_size - 1) // chunk_size

    for batch_idx, start in enumerate(range(0, len(specs), chunk_size)):
        chunk = specs[start : start + chunk_size]
        logger.info(f"  Processing batch {batch_idx + 1}/{total_batches} ({len(chunk)} windows)")
        sys.stdout.flush()
        
        past_frames: list[pd.DataFrame] = []
        future_frames: list[pd.DataFrame] = []
        for spec in chunk:
            source = frames[spec.source_item_id]
            context_start = spec.origin_index - context_steps + 1
            past_window = source.iloc[context_start : spec.origin_index + 1].copy()
            past_window["item_id"] = spec.item_id
            # Fill NaN in target column so Chronos does not degenerate to a constant forecast.
            past_window[target] = past_window[target].ffill().bfill()
            past_frames.append(past_window)

            future_window = source.iloc[spec.origin_index + 1 : spec.origin_index + prediction_length + 1].copy()
            future_window["item_id"] = spec.item_id
            future_frames.append(future_window[["item_id", "timestamp", *base_known_covariates]])

        past_ts = to_tsdf(pd.concat(past_frames, ignore_index=True))
        if base_known_covariates:
            future_known_ts = to_tsdf(pd.concat(future_frames, ignore_index=True))
            preds = predictor.predict(past_ts, known_covariates=future_known_ts, model=model_name)
        else:
            preds = predictor.predict(past_ts, model=model_name)
        pred_df = preds.reset_index().rename(columns={"mean": "teacher_mean"})
        rows.append(pred_df[["item_id", "timestamp", "teacher_mean"]].copy())

    if not rows:
        return pd.DataFrame(columns=["item_id", "timestamp", "teacher_mean"])
    out = pd.concat(rows, ignore_index=True)
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    return out.sort_values(["item_id", "timestamp"]).reset_index(drop=True)


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
    context_steps = args.context_hours * steps_per_h
    stride_steps = args.window_stride_hours * steps_per_h

    logger.info("=" * 72)
    logger.info("Chronos teacher cache generation")
    logger.info(f"  CUDA         : {env['cuda_available']} | device: {env['device_name']}")
    logger.info(f"  Split        : {args.train_ratio:.0%} / {args.val_ratio:.0%} / {args.test_ratio:.0%}")
    logger.info(f"  Short horizon: {args.short_horizon_hours}h ({short_pred_len} steps)")
    logger.info(f"  Context      : {args.context_hours}h ({context_steps} steps)")
    logger.info(f"  Teacher path : {args.teacher_model_path}")
    logger.info("=" * 72)
    sys.stdout.flush()

    shared_start, shared_end = load_shared_window(args.metadata_path, args.no_shared_window)
    station_files = sorted(args.input_dir.glob("*_aligned_10min.csv"))
    if not station_files:
        raise FileNotFoundError(f"No *_aligned_10min.csv files found in {args.input_dir}")
    logger.info(f"Found {len(station_files)} station files")
    sys.stdout.flush()

    logger.info("Loading stations ...")
    frames, station_summaries, base_known_covariates, _ = build_all_stations(
        files=station_files,
        target=args.target,
        shared_start=shared_start,
        shared_end=shared_end,
        limit=args.limit_stations,
        max_steps=args.max_steps,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    for summary in station_summaries:
        logger.info(f"  {summary.item_id}: {summary.rows_after_window} rows | train {summary.train_rows} | val {summary.val_rows} | test {summary.test_rows}")

    split_csv = args.output_root / "station_split_summary.csv"
    pd.DataFrame([asdict(summary) for summary in station_summaries]).to_csv(
        split_csv, index=False, encoding="utf-8-sig"
    )

    train_ts = to_tsdf(make_base_split(frames, station_summaries, "train"))
    val_ts = to_tsdf(make_base_split(frames, station_summaries, "val"))

    logger.info("Preparing Chronos teacher ...")
    sys.stdout.flush()
    t0 = time.perf_counter()
    teacher_predictor, teacher_model_name, teacher_model_dir = fit_or_load_teacher(
        train_ts=train_ts,
        val_ts=val_ts,
        args=args,
        base_known_covariates=base_known_covariates,
        prediction_length=short_pred_len,
    )
    teacher_prepare_elapsed = time.perf_counter() - t0
    logger.info(f"  Teacher model : {teacher_model_name}")
    logger.info(f"  Elapsed: {teacher_prepare_elapsed:.1f}s")
    sys.stdout.flush()

    split_specs: dict[str, list[WindowSpec]] = {
        "train": build_window_specs(
            frames=frames,
            summaries=station_summaries,
            split_name="train",
            horizon_hours=args.short_horizon_hours,
            prediction_length=short_pred_len,
            context_steps=context_steps,
            stride_steps=stride_steps,
            max_windows_per_station=args.train_max_windows_per_station,
        ),
        "val": build_window_specs(
            frames=frames,
            summaries=station_summaries,
            split_name="val",
            horizon_hours=args.short_horizon_hours,
            prediction_length=short_pred_len,
            context_steps=context_steps,
            stride_steps=stride_steps,
            max_windows_per_station=args.val_max_windows_per_station,
        ),
        "test": build_window_specs(
            frames=frames,
            summaries=station_summaries,
            split_name="test",
            horizon_hours=args.short_horizon_hours,
            prediction_length=short_pred_len,
            context_steps=context_steps,
            stride_steps=stride_steps,
            max_windows_per_station=args.test_max_windows_per_station,
        ),
    }

    split_summaries: list[dict[str, Any]] = []
    for split_name, specs in split_specs.items():
        logger.info(f"Caching teacher forecasts for {split_name} ({len(specs)} windows) ...")
        sys.stdout.flush()
        specs_df = specs_to_frame(specs)
        specs_df.to_csv(args.output_root / f"window_index_{split_name}.csv", index=False, encoding="utf-8-sig")

        t1 = time.perf_counter()
        forecast_df = build_teacher_predictions(
            specs=specs,
            frames=frames,
            predictor=teacher_predictor,
            model_name=teacher_model_name,
            prediction_length=short_pred_len,
            context_steps=context_steps,
            base_known_covariates=base_known_covariates,
            batch_size=args.teacher_batch_size,
            target=args.target,
        )
        elapsed = time.perf_counter() - t1
        forecast_csv = args.output_root / f"teacher_forecasts_{split_name}.csv"
        forecast_df.to_csv(forecast_csv, index=False, encoding="utf-8-sig")
        logger.info(f"  Saved {len(forecast_df)} forecasts in {elapsed:.1f}s")
        sys.stdout.flush()
        split_summaries.append(
            {
                "split_name": split_name,
                "windows": len(specs),
                "forecast_rows": len(forecast_df),
                "forecast_csv": str(forecast_csv.resolve()),
                "elapsed_seconds": elapsed,
            }
        )

    pd.DataFrame(split_summaries).to_csv(
        args.output_root / "teacher_cache_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config": vars(args)
        | {
            "input_dir": str(args.input_dir.resolve()),
            "output_root": str(args.output_root.resolve()),
            "base_known_covariates": base_known_covariates,
        },
        "environment": env,
        "teacher_model_name": teacher_model_name,
        "teacher_model_dir": teacher_model_dir,
        "teacher_prepare_elapsed_seconds": teacher_prepare_elapsed,
        "stations": [asdict(summary) for summary in station_summaries],
        "splits": split_summaries,
    }
    args.log_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    logger.info("=" * 72)
    logger.info("Done.")
    logger.info(f"  Teacher cache : {args.output_root.resolve()}")
    logger.info(f"  Train log     : {args.log_path.resolve()}")
    logger.info("=" * 72)
    sys.stdout.flush()


if __name__ == "__main__":
    main()

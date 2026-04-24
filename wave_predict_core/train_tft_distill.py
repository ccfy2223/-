"""
train_tft_distill.py

Train a TFT student using Chronos teacher forecasts as blended soft targets,
and optionally run iterative rollout inference on long horizons.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import warnings
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any


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

from train_tft_chronos2_cascade import (
    WindowSpec,
    build_all_stations,
    collect_env,
    load_shared_window,
    train_short_tft,
    steps_per_hour,
    to_tsdf,
)
from train_tft_iterative import build_scale_map, evaluate_horizon, infer_seasonal_period


DEFAULT_LONG_HORIZONS = [12, 24, 48, 72, 120]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a TFT student with Chronos soft targets, then optionally run iterative rollout."
    )
    parser.add_argument("--input-dir", type=Path, default=Path("processed_csv/aligned_stations"))
    parser.add_argument("--metadata-path", type=Path, default=Path("processed_csv/shared_timeline_metadata.json"))
    parser.add_argument("--teacher-cache-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("tft_distill_runs"))
    parser.add_argument("--log-path", type=Path, default=Path("tft_distill_runs/distill_log.json"))
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
    parser.add_argument(
        "--teacher-weight",
        type=float,
        default=0.5,
        help="Blended target = (1 - teacher_weight) * y_true + teacher_weight * y_teacher",
    )
    parser.add_argument("--tft-time-limit", type=int, default=3600)
    parser.add_argument(
        "--student-model",
        default="PatchTST",
        choices=["TemporalFusionTransformer", "PatchTST"],
        help="Student model architecture.",
    )
    parser.add_argument(
        "--allow-no-past-covariates-fallback",
        action="store_true",
        default=False,
        help="Allow disabling past covariates as final fallback if robust TFT attempts all fail.",
    )
    parser.add_argument("--run-iterative-rollout", action="store_true")
    parser.add_argument(
        "--long-horizons",
        default=",".join(str(h) for h in DEFAULT_LONG_HORIZONS),
        help="Comma-separated long-horizon hours for rollout evaluation.",
    )
    parser.add_argument("--window-stride-hours", type=int, default=6)
    parser.add_argument("--val-max-windows-per-station", type=int, default=32)
    parser.add_argument("--test-max-windows-per-station", type=int, default=32)
    parser.add_argument("--rollout-batch-size", type=int, default=32)
    parser.add_argument("--rollout-past-covariates", default="last", choices=["last", "actual", "zero"])
    parser.add_argument("--seasonal-period", type=int, default=0)
    parser.add_argument("--sample-windows-per-split", type=int, default=5)
    parser.add_argument(
        "--encoder-feature-dir",
        type=Path,
        default=None,
        help="Directory with encoder_features_{train,val,test}.csv from generate_chronos_encoder_features.py. "
             "If provided, encoder embeddings are appended as past covariates.",
    )
    return parser


def load_window_specs(path: Path) -> list[WindowSpec]:
    if not path.exists():
        raise FileNotFoundError(f"Missing teacher window index: {path}")
    df = pd.read_csv(path)
    records: list[WindowSpec] = []
    for row in df.to_dict(orient="records"):
        records.append(
            WindowSpec(
                item_id=str(row["item_id"]),
                source_item_id=str(row["source_item_id"]),
                split_name=str(row["split_name"]),
                horizon_hours=int(row["horizon_hours"]),
                origin_index=int(row["origin_index"]),
                origin_timestamp=str(row["origin_timestamp"]),
            )
        )
    return records


def load_teacher_map(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Missing teacher forecast cache: {path}")
    df = pd.read_csv(path, parse_dates=["timestamp"])
    mapping: dict[str, np.ndarray] = {}
    for item_id, group in df.groupby("item_id"):
        ordered = group.sort_values("timestamp")
        mapping[str(item_id)] = ordered["teacher_mean"].to_numpy(dtype=np.float32)
    return mapping


def build_student_window_dataset(
    specs: list[WindowSpec],
    frames: dict[str, pd.DataFrame],
    teacher_map: dict[str, np.ndarray],
    target: str,
    context_steps: int,
    prediction_length: int,
    teacher_weight: float,
    apply_distillation: bool,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for spec in specs:
        source = frames[spec.source_item_id]
        context_start = spec.origin_index - context_steps + 1
        window_stop = spec.origin_index + prediction_length + 1
        window_df = source.iloc[context_start:window_stop].copy().reset_index(drop=True)
        expected_len = context_steps + prediction_length
        if len(window_df) != expected_len:
            raise ValueError(f"Window length mismatch for {spec.item_id}: got {len(window_df)}, expected {expected_len}")
        window_df["item_id"] = spec.source_item_id
        if apply_distillation:
            teacher_values = teacher_map.get(spec.item_id)
            if teacher_values is None:
                raise ValueError(f"Missing teacher predictions for {spec.item_id}")
            future_index = window_df.index[context_steps:]
            y_true = window_df.loc[future_index, target].to_numpy(dtype=np.float32)
            teacher_values = teacher_values[:prediction_length]
            if len(teacher_values) < prediction_length:
                padded = y_true.copy()
                padded[: len(teacher_values)] = teacher_values
                teacher_values = padded
            finite_mask = np.isfinite(teacher_values)
            if not finite_mask.all():
                # Guard against bad teacher cache values: fall back to ground-truth on invalid positions.
                teacher_values = teacher_values.copy()
                teacher_values[~finite_mask] = y_true[~finite_mask]
            blended = (1.0 - teacher_weight) * y_true + teacher_weight * teacher_values
            window_df.loc[future_index, target] = blended.astype(np.float32)
        rows.append(window_df)
    if not rows:
        raise ValueError("No student windows were created.")
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    args = build_parser().parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.log_path.parent.mkdir(parents=True, exist_ok=True)

    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if not math.isclose(total_ratio, 1.0, abs_tol=1e-6):
        raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}.")
    if not (0.0 <= args.teacher_weight <= 1.0):
        raise ValueError(f"teacher_weight must be in [0, 1], got {args.teacher_weight}.")

    env = collect_env()
    steps_per_h = steps_per_hour(args.freq)
    short_pred_len = args.short_horizon_hours * steps_per_h
    context_steps = args.context_hours * steps_per_h
    seasonal_period = infer_seasonal_period(args.freq, args.seasonal_period)

    print("=" * 72)
    print("TFT student distillation")
    print(f"  CUDA         : {env['cuda_available']} | device: {env['device_name']}")
    print(f"  Split        : {args.train_ratio:.0%} / {args.val_ratio:.0%} / {args.test_ratio:.0%}")
    print(f"  Short horizon: {args.short_horizon_hours}h ({short_pred_len} steps)")
    print(f"  Context      : {args.context_hours}h ({context_steps} steps)")
    print(f"  Teacher cache: {args.teacher_cache_dir}")
    print(f"  Teacher w    : {args.teacher_weight:.3f}")
    print(f"  No-past FB   : {args.allow_no_past_covariates_fallback}")
    print("=" * 72)

    shared_start, shared_end = load_shared_window(args.metadata_path, args.no_shared_window)
    station_files = sorted(args.input_dir.glob("*_aligned_10min.csv"))
    if not station_files:
        raise FileNotFoundError(f"No *_aligned_10min.csv files found in {args.input_dir}")
    print(f"\nFound {len(station_files)} station files")

    print("\nLoading stations ...")
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

    split_csv = args.output_root / "station_split_summary.csv"
    pd.DataFrame([asdict(summary) for summary in station_summaries]).to_csv(
        split_csv, index=False, encoding="utf-8-sig"
    )

    teacher_root = args.teacher_cache_dir
    train_specs = load_window_specs(teacher_root / "window_index_train.csv")
    val_specs = load_window_specs(teacher_root / "window_index_val.csv")
    test_specs = load_window_specs(teacher_root / "window_index_test.csv")

    train_teacher_map = load_teacher_map(teacher_root / "teacher_forecasts_train.csv")
    val_teacher_map = load_teacher_map(teacher_root / "teacher_forecasts_val.csv")
    test_teacher_map = load_teacher_map(teacher_root / "teacher_forecasts_test.csv")

    print("\nBuilding distillation windows ...")
    train_df = build_student_window_dataset(
        specs=train_specs,
        frames=frames,
        teacher_map=train_teacher_map,
        target=args.target,
        context_steps=context_steps,
        prediction_length=short_pred_len,
        teacher_weight=args.teacher_weight,
        apply_distillation=True,
    )
    val_df = build_student_window_dataset(
        specs=val_specs,
        frames=frames,
        teacher_map=val_teacher_map,
        target=args.target,
        context_steps=context_steps,
        prediction_length=short_pred_len,
        teacher_weight=0.0,
        apply_distillation=False,
    )
    test_df = build_student_window_dataset(
        specs=test_specs,
        frames=frames,
        teacher_map=test_teacher_map,
        target=args.target,
        context_steps=context_steps,
        prediction_length=short_pred_len,
        teacher_weight=0.0,
        apply_distillation=False,
    )

    train_df.to_csv(args.output_root / "student_windows_train.csv", index=False, encoding="utf-8-sig")
    val_df.to_csv(args.output_root / "student_windows_val.csv", index=False, encoding="utf-8-sig")
    test_df.to_csv(args.output_root / "student_windows_test.csv", index=False, encoding="utf-8-sig")

    # Optionally merge encoder features as past covariates
    if args.encoder_feature_dir is not None:
        print("\nMerging encoder features ...")
        for split_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
            feat_path = args.encoder_feature_dir / f"encoder_features_{split_name}.csv"
            if not feat_path.exists():
                raise FileNotFoundError(f"Encoder feature file not found: {feat_path}")
            feat_df = pd.read_csv(feat_path)
            enc_cols = [c for c in feat_df.columns if c != "item_id"]
            # Each encoder feature is a per-window scalar; broadcast to all rows of that window
            merged = df.merge(feat_df[["item_id"] + enc_cols], on="item_id", how="left")
            df[enc_cols] = merged[enc_cols].values
            print(f"  {split_name}: added {len(enc_cols)} encoder feature columns")
        print("  Encoder features merged.")

    train_ts = to_tsdf(train_df)
    val_ts = to_tsdf(val_df)
    test_ts = to_tsdf(test_df)

    print(f"\nTraining {args.student_model} student ...")
    args.tft_time_limit = None if args.tft_time_limit <= 0 else args.tft_time_limit
    tft_predictor, short_result = train_short_tft(
        train_ts=train_ts,
        val_ts=val_ts,
        test_ts=test_ts,
        args=args,
        base_known_covariates=base_known_covariates,
        prediction_length=short_pred_len,
        model_name=args.student_model,
    )

    print(
        f"  Student model  : {short_result.best_model}\n"
        f"  Val MASE       : {short_result.val_mase:.6f}\n"
        f"  Test MASE      : {short_result.metrics['MASE']:.6f}\n"
        f"  Test RMSE      : {short_result.metrics['RMSE']:.6f}\n"
        f"  Student path   : {short_result.model_dir}"
    )

    rollout_rows: list[dict[str, Any]] = []
    if args.run_iterative_rollout:
        # Encoder features are per-window scalars, not available during rollout.
        # Inject zero-filled enc columns into frames so history_df slices include them.
        enc_cols: set[str] = set()
        if args.encoder_feature_dir is not None:
            feat_path = args.encoder_feature_dir / "encoder_features_train.csv"
            if feat_path.exists():
                enc_cols = set(pd.read_csv(feat_path, nrows=0).columns) - {"item_id"}
                for station_id, df in frames.items():
                    for col in enc_cols:
                        if col not in df.columns:
                            df[col] = np.float32(0.0)

        scale_map = build_scale_map(
            frames=frames,
            summaries=station_summaries,
            target=args.target,
            seasonal_period=seasonal_period,
        )
        stride_steps = args.window_stride_hours * steps_per_h
        long_horizons = [int(x) for x in args.long_horizons.split(",") if x.strip()]

        for horizon_hours in sorted(set(long_horizons)):
            long_pred_len = horizon_hours * steps_per_h
            print(f"\nRunning iterative rollout for {horizon_hours}h ...")
            result = evaluate_horizon(
                args=args,
                frames=frames,
                summaries=station_summaries,
                target=args.target,
                base_known_covariates=base_known_covariates,
                past_covariates=[c for c in train_df.columns if c not in {"item_id", "timestamp", args.target, *base_known_covariates}],
                predictor=tft_predictor,
                model_name=short_result.best_model,
                short_pred_len=short_pred_len,
                horizon_hours=horizon_hours,
                long_pred_len=long_pred_len,
                context_steps=context_steps,
                stride_steps=stride_steps,
                scale_map=scale_map,
            )
            rollout_rows.append(
                {
                    "horizon_hours": result.horizon_hours,
                    "prediction_length": result.prediction_length,
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
            )

    if rollout_rows:
        pd.DataFrame(rollout_rows).to_csv(
            args.output_root / "distill_iterative_metrics_summary.csv",
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
            "seasonal_period": seasonal_period,
        },
        "environment": env,
        "teacher_cache_dir": str(args.teacher_cache_dir.resolve()),
        "stations": [asdict(summary) for summary in station_summaries],
        "short_tft_student": asdict(short_result),
        "iterative_rollout": rollout_rows,
    }
    args.log_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print("\nDone.")
    print(f"  Student path : {short_result.model_dir}")
    print(f"  Train log    : {args.log_path.resolve()}")


if __name__ == "__main__":
    main()

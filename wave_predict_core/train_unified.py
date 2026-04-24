"""
train_unified.py
================
多站点 + 多 Horizon 联合训练（统一模型）

与基线 train_autogluon.py 的区别：
  - 基线：每个 horizon 单独训练一个模型，8 站作为独立 item_id
  - 本脚本：一个模型，8站 × 8 horizon = 64 个 item_id 联合训练
    item_id 格式："{station_id}_h{horizon:03d}"
    prediction_length 固定为最长 horizon（720 steps = 120h）
    horizon 信息通过 horizon_sin / horizon_cos 编码为 known_covariate

模型：
  - PatchTST（主实验，监督学习）
  - Chronos fine-tune（对比实验，预训练大模型迁移）
  两者同时在一次 predictor.fit() 中训练，最后 leaderboard 对比

评估：
  训练完后对每个 horizon 单独评估（截取预测序列的前 N 步），
  与 autogluon_runs/ 基线各 horizon 最优 MASE 对比。
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
from typing import Any

warnings.filterwarnings("ignore", category=FutureWarning)
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))


def _read_cli_value(flag: str, default: str) -> str:
    try:
        idx = sys.argv.index(flag)
    except ValueError:
        return default
    if idx + 1 >= len(sys.argv):
        return default
    return sys.argv[idx + 1]


os.environ["CUDA_VISIBLE_DEVICES"] = _read_cli_value("--gpu-id", "0")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np
import pandas as pd
import torch
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

# ── constants ─────────────────────────────────────────────────────────────────

HORIZON_HOURS = [1, 3, 6, 12, 24, 48, 72, 120]
STEPS_PER_HOUR = 6  # 10min frequency
MAX_HORIZON_STEPS = max(HORIZON_HOURS) * STEPS_PER_HOUR  # 720

KNOWN_COVARIATES = [
    "time_sin_hour", "time_cos_hour",
    "time_sin_doy",  "time_cos_doy",
    "month", "day_of_week",
    "horizon_sin", "horizon_cos",   # horizon encoding
]

DYNAMIC_COLUMNS = [
    "WDIR", "WSPD", "GST", "WVHT",
    "DPD", "APD", "MWD",
    "PRES", "ATMP", "WTMP", "DEWP", "VIS", "TIDE",
]

EXTRA_METRICS = ["MAE", "MASE", "RMSE", "RMSLE", "SMAPE"]

# 基线各 horizon 最优 MASE（来自 autogluon_runs/）
BASELINE_BEST_MASE = {
    1:   0.071128,
    3:   0.083864,
    6:   0.110672,
    12:  0.198754,
    24:  0.449000,
    48:  0.539210,
    72:  0.523886,
    120: 0.659477,
}

BASELINE_WE_MASE = {
    1:   0.074371,
    3:   0.091498,
    6:   0.110672,
    12:  0.198754,
    24:  0.499039,
    48:  0.599753,
    72:  0.538715,
    120: 0.750106,
}


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Unified multi-horizon multi-station wave height forecasting"
    )
    p.add_argument("--input-dir",   type=Path,
                   default=Path("processed_csv/aligned_stations"))
    p.add_argument("--metadata-path", type=Path,
                   default=Path("processed_csv/shared_timeline_metadata.json"))
    p.add_argument("--output-root", type=Path, default=Path("sota_runs/unified"))
    p.add_argument("--log-path",    type=Path, default=Path("sota_runs/unified/unified_log.json"))
    p.add_argument("--target",      default="WVHT")
    p.add_argument("--freq",        default="10min")
    p.add_argument("--gpu-id",      default="0")
    p.add_argument("--train-ratio", type=float, default=0.70)
    p.add_argument("--val-ratio",   type=float, default=0.20)
    p.add_argument("--presets",     default="best_quality")
    p.add_argument("--time-limit",  type=int, default=14400,
                   help="Total training time limit in seconds (0=unlimited).")
    p.add_argument("--random-seed", type=int, default=42)
    p.add_argument("--limit-stations", type=int, default=0,
                   help="Use only first N stations (0=all, for smoke test).")
    p.add_argument("--prediction-hours", default=",".join(str(h) for h in HORIZON_HOURS),
                   help="Comma-separated horizon hours to include.")
    p.add_argument("--verbosity",   type=int, default=2)
    # PatchTST params
    p.add_argument("--patchtst-context-length", type=int, default=1008)
    p.add_argument("--patchtst-d-model",        type=int, default=128)
    p.add_argument("--patchtst-nhead",           type=int, default=8)
    p.add_argument("--patchtst-layers",          type=int, default=4)
    p.add_argument("--patchtst-epochs",          type=int, default=50)
    p.add_argument("--patchtst-batch-size",      type=int, default=32)
    p.add_argument("--patchtst-num-batches",     type=int, default=200)
    p.add_argument("--patchtst-lr",              type=float, default=1e-4)
    p.add_argument("--patchtst-patience",        type=int, default=10)
    # Chronos params
    p.add_argument("--chronos-model-path",       default="amazon/chronos-t5-large")
    p.add_argument("--chronos-context-length",   type=int, default=2048)
    p.add_argument("--chronos-batch-size",       type=int, default=8)
    p.add_argument("--chronos-fine-tune",        action="store_true",
                   help="Fine-tune Chronos (otherwise zero-shot).")
    p.add_argument("--chronos-fine-tune-steps",  type=int, default=3000)
    p.add_argument("--chronos-fine-tune-lr",     type=float, default=1e-5)
    p.add_argument("--no-patchtst",              action="store_true",
                   help="Skip PatchTST training.")
    p.add_argument("--no-chronos",               action="store_true",
                   help="Skip Chronos training.")
    return p


# ── data loading ──────────────────────────────────────────────────────────────

def load_station(
    path: Path,
    target: str,
    shared_start: pd.Timestamp | None,
    shared_end: pd.Timestamp | None,
) -> pd.DataFrame:
    item_id = path.stem.split("_")[0]
    cols_all = pd.read_csv(path, nrows=0).columns.tolist()
    base_known = [c for c in KNOWN_COVARIATES
                  if c not in {"horizon_sin", "horizon_cos"} and c in cols_all]
    dyn = [c for c in DYNAMIC_COLUMNS if c in cols_all]

    df = pd.read_csv(
        path,
        usecols=["datetime"] + dyn + base_known,
        parse_dates=["datetime"],
        low_memory=False,
    )
    df = (df.sort_values("datetime")
            .drop_duplicates("datetime", keep="last")
            .reset_index(drop=True))

    if shared_start is not None:
        df = df[(df["datetime"] >= shared_start) &
                (df["datetime"] <= shared_end)].reset_index(drop=True)

    if len(df) == 0:
        raise ValueError(f"{path.name}: no rows after window filter")

    num_cols = [c for c in df.columns if c != "datetime"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
    df[num_cols] = df[num_cols].ffill().bfill()

    if df[target].isna().any():
        raise ValueError(f"Target '{target}' still has NaN in {path.name}")

    df.insert(0, "item_id", item_id)
    df = df.rename(columns={"datetime": "timestamp"})
    return df


def build_unified_frames(
    files: list[Path],
    target: str,
    shared_start: pd.Timestamp | None,
    shared_end: pd.Timestamp | None,
    horizon_hours: list[int],
    limit: int,
    train_ratio: float,
    val_ratio: float,
) -> tuple[dict[str, pd.DataFrame], dict[str, tuple[int, int]]]:
    """
    Returns:
        frames: dict[item_id -> DataFrame]  (item_id = "{station}_h{horizon:03d}")
        splits: dict[item_id -> (train_end_idx, val_end_idx)]
    """
    selected = files[:limit] if limit > 0 else files

    # Load raw station data
    raw: dict[str, pd.DataFrame] = {}
    for p in selected:
        df = load_station(p, target, shared_start, shared_end)
        station_id = df["item_id"].iloc[0]
        raw[station_id] = df
        print(f"  {station_id}: {len(df)} rows")

    if not raw:
        raise ValueError("No station data loaded.")

    # Drop past-covariate columns with residual NaN across all stations
    combined = pd.concat(raw.values(), ignore_index=True)
    past_candidates = [
        c for c in combined.columns
        if c not in {"item_id", "timestamp", target, *KNOWN_COVARIATES}
    ]
    bad = {c for c in past_candidates if combined[c].isna().any()}
    if bad:
        keep_cols = (
            ["item_id", "timestamp", target]
            + [c for c in past_candidates if c not in bad]
            + [c for c in KNOWN_COVARIATES
               if c not in {"horizon_sin", "horizon_cos"} and c in combined.columns]
        )
        for k in raw:
            raw[k] = raw[k][[c for c in keep_cols if c in raw[k].columns]].copy()
        print(f"  Dropped past covariates with residual NaN: {sorted(bad)}")

    # Build unified frames: replicate each station for each horizon
    frames: dict[str, pd.DataFrame] = {}
    splits: dict[str, tuple[int, int]] = {}

    for h_idx, h in enumerate(horizon_hours):
        h_sin = np.float32(math.sin(2 * math.pi * h_idx / len(horizon_hours)))
        h_cos = np.float32(math.cos(2 * math.pi * h_idx / len(horizon_hours)))

        for station_id, df in raw.items():
            item_id = f"{station_id}_h{h:03d}"
            sub = df.copy()
            sub["item_id"] = item_id
            sub["horizon_sin"] = h_sin
            sub["horizon_cos"] = h_cos

            n = len(sub)
            train_end = max(1, int(n * train_ratio))
            val_end   = max(train_end + 1, int(n * (train_ratio + val_ratio)))
            val_end   = min(val_end, n - 1)

            frames[item_id] = sub
            splits[item_id] = (train_end, val_end)

    print(f"\n  Total item_ids: {len(frames)} "
          f"({len(selected)} stations × {len(horizon_hours)} horizons)")
    return frames, splits


def make_split(
    frames: dict[str, pd.DataFrame],
    splits: dict[str, tuple[int, int]],
    split: str,
) -> pd.DataFrame:
    parts = []
    for iid, df in frames.items():
        train_end, val_end = splits[iid]
        if split == "train":
            parts.append(df.iloc[:train_end].copy())
        elif split == "val":
            parts.append(df.iloc[:val_end].copy())   # cumulative for AutoGluon tuning
        else:
            parts.append(df.copy())                  # full series for test
    return pd.concat(parts, ignore_index=True)


def to_tsdf(df: pd.DataFrame) -> TimeSeriesDataFrame:
    return TimeSeriesDataFrame.from_data_frame(
        df=df, id_column="item_id", timestamp_column="timestamp"
    )


# ── hyperparameters ───────────────────────────────────────────────────────────

def build_hyperparameters(args: argparse.Namespace) -> dict[str, Any]:
    hp: dict[str, Any] = {}

    if not args.no_patchtst:
        hp["PatchTST"] = {
            "context_length": args.patchtst_context_length,
            "patch_len": 16,
            "stride": 8,
            "d_model": args.patchtst_d_model,
            "nhead": args.patchtst_nhead,
            "num_encoder_layers": args.patchtst_layers,
            "dropout": 0.1,
            "lr": args.patchtst_lr,
            "batch_size": args.patchtst_batch_size,
            "max_epochs": args.patchtst_epochs,
            "num_batches_per_epoch": args.patchtst_num_batches,
            "early_stopping_patience": args.patchtst_patience,
            "trainer_kwargs": {"gradient_clip_val": 1.0},
        }

    if not args.no_chronos:
        chronos_cfg: dict[str, Any] = {
            "model_path": args.chronos_model_path,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "batch_size": args.chronos_batch_size,
            "context_length": args.chronos_context_length,
        }
        if args.chronos_fine_tune:
            chronos_cfg.update({
                "fine_tune": True,
                "fine_tune_steps": args.chronos_fine_tune_steps,
                "fine_tune_batch_size": args.chronos_batch_size,
                "fine_tune_lr": args.chronos_fine_tune_lr,
                "eval_during_fine_tune": True,
                "fine_tune_trainer_kwargs": {
                    "logging_steps": 20,
                    "save_steps": 100,
                    "eval_steps": 100,
                    "gradient_accumulation_steps": 4,
                    "overwrite_output_dir": True,
                },
            })
        hp["Chronos"] = [chronos_cfg]

    return hp


# ── per-horizon evaluation ────────────────────────────────────────────────────

def evaluate_per_horizon(
    predictor: TimeSeriesPredictor,
    test_ts: TimeSeriesDataFrame,
    horizon_hours: list[int],
) -> list[dict]:
    """
    For each horizon h, build a sub-predictor with prediction_length=h*6,
    evaluate with leaderboard() on item_ids matching _h{h:03d}.
    This uses AutoGluon's correct MASE normalisation.
    """
    print("\n" + "=" * 70)
    print("Per-horizon evaluation")
    print("=" * 70)

    results = []
    for h in horizon_hours:
        suffix = f"_h{h:03d}"
        pred_steps = h * STEPS_PER_HOUR

        item_ids_h = [iid for iid in test_ts.item_ids if iid.endswith(suffix)]
        if not item_ids_h:
            print(f"  {h:3d}h: no item_ids found, skipping")
            continue

        sub_test  = test_ts.loc[item_ids_h]

        try:
            lb = predictor.leaderboard(
                data=sub_test,
                extra_metrics=["MAE", "MASE", "RMSE"],
                display=False,
            )
            lb = lb.copy()
            lb["MASE"] = lb["MASE"].abs()
            lb["MAE"]  = lb["MAE"].abs()
            lb["RMSE"] = lb["RMSE"].abs()
            lb["score_test"] = lb["score_test"].abs()

            best_row = lb.iloc[0]
            mase = float(best_row["MASE"])
            mae  = float(best_row["MAE"])
            rmse = float(best_row["RMSE"])
            model = str(best_row["model"])
        except Exception as e:
            print(f"  {h:3d}h: leaderboard failed: {e}")
            results.append({"horizon_hours": h, "error": str(e)})
            continue

        baseline_best = BASELINE_BEST_MASE.get(h, float("nan"))
        baseline_we   = BASELINE_WE_MASE.get(h, float("nan"))
        delta_best = mase - baseline_best
        delta_we   = mase - baseline_we

        print(f"  {h:3d}h | MASE={mase:.4f} | MAE={mae:.4f} | RMSE={rmse:.4f} "
              f"| vs best_baseline {delta_best:+.4f} | vs WE {delta_we:+.4f}")

        results.append({
            "horizon_hours": h,
            "pred_steps": pred_steps,
            "n_items": len(item_ids_h),
            "MASE": mase,
            "MAE": mae,
            "RMSE": rmse,
            "baseline_best_MASE": baseline_best,
            "baseline_WE_MASE": baseline_we,
            "delta_vs_best": delta_best,
            "delta_vs_WE": delta_we,
        })

    return results


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = build_parser().parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.log_path.parent.mkdir(parents=True, exist_ok=True)

    horizon_hours = [int(h) for h in args.prediction_hours.split(",") if h.strip()]
    pred_len = max(horizon_hours) * STEPS_PER_HOUR

    device_name = (
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    )
    print("=" * 70)
    print("Unified multi-horizon multi-station wave height forecasting")
    print(f"  CUDA         : {torch.cuda.is_available()} | device: {device_name}")
    print(f"  Horizons     : {horizon_hours}")
    print(f"  pred_length  : {pred_len} steps ({max(horizon_hours)}h)")
    print(f"  Output       : {args.output_root}")
    print("=" * 70)

    # Load metadata for shared window
    meta = args.metadata_path
    shared_start = shared_end = None
    if meta.exists():
        d = json.loads(meta.read_text(encoding="utf-8"))
        s, e = d.get("shared_start"), d.get("shared_end")
        shared_start = pd.Timestamp(s) if s else None
        shared_end   = pd.Timestamp(e) if e else None

    files = sorted(args.input_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files in {args.input_dir}")

    print("\nLoading stations ...")
    frames, splits = build_unified_frames(
        files=files,
        target=args.target,
        shared_start=shared_start,
        shared_end=shared_end,
        horizon_hours=horizon_hours,
        limit=args.limit_stations,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )

    print("\nBuilding splits ...")
    train_df = make_split(frames, splits, "train")
    val_df   = make_split(frames, splits, "val")
    test_df  = make_split(frames, splits, "test")

    train_ts = to_tsdf(train_df)
    val_ts   = to_tsdf(val_df)
    test_ts  = to_tsdf(test_df)

    print(f"  train: {train_ts.num_items} items, {len(train_df)} rows")
    print(f"  val  : {val_ts.num_items} items, {len(val_df)} rows")
    print(f"  test : {test_ts.num_items} items, {len(test_df)} rows")

    model_dir = args.output_root / "unified_model"
    predictor = TimeSeriesPredictor(
        target=args.target,
        prediction_length=pred_len,
        freq=args.freq,
        eval_metric="MASE",
        known_covariates_names=KNOWN_COVARIATES,
        path=str(model_dir),
        verbosity=args.verbosity,
    )

    hp = build_hyperparameters(args)
    print(f"\nModels to train: {list(hp.keys())}")

    time_limit = None if args.time_limit <= 0 else args.time_limit
    t0 = time.perf_counter()
    predictor.fit(
        train_data=train_ts,
        tuning_data=val_ts,
        presets=args.presets,
        hyperparameters=hp,
        time_limit=time_limit,
        random_seed=args.random_seed,
        enable_ensemble=False,   # 对比单模型，不做集成
    )
    elapsed = time.perf_counter() - t0
    print(f"\nTraining done in {elapsed/3600:.2f}h")

    # Global leaderboard on test set
    print("\nGlobal leaderboard (prediction_length=720, all horizons pooled):")
    try:
        lb = predictor.leaderboard(data=test_ts, extra_metrics=EXTRA_METRICS, display=False)
        lb_abs = lb.copy()
        for c in EXTRA_METRICS:
            if c in lb_abs.columns:
                lb_abs[c] = lb_abs[c].abs()
        lb_abs["score_test"] = lb_abs["score_test"].abs()
        print(lb_abs[["model", "score_test"] + [c for c in EXTRA_METRICS if c in lb_abs.columns]].to_string(index=False))
        lb_path = args.output_root / "leaderboard_test_global.csv"
        lb_abs.to_csv(lb_path, index=False, encoding="utf-8-sig")
    except Exception as e:
        print(f"  leaderboard failed: {e}")
        lb_abs = None

    # Per-horizon evaluation
    per_h = evaluate_per_horizon(predictor, test_ts, horizon_hours)

    # Summary vs baseline
    print("\n" + "=" * 70)
    print("Summary vs baseline")
    print(f"{'Horizon':>8} {'Unified MASE':>14} {'Best Baseline':>14} {'WeightedEnsemble':>17} {'vs Best':>8} {'vs WE':>7}")
    print("-" * 70)
    for r in per_h:
        if "error" in r:
            continue
        h = r["horizon_hours"]
        print(f"{h:>7}h {r['MASE']:>14.4f} {r['baseline_best_MASE']:>14.4f} "
              f"{r['baseline_WE_MASE']:>17.4f} {r['delta_vs_best']:>+8.4f} {r['delta_vs_WE']:>+7.4f}")

    # Save log
    log = {
        "created_at": datetime.now().isoformat(),
        "horizon_hours": horizon_hours,
        "prediction_length": pred_len,
        "elapsed_seconds": elapsed,
        "models_trained": list(hp.keys()),
        "per_horizon_results": per_h,
    }
    args.log_path.write_text(
        json.dumps(log, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"\nLog saved to {args.log_path}")


if __name__ == "__main__":
    main()

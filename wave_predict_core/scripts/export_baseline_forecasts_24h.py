"""
Export a common 24h truth-vs-prediction window for the main baseline models.

This script reloads the local AutoGluon predictor and computes one backtest
window for a representative station. It is intentionally small because the
result is used only for thesis visualisation.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys
from pathlib import Path

import pandas as pd


if os.name == "nt":
    # The downloaded AutoGluon artifacts were trained on Linux and contain
    # PosixPath objects. This compatibility shim is needed before loading them.
    pathlib.PosixPath = pathlib.WindowsPath  # type: ignore[attr-defined,assignment]


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autogluon.timeseries import TimeSeriesPredictor  # noqa: E402
from train_autogluon import build_all_stations, load_shared_window, to_tsdf  # noqa: E402


MODEL_NAMES = {
    "Chronos[amazon__chronos-t5-large]": "Chronos",
    "TemporalFusionTransformer": "TFT",
    "PatchTST": "PatchTST",
    "DLinear": "DLinear",
    "DeepAR": "DeepAR",
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", type=Path, default=ROOT / "autogluon_runs" / "horizon_024h" / "model")
    p.add_argument("--input-dir", type=Path, default=ROOT / "processed_csv" / "aligned_stations")
    p.add_argument("--metadata-path", type=Path, default=ROOT / "processed_csv" / "shared_timeline_metadata.json")
    p.add_argument("--station-id", default="41010")
    p.add_argument("--output", type=Path, default=ROOT / "thesis_assets" / "tables" / "baseline_024h_prediction_compare.csv")
    return p


def main() -> None:
    args = build_parser().parse_args()
    station_file = args.input_dir / f"{args.station_id}_aligned_10min.csv"
    if not station_file.exists():
        raise FileNotFoundError(station_file)

    shared_start, shared_end = load_shared_window(args.metadata_path, disabled=False)
    frames, _ = build_all_stations(
        files=[station_file],
        target="WVHT",
        shared_start=shared_start,
        shared_end=shared_end,
        limit=0,
        max_steps=0,
        train_ratio=0.70,
        val_ratio=0.20,
    )
    ts = to_tsdf(pd.concat(frames.values(), ignore_index=True))

    predictor = TimeSeriesPredictor.load(str(args.model_dir))
    targets = predictor.backtest_targets(data=ts, num_val_windows=1)[0]
    target_df = (
        targets.reset_index()[["item_id", "timestamp", "WVHT"]]
        .rename(columns={"WVHT": "y_true"})
    )
    target_df["item_id"] = target_df["item_id"].astype(str)

    rows: list[pd.DataFrame] = []
    for model, display in MODEL_NAMES.items():
        print(f"[export] {display}")
        pred = predictor.backtest_predictions(
            data=ts,
            model=model,
            num_val_windows=1,
            use_cache=False,
        )[0]
        pred_df = (
            pred.reset_index()[["item_id", "timestamp", "mean"]]
            .rename(columns={"mean": "y_pred"})
        )
        pred_df["item_id"] = pred_df["item_id"].astype(str)
        merged = target_df.merge(pred_df, on=["item_id", "timestamp"], how="inner")
        merged["model"] = display
        merged["horizon_hours"] = 24
        merged["forecast_step"] = merged.groupby(["item_id", "model"]).cumcount() + 1
        rows.append(merged)

    out = pd.concat(rows, ignore_index=True)
    out = out[["item_id", "timestamp", "forecast_step", "horizon_hours", "model", "y_true", "y_pred"]]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"[done] {args.output} rows={len(out)}")


if __name__ == "__main__":
    main()

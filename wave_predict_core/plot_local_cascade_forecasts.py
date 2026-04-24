"""
plot_local_cascade_forecasts.py
===============================

Read batch inference outputs produced by local_pretrained_cascade_infer.py and
draw forecast curves for:
- TFT
- Chronos
- Cascade

Each figure also includes a short history tail for context.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.unicode_minus": False,
        "figure.dpi": 150,
    }
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot local TFT / Chronos / Cascade forecasts.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("local_cascade_batch_output"),
        help="Batch inference output directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Plot output directory. Default: <run-dir>/plots",
    )
    parser.add_argument(
        "--horizons",
        default=None,
        help="Comma-separated horizons to plot, e.g. 24,48,72,120. Default: auto-discover.",
    )
    parser.add_argument(
        "--stations",
        default=None,
        help="Comma-separated station ids to plot. Default: all stations found in history_input.csv.",
    )
    parser.add_argument(
        "--history-steps",
        type=int,
        default=288,
        help="How many latest history steps to display before the forecast start.",
    )
    parser.add_argument(
        "--value-column",
        default="mean",
        help="Forecast value column to plot. Default: mean. Falls back to 0.5 when needed.",
    )
    parser.add_argument("--format", default="png", choices=["png", "pdf", "svg"])
    return parser


def parse_int_list(raw: str) -> list[int]:
    return sorted({int(chunk.strip()) for chunk in raw.split(",") if chunk.strip()})


def parse_str_list(raw: str) -> list[str]:
    return [chunk.strip() for chunk in raw.split(",") if chunk.strip()]


def discover_horizons(run_dir: Path) -> list[int]:
    pattern = re.compile(r"horizon_(\d+)h")
    found: list[int] = []
    for path in sorted(run_dir.iterdir()):
        match = pattern.fullmatch(path.name)
        if match and path.is_dir():
            found.append(int(match.group(1)))
    return found


def resolve_tft_csv(run_dir: Path) -> Path:
    candidates = [
        run_dir / "tft_006h_forecast.csv",
        run_dir / "tft_6h_forecast.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Unable to find TFT forecast CSV in {run_dir}")


def detect_value_column(path: Path, preferred: str) -> str:
    columns = pd.read_csv(path, nrows=0).columns.tolist()
    if preferred in columns:
        return preferred
    if "mean" in columns:
        return "mean"
    if "0.5" in columns:
        return "0.5"
    raise KeyError(f"No usable value column found in {path}. Available columns: {columns}")


def read_csv(
    path: Path,
    usecols: list[str] | None = None,
    station_ids: set[str] | None = None,
    chunksize: int = 200_000,
) -> pd.DataFrame:
    if station_ids:
        chunks: list[pd.DataFrame] = []
        for chunk in pd.read_csv(
            path,
            usecols=usecols,
            parse_dates=["timestamp"],
            chunksize=chunksize,
            low_memory=False,
        ):
            if "item_id" in chunk.columns:
                chunk["item_id"] = chunk["item_id"].astype(str)
                chunk = chunk[chunk["item_id"].isin(station_ids)]
            if not chunk.empty:
                chunks.append(chunk)
        if not chunks:
            empty_columns = usecols if usecols is not None else pd.read_csv(path, nrows=0).columns.tolist()
            return pd.DataFrame(columns=empty_columns)
        return pd.concat(chunks, ignore_index=True)

    df = pd.read_csv(path, usecols=usecols, parse_dates=["timestamp"], low_memory=False)
    if "item_id" in df.columns:
        df["item_id"] = df["item_id"].astype(str)
    return df


def select_value_column(df: pd.DataFrame, preferred: str) -> str:
    if preferred in df.columns:
        return preferred
    if "mean" in df.columns:
        return "mean"
    if "0.5" in df.columns:
        return "0.5"
    raise KeyError(f"No usable value column found. Available columns: {df.columns.tolist()}")


def load_target_name(run_dir: Path) -> str:
    meta_path = run_dir / "batch_infer_meta.json"
    if not meta_path.exists():
        return "WVHT"
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    return str(payload.get("target", "WVHT"))


def save_figure(fig: plt.Figure, path: Path, fmt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(f".{fmt}"), bbox_inches="tight")
    plt.close(fig)


def plot_one_station(
    station_id: str,
    horizon_hours: int,
    history_df: pd.DataFrame,
    tft_df: pd.DataFrame,
    chronos_df: pd.DataFrame,
    cascade_df: pd.DataFrame,
    target_column: str,
    history_steps: int,
    output_path: Path,
    fmt: str,
) -> bool:
    history_station = history_df[history_df["item_id"] == station_id].sort_values("timestamp")
    tft_station = tft_df[tft_df["item_id"] == station_id].sort_values("timestamp")
    chronos_station = chronos_df[chronos_df["item_id"] == station_id].sort_values("timestamp")
    cascade_station = cascade_df[cascade_df["item_id"] == station_id].sort_values("timestamp")

    if history_station.empty or chronos_station.empty or cascade_station.empty:
        return False

    if history_steps > 0:
        history_station = history_station.tail(history_steps)

    forecast_start = chronos_station["timestamp"].min()

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(
        history_station["timestamp"],
        history_station[target_column],
        color="#6b7280",
        linewidth=1.8,
        label="History",
    )
    ax.plot(
        tft_station["timestamp"],
        tft_station["plot_value"],
        color="#2563eb",
        linewidth=2.0,
        label="TFT",
    )
    ax.plot(
        chronos_station["timestamp"],
        chronos_station["plot_value"],
        color="#f59e0b",
        linewidth=2.0,
        label="Chronos",
    )
    ax.plot(
        cascade_station["timestamp"],
        cascade_station["plot_value"],
        color="#16a34a",
        linewidth=2.2,
        label="Cascade",
    )
    ax.axvline(forecast_start, color="#9ca3af", linestyle="--", linewidth=1.2)

    ax.set_title(f"Station {station_id} | {horizon_hours}h Forecast")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel(target_column)
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(loc="best", framealpha=0.9)
    fig.autofmt_xdate()
    fig.tight_layout()
    save_figure(fig, output_path, fmt)
    return True


def main() -> None:
    args = build_parser().parse_args()
    run_dir = args.run_dir
    if not run_dir.exists():
        raise FileNotFoundError(
            f"Run directory not found: {run_dir}\n"
            f"Please run local_pretrained_cascade_infer.py first to generate forecast outputs."
        )

    output_dir = args.output_dir if args.output_dir else run_dir / "plots"
    target_column = load_target_name(run_dir)
    station_ids = set(parse_str_list(args.stations)) if args.stations else None

    history_path = run_dir / "history_plot_input.csv"
    if not history_path.exists():
        history_path = run_dir / "history_input.csv"
    history_usecols = ["item_id", "timestamp", target_column]
    history_df = read_csv(history_path, usecols=history_usecols, station_ids=station_ids)

    tft_path = resolve_tft_csv(run_dir)
    tft_value_column = detect_value_column(tft_path, args.value_column)
    tft_df = read_csv(
        tft_path,
        usecols=["item_id", "timestamp", tft_value_column],
        station_ids=station_ids,
    )
    tft_df["plot_value"] = tft_df[tft_value_column]

    if args.horizons:
        horizons = parse_int_list(args.horizons)
    else:
        horizons = discover_horizons(run_dir)
    if not horizons:
        raise FileNotFoundError(f"No horizon_XXXh folders found in {run_dir}")

    if station_ids is None:
        station_list = sorted(history_df["item_id"].astype(str).unique().tolist())
    else:
        station_list = sorted(station_ids)

    total_saved = 0
    for horizon_hours in horizons:
        horizon_dir = run_dir / f"horizon_{horizon_hours:03d}h"
        chronos_csv = horizon_dir / f"chronos_{horizon_hours:03d}h_forecast.csv"
        cascade_csv = horizon_dir / f"cascade_{horizon_hours:03d}h_forecast.csv"
        if not chronos_csv.exists() or not cascade_csv.exists():
            print(f"Skipping {horizon_hours}h: missing forecast CSVs.")
            continue

        chronos_value_column = detect_value_column(chronos_csv, args.value_column)
        cascade_value_column = detect_value_column(cascade_csv, args.value_column)
        chronos_df = read_csv(
            chronos_csv,
            usecols=["item_id", "timestamp", chronos_value_column],
            station_ids=station_ids,
        )
        cascade_df = read_csv(
            cascade_csv,
            usecols=["item_id", "timestamp", cascade_value_column],
            station_ids=station_ids,
        )
        chronos_df["plot_value"] = chronos_df[chronos_value_column]
        cascade_df["plot_value"] = cascade_df[cascade_value_column]

        for station_id in station_list:
            output_path = output_dir / f"horizon_{horizon_hours:03d}h" / f"{station_id}_forecast"
            saved = plot_one_station(
                station_id=station_id,
                horizon_hours=horizon_hours,
                history_df=history_df,
                tft_df=tft_df,
                chronos_df=chronos_df,
                cascade_df=cascade_df,
                target_column=target_column,
                history_steps=args.history_steps,
                output_path=output_path,
                fmt=args.format,
            )
            if saved:
                total_saved += 1

    print("Done.")
    print(f"  Plot directory : {output_dir.resolve()}")
    print(f"  Figures saved  : {total_saved}")


if __name__ == "__main__":
    main()

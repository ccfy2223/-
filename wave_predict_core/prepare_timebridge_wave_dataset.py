from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


DYNAMIC_COLUMNS = [
    "WDIR", "WSPD", "GST", "WVHT", "DPD", "APD", "MWD",
    "PRES", "ATMP", "WTMP", "DEWP", "VIS", "TIDE",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert aligned multi-station wave CSVs into a wide TimeBridge dataset."
    )
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--output-csv", type=Path, required=True)
    p.add_argument("--metadata-path", type=Path, default=None)
    p.add_argument(
        "--features",
        type=str,
        default="WVHT",
        help="Comma-separated dynamic features to export, e.g. WVHT or WVHT,WSPD,DPD.",
    )
    p.add_argument(
        "--target-features",
        type=str,
        default="WVHT",
        help="Comma-separated features used as supervised targets for evaluation/loss.",
    )
    p.add_argument(
        "--limit-stations",
        type=int,
        default=0,
        help="If > 0, only use the first N sorted stations for quick smoke tests.",
    )
    return p.parse_args()


def load_shared_range(metadata_path: Path | None) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    if metadata_path is None or not metadata_path.exists():
        return None, None
    meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    return pd.Timestamp(meta["shared_start"]), pd.Timestamp(meta["shared_end"])


def read_station_frame(
    path: Path,
    selected_features: list[str],
    shared_start: pd.Timestamp | None,
    shared_end: pd.Timestamp | None,
) -> pd.DataFrame:
    station_id = path.stem.split("_")[0]
    header = pd.read_csv(path, nrows=0)
    available = [c for c in selected_features if c in header.columns]
    missing = [c for c in selected_features if c not in header.columns]
    if missing:
        raise ValueError(f"{station_id} is missing requested features: {missing}")

    df = pd.read_csv(
        path,
        usecols=["datetime"] + available,
        parse_dates=["datetime"],
        low_memory=False,
    )
    df = df.sort_values("datetime").drop_duplicates("datetime", keep="last").reset_index(drop=True)
    if shared_start is not None:
        df = df[(df["datetime"] >= shared_start) & (df["datetime"] <= shared_end)].reset_index(drop=True)

    renamed = {"datetime": "date"}
    renamed.update({feature: f"{station_id}_{feature}" for feature in available})
    df = df.rename(columns=renamed)

    value_cols = [c for c in df.columns if c != "date"]
    for col in value_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")
    if value_cols:
        df[value_cols] = df[value_cols].ffill().bfill()
    return df


def main() -> None:
    args = parse_args()
    selected_features = [x.strip() for x in args.features.split(",") if x.strip()]
    target_features = [x.strip() for x in args.target_features.split(",") if x.strip()]

    bad_features = [x for x in selected_features + target_features if x not in DYNAMIC_COLUMNS]
    if bad_features:
        raise ValueError(f"Unsupported features: {bad_features}. Allowed: {DYNAMIC_COLUMNS}")
    if any(x not in selected_features for x in target_features):
        raise ValueError("--target-features must be a subset of --features")

    shared_start, shared_end = load_shared_range(args.metadata_path)
    files = sorted(args.input_dir.glob("*.csv"))
    if args.limit_stations > 0:
        files = files[:args.limit_stations]
    if not files:
        raise FileNotFoundError(f"No CSV files found in {args.input_dir}")

    wide_df: pd.DataFrame | None = None
    station_ids: list[str] = []
    for path in files:
        station_id = path.stem.split("_")[0]
        station_ids.append(station_id)
        print(f"Loading {station_id} from {path.name} ...")
        station_df = read_station_frame(path, selected_features, shared_start, shared_end)
        if wide_df is None:
            wide_df = station_df
        else:
            wide_df = wide_df.merge(station_df, on="date", how="inner", validate="one_to_one")

    assert wide_df is not None
    wide_df = wide_df.sort_values("date").reset_index(drop=True)

    value_columns = [c for c in wide_df.columns if c != "date"]
    wide_df[value_columns] = wide_df[value_columns].ffill().bfill()

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    wide_df.to_csv(args.output_csv, index=False)

    target_columns = [
        f"{station_id}_{feature}"
        for station_id in station_ids
        for feature in target_features
        if f"{station_id}_{feature}" in wide_df.columns
    ]
    info = {
        "rows": int(len(wide_df)),
        "enc_in": int(len(value_columns)),
        "station_ids": station_ids,
        "selected_features": selected_features,
        "target_features": target_features,
        "target_columns": target_columns,
        "all_value_columns": value_columns,
        "date_start": str(wide_df["date"].iloc[0]),
        "date_end": str(wide_df["date"].iloc[-1]),
        "source_dir": str(args.input_dir),
        "output_csv": str(args.output_csv),
    }
    info_path = args.output_csv.with_suffix(args.output_csv.suffix + ".meta.json")
    info_path.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nDone.")
    print(f"Output CSV: {args.output_csv}")
    print(f"Rows: {info['rows']}")
    print(f"enc_in: {info['enc_in']}")
    print(f"Target columns ({len(target_columns)}): {target_columns[:8]}{' ...' if len(target_columns) > 8 else ''}")
    print(f"Meta JSON: {info_path}")


if __name__ == "__main__":
    main()

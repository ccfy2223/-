from __future__ import annotations

import gzip
import json
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data"
STATIONS_PATH = ROOT / "selected_stations.csv"
OUTPUT_ROOT = ROOT / "processed"
ALIGNED_STATION_ROOT = OUTPUT_ROOT / "aligned_stations"
WIDE_OUTPUT_PATH = OUTPUT_ROOT / "shared_timeline_10min.csv.gz"
METADATA_PATH = OUTPUT_ROOT / "shared_timeline_metadata.json"

FREQ = "10min"
INTERP_LIMIT_STEPS = 6

CANONICAL_NAME_MAP = {
    "#YY": "year",
    "YY": "year",
    "YYYY": "year",
    "#YR": "year",
    "YR": "year",
    "MM": "month",
    "DD": "day",
    "hh": "hour",
    "HH": "hour",
    "mm": "minute",
    "MN": "minute",
    "WD": "WDIR",
    "WDIR": "WDIR",
    "WSPD": "WSPD",
    "GST": "GST",
    "WVHT": "WVHT",
    "DPD": "DPD",
    "APD": "APD",
    "MWD": "MWD",
    "BAR": "PRES",
    "PRES": "PRES",
    "PRES.": "PRES",
    "ATMP": "ATMP",
    "WTMP": "WTMP",
    "DEWP": "DEWP",
    "VIS": "VIS",
    "TIDE": "TIDE",
}

FEATURE_COLUMNS = [
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

MISSING_SENTINELS = {
    "WDIR": {99.0, 999.0, 9999.0},
    "MWD": {99.0, 999.0, 9999.0},
    "WSPD": {99.0, 999.0, 9999.0},
    "GST": {99.0, 999.0, 9999.0},
    "WVHT": {99.0, 999.0, 9999.0},
    "DPD": {99.0, 999.0, 9999.0},
    "APD": {99.0, 999.0, 9999.0},
    "PRES": {99.0, 999.0, 9999.0},
    "ATMP": {99.0, 999.0, 9999.0},
    "WTMP": {99.0, 999.0, 9999.0},
    "DEWP": {99.0, 999.0, 9999.0},
    "VIS": {99.0, 999.0, 9999.0},
    "TIDE": {99.0, 999.0, 9999.0},
}


def read_station_table(station_id: str) -> pd.DataFrame:
    station_dir = DATA_ROOT / station_id
    if not station_dir.exists():
        raise FileNotFoundError(f"Station directory not found: {station_dir}")

    frames: list[pd.DataFrame] = []
    for gz_path in sorted(station_dir.glob("*/*.txt.gz")):
        frames.append(read_single_gzip(gz_path))

    if not frames:
        raise ValueError(f"No gzip files found for station {station_id}")

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values("datetime").drop_duplicates(subset=["datetime"], keep="last").reset_index(drop=True)
    return df


def read_single_gzip(path: Path) -> pd.DataFrame:
    with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as handle:
        raw_lines = [line.strip() for line in handle if line.strip()]

    header_line = raw_lines[0]
    columns = normalize_columns(header_line.split())
    data_start_idx = 1
    if header_line.startswith("#") and len(raw_lines) > 1 and raw_lines[1].startswith("#"):
        data_start_idx = 2

    data_text = "\n".join(raw_lines[data_start_idx:])
    frame = pd.read_csv(StringIO(data_text), sep=r"\s+", names=columns, engine="python")

    if "minute" not in frame.columns:
        frame["minute"] = 0

    frame["year"] = frame["year"].astype(int).map(normalize_year)
    datetime_parts = frame[["year", "month", "day", "hour", "minute"]].rename(
        columns={"year": "year", "month": "month", "day": "day", "hour": "hour", "minute": "minute"}
    )
    frame["datetime"] = pd.to_datetime(datetime_parts, errors="coerce", utc=False)
    frame = frame.dropna(subset=["datetime"]).copy()

    for column in FEATURE_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame.loc[frame[column].isin(MISSING_SENTINELS[column]), column] = np.nan

    return frame[["datetime", *FEATURE_COLUMNS]]


def normalize_columns(raw_columns: list[str]) -> list[str]:
    normalized = []
    for column in raw_columns:
        normalized.append(CANONICAL_NAME_MAP.get(column, column))
    return normalized


def normalize_year(year_value: int) -> int:
    if year_value >= 100:
        return year_value
    return 1900 + year_value if year_value >= 70 else 2000 + year_value


def align_station_frame(station_frame: pd.DataFrame, station_meta: pd.Series) -> pd.DataFrame:
    station_frame = station_frame.set_index("datetime").sort_index()
    full_index = pd.date_range(
        station_frame.index.min().floor(FREQ),
        station_frame.index.max().ceil(FREQ),
        freq=FREQ,
    )
    aligned = station_frame.reindex(full_index)
    aligned.index.name = "datetime"

    aligned = aligned.interpolate(
        method="time",
        limit=INTERP_LIMIT_STEPS,
        limit_direction="both",
        limit_area="inside",
    )

    aligned["station_id"] = station_meta["station_id"]
    aligned["region_key"] = station_meta["region_key"]
    aligned["station_name"] = station_meta.get("station_name", "")
    aligned["time_sin_hour"] = np.sin(2 * np.pi * (aligned.index.hour * 60 + aligned.index.minute) / (24 * 60))
    aligned["time_cos_hour"] = np.cos(2 * np.pi * (aligned.index.hour * 60 + aligned.index.minute) / (24 * 60))
    aligned["time_sin_doy"] = np.sin(2 * np.pi * aligned.index.dayofyear / 366.0)
    aligned["time_cos_doy"] = np.cos(2 * np.pi * aligned.index.dayofyear / 366.0)
    aligned["month"] = aligned.index.month
    aligned["day_of_week"] = aligned.index.dayofweek
    return aligned.reset_index()


def build_shared_timeline(aligned_station_frames: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict[str, str]]:
    start = max(frame["datetime"].min() for frame in aligned_station_frames.values())
    end = min(frame["datetime"].max() for frame in aligned_station_frames.values())
    shared_index = pd.date_range(start.ceil(FREQ), end.floor(FREQ), freq=FREQ)

    wide_frames = []
    feature_coverage: dict[str, str] = {}
    for station_id, frame in aligned_station_frames.items():
        station_frame = frame.set_index("datetime").reindex(shared_index)
        station_features = station_frame[FEATURE_COLUMNS].add_prefix(f"{station_id}_")
        wide_frames.append(station_features)
        feature_coverage[station_id] = f"{station_frame.index.min()} -> {station_frame.index.max()}"

    wide = pd.concat(wide_frames, axis=1)
    wide.index.name = "datetime"
    time_features = pd.DataFrame(
        {
            "time_sin_hour": np.sin(2 * np.pi * (wide.index.hour * 60 + wide.index.minute) / (24 * 60)),
            "time_cos_hour": np.cos(2 * np.pi * (wide.index.hour * 60 + wide.index.minute) / (24 * 60)),
            "time_sin_doy": np.sin(2 * np.pi * wide.index.dayofyear / 366.0),
            "time_cos_doy": np.cos(2 * np.pi * wide.index.dayofyear / 366.0),
            "month": wide.index.month,
            "day_of_week": wide.index.dayofweek,
        },
        index=wide.index,
    )
    wide = pd.concat([wide, time_features], axis=1)
    return wide.reset_index(), feature_coverage


def main() -> None:
    stations = pd.read_csv(STATIONS_PATH, encoding="utf-8-sig")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    ALIGNED_STATION_ROOT.mkdir(parents=True, exist_ok=True)

    aligned_station_frames: dict[str, pd.DataFrame] = {}
    station_stats: list[dict[str, object]] = []

    for _, station_meta in stations.iterrows():
        station_id = str(station_meta["station_id"])
        raw_frame = read_station_table(station_id)
        aligned_frame = align_station_frame(raw_frame, station_meta)
        aligned_station_frames[station_id] = aligned_frame

        output_path = ALIGNED_STATION_ROOT / f"{station_id}_aligned_10min.csv.gz"
        aligned_frame.to_csv(output_path, index=False, compression="gzip")
        station_stats.append(
            {
                "station_id": station_id,
                "region_key": station_meta["region_key"],
                "raw_rows": int(raw_frame.shape[0]),
                "aligned_rows": int(aligned_frame.shape[0]),
                "start": aligned_frame["datetime"].min().isoformat(),
                "end": aligned_frame["datetime"].max().isoformat(),
                "output_path": str(output_path),
            }
        )
        print(
            f"{station_id}: raw={raw_frame.shape[0]} rows, aligned={aligned_frame.shape[0]} rows, "
            f"range={aligned_frame['datetime'].min()} -> {aligned_frame['datetime'].max()}"
        )

    wide_frame, coverage = build_shared_timeline(aligned_station_frames)
    wide_frame.to_csv(WIDE_OUTPUT_PATH, index=False, compression="gzip")

    metadata = {
        "frequency": FREQ,
        "interpolation_limit_steps": INTERP_LIMIT_STEPS,
        "feature_columns": FEATURE_COLUMNS,
        "shared_output_path": str(WIDE_OUTPUT_PATH),
        "shared_start": wide_frame["datetime"].min().isoformat(),
        "shared_end": wide_frame["datetime"].max().isoformat(),
        "shared_rows": int(wide_frame.shape[0]),
        "stations": station_stats,
        "coverage": coverage,
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print()
    print(f"Shared timeline saved to: {WIDE_OUTPUT_PATH}")
    print(f"Metadata saved to: {METADATA_PATH}")
    print(f"Shared rows: {wide_frame.shape[0]}")


if __name__ == "__main__":
    main()

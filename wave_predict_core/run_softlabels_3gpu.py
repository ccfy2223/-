from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd


REQUIRED_SPLITS = {"train", "val", "test"}
DEFAULT_STATION_GROUPS = [
    ["41010", "41043", "42040"],
    ["44025", "46026", "46050"],
    ["46061", "51001"],
]
DEFAULT_STRIDE_BY_HORIZON = {
    12: 24,
    24: 24,
    48: 48,
    72: 48,
    120: 48,
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate Chronos soft labels with 3 GPUs by splitting stations into 3 groups."
    )
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--metadata-path", type=Path, required=True)
    p.add_argument("--predictor-root", type=Path, required=True,
                   help="Root directory that contains horizon_XXXh/model subdirectories.")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Directory to save merged labels_XXXh.csv files.")
    p.add_argument("--generator-script", type=Path, default=Path("generate_chronos_fullseq_labels.py"))
    p.add_argument("--horizons", default="12,24,120",
                   help="Comma-separated horizons to generate.")
    p.add_argument("--context-hours", type=int, default=168)
    p.add_argument("--freq", default="10min")
    p.add_argument("--train-ratio", type=float, default=0.70)
    p.add_argument("--val-ratio", type=float, default=0.20)
    p.add_argument("--splits", default="train,val,test")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--gpu-ids", default="0,1,2",
                   help="Exactly 3 GPU ids, for example 0,1,2")
    p.add_argument("--station-groups", default="",
                   help="Optional custom station groups. Format: 41010+41043+42040,44025+46026+46050,46061+51001")
    p.add_argument("--workspace-dir", type=Path, default=Path("softlabel_3gpu_workspace"))
    p.add_argument("--keep-parts", action="store_true",
                   help="Keep per-GPU part csv files after merge.")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip a horizon if merged labels_XXXh.csv already exists and is valid.")
    return p


def parse_csv_ints(raw: str) -> list[int]:
    values = []
    for part in raw.split(","):
        part = part.strip()
        if part:
            values.append(int(part))
    if not values:
        raise ValueError("At least one horizon is required.")
    return values


def parse_gpu_ids(raw: str) -> list[str]:
    gpu_ids = [part.strip() for part in raw.split(",") if part.strip()]
    if len(gpu_ids) != 3:
        raise ValueError("--gpu-ids must contain exactly 3 entries.")
    return gpu_ids


def parse_station_groups(raw: str) -> list[list[str]]:
    if not raw.strip():
        return DEFAULT_STATION_GROUPS
    groups = []
    for group_raw in raw.split(","):
        stations = [item.strip() for item in group_raw.split("+") if item.strip()]
        if not stations:
            continue
        groups.append(stations)
    if len(groups) != 3:
        raise ValueError("--station-groups must define exactly 3 groups.")
    return groups


def stride_for_horizon(horizon: int) -> int:
    return DEFAULT_STRIDE_BY_HORIZON.get(horizon, max(horizon, 24))


def predictor_path(predictor_root: Path, horizon: int) -> Path:
    return predictor_root / f"horizon_{horizon:03d}h" / "model"


def merged_output_path(output_dir: Path, horizon: int) -> Path:
    return output_dir / f"labels_{horizon:03d}h.csv"


def part_output_path(output_dir: Path, horizon: int, gpu_index: int) -> Path:
    return output_dir / f"labels_{horizon:03d}h.part_gpu{gpu_index}.csv"


def validate_labels_csv(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, "missing"
    if path.stat().st_size == 0:
        return False, "empty"

    try:
        df = pd.read_csv(path, usecols=["item_id", "split", "timestamp", "chronos_pred"])
    except Exception as exc:
        return False, f"bad_csv: {exc}"

    if df.empty:
        return False, "no_rows"

    splits = set(df["split"].dropna().astype(str).unique())
    missing = sorted(REQUIRED_SPLITS - splits)
    if missing:
        return False, f"missing_splits:{','.join(missing)}"

    if df["item_id"].isna().any():
        return False, "null_item_id"
    if df["timestamp"].isna().any():
        return False, "null_timestamp"
    if df["chronos_pred"].isna().all():
        return False, "all_predictions_nan"

    return True, f"ok rows={len(df)} items={df['item_id'].nunique()}"


def station_files_by_id(input_dir: Path) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for path in sorted(input_dir.glob("*.csv")):
        station_id = path.stem.split("_")[0]
        mapping[station_id] = path
    if not mapping:
        raise FileNotFoundError(f"No station csv files found in {input_dir}")
    return mapping


def prepare_group_dirs(workspace_dir: Path, input_dir: Path, station_groups: list[list[str]]) -> list[Path]:
    station_map = station_files_by_id(input_dir)
    workspace_dir.mkdir(parents=True, exist_ok=True)
    group_dirs: list[Path] = []

    for group_index, station_ids in enumerate(station_groups):
        group_dir = workspace_dir / f"stations_gpu{group_index}"
        if group_dir.exists():
            shutil.rmtree(group_dir)
        group_dir.mkdir(parents=True, exist_ok=True)

        for station_id in station_ids:
            if station_id not in station_map:
                raise FileNotFoundError(f"Station {station_id} not found in {input_dir}")
            source = station_map[station_id]
            target = group_dir / source.name
            target.symlink_to(source.resolve())

        group_dirs.append(group_dir)

    return group_dirs


def build_worker_command(
    args: argparse.Namespace,
    group_dir: Path,
    horizon: int,
    gpu_id: str,
    gpu_index: int,
) -> list[str]:
    cmd = [
        sys.executable,
        str(args.generator_script),
        "--input-dir", str(group_dir),
        "--metadata-path", str(args.metadata_path),
        "--predictor-path", str(predictor_path(args.predictor_root, horizon)),
        "--output-path", str(part_output_path(args.output_dir, horizon, gpu_index)),
        "--horizon-hours", str(horizon),
        "--context-hours", str(args.context_hours),
        "--stride-hours", str(stride_for_horizon(horizon)),
        "--freq", args.freq,
        "--gpu-id", gpu_id,
        "--train-ratio", str(args.train_ratio),
        "--val-ratio", str(args.val_ratio),
        "--splits", args.splits,
        "--batch-size", str(args.batch_size),
    ]
    return cmd


def run_workers_for_horizon(args: argparse.Namespace, group_dirs: list[Path], gpu_ids: list[str], horizon: int) -> None:
    processes: list[tuple[int, subprocess.Popen[bytes], Path]] = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = args.workspace_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    for gpu_index, (group_dir, gpu_id) in enumerate(zip(group_dirs, gpu_ids)):
        cmd = build_worker_command(args, group_dir, horizon, gpu_id, gpu_index)
        log_path = log_dir / f"softlabels_{horizon:03d}h_gpu{gpu_index}.log"
        print(f"[RUN] horizon={horizon}h gpu={gpu_id} group_dir={group_dir}")
        print("      " + " ".join(cmd))
        log_handle = log_path.open("w", encoding="utf-8")
        proc = subprocess.Popen(cmd, stdout=log_handle, stderr=subprocess.STDOUT)
        processes.append((gpu_index, proc, log_path))

    failed = []
    for gpu_index, proc, log_path in processes:
        code = proc.wait()
        if code != 0:
            failed.append((gpu_index, code, log_path))

    if failed:
        details = ", ".join([f"gpu{gpu_index}:code={code}:log={log_path}" for gpu_index, code, log_path in failed])
        raise RuntimeError(f"Soft label generation failed for horizon={horizon}h: {details}")


def merge_parts(output_dir: Path, horizon: int, keep_parts: bool) -> Path:
    parts = [part_output_path(output_dir, horizon, i) for i in range(3)]
    frames = []
    for part in parts:
        ok, message = validate_labels_csv(part)
        if not ok:
            raise RuntimeError(f"Part file invalid: {part} ({message})")
        frames.append(pd.read_csv(part))

    out = pd.concat(frames, ignore_index=True)
    out["item_id"] = out["item_id"].astype(str)
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    out = out.sort_values(["item_id", "split", "timestamp"]).reset_index(drop=True)

    merged_path = merged_output_path(output_dir, horizon)
    out.to_csv(merged_path, index=False)

    if not keep_parts:
        for part in parts:
            part.unlink(missing_ok=True)

    return merged_path


def run_one_horizon(args: argparse.Namespace, group_dirs: list[Path], gpu_ids: list[str], horizon: int) -> None:
    merged_path = merged_output_path(args.output_dir, horizon)
    if args.skip_existing:
        ok, message = validate_labels_csv(merged_path)
        if ok:
            print(f"[SKIP] horizon={horizon}h {merged_path.name} {message}")
            return

    pred_path = predictor_path(args.predictor_root, horizon)
    if not pred_path.exists():
        raise FileNotFoundError(f"Predictor path not found for horizon={horizon}h: {pred_path}")

    run_workers_for_horizon(args, group_dirs, gpu_ids, horizon)
    merged_path = merge_parts(args.output_dir, horizon, args.keep_parts)
    ok, message = validate_labels_csv(merged_path)
    if not ok:
        raise RuntimeError(f"Merged labels invalid for horizon={horizon}h: {message}")
    print(f"[DONE] horizon={horizon}h {merged_path.name} {message}")


def main() -> None:
    args = build_parser().parse_args()
    horizons = parse_csv_ints(args.horizons)
    gpu_ids = parse_gpu_ids(args.gpu_ids)
    station_groups = parse_station_groups(args.station_groups)
    group_dirs = prepare_group_dirs(args.workspace_dir, args.input_dir, station_groups)

    print(f"Horizons: {horizons}")
    print(f"GPU ids: {gpu_ids}")
    print(f"Station groups: {station_groups}")
    print(f"Workspace: {args.workspace_dir}")
    print(f"Output dir: {args.output_dir}")

    for horizon in horizons:
        run_one_horizon(args, group_dirs, gpu_ids, horizon)

    print("[ALL DONE] Soft label generation finished.")


if __name__ == "__main__":
    main()

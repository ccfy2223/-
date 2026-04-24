from __future__ import annotations

import argparse
import json
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


def resolve_path(path: Path, base_dir: Path) -> Path:
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate Chronos soft labels on 3 GPUs, then run distillation on 3 GPUs."
    )
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--metadata-path", type=Path, required=True)
    p.add_argument("--predictor-root", type=Path, required=True,
                   help="Directory containing horizon_XXXh/model folders for soft-label generation.")
    p.add_argument("--soft-labels-dir", type=Path, required=True)
    p.add_argument("--distill-output-dir", type=Path, required=True)
    p.add_argument("--generator-script", type=Path, default=Path("generate_chronos_fullseq_labels.py"))
    p.add_argument("--distill-script", type=Path, default=Path("train_distill_batch.py"))
    p.add_argument("--softlabel-horizons", default="12,24,48,72,120")
    p.add_argument("--distill-horizons", default="12,24,48,72,120")
    p.add_argument("--context-hours", type=int, default=168)
    p.add_argument("--freq", default="10min")
    p.add_argument("--train-ratio", type=float, default=0.70)
    p.add_argument("--val-ratio", type=float, default=0.20)
    p.add_argument("--splits", default="train,val,test")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--alphas", default="0.3,0.5,0.7,1.0")
    p.add_argument("--time-limit", type=int, default=3600)
    p.add_argument("--limit-stations", type=int, default=0)
    p.add_argument("--gpu-ids", default="0,1,2",
                   help="Exactly 3 GPU ids, for example 0,1,2")
    p.add_argument("--station-groups", default="",
                   help="Optional custom station groups. Format: 41010+41043+42040,44025+46026+46050,46061+51001")
    p.add_argument("--workspace-dir", type=Path, default=Path("pipeline_3gpu_workspace"))
    p.add_argument("--keep-parts", action="store_true")
    p.add_argument("--skip-existing-softlabels", action="store_true")
    p.add_argument("--skip-existing-distill", action="store_true")
    p.add_argument("--_worker-mode", choices=["distill"], default=None)
    p.add_argument("--_worker-gpu-id", default=None)
    p.add_argument("--_worker-horizons", default=None)
    p.add_argument("--_worker-index", type=int, default=None)
    return p


def parse_int_list(raw: str, exclude_six: bool = False) -> list[int]:
    values = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if exclude_six and value == 6:
            continue
        values.append(value)
    if not values:
        raise ValueError("No valid horizons were provided.")
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
        if stations:
            groups.append(stations)
    if len(groups) != 3:
        raise ValueError("--station-groups must define exactly 3 groups.")
    return groups


def stride_for_horizon(horizon: int) -> int:
    return DEFAULT_STRIDE_BY_HORIZON.get(horizon, max(horizon, 24))


def softlabel_predictor_path(predictor_root: Path, horizon: int) -> Path:
    return predictor_root / f"horizon_{horizon:03d}h" / "model"


def merged_softlabel_path(soft_labels_dir: Path, horizon: int) -> Path:
    return soft_labels_dir / f"labels_{horizon:03d}h.csv"


def part_softlabel_path(soft_labels_dir: Path, horizon: int, gpu_index: int) -> Path:
    return soft_labels_dir / f"labels_{horizon:03d}h.part_gpu{gpu_index}.csv"


def distill_summary_path(distill_output_dir: Path, horizon: int) -> Path:
    return distill_output_dir / f"horizon_{horizon:03d}h" / "summary.json"


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


def build_softlabel_command(
    args: argparse.Namespace,
    group_dir: Path,
    horizon: int,
    gpu_id: str,
    gpu_index: int,
) -> list[str]:
    cmd = [
        sys.executable,
        "-u",
        str(args.generator_script),
        "--input-dir", str(group_dir),
        "--metadata-path", str(args.metadata_path),
        "--predictor-path", str(softlabel_predictor_path(args.predictor_root, horizon)),
        "--output-path", str(part_softlabel_path(args.soft_labels_dir, horizon, gpu_index)),
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
    if args.limit_stations > 0:
        cmd.extend(["--limit-stations", str(args.limit_stations)])
    return cmd


def run_softlabel_workers(args: argparse.Namespace, group_dirs: list[Path], gpu_ids: list[str], horizon: int) -> None:
    log_dir = args.workspace_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    processes: list[tuple[int, subprocess.Popen[bytes], object, Path]] = []

    for gpu_index, (group_dir, gpu_id) in enumerate(zip(group_dirs, gpu_ids)):
        cmd = build_softlabel_command(args, group_dir, horizon, gpu_id, gpu_index)
        log_path = log_dir / f"softlabels_{horizon:03d}h_gpu{gpu_index}.log"
        print(f"[SOFTLABEL RUN] horizon={horizon}h gpu={gpu_id}")
        print("  " + " ".join(cmd))
        log_handle = log_path.open("w", encoding="utf-8")
        proc = subprocess.Popen(cmd, stdout=log_handle, stderr=subprocess.STDOUT)
        processes.append((gpu_index, proc, log_handle, log_path))

    failures = []
    for gpu_index, proc, log_handle, log_path in processes:
        code = proc.wait()
        log_handle.close()
        if code != 0:
            failures.append((gpu_index, code, log_path))

    if failures:
        details = ", ".join(
            [f"gpu{gpu_index}:code={code}:log={log_path}" for gpu_index, code, log_path in failures]
        )
        raise RuntimeError(f"Soft-label generation failed for horizon={horizon}h: {details}")


def merge_softlabel_parts(args: argparse.Namespace, horizon: int) -> Path:
    parts = [part_softlabel_path(args.soft_labels_dir, horizon, i) for i in range(3)]
    frames = []
    for part in parts:
        ok, message = validate_labels_csv(part)
        if not ok:
            raise RuntimeError(f"Part file invalid: {part} ({message})")
        frames.append(pd.read_csv(part))

    merged = pd.concat(frames, ignore_index=True)
    merged["item_id"] = merged["item_id"].astype(str)
    merged["timestamp"] = pd.to_datetime(merged["timestamp"])
    merged = merged.sort_values(["item_id", "split", "timestamp"]).reset_index(drop=True)

    merged_path = merged_softlabel_path(args.soft_labels_dir, horizon)
    merged_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(merged_path, index=False)

    if not args.keep_parts:
        for part in parts:
            part.unlink(missing_ok=True)

    return merged_path


def run_softlabels_stage(args: argparse.Namespace, group_dirs: list[Path], gpu_ids: list[str], horizons: list[int]) -> None:
    for horizon in horizons:
        merged_path = merged_softlabel_path(args.soft_labels_dir, horizon)
        if args.skip_existing_softlabels:
            ok, message = validate_labels_csv(merged_path)
            if ok:
                print(f"[SOFTLABEL SKIP] horizon={horizon}h {merged_path.name} {message}")
                continue

        pred_path = softlabel_predictor_path(args.predictor_root, horizon)
        if not pred_path.exists():
            raise FileNotFoundError(f"Predictor path not found for horizon={horizon}h: {pred_path}")

        run_softlabel_workers(args, group_dirs, gpu_ids, horizon)
        merged_path = merge_softlabel_parts(args, horizon)
        ok, message = validate_labels_csv(merged_path)
        if not ok:
            raise RuntimeError(f"Merged soft labels invalid for horizon={horizon}h: {message}")
        print(f"[SOFTLABEL DONE] horizon={horizon}h {merged_path.name} {message}")


def build_distill_command(args: argparse.Namespace, horizon: int, gpu_id: str) -> list[str]:
    cmd = [
        sys.executable,
        "-u",
        str(args.distill_script),
        "--input-dir", str(args.input_dir),
        "--soft-labels-dir", str(args.soft_labels_dir),
        "--metadata-path", str(args.metadata_path),
        "--output-dir", str(args.distill_output_dir),
        "--horizon-hours", str(horizon),
        "--alphas", args.alphas,
        "--context-hours", str(args.context_hours),
        "--freq", args.freq,
        "--train-ratio", str(args.train_ratio),
        "--val-ratio", str(args.val_ratio),
        "--time-limit", str(args.time_limit),
        "--gpu-id", gpu_id,
    ]
    if args.limit_stations > 0:
        cmd.extend(["--limit-stations", str(args.limit_stations)])
    return cmd


def assign_horizons_to_gpus(horizons: list[int], gpu_ids: list[str]) -> dict[str, list[int]]:
    assignments = {gpu_id: [] for gpu_id in gpu_ids}
    for index, horizon in enumerate(horizons):
        gpu_id = gpu_ids[index % len(gpu_ids)]
        assignments[gpu_id].append(horizon)
    return assignments


def run_distill_worker(args: argparse.Namespace, gpu_id: str, horizons: list[int], worker_index: int) -> None:
    log_dir = args.workspace_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    queue_log = log_dir / f"distill_gpu{worker_index}.log"
    with queue_log.open("w", encoding="utf-8") as queue_handle:
        queue_handle.write(f"gpu={gpu_id} horizons={horizons}\n")
        queue_handle.flush()

        for horizon in horizons:
            label_ok, label_message = validate_labels_csv(merged_softlabel_path(args.soft_labels_dir, horizon))
            if not label_ok:
                raise RuntimeError(f"Soft labels not ready for horizon={horizon}h: {label_message}")

            summary = distill_summary_path(args.distill_output_dir, horizon)
            if args.skip_existing_distill and summary.exists():
                queue_handle.write(f"[SKIP] horizon={horizon}h summary exists: {summary}\n")
                queue_handle.flush()
                continue

            cmd = build_distill_command(args, horizon, gpu_id)
            horizon_log = log_dir / f"distill_{horizon:03d}h_gpu{worker_index}.log"
            queue_handle.write("[RUN] " + " ".join(cmd) + "\n")
            queue_handle.flush()
            with horizon_log.open("w", encoding="utf-8") as horizon_handle:
                result = subprocess.run(cmd, stdout=horizon_handle, stderr=subprocess.STDOUT, check=False)

            if result.returncode != 0:
                raise RuntimeError(
                    f"Distill failed for horizon={horizon}h on gpu={gpu_id}. Log: {horizon_log}"
                )
            if not summary.exists():
                raise FileNotFoundError(f"summary.json not found for horizon={horizon}h: {summary}")

            try:
                payload = json.loads(summary.read_text(encoding="utf-8"))
            except Exception as exc:
                raise RuntimeError(f"Failed to parse summary for horizon={horizon}h: {exc}") from exc

            queue_handle.write(
                f"[DONE] horizon={horizon}h results={len(payload.get('results', []))} summary={summary}\n"
            )
            queue_handle.flush()


def run_distill_stage(args: argparse.Namespace, gpu_ids: list[str], horizons: list[int]) -> None:
    assignments = assign_horizons_to_gpus(horizons, gpu_ids)
    print(f"[DISTILL ASSIGNMENTS] {assignments}")

    processes: list[tuple[int, str, subprocess.Popen[bytes], object, Path]] = []
    log_dir = args.workspace_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    for worker_index, gpu_id in enumerate(gpu_ids):
        queue = assignments[gpu_id]
        if not queue:
            continue
        log_path = log_dir / f"distill_worker_gpu{worker_index}.log"
        log_handle = log_path.open("w", encoding="utf-8")
        proc = subprocess.Popen(
            [
                sys.executable,
                "-u",
                str(Path(__file__).resolve()),
                *sys.argv[1:],
                "--_worker-mode", "distill",
                "--_worker-gpu-id", gpu_id,
                "--_worker-horizons", ",".join(str(h) for h in queue),
                "--_worker-index", str(worker_index),
            ],
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        processes.append((worker_index, gpu_id, proc, log_handle, log_path))

    failures = []
    for worker_index, gpu_id, proc, log_handle, log_path in processes:
        code = proc.wait()
        log_handle.close()
        if code != 0:
            failures.append((worker_index, gpu_id, code, log_path))

    if failures:
        details = ", ".join(
            [
                f"worker={worker_index}:gpu={gpu_id}:code={code}:log={log_path}"
                for worker_index, gpu_id, code, log_path in failures
            ]
        )
        raise RuntimeError(f"Distill stage failed: {details}")


def main() -> None:
    args = build_parser().parse_args()
    base_dir = Path(__file__).resolve().parent
    args.workspace_dir = resolve_path(args.workspace_dir, base_dir)
    args.generator_script = resolve_path(args.generator_script, base_dir)
    args.distill_script = resolve_path(args.distill_script, base_dir)
    args.input_dir = resolve_path(args.input_dir, base_dir)
    args.metadata_path = resolve_path(args.metadata_path, base_dir)
    args.predictor_root = resolve_path(args.predictor_root, base_dir)
    args.soft_labels_dir = resolve_path(args.soft_labels_dir, base_dir)
    args.distill_output_dir = resolve_path(args.distill_output_dir, base_dir)

    if args._worker_mode == "distill":
        if args._worker_gpu_id is None or args._worker_horizons is None or args._worker_index is None:
            raise ValueError("Missing internal distill worker arguments.")
        worker_horizons = parse_int_list(args._worker_horizons, exclude_six=False)
        run_distill_worker(args, args._worker_gpu_id, worker_horizons, args._worker_index)
        return

    softlabel_horizons = parse_int_list(args.softlabel_horizons, exclude_six=True)
    distill_horizons = parse_int_list(args.distill_horizons, exclude_six=True)
    gpu_ids = parse_gpu_ids(args.gpu_ids)
    station_groups = parse_station_groups(args.station_groups)
    group_dirs = prepare_group_dirs(args.workspace_dir, args.input_dir, station_groups)

    print(f"Softlabel horizons: {softlabel_horizons}")
    print(f"Distill horizons: {distill_horizons}")
    print(f"GPU ids: {gpu_ids}")
    print(f"Station groups: {station_groups}")
    print(f"Workspace: {args.workspace_dir}")
    print(f"Soft labels dir: {args.soft_labels_dir}")
    print(f"Distill output dir: {args.distill_output_dir}")

    run_softlabels_stage(args, group_dirs, gpu_ids, softlabel_horizons)
    run_distill_stage(args, gpu_ids, distill_horizons)
    print("[ALL DONE] Soft labels and distill pipeline finished.")


if __name__ == "__main__":
    main()

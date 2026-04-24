from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


REQUIRED_SPLITS = {"train", "val", "test"}
DEFAULT_HORIZONS = "3,6,12"
DEFAULT_GPU_IDS = "0,1,2"


def resolve_path(path: Path, base_dir: Path) -> Path:
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run short-horizon Chronos soft labels and distillation with 3 GPUs, one horizon per GPU."
    )
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--metadata-path", type=Path, required=True)
    p.add_argument("--predictor-root", type=Path, required=True,
                   help="Directory containing horizon_XXXh/model folders.")
    p.add_argument("--soft-labels-dir", type=Path, required=True)
    p.add_argument("--distill-output-dir", type=Path, required=True)
    p.add_argument("--generator-script", type=Path, default=Path("generate_chronos_fullseq_labels.py"))
    p.add_argument("--distill-script", type=Path, default=Path("train_distill_batch.py"))
    p.add_argument("--horizons", default=DEFAULT_HORIZONS,
                   help="Comma-separated horizons. Defaults to 3,6,12.")
    p.add_argument("--gpu-ids", default=DEFAULT_GPU_IDS,
                   help="Comma-separated GPU ids for soft labels. Defaults to 0,1,2.")
    p.add_argument("--context-hours", type=int, default=168)
    p.add_argument("--freq", default="10min")
    p.add_argument("--train-ratio", type=float, default=0.70)
    p.add_argument("--val-ratio", type=float, default=0.20)
    p.add_argument("--splits", default="train")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--alphas", default="0.3,0.5,0.7,1.0")
    p.add_argument("--time-limit", type=int, default=3600)
    p.add_argument("--distill-gpu-id", default="3",
                   help="GPU id reserved for distillation. Defaults to 3.")
    p.add_argument("--speedup-factor", type=float, default=2.5)
    p.add_argument("--limit-stations", type=int, default=0)
    p.add_argument("--workspace-dir", type=Path, default=Path("pipeline_3gpu_short_workspace"))
    p.add_argument("--skip-existing-softlabels", action="store_true")
    p.add_argument("--skip-existing-distill", action="store_true")
    p.add_argument("--_worker-mode", choices=["pipeline"], default=None)
    p.add_argument("--_worker-gpu-id", default=None)
    p.add_argument("--_worker-horizon", type=int, default=None)
    p.add_argument("--_worker-index", type=int, default=None)
    return p


def parse_int_list(raw: str) -> list[int]:
    values = []
    for part in raw.split(","):
        part = part.strip()
        if part:
            values.append(int(part))
    if not values:
        raise ValueError("No valid integer values were provided.")
    return values


def parse_gpu_ids(raw: str) -> list[str]:
    gpu_ids = [part.strip() for part in raw.split(",") if part.strip()]
    if not gpu_ids:
        raise ValueError("No GPU ids were provided.")
    return gpu_ids


def validate_pairing(horizons: list[int], gpu_ids: list[str]) -> None:
    if len(horizons) != len(gpu_ids):
        raise ValueError(
            f"--horizons count ({len(horizons)}) must match --gpu-ids count ({len(gpu_ids)})."
        )


def stride_for_horizon(horizon: int) -> int:
    mapping = {
        3: 12,
        6: 12,
        12: 24,
    }
    if horizon in mapping:
        return mapping[horizon]
    if horizon <= 24:
        return max(horizon, 12)
    return min(horizon, 48)


def softlabel_predictor_path(predictor_root: Path, horizon: int) -> Path:
    return predictor_root / f"horizon_{horizon:03d}h" / "model"


def merged_softlabel_path(soft_labels_dir: Path, horizon: int) -> Path:
    return soft_labels_dir / f"labels_{horizon:03d}h.csv"


def distill_summary_path(distill_output_dir: Path, horizon: int) -> Path:
    return distill_output_dir / f"horizon_{horizon:03d}h" / "summary.json"


def validate_labels_csv(path: Path, required_splits: set[str] | None = None) -> tuple[bool, str]:
    if required_splits is None:
        required_splits = REQUIRED_SPLITS
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
    missing = sorted(required_splits - splits)
    if missing:
        return False, f"missing_splits:{','.join(missing)}"

    if df["chronos_pred"].isna().all():
        return False, "all_predictions_nan"

    return True, f"ok rows={len(df)} items={df['item_id'].nunique()}"


def build_softlabel_command(args: argparse.Namespace, horizon: int, gpu_id: str) -> list[str]:
    cmd = [
        sys.executable,
        "-u",
        str(args.generator_script),
        "--input-dir", str(args.input_dir),
        "--metadata-path", str(args.metadata_path),
        "--predictor-path", str(softlabel_predictor_path(args.predictor_root, horizon)),
        "--output-path", str(merged_softlabel_path(args.soft_labels_dir, horizon)),
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
        "--speedup-factor", str(args.speedup_factor),
    ]
    if args.limit_stations > 0:
        cmd.extend(["--limit-stations", str(args.limit_stations)])
    return cmd


def run_pipeline_worker(args: argparse.Namespace, gpu_id: str, horizon: int, worker_index: int) -> None:
    log_dir = args.workspace_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    queue_log = log_dir / f"pipeline_gpu{worker_index}.log"
    softlabel_log = log_dir / f"softlabels_{horizon:03d}h_gpu{worker_index}.log"
    distill_log = log_dir / f"distill_{horizon:03d}h_gpu{worker_index}.log"
    required_splits = {s.strip() for s in args.splits.split(",") if s.strip()}

    labels_path = merged_softlabel_path(args.soft_labels_dir, horizon)
    summary_path = distill_summary_path(args.distill_output_dir, horizon)

    with queue_log.open("w", encoding="utf-8") as queue_handle:
        queue_handle.write(f"gpu={gpu_id} horizon={horizon}\n")
        queue_handle.flush()

        if args.skip_existing_softlabels:
            ok, message = validate_labels_csv(labels_path, required_splits)
            if ok:
                queue_handle.write(f"[SOFTLABEL SKIP] {labels_path} {message}\n")
                queue_handle.flush()
            else:
                predictor_path = softlabel_predictor_path(args.predictor_root, horizon)
                if not predictor_path.exists():
                    raise FileNotFoundError(f"Predictor path not found for horizon={horizon}h: {predictor_path}")

                cmd = build_softlabel_command(args, horizon, gpu_id)
                queue_handle.write("[SOFTLABEL RUN] " + " ".join(cmd) + "\n")
                queue_handle.flush()
                with softlabel_log.open("w", encoding="utf-8") as handle:
                    result = subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT, check=False)
                if result.returncode != 0:
                    raise RuntimeError(
                        f"Soft-label generation failed for horizon={horizon}h on gpu={gpu_id}. Log: {softlabel_log}"
                    )
                ok, message = validate_labels_csv(labels_path, required_splits)
                if not ok:
                    raise RuntimeError(f"Soft labels invalid for horizon={horizon}h: {message}")
                queue_handle.write(f"[SOFTLABEL DONE] {labels_path} {message}\n")
                queue_handle.flush()
        else:
            predictor_path = softlabel_predictor_path(args.predictor_root, horizon)
            if not predictor_path.exists():
                raise FileNotFoundError(f"Predictor path not found for horizon={horizon}h: {predictor_path}")

            cmd = build_softlabel_command(args, horizon, gpu_id)
            queue_handle.write("[SOFTLABEL RUN] " + " ".join(cmd) + "\n")
            queue_handle.flush()
            with softlabel_log.open("w", encoding="utf-8") as handle:
                result = subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT, check=False)
            if result.returncode != 0:
                raise RuntimeError(
                    f"Soft-label generation failed for horizon={horizon}h on gpu={gpu_id}. Log: {softlabel_log}"
                )
            ok, message = validate_labels_csv(labels_path, required_splits)
            if not ok:
                raise RuntimeError(f"Soft labels invalid for horizon={horizon}h: {message}")
            queue_handle.write(f"[SOFTLABEL DONE] {labels_path} {message}\n")
            queue_handle.flush()

        if args.skip_existing_distill and summary_path.exists():
            queue_handle.write(f"[DISTILL SKIP] summary exists: {summary_path}\n")
            queue_handle.flush()
            return

        cmd = build_distill_command(args, horizon, args.distill_gpu_id)
        queue_handle.write("[DISTILL RUN] " + " ".join(cmd) + "\n")
        queue_handle.flush()
        with distill_log.open("w", encoding="utf-8") as handle:
            result = subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"Distill failed for horizon={horizon}h on gpu={gpu_id}. Log: {distill_log}")
        if not summary_path.exists():
            raise FileNotFoundError(f"summary.json not found for horizon={horizon}h: {summary_path}")

        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise RuntimeError(f"Failed to parse summary for horizon={horizon}h: {exc}") from exc

        queue_handle.write(
            f"[DISTILL DONE] horizon={horizon}h results={len(payload.get('results', []))} summary={summary_path}\n"
        )
        queue_handle.flush()


def run_all_workers(args: argparse.Namespace, horizons: list[int], gpu_ids: list[str]) -> None:
    log_dir = args.workspace_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    processes: list[tuple[int, int, str, subprocess.Popen[bytes], object, Path]] = []
    for worker_index, (horizon, gpu_id) in enumerate(zip(horizons, gpu_ids)):
        log_path = log_dir / f"worker_gpu{worker_index}.log"
        log_handle = log_path.open("w", encoding="utf-8")
        proc = subprocess.Popen(
            [
                sys.executable,
                "-u",
                str(Path(__file__).resolve()),
                *sys.argv[1:],
                "--_worker-mode", "pipeline",
                "--_worker-gpu-id", gpu_id,
                "--_worker-horizon", str(horizon),
                "--_worker-index", str(worker_index),
            ],
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        processes.append((worker_index, horizon, gpu_id, proc, log_handle, log_path))

    failures = []
    for worker_index, horizon, gpu_id, proc, log_handle, log_path in processes:
        code = proc.wait()
        log_handle.close()
        if code != 0:
            failures.append((worker_index, horizon, gpu_id, code, log_path))

    if failures:
        details = ", ".join(
            [
                f"worker={worker_index}:horizon={horizon}:gpu={gpu_id}:code={code}:log={log_path}"
                for worker_index, horizon, gpu_id, code, log_path in failures
            ]
        )
        raise RuntimeError(f"Pipeline failed: {details}")


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

    if args._worker_mode == "pipeline":
        if args._worker_gpu_id is None or args._worker_horizon is None or args._worker_index is None:
            raise ValueError("Missing internal worker arguments.")
        run_pipeline_worker(args, args._worker_gpu_id, args._worker_horizon, args._worker_index)
        return

    horizons = parse_int_list(args.horizons)
    gpu_ids = parse_gpu_ids(args.gpu_ids)
    validate_pairing(horizons, gpu_ids)

    print(f"Horizons: {horizons}")
    print(f"GPU ids: {gpu_ids}")
    print(f"Workspace: {args.workspace_dir}")
    print(f"Soft labels dir: {args.soft_labels_dir}")
    print(f"Distill output dir: {args.distill_output_dir}")
    stride_map = {h: stride_for_horizon(h) for h in horizons}
    print(f"Stride map: {stride_map}")

    run_all_workers(args, horizons, gpu_ids)
    print("[ALL DONE] Short-horizon 3GPU pipeline finished.")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd


REQUIRED_SPLITS = {"train", "val", "test"}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Wait for Chronos soft labels, validate them, then run distillation for selected horizons."
    )
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--soft-labels-dir", type=Path, required=True)
    p.add_argument("--metadata-path", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--horizons", default="12,24,48,72,120",
                   help="Horizons to run after labels are ready. 6h is excluded by default.")
    p.add_argument("--alphas", default="0.3,0.5,0.7,1.0")
    p.add_argument("--context-hours", type=int, default=168)
    p.add_argument("--freq", default="10min")
    p.add_argument("--train-ratio", type=float, default=0.70)
    p.add_argument("--val-ratio", type=float, default=0.20)
    p.add_argument("--time-limit", type=int, default=3600)
    p.add_argument("--gpu-id", default="0")
    p.add_argument("--limit-stations", type=int, default=0)
    p.add_argument("--poll-seconds", type=int, default=300,
                   help="Polling interval while waiting for soft labels.")
    p.add_argument("--timeout-hours", type=float, default=0,
                   help="Overall wait timeout in hours. 0 means wait indefinitely.")
    p.add_argument("--train-script", type=Path, default=Path("train_distill_batch.py"))
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip a horizon if summary.json already exists.")
    return p


def parse_horizons(raw: str) -> list[int]:
    horizons = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value == 6:
            continue
        horizons.append(value)
    if not horizons:
        raise ValueError("No valid horizons to run.")
    return horizons


def label_path(soft_labels_dir: Path, horizon: int) -> Path:
    return soft_labels_dir / f"labels_{horizon:03d}h.csv"


def validate_soft_labels(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, "missing"
    if path.stat().st_size == 0:
        return False, "empty"

    try:
        df = pd.read_csv(path, usecols=["item_id", "split", "timestamp"])
    except Exception as exc:
        return False, f"bad_csv: {exc}"

    if df.empty:
        return False, "no_rows"

    if "split" not in df.columns:
        return False, "no_split_column"

    splits = set(df["split"].dropna().astype(str).unique())
    missing = sorted(REQUIRED_SPLITS - splits)
    if missing:
        return False, f"missing_splits:{','.join(missing)}"

    if df["item_id"].isna().all():
        return False, "no_item_id"

    if df["timestamp"].isna().all():
        return False, "no_timestamp"

    return True, f"ok rows={len(df)} items={df['item_id'].nunique()}"


def wait_for_all_labels(soft_labels_dir: Path, horizons: list[int], poll_seconds: int, timeout_hours: float) -> None:
    start = time.time()
    timeout_seconds = timeout_hours * 3600 if timeout_hours > 0 else 0

    while True:
        all_ok = True
        print(f"[CHECK] {time.strftime('%Y-%m-%d %H:%M:%S')}")
        for horizon in horizons:
            path = label_path(soft_labels_dir, horizon)
            ok, message = validate_soft_labels(path)
            print(f"  {path.name}: {message}")
            if not ok:
                all_ok = False

        if all_ok:
            print(f"[READY] All required soft labels are valid.")
            return

        if timeout_seconds and (time.time() - start) > timeout_seconds:
            raise TimeoutError("Timed out while waiting for soft labels.")

        time.sleep(poll_seconds)


def summary_path(output_dir: Path, horizon: int) -> Path:
    return output_dir / f"horizon_{horizon:03d}h" / "summary.json"


def build_command(args: argparse.Namespace, horizon: int) -> list[str]:
    cmd = [
        sys.executable,
        str(args.train_script),
        "--input-dir", str(args.input_dir),
        "--soft-labels-dir", str(args.soft_labels_dir),
        "--output-dir", str(args.output_dir),
        "--horizon-hours", str(horizon),
        "--alphas", args.alphas,
        "--context-hours", str(args.context_hours),
        "--freq", args.freq,
        "--train-ratio", str(args.train_ratio),
        "--val-ratio", str(args.val_ratio),
        "--time-limit", str(args.time_limit),
        "--gpu-id", str(args.gpu_id),
    ]
    if args.metadata_path:
        cmd.extend(["--metadata-path", str(args.metadata_path)])
    if args.limit_stations > 0:
        cmd.extend(["--limit-stations", str(args.limit_stations)])
    return cmd


def run_one_horizon(args: argparse.Namespace, horizon: int) -> None:
    target_summary = summary_path(args.output_dir, horizon)
    if args.skip_existing and target_summary.exists():
        print(f"[SKIP] horizon={horizon}h summary exists: {target_summary}")
        return

    cmd = build_command(args, horizon)
    print(f"[RUN] horizon={horizon}h")
    print("      " + " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Distill training failed for horizon={horizon}h with code {result.returncode}")

    if not target_summary.exists():
        raise FileNotFoundError(f"summary.json not found after horizon={horizon}h: {target_summary}")

    try:
        summary = json.loads(target_summary.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Failed to read summary for horizon={horizon}h: {exc}") from exc

    print(f"[DONE] horizon={horizon}h results={len(summary.get('results', []))}")


def main() -> None:
    args = build_parser().parse_args()
    horizons = parse_horizons(args.horizons)

    print(f"Target horizons: {horizons}")
    print(f"Alphas: {args.alphas}")
    print(f"Soft labels dir: {args.soft_labels_dir}")
    print(f"Output dir: {args.output_dir}")

    wait_for_all_labels(
        soft_labels_dir=args.soft_labels_dir,
        horizons=horizons,
        poll_seconds=args.poll_seconds,
        timeout_hours=args.timeout_hours,
    )

    for horizon in horizons:
        run_one_horizon(args, horizon)

    print("[ALL DONE] Distillation pipeline finished.")


if __name__ == "__main__":
    main()

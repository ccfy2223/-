from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run TimeEmb on prepared wave wide-table dataset.")
    p.add_argument("--timeemb-root", type=Path, required=True)
    p.add_argument("--dataset-csv", type=Path, required=True)
    p.add_argument("--meta-json", type=Path, required=True)
    p.add_argument("--checkpoints", type=Path, default=None)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seq-len", type=int, default=1008, help="Lookback length in 10-min steps. 1008 = 7 days.")
    p.add_argument("--label-len", type=int, default=0)
    p.add_argument("--pred-lens", type=str, default="36,72,144",
                   help="Comma-separated prediction lengths in 10-min steps.")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--eval-batch-size", type=int, default=256)
    p.add_argument("--train-epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--use-revin", type=int, default=1)
    p.add_argument("--use-hour-index", type=int, default=1)
    p.add_argument("--use-day-index", type=int, default=1)
    p.add_argument("--hour-length", type=int, default=24)
    p.add_argument("--day-length", type=int, default=7)
    p.add_argument("--rec-lambda", type=float, default=1.0)
    p.add_argument("--auxi-lambda", type=float, default=1.0)
    p.add_argument("--auxi-loss", type=str, default="MAE", choices=["MAE", "MSE"])
    p.add_argument("--auxi-mode", type=str, default="rfft")
    p.add_argument("--auxi-type", type=str, default="mag")
    p.add_argument("--module-first", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--model-id-prefix", type=str, default="wave_timeemb")
    p.add_argument("--des", type=str, default="wave")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    meta = json.loads(args.meta_json.read_text(encoding="utf-8"))
    enc_in = int(meta["enc_in"])
    target_cols = ",".join(meta["target_columns"])

    te_root = args.timeemb_root.resolve()
    run_py = te_root / "run.py"
    if not run_py.exists():
        raise FileNotFoundError(f"Cannot find TimeEmb run.py at {run_py}")

    pred_lens = [int(x.strip()) for x in args.pred_lens.split(",") if x.strip()]
    checkpoints = args.checkpoints or (te_root / "checkpoints_wave")

    for pred_len in pred_lens:
        model_id = f"{args.model_id_prefix}_sl{args.seq_len}_pl{pred_len}"
        cmd = [
            sys.executable,
            str(run_py),
            "--is_training", "1",
            "--root_path", str(args.dataset_csv.parent.resolve()),
            "--data_path", args.dataset_csv.name,
            "--model_id", model_id,
            "--model", "TimeEmb",
            "--data", "custom",
            "--features", "M",
            "--target", meta["target_columns"][0],
            "--target_cols", target_cols,
            "--freq", "10min",
            "--seq_len", str(args.seq_len),
            "--label_len", str(args.label_len),
            "--pred_len", str(pred_len),
            "--enc_in", str(enc_in),
            "--dec_in", str(enc_in),
            "--c_out", str(enc_in),
            "--d_model", str(args.d_model),
            "--use_revin", str(args.use_revin),
            "--use_hour_index", str(args.use_hour_index),
            "--use_day_index", str(args.use_day_index),
            "--hour_length", str(args.hour_length),
            "--day_length", str(args.day_length),
            "--rec_lambda", str(args.rec_lambda),
            "--auxi_lambda", str(args.auxi_lambda),
            "--auxi_loss", args.auxi_loss,
            "--auxi_mode", args.auxi_mode,
            "--auxi_type", args.auxi_type,
            "--module_first", str(args.module_first),
            "--batch_size", str(args.batch_size),
            "--eval_batch_size", str(args.eval_batch_size),
            "--learning_rate", str(args.learning_rate),
            "--train_epochs", str(args.train_epochs),
            "--patience", str(args.patience),
            "--num_workers", str(args.num_workers),
            "--checkpoints", str(checkpoints.resolve()),
            "--gpu", str(args.gpu),
            "--des", args.des,
            "--itr", "1",
        ]

        print("\n" + "=" * 80)
        print("Running:", " ".join(cmd))
        print("=" * 80)
        if not args.dry_run:
            subprocess.run(cmd, check=True, cwd=str(te_root))


if __name__ == "__main__":
    main()

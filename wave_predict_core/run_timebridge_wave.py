from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run TimeBridge on prepared wave wide-table dataset.")
    p.add_argument("--timebridge-root", type=Path, required=True)
    p.add_argument("--dataset-csv", type=Path, required=True)
    p.add_argument("--meta-json", type=Path, required=True)
    p.add_argument("--checkpoints", type=Path, default=None)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seq-len", type=int, default=1008, help="Lookback length in 10-min steps. 1008 = 7 days.")
    p.add_argument("--label-len", type=int, default=144)
    p.add_argument("--pred-lens", type=str, default="36,72,144,288,432,720",
                   help="Comma-separated prediction lengths in 10-min steps.")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--eval-batch-size", type=int, default=128)
    p.add_argument("--train-epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--d-ff", type=int, default=256)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--period", type=int, default=144, help="Patch period in 10-min steps. 144 = 24h.")
    p.add_argument("--num-p", type=int, default=8)
    p.add_argument("--alpha", type=float, default=0.2)
    p.add_argument("--ia-layers", type=int, default=2)
    p.add_argument("--pd-layers", type=int, default=1)
    p.add_argument("--ca-layers", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=12)
    p.add_argument("--embed", type=str, default="fixed", choices=["timeF", "fixed", "learned"])
    p.add_argument("--target-feature", type=str, default="WVHT")
    p.add_argument("--model-id-prefix", type=str, default="wave_timebridge")
    p.add_argument("--des", type=str, default="wave")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    meta = json.loads(args.meta_json.read_text(encoding="utf-8"))
    enc_in = int(meta["enc_in"])
    target_cols = ",".join(meta["target_columns"])

    tb_root = args.timebridge_root.resolve()
    run_py = tb_root / "run.py"
    if not run_py.exists():
        raise FileNotFoundError(f"Cannot find TimeBridge run.py at {run_py}")

    pred_lens = [int(x.strip()) for x in args.pred_lens.split(",") if x.strip()]
    checkpoints = args.checkpoints or (tb_root / "checkpoints_wave")

    for pred_len in pred_lens:
        model_id = f"{args.model_id_prefix}_sl{args.seq_len}_pl{pred_len}"
        cmd = [
            sys.executable,
            str(run_py),
            "--is_training", "1",
            "--root_path", str(args.dataset_csv.parent.resolve()),
            "--data_path", args.dataset_csv.name,
            "--model_id", model_id,
            "--model", "TimeBridge",
            "--data", "custom",
            "--features", "M",
            "--target", target_cols.split(",")[0],
            "--target_cols", target_cols,
            "--freq", "10min",
            "--embed", args.embed,
            "--seq_len", str(args.seq_len),
            "--label_len", str(args.label_len),
            "--pred_len", str(pred_len),
            "--enc_in", str(enc_in),
            "--period", str(args.period),
            "--num_p", str(args.num_p),
            "--ca_layers", str(args.ca_layers),
            "--pd_layers", str(args.pd_layers),
            "--ia_layers", str(args.ia_layers),
            "--d_model", str(args.d_model),
            "--d_ff", str(args.d_ff),
            "--n_heads", str(args.n_heads),
            "--batch_size", str(args.batch_size),
            "--eval_batch_size", str(args.eval_batch_size),
            "--learning_rate", str(args.learning_rate),
            "--train_epochs", str(args.train_epochs),
            "--patience", str(args.patience),
            "--alpha", str(args.alpha),
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
            subprocess.run(cmd, check=True, cwd=str(tb_root))


if __name__ == "__main__":
    main()

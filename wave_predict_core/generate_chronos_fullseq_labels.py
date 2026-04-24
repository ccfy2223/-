"""
generate_chronos_fullseq_labels.py
===================================
用已训练的 Chronos predictor 对全序列做滚动推理，
生成每个时间步的软标签，保存为 CSV。

输出格式：
  item_id, timestamp, chronos_pred
  41010, 2007-04-11 16:00:00, 1.23
  ...

用法：
  python generate_chronos_fullseq_labels.py \
    --input-dir /root/autodl-tmp/processed_csv/aligned_stations \
    --metadata-path /root/autodl-tmp/processed_csv/shared_timeline_metadata.json \
    --predictor-path /root/autodl-tmp/autogluon_runs/horizon_024h/model \
    --output-path /root/sota_runs/chronos_fullseq_labels/labels_24h.csv \
    --horizon-hours 24 \
    --context-hours 168 \
    --stride-hours 48 \
    --gpu-id 0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))


def _read_cli_value(flag: str, default: str) -> str:
    try:
        idx = sys.argv.index(flag)
    except ValueError:
        return default
    if idx + 1 >= len(sys.argv):
        return default
    return sys.argv[idx + 1]


os.environ["CUDA_VISIBLE_DEVICES"] = _read_cli_value("--gpu-id", "0")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np
import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

DYNAMIC_COLUMNS = [
    "WDIR", "WSPD", "GST", "WVHT", "DPD", "APD", "MWD",
    "PRES", "ATMP", "WTMP", "DEWP", "VIS", "TIDE",
]
KNOWN_COVARIATES = [
    "time_sin_hour", "time_cos_hour",
    "time_sin_doy", "time_cos_doy",
    "month", "day_of_week",
]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir",      type=Path, required=True)
    p.add_argument("--metadata-path",  type=Path, required=True)
    p.add_argument("--predictor-path", type=Path, required=True,
                   help="已训练的 AutoGluon TimeSeriesPredictor 路径")
    p.add_argument("--output-path",    type=Path, required=True,
                   help="输出 CSV 路径，格式：item_id,timestamp,chronos_pred")
    p.add_argument("--horizon-hours",  type=int, default=24)
    p.add_argument("--context-hours",  type=int, default=168)
    p.add_argument("--stride-hours",   type=int, default=48,
                   help="滚动步长（小时），越小覆盖越密但越慢")
    p.add_argument("--freq",           default="10min")
    p.add_argument("--gpu-id",         default="0")
    p.add_argument("--train-ratio",    type=float, default=0.70)
    p.add_argument("--val-ratio",      type=float, default=0.20)
    p.add_argument("--splits",         default="train",
                   help="生成哪些 split 的标签，逗号分隔：train,val,test")
    p.add_argument("--limit-stations", type=int, default=0)
    p.add_argument("--batch-size",     type=int, default=8)
    return p


def load_station(path: Path, shared_start, shared_end) -> pd.DataFrame:
    item_id = path.stem.split("_")[0]
    cols_all = pd.read_csv(path, nrows=0).columns.tolist()
    dyn = [c for c in DYNAMIC_COLUMNS if c in cols_all]
    kno = [c for c in KNOWN_COVARIATES if c in cols_all]

    df = pd.read_csv(
        path,
        usecols=["datetime"] + dyn + kno,
        parse_dates=["datetime"],
        low_memory=False,
    )
    df = df.sort_values("datetime").drop_duplicates("datetime", keep="last").reset_index(drop=True)
    if shared_start is not None:
        df = df[(df["datetime"] >= shared_start) & (df["datetime"] <= shared_end)].reset_index(drop=True)

    num_cols = [c for c in df.columns if c != "datetime"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
    df[num_cols] = df[num_cols].ffill().bfill()

    df.insert(0, "item_id", item_id)
    df = df.rename(columns={"datetime": "timestamp"})
    return df


def rolling_predict(
    predictor: TimeSeriesPredictor,
    df: pd.DataFrame,
    context_steps: int,
    pred_steps: int,
    stride_steps: int,
    model_name: str,
) -> pd.Series:
    """
    对单个站点的全序列做滚动推理。
    返回 Series，index 为 timestamp，值为 chronos_pred（均值）。
    """
    n = len(df)
    timestamps = df["timestamp"].values
    pred_map: dict = {}  # timestamp -> list of predictions (多次覆盖取均值)

    origins = list(range(context_steps, n - pred_steps + 1, stride_steps))
    if not origins:
        return pd.Series(dtype=float)

    item_id = df["item_id"].iloc[0]
    kc_cols = [c for c in KNOWN_COVARIATES if c in df.columns]

    total = len(origins)
    t0 = time.perf_counter()

    for batch_start in range(0, total, 8):
        batch_origins = origins[batch_start: batch_start + 8]
        parts = []
        kc_parts = []

        for origin in batch_origins:
            # context window: [origin-context_steps : origin]
            ctx = df.iloc[origin - context_steps: origin].copy()
            ctx["item_id"] = f"{item_id}__o{origin}"
            parts.append(ctx)

            # known covariates: future [origin : origin+pred_steps]
            fut = df.iloc[origin: origin + pred_steps].copy()
            fut["item_id"] = f"{item_id}__o{origin}"
            kc_parts.append(fut[["item_id", "timestamp"] + kc_cols])

        ctx_df = pd.concat(parts, ignore_index=True)
        kc_df  = pd.concat(kc_parts, ignore_index=True)

        ctx_ts = TimeSeriesDataFrame.from_data_frame(ctx_df, id_column="item_id", timestamp_column="timestamp")
        kc_ts  = TimeSeriesDataFrame.from_data_frame(kc_df,  id_column="item_id", timestamp_column="timestamp")

        try:
            preds = predictor.predict(ctx_ts, known_covariates=kc_ts, model=model_name)
        except Exception:
            try:
                preds = predictor.predict(ctx_ts, model=model_name)
            except Exception as e:
                print(f"    predict failed at batch {batch_start}: {e}")
                continue

        for origin in batch_origins:
            iid = f"{item_id}__o{origin}"
            if iid not in preds.index.get_level_values(0):
                continue
            p = preds.loc[iid]["mean"].values
            future_ts = timestamps[origin: origin + len(p)]
            for ts, val in zip(future_ts, p):
                if ts not in pred_map:
                    pred_map[ts] = []
                pred_map[ts].append(float(val))

        if (batch_start // 8 + 1) % 20 == 0:
            elapsed = time.perf_counter() - t0
            done = batch_start + len(batch_origins)
            eta = elapsed / done * (total - done)
            print(f"    {done}/{total} origins | {elapsed:.0f}s elapsed | ETA {eta:.0f}s")

    # 多次覆盖取均值
    result = {ts: float(np.mean(vals)) for ts, vals in pred_map.items()}
    return pd.Series(result, name="chronos_pred")


def main() -> None:
    args = build_parser().parse_args()

    # 读 metadata
    meta = json.loads(args.metadata_path.read_text(encoding="utf-8"))
    shared_start = pd.Timestamp(meta["shared_start"])
    shared_end   = pd.Timestamp(meta["shared_end"])

    steps_per_hour = {"10min": 6, "1h": 1, "30min": 2}.get(args.freq, 6)
    context_steps = args.context_hours * steps_per_hour
    pred_steps    = args.horizon_hours * steps_per_hour
    stride_steps  = args.stride_hours  * steps_per_hour

    splits_to_run = [s.strip() for s in args.splits.split(",")]

    # 加载 predictor
    print(f"Loading predictor from {args.predictor_path} ...")
    predictor = TimeSeriesPredictor.load(str(args.predictor_path))
    lb = predictor.leaderboard(display=False)
    # 选 Chronos 模型
    chronos_rows = lb[lb["model"].str.contains("Chronos", case=False)]
    if chronos_rows.empty:
        model_name = lb.iloc[0]["model"]
        print(f"  No Chronos model found, using best: {model_name}")
    else:
        model_name = chronos_rows.iloc[0]["model"]
    print(f"  Using model: {model_name}")

    # 加载站点
    files = sorted(args.input_dir.glob("*.csv"))
    if args.limit_stations > 0:
        files = files[:args.limit_stations]
    print(f"Found {len(files)} station files")

    all_labels = []

    for path in files:
        item_id = path.stem.split("_")[0]
        print(f"\n[{item_id}] Loading ...")
        df = load_station(path, shared_start, shared_end)
        n = len(df)
        train_end = int(n * args.train_ratio)
        val_end   = int(n * (args.train_ratio + args.val_ratio))

        split_ranges = {
            "train": (0, train_end),
            "val":   (train_end, val_end),
            "test":  (val_end, n),
        }

        for split in splits_to_run:
            s, e = split_ranges[split]
            # context 需要往前借 context_steps 行
            ctx_start = max(0, s - context_steps)
            sub = df.iloc[ctx_start:e].reset_index(drop=True)
            # 重新对齐 origin 范围
            actual_context = s - ctx_start
            sub_origins_offset = actual_context  # origins 从这里开始

            print(f"  [{split}] rows {s}~{e} | context_start={ctx_start} | "
                  f"sub_len={len(sub)} | stride={stride_steps}")

            t0 = time.perf_counter()
            preds = rolling_predict(
                predictor=predictor,
                df=sub,
                context_steps=actual_context if actual_context >= context_steps else context_steps,
                pred_steps=pred_steps,
                stride_steps=stride_steps,
                model_name=model_name,
            )
            elapsed = time.perf_counter() - t0
            print(f"  [{split}] done in {elapsed:.0f}s | {len(preds)} timestamps covered")

            if preds.empty:
                continue

            label_df = preds.reset_index()
            label_df.columns = ["timestamp", "chronos_pred"]
            label_df.insert(0, "item_id", item_id)
            label_df.insert(1, "split", split)
            all_labels.append(label_df)

    if not all_labels:
        print("No labels generated.")
        return

    out_df = pd.concat(all_labels, ignore_index=True)
    out_df["timestamp"] = pd.to_datetime(out_df["timestamp"])
    out_df = out_df.sort_values(["item_id", "split", "timestamp"]).reset_index(drop=True)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output_path, index=False)
    print(f"\nSaved {len(out_df)} rows to {args.output_path}")

    # 覆盖率统计
    print("\nCoverage stats:")
    for (iid, split), grp in out_df.groupby(["item_id", "split"]):
        print(f"  {iid} [{split}]: {len(grp)} timestamps")


if __name__ == "__main__":
    main()

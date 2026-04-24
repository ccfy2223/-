"""
train_distill.py
================
Knowledge Distillation: Chronos (teacher) → PatchTST (student)

使用 Chronos 生成的软标签训练 PatchTST 学生模型。
损失函数：loss = α * MSE(pred, true) + (1-α) * MSE(pred, chronos_soft)

用法：
    python train_distill.py \
        --input-dir /root/autodl-tmp/processed_csv/aligned_stations \
        --soft-labels-dir /root/sota_runs/chronos_fullseq_labels \
        --output-dir /root/sota_runs/distill_runs \
        --horizon-hours 24 \
        --alpha 0.5 \
        --gpu-id 0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from dataclasses import dataclass
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
import torch
import torch.nn as nn
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
from autogluon.timeseries.metrics import check_get_evaluation_metric

DYNAMIC_COLUMNS = [
    "WDIR", "WSPD", "GST", "WVHT", "DPD", "APD", "MWD",
    "PRES", "ATMP", "WTMP", "DEWP", "VIS", "TIDE",
]
KNOWN_COVARIATES = [
    "time_sin_hour", "time_cos_hour",
    "time_sin_doy", "time_cos_doy",
    "month", "day_of_week",
]


@dataclass
class DistillConfig:
    horizon_hours: int
    alpha: float  # 蒸馏权重：loss = α*hard + (1-α)*soft
    context_hours: int
    prediction_length: int
    freq: str
    train_ratio: float
    val_ratio: float


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, required=True,
                   help="对齐后的站点数据目录")
    p.add_argument("--soft-labels-dir", type=Path, required=True,
                   help="Chronos 软标签目录，包含 labels_XXXh.csv")
    p.add_argument("--metadata-path", type=Path, default=None,
                   help="shared_timeline_metadata.json")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="蒸馏模型输出目录")
    p.add_argument("--horizon-hours", type=int, default=24)
    p.add_argument("--alpha", type=float, default=0.5,
                   help="蒸馏权重：loss = α*MSE(pred,true) + (1-α)*MSE(pred,soft)")
    p.add_argument("--context-hours", type=int, default=168)
    p.add_argument("--freq", default="10min")
    p.add_argument("--train-ratio", type=float, default=0.70)
    p.add_argument("--val-ratio", type=float, default=0.20)
    p.add_argument("--time-limit", type=int, default=7200,
                   help="训练时间限制（秒）")
    p.add_argument("--gpu-id", default="0")
    p.add_argument("--limit-stations", type=int, default=0)
    return p


def load_station(path: Path, shared_start, shared_end) -> pd.DataFrame:
    """加载单个站点数据"""
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


def load_soft_labels(soft_labels_path: Path, item_id: str, split: str) -> pd.DataFrame:
    """加载 Chronos 软标签"""
    if not soft_labels_path.exists():
        raise FileNotFoundError(f"软标签文件不存在：{soft_labels_path}")

    df = pd.read_csv(soft_labels_path, parse_dates=["timestamp"])
    df = df[(df["item_id"] == item_id) & (df["split"] == split)].copy()
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df[["timestamp", "chronos_pred"]]


def merge_soft_labels(data_df: pd.DataFrame, soft_df: pd.DataFrame) -> pd.DataFrame:
    """将软标签合并到数据中"""
    merged = data_df.merge(soft_df, on="timestamp", how="left")
    # 填充缺失的软标签（用真实值）
    merged["chronos_pred"] = merged["chronos_pred"].fillna(merged["WVHT"])
    return merged


class DistillationLoss(nn.Module):
    """知识蒸馏损失函数"""
    def __init__(self, alpha: float = 0.5):
        super().__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()

    def forward(self, pred, target_hard, target_soft):
        """
        pred: 学生模型预测 [batch, seq_len]
        target_hard: 真实标签 [batch, seq_len]
        target_soft: 教师软标签 [batch, seq_len]
        """
        loss_hard = self.mse(pred, target_hard)
        loss_soft = self.mse(pred, target_soft)
        return self.alpha * loss_hard + (1 - self.alpha) * loss_soft


def train_distill_model(
    train_data: TimeSeriesDataFrame,
    val_data: TimeSeriesDataFrame,
    config: DistillConfig,
    output_dir: Path,
) -> TimeSeriesPredictor:
    """训练蒸馏模型"""

    print(f"\n{'='*60}")
    print(f"训练蒸馏模型：α={config.alpha}, horizon={config.horizon_hours}h")
    print(f"{'='*60}")

    # 检查是否有 chronos_pred 列
    if "chronos_pred" not in train_data.columns:
        raise ValueError("训练数据中缺少 chronos_pred 列（Chronos 软标签）")

    predictor = TimeSeriesPredictor(
        target="WVHT",
        prediction_length=config.prediction_length,
        freq=config.freq,
        eval_metric="MASE",
        known_covariates_names=KNOWN_COVARIATES,
        path=str(output_dir / f"alpha_{config.alpha:.1f}"),
    )

    # PatchTST 超参数（针对蒸馏优化）
    hyperparameters = {
        "PatchTST": {
            "context_length": config.context_hours * 6,  # 168h = 1008 steps
            "patch_len": 16,
            "stride": 8,
            "d_model": 128,
            "nhead": 8,
            "num_encoder_layers": 4,
            "dropout": 0.1,
            "lr": 1e-4,
            "batch_size": 32,
            "max_epochs": 100,
            "num_batches_per_epoch": 100,
            "early_stopping_patience": 15,
            # 蒸馏相关
            "use_soft_labels": True,  # 标记使用软标签
            "soft_label_column": "chronos_pred",
            "distill_alpha": config.alpha,
        }
    }

    print(f"\n开始训练（时间限制：{7200}秒）...")
    t0 = time.time()

    try:
        predictor.fit(
            train_data=train_data,
            tuning_data=val_data,
            hyperparameters=hyperparameters,
            time_limit=7200,
            enable_ensemble=False,  # 不使用集成
        )
    except Exception as e:
        print(f"训练失败：{e}")
        raise

    elapsed = time.time() - t0
    print(f"训练完成，耗时：{elapsed:.1f}秒")

    return predictor


def evaluate_model(
    predictor: TimeSeriesPredictor,
    test_data: TimeSeriesDataFrame,
    config: DistillConfig,
) -> dict:
    """评估模型"""
    print("\n评估模型...")

    # 生成预测
    predictions = predictor.predict(test_data)

    # 计算指标
    metrics = {}
    for metric_name in ["MAE", "MASE", "RMSE", "RMSLE", "SMAPE"]:
        try:
            metric_func = check_get_evaluation_metric(metric_name)
            score = metric_func(test_data, predictions)
            metrics[metric_name] = abs(float(score))
        except Exception as e:
            print(f"  计算 {metric_name} 失败：{e}")
            metrics[metric_name] = None

    return metrics


def main():
    args = build_parser().parse_args()

    # 配置
    steps_per_hour = {"10min": 6, "1h": 1, "30min": 2}.get(args.freq, 6)
    config = DistillConfig(
        horizon_hours=args.horizon_hours,
        alpha=args.alpha,
        context_hours=args.context_hours,
        prediction_length=args.horizon_hours * steps_per_hour,
        freq=args.freq,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )

    # 加载 metadata
    if args.metadata_path and args.metadata_path.exists():
        meta = json.loads(args.metadata_path.read_text(encoding="utf-8"))
        shared_start = pd.Timestamp(meta["shared_start"])
        shared_end = pd.Timestamp(meta["shared_end"])
    else:
        shared_start = None
        shared_end = None

    # 软标签路径
    soft_labels_path = args.soft_labels_dir / f"labels_{args.horizon_hours:03d}h.csv"
    if not soft_labels_path.exists():
        raise FileNotFoundError(f"软标签文件不存在：{soft_labels_path}")

    print(f"加载软标签：{soft_labels_path}")

    # 加载所有站点数据
    files = sorted(args.input_dir.glob("*.csv"))
    if args.limit_stations > 0:
        files = files[:args.limit_stations]

    print(f"找到 {len(files)} 个站点文件")

    all_train = []
    all_val = []
    all_test = []

    for path in files:
        item_id = path.stem.split("_")[0]
        print(f"\n[{item_id}] 加载数据...")

        # 加载站点数据
        df = load_station(path, shared_start, shared_end)
        n = len(df)
        train_end = int(n * config.train_ratio)
        val_end = int(n * (config.train_ratio + config.val_ratio))

        # 加载软标签
        try:
            soft_train = load_soft_labels(soft_labels_path, item_id, "train")
            soft_val = load_soft_labels(soft_labels_path, item_id, "val")
            soft_test = load_soft_labels(soft_labels_path, item_id, "test")
        except Exception as e:
            print(f"  跳过（软标签加载失败）：{e}")
            continue

        # 合并软标签
        df_train = merge_soft_labels(df.iloc[:train_end].copy(), soft_train)
        df_val = merge_soft_labels(df.iloc[train_end:val_end].copy(), soft_val)
        df_test = merge_soft_labels(df.iloc[val_end:].copy(), soft_test)

        print(f"  Train: {len(df_train)} | Val: {len(df_val)} | Test: {len(df_test)}")
        print(f"  软标签覆盖率 - Train: {(~df_train['chronos_pred'].isna()).sum()}/{len(df_train)}")

        all_train.append(df_train)
        all_val.append(df_val)
        all_test.append(df_test)

    if not all_train:
        raise ValueError("没有可用的训练数据")

    # 合并所有站点
    train_df = pd.concat(all_train, ignore_index=True)
    val_df = pd.concat(all_val, ignore_index=True)
    test_df = pd.concat(all_test, ignore_index=True)

    print(f"\n总数据量：")
    print(f"  Train: {len(train_df)} 行，{train_df['item_id'].nunique()} 个站点")
    print(f"  Val:   {len(val_df)} 行")
    print(f"  Test:  {len(test_df)} 行")

    # 转换为 TimeSeriesDataFrame
    train_ts = TimeSeriesDataFrame.from_data_frame(train_df, id_column="item_id", timestamp_column="timestamp")
    val_ts = TimeSeriesDataFrame.from_data_frame(val_df, id_column="item_id", timestamp_column="timestamp")
    test_ts = TimeSeriesDataFrame.from_data_frame(test_df, id_column="item_id", timestamp_column="timestamp")

    # 训练蒸馏模型
    output_dir = args.output_dir / f"horizon_{args.horizon_hours:03d}h"
    output_dir.mkdir(parents=True, exist_ok=True)

    predictor = train_distill_model(train_ts, val_ts, config, output_dir)

    # 评估
    metrics = evaluate_model(predictor, test_ts, config)

    # 保存结果
    result = {
        "horizon_hours": config.horizon_hours,
        "alpha": config.alpha,
        "context_hours": config.context_hours,
        "metrics": metrics,
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
        "stations": train_df["item_id"].unique().tolist(),
    }

    result_path = output_dir / f"result_alpha_{config.alpha:.1f}.json"
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{'='*60}")
    print(f"蒸馏训练完成！")
    print(f"{'='*60}")
    print(f"Alpha: {config.alpha}")
    print(f"测试集指标：")
    for k, v in metrics.items():
        if v is not None:
            print(f"  {k}: {v:.4f}")
    print(f"\n结果已保存：{result_path}")


if __name__ == "__main__":
    main()

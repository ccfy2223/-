"""
hpo_distill.py
==============
用 Optuna 自动调参知识蒸馏 PatchTST

搜索空间：
- lr: 1e-5 ~ 1e-3
- d_model: 64, 96, 128, 192
- nhead: 4, 8
- num_layers: 2, 3, 4
- dropout: 0.05 ~ 0.3
- batch_size: 32, 64, 128
- alpha: 0.1 ~ 0.9

用法：
    python hpo_distill.py \
        --input-dir /root/autodl-tmp/processed_csv/aligned_stations \
        --soft-labels-dir /root/autodl-tmp/sota_runs/chronos_fullseq_labels \
        --metadata-path /root/autodl-tmp/processed_csv/shared_timeline_metadata.json \
        --output-dir /root/autodl-tmp/sota_runs/hpo_distill \
        --horizon-hours 24 \
        --n-trials 30 \
        --trial-epochs 30 \
        --limit-stations 3 \
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

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import optuna
from optuna.samplers import TPESampler

def _read_cli_value(flag, default):
    try:
        idx = sys.argv.index(flag)
        if idx + 1 < len(sys.argv):
            return sys.argv[idx + 1]
    except ValueError:
        pass
    return default

os.environ["CUDA_VISIBLE_DEVICES"] = _read_cli_value("--gpu-id", "0")

DYNAMIC_COLUMNS = ["WDIR", "WSPD", "GST", "WVHT", "DPD", "APD", "MWD", "PRES", "ATMP", "WTMP", "DEWP", "VIS", "TIDE"]
KNOWN_COVARIATES = ["time_sin_hour", "time_cos_hour", "time_sin_doy", "time_cos_doy", "month", "day_of_week"]


# ── 模型 ──────────────────────────────────────────────────────────────────────

class PatchTST(nn.Module):
    def __init__(self, context_length, prediction_length, patch_len=16, stride=8,
                 d_model=96, nhead=8, num_layers=4, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.context_length = context_length
        self.patch_len = patch_len
        self.stride = stride
        self.num_patches = (context_length - patch_len) // stride + 1

        self.patch_proj = nn.Linear(patch_len, d_model)
        self.positional_encoding = nn.Parameter(torch.randn(self.num_patches, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, prediction_length)

        # 初始化
        nn.init.xavier_uniform_(self.patch_proj.weight)
        nn.init.xavier_uniform_(self.output_proj.weight)

    def forward(self, x):
        # x: [batch, context_length]
        batch = x.size(0)
        patches = torch.stack([
            x[:, i*self.stride: i*self.stride + self.patch_len]
            for i in range(self.num_patches)
        ], dim=1)  # [batch, num_patches, patch_len]

        x = self.patch_proj(patches) + self.positional_encoding
        x = self.encoder(x)
        x = self.norm(x.mean(dim=1))
        return self.output_proj(x)


class DistillLoss(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def forward(self, pred, hard, soft):
        loss_hard = F.mse_loss(pred, hard)
        loss_soft = F.mse_loss(pred, soft)
        return self.alpha * loss_hard + (1 - self.alpha) * loss_soft


import torch.nn.functional as F


# ── 数据集 ────────────────────────────────────────────────────────────────────

class TSDataset(Dataset):
    def __init__(self, data_df, soft_df, context_len, pred_len, split):
        self.context_len = context_len
        self.pred_len = pred_len
        self.groups = {}

        for item_id in data_df["item_id"].unique():
            g = data_df[data_df["item_id"] == item_id].sort_values("timestamp").reset_index(drop=True)
            soft_sub = soft_df[
                (soft_df["item_id"] == int(item_id)) & (soft_df["split"] == split)
            ].set_index("timestamp")["chronos_pred"]

            wvht = g["WVHT"].values.astype(np.float32)
            soft = g["timestamp"].map(soft_sub).fillna(pd.Series(wvht, index=g.index)).values.astype(np.float32)
            self.groups[item_id] = (wvht, soft)

        self.samples = [
            (iid, i)
            for iid, (w, _) in self.groups.items()
            for i in range(0, len(w) - context_len - pred_len + 1, pred_len)
        ]

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        iid, s = self.samples[idx]
        w, soft = self.groups[iid]
        e = s + self.context_len
        return {
            "x": torch.from_numpy(w[s:e]),
            "y": torch.from_numpy(w[e:e+self.pred_len]),
            "s": torch.from_numpy(soft[e:e+self.pred_len]),
        }


# ── 训练 / 评估 ───────────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train() if train else model.eval()
    total = 0
    with torch.set_grad_enabled(train):
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            s = batch["s"].to(device)
            pred = model(x)
            loss = criterion(pred, y, s)
            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total += loss.item()
    return total / len(loader)


def eval_mae(model, loader, device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            preds.append(model(batch["x"].to(device)).cpu().numpy())
            targets.append(batch["y"].numpy())
    p = np.concatenate(preds)
    t = np.concatenate(targets)
    return float(np.mean(np.abs(p - t)))


# ── Optuna objective ──────────────────────────────────────────────────────────

def make_objective(train_df, val_df, soft_df, context_len, pred_len, trial_epochs, device):
    def objective(trial):
        # 搜索空间
        lr        = trial.suggest_float("lr", 1e-5, 5e-4, log=True)
        d_model   = trial.suggest_categorical("d_model", [64, 96, 128])
        nhead     = trial.suggest_categorical("nhead", [4, 8])
        num_layers= trial.suggest_int("num_layers", 2, 5)
        dropout   = trial.suggest_float("dropout", 0.05, 0.3)
        batch_size= trial.suggest_categorical("batch_size", [32, 64, 128])
        alpha     = trial.suggest_float("alpha", 0.1, 0.9)

        # d_model 必须能被 nhead 整除
        if d_model % nhead != 0:
            raise optuna.TrialPruned()

        dim_ff = d_model * 4

        train_ds = TSDataset(train_df, soft_df, context_len, pred_len, "train")
        val_ds   = TSDataset(val_df,   soft_df, context_len, pred_len, "val")
        train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=True)
        val_ld   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

        model = PatchTST(
            context_length=context_len, prediction_length=pred_len,
            d_model=d_model, nhead=nhead, num_layers=num_layers,
            dim_feedforward=dim_ff, dropout=dropout,
        ).to(device)

        criterion = DistillLoss(alpha=alpha)
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

        best_mae = float("inf")
        patience, no_imp = 8, 0

        for epoch in range(trial_epochs):
            run_epoch(model, train_ld, criterion, optimizer, device, train=True)
            mae = eval_mae(model, val_ld, device)

            if mae < best_mae:
                best_mae = mae
                no_imp = 0
            else:
                no_imp += 1

            # Optuna pruning
            trial.report(mae, epoch)
            if trial.should_prune() or no_imp >= patience:
                raise optuna.TrialPruned()

        return best_mae

    return objective


# ── main ──────────────────────────────────────────────────────────────────────

def load_station(path, shared_start, shared_end):
    item_id = path.stem.split("_")[0]
    cols = pd.read_csv(path, nrows=0).columns.tolist()
    dyn = [c for c in DYNAMIC_COLUMNS if c in cols]
    kno = [c for c in KNOWN_COVARIATES if c in cols]
    df = pd.read_csv(path, usecols=["datetime"] + dyn + kno, parse_dates=["datetime"], low_memory=False)
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


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir",      type=Path, required=True)
    p.add_argument("--soft-labels-dir",type=Path, required=True)
    p.add_argument("--metadata-path",  type=Path, default=None)
    p.add_argument("--output-dir",     type=Path, required=True)
    p.add_argument("--horizon-hours",  type=int,  default=24)
    p.add_argument("--context-hours",  type=int,  default=168)
    p.add_argument("--freq",           default="10min")
    p.add_argument("--train-ratio",    type=float, default=0.70)
    p.add_argument("--val-ratio",      type=float, default=0.20)
    p.add_argument("--n-trials",       type=int,  default=30)
    p.add_argument("--trial-epochs",   type=int,  default=30)
    p.add_argument("--limit-stations", type=int,  default=0)
    p.add_argument("--gpu-id",         default="0")
    return p


def main():
    args = build_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备：{device}")

    steps = {"10min": 6, "1h": 1, "30min": 2}.get(args.freq, 6)
    context_len = args.context_hours * steps
    pred_len    = args.horizon_hours  * steps

    # metadata
    shared_start = shared_end = None
    if args.metadata_path and args.metadata_path.exists():
        meta = json.loads(args.metadata_path.read_text())
        shared_start = pd.Timestamp(meta["shared_start"])
        shared_end   = pd.Timestamp(meta["shared_end"])

    # 软标签
    soft_path = args.soft_labels_dir / f"labels_{args.horizon_hours:03d}h.csv"
    if not soft_path.exists():
        soft_path = args.soft_labels_dir / f"labels_{args.horizon_hours}h.csv"
    print(f"软标签：{soft_path}")
    soft_df = pd.read_csv(soft_path, parse_dates=["timestamp"])

    # 站点数据
    files = sorted(args.input_dir.glob("*.csv"))
    if args.limit_stations > 0:
        files = files[:args.limit_stations]
    print(f"站点数：{len(files)}")

    all_data = [load_station(f, shared_start, shared_end) for f in files]
    combined = pd.concat(all_data, ignore_index=True)

    train_dfs, val_dfs = [], []
    for iid in combined["item_id"].unique():
        g = combined[combined["item_id"] == iid].sort_values("timestamp").reset_index(drop=True)
        n = len(g)
        te = int(n * args.train_ratio)
        ve = int(n * (args.train_ratio + args.val_ratio))
        train_dfs.append(g.iloc[:te])
        val_dfs.append(g.iloc[te:ve])

    train_df = pd.concat(train_dfs, ignore_index=True)
    val_df   = pd.concat(val_dfs,   ignore_index=True)
    print(f"Train: {len(train_df)} | Val: {len(val_df)}")

    # Optuna study
    args.output_dir.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
        study_name=f"distill_{args.horizon_hours}h",
        storage=f"sqlite:///{args.output_dir}/hpo.db",
        load_if_exists=True,
    )

    objective = make_objective(train_df, val_df, soft_df, context_len, pred_len, args.trial_epochs, device)

    print(f"\n开始 HPO：{args.n_trials} trials × {args.trial_epochs} epochs")
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    # 结果
    print(f"\n{'='*60}")
    print(f"最佳 Val MAE: {study.best_value:.4f}")
    print(f"最佳参数：")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    # 保存
    result = {
        "best_val_mae": study.best_value,
        "best_params": study.best_params,
        "n_trials": len(study.trials),
    }
    (args.output_dir / "best_params.json").write_text(json.dumps(result, indent=2))

    # 打印 top-5
    print(f"\nTop-5 trials:")
    trials_df = study.trials_dataframe().sort_values("value").head(5)
    print(trials_df[["number", "value"] + [c for c in trials_df.columns if c.startswith("params_")]].to_string(index=False))


if __name__ == "__main__":
    main()

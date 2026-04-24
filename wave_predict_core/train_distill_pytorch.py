"""
train_distill_pytorch.py
========================
使用 PyTorch 原生实现知识蒸馏，支持自定义损失函数

核心创新：
- 从 GluonTS PatchTST 提取模型结构和权重
- 自定义蒸馏损失：loss = α * MSE(pred, true) + (1-α) * MSE(pred, soft)
- 直接在真实标签上评估，避免训练-评估不匹配

用法：
    python train_distill_pytorch.py \
        --input-dir /root/autodl-tmp/processed_csv/aligned_stations \
        --soft-labels-dir /root/autodl-tmp/sota_runs/chronos_fullseq_labels \
        --pretrained-model /root/autodl-tmp/autogluon_runs/horizon_024h/model/models/PatchTST/gluon_ts/prediction-net-state.pt \
        --output-dir /root/autodl-tmp/sota_runs/distill_pytorch \
        --horizon-hours 24 \
        --alphas 0.3,0.5,0.7 \
        --epochs 50 \
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

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

# GPU 设置
def _read_cli_value(flag: str, default: str) -> str:
    try:
        idx = sys.argv.index(flag)
        if idx + 1 < len(sys.argv):
            return sys.argv[idx + 1]
    except ValueError:
        pass
    return default

os.environ["CUDA_VISIBLE_DEVICES"] = _read_cli_value("--gpu-id", "0")

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = True
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")

DYNAMIC_COLUMNS = [
    "WDIR", "WSPD", "GST", "WVHT", "DPD", "APD", "MWD",
    "PRES", "ATMP", "WTMP", "DEWP", "VIS", "TIDE",
]
KNOWN_COVARIATES = [
    "time_sin_hour", "time_cos_hour",
    "time_sin_doy", "time_cos_doy",
    "month", "day_of_week",
]

DEFAULT_NUM_WORKERS = min(32, max(8, (os.cpu_count() or 8) // 2))


# ============================================================================
# PatchTST Model (从 GluonTS 提取的结构)
# ============================================================================

class PatchTST(nn.Module):
    """
    PatchTST 模型（简化版，匹配 GluonTS 权重结构）

    参数从 AutoGluon 训练的模型中提取：
    - patch_len: 16
    - stride: 8
    - d_model: 32 (从权重形状推断)
    - nhead: 3 (96 / 32 = 3)
    - num_layers: 4
    """
    def __init__(
        self,
        context_length: int,
        prediction_length: int,
        num_features: int,
        num_known_covariates: int,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 32,
        nhead: int = 3,
        num_layers: int = 4,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model

        # 计算 patch 数量
        self.num_patches = (context_length - patch_len) // stride + 1

        # Patch embedding: [batch, num_patches, patch_len * num_features] -> [batch, num_patches, d_model]
        self.patch_proj = nn.Linear(patch_len * num_features, d_model)

        # Positional encoding
        self.positional_encoding = nn.Parameter(torch.randn(self.num_patches, d_model))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection
        self.output_proj = nn.Linear(d_model, prediction_length)

    def forward(self, past_target, known_covariates=None):
        """
        past_target: [batch, context_length, num_features]
        known_covariates: [batch, prediction_length, num_known_covariates] (未使用)

        返回: [batch, prediction_length]
        """
        batch_size = past_target.size(0)

        # 创建 patches: [batch, num_patches, patch_len * num_features]
        patches = []
        for i in range(self.num_patches):
            start = i * self.stride
            end = start + self.patch_len
            patch = past_target[:, start:end, :].reshape(batch_size, -1)
            patches.append(patch)
        patches = torch.stack(patches, dim=1)  # [batch, num_patches, patch_len * num_features]

        # Patch embedding
        x = self.patch_proj(patches)  # [batch, num_patches, d_model]

        # Add positional encoding
        x = x + self.positional_encoding.unsqueeze(0)

        # Transformer encoder
        x = self.encoder(x)  # [batch, num_patches, d_model]

        # Global pooling
        x = x.mean(dim=1)  # [batch, d_model]

        # Output projection
        output = self.output_proj(x)  # [batch, prediction_length]

        return output


class DistillationLoss(nn.Module):
    """知识蒸馏损失函数（支持动态 alpha + 频域一致性约束）"""
    def __init__(self, alpha: float = 0.5, lambda_freq: float = 0.0):
        super().__init__()
        self.alpha = alpha
        self.lambda_freq = lambda_freq
        self.mae = nn.L1Loss()

    def forward(self, pred, target_hard, target_soft, alpha_dynamic=None):
        alpha = alpha_dynamic if alpha_dynamic is not None else self.alpha
        loss_hard = self.mae(pred, target_hard)
        loss_soft = self.mae(pred, target_soft)
        loss = (1 - alpha) * loss_hard + alpha * loss_soft

        if self.lambda_freq > 0:
            pred_freq = torch.abs(torch.fft.rfft(pred.float(), dim=-1))
            true_freq = torch.abs(torch.fft.rfft(target_hard.float(), dim=-1))
            loss_freq = torch.mean(torch.abs(pred_freq - true_freq))
            loss = loss + self.lambda_freq * loss_freq
        else:
            loss_freq = torch.tensor(0.0)

        return loss, loss_hard, loss_soft, loss_freq


class TimeSeriesDataset(Dataset):
    """时间序列数据集（多特征版）"""
    def __init__(self, data_df, soft_labels_df, context_length, pred_length, split="train"):
        self.context_length = context_length
        self.pred_length = pred_length

        # 确定实际存在的特征列
        feat_cols = [c for c in DYNAMIC_COLUMNS if c in data_df.columns]
        self.feat_cols = feat_cols
        self.num_features = len(feat_cols)

        self.groups = {}
        for item_id in data_df["item_id"].unique():
            group = data_df[data_df["item_id"] == item_id].sort_values("timestamp").reset_index(drop=True)

            # 软标签对齐（确保类型一致）
            soft_sub = soft_labels_df[
                (soft_labels_df["item_id"].astype(str) == str(item_id)) &
                (soft_labels_df["split"] == split)
            ].set_index("timestamp")["chronos_pred"]

            wvht = group["WVHT"].values.astype(np.float32)
            soft = group["timestamp"].map(soft_sub).fillna(
                pd.Series(wvht, index=group.index)
            ).values.astype(np.float32)

            # 全部动态特征 [T, num_features]
            feats = group[feat_cols].values.astype(np.float32)
            if not np.isfinite(feats).all():
                bad = np.size(feats) - np.isfinite(feats).sum()
                raise ValueError(f"{item_id} 的输入特征存在 {bad} 个非有限值，请检查特征对齐与缺失值处理")
            if not np.isfinite(wvht).all():
                raise ValueError(f"{item_id} 的 WVHT 存在非有限值")
            if not np.isfinite(soft).all():
                raise ValueError(f"{item_id} 的软标签存在非有限值")

            self.groups[item_id] = (wvht, soft, feats)

        # 步长设为 pred_length // 4，增加样本密度（4倍重叠）
        step = max(1, pred_length // 4)
        self.samples = [
            (iid, i)
            for iid, (w, _, _) in self.groups.items()
            for i in range(0, len(w) - context_length - pred_length + 1, step)
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item_id, s = self.samples[idx]
        wvht, soft, feats = self.groups[item_id]
        e = s + self.context_length
        p = e + self.pred_length

        return {
            "past_target": torch.from_numpy(feats[s:e]),       # [ctx, num_features]
            "target_hard": torch.from_numpy(wvht[e:p]),        # [pred]
            "target_soft": torch.from_numpy(soft[e:p]),        # [pred]
        }


def _get_amp_dtype(amp_dtype: str) -> torch.dtype:
    if amp_dtype == "fp16":
        return torch.float16
    return torch.bfloat16


def _tensor_stats(name: str, tensor: torch.Tensor) -> str:
    tensor = tensor.detach()
    finite_mask = torch.isfinite(tensor)
    finite_count = int(finite_mask.sum().item())
    total_count = tensor.numel()
    if finite_count == 0:
        return (
            f"{name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}, "
            f"device={tensor.device}, finite=0/{total_count}"
        )

    finite_vals = tensor[finite_mask].to(torch.float32)
    return (
        f"{name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}, device={tensor.device}, "
        f"finite={finite_count}/{total_count}, min={finite_vals.min().item():.6g}, "
        f"max={finite_vals.max().item():.6g}, mean={finite_vals.mean().item():.6g}"
    )


def train_one_epoch(model, dataloader, criterion, optimizer, device, scaler, use_amp: bool, amp_dtype: str, alpha_dynamic: float = None):
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    total_hard = 0
    total_soft = 0
    total_freq = 0
    num_batches = 0

    for batch in dataloader:
        past_target = batch["past_target"].to(device, non_blocking=True)
        target_hard = batch["target_hard"].to(device, non_blocking=True)
        target_soft = batch["target_soft"].to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=_get_amp_dtype(amp_dtype), enabled=use_amp):
            pred = model(past_target)
            loss, loss_hard, loss_soft, loss_freq = criterion(pred, target_hard, target_soft, alpha_dynamic=alpha_dynamic)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        total_hard += loss_hard.item()
        total_soft += loss_soft.item()
        total_freq += loss_freq.item()
        num_batches += 1

    return {
        "loss": total_loss / num_batches,
        "loss_hard": total_hard / num_batches,
        "loss_soft": total_soft / num_batches,
        "loss_freq": total_freq / num_batches,
    }


def evaluate(model, dataloader, device, use_amp: bool, amp_dtype: str, naive_mae: float = None):
    """评估模型（只用真实标签）"""
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            past_target = batch["past_target"].to(device, non_blocking=True)
            target_hard = batch["target_hard"].to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, dtype=_get_amp_dtype(amp_dtype), enabled=use_amp):
                pred = model(past_target)

            all_preds.append(pred.detach().to(torch.float32).cpu().numpy())
            all_targets.append(target_hard.detach().to(torch.float32).cpu().numpy())

    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    mae = float(np.mean(np.abs(preds - targets)))
    rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))

    # MASE 分母用传入的训练集 naive MAE，没传则用当前集合自身（仅用于 val 阶段快速比较）
    if naive_mae is None:
        naive_mae = float(np.mean(np.abs(targets[:, 1:] - targets[:, :-1])))
    mase = mae / (naive_mae + 1e-8)

    return {"MAE": mae, "RMSE": rmse, "MASE": mase}


def _build_model(trial_params: dict, num_features: int, context_length: int, pred_length: int) -> "PatchTST":
    """根据超参数字典构建模型"""
    d_model = trial_params["d_model"]
    nhead = trial_params["nhead"]
    return PatchTST(
        context_length=context_length,
        prediction_length=pred_length,
        num_features=num_features,
        num_known_covariates=len(KNOWN_COVARIATES),
        patch_len=trial_params.get("patch_len", 16),
        stride=trial_params.get("stride", 8),
        d_model=d_model,
        nhead=nhead,
        num_layers=trial_params["num_layers"],
        dim_feedforward=d_model * 4,
        dropout=trial_params["dropout"],
    )


def run_hpo(
    train_df, val_df,
    soft_labels_df,
    context_length: int,
    pred_length: int,
    device: torch.device,
    use_amp: bool,
    amp_dtype: str,
    num_workers: int,
    n_trials: int,
    hpo_epochs: int,
    hpo_batch_size: int,
) -> dict:
    """
    用 Optuna 搜索最优超参数（alpha + 模型结构 + 优化器）。
    每个 trial 只跑 hpo_epochs 轮，用 val MAE 作为目标。
    返回 best_params 字典。
    """
    if not HAS_OPTUNA:
        raise RuntimeError("请先安装 optuna：pip install optuna")

    # 预先构建数据集（所有 trial 共用，避免重复 IO）
    train_dataset = TimeSeriesDataset(train_df, soft_labels_df, context_length, pred_length, split="train")
    val_dataset   = TimeSeriesDataset(val_df,   soft_labels_df, context_length, pred_length, split="val")
    num_features  = train_dataset.num_features

    loader_kw = dict(num_workers=num_workers, pin_memory=(device.type == "cuda"),
                     persistent_workers=(num_workers > 0))
    if num_workers > 0:
        loader_kw["prefetch_factor"] = 4

    def objective(trial: "optuna.Trial") -> float:
        # ── 搜索空间（针对 4090 24G 优化）──────────────────────────
        alpha      = trial.suggest_float("alpha",      0.0,  1.0)
        lr         = trial.suggest_float("lr",         1e-5, 3e-4, log=True)
        d_model    = trial.suggest_categorical("d_model",    [128, 192, 256, 320])
        nhead      = trial.suggest_categorical("nhead",      [4, 8, 16])
        num_layers = trial.suggest_int("num_layers",   3, 8)
        dropout    = trial.suggest_float("dropout",    0.05, 0.40)
        batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024])

        # nhead 必须整除 d_model
        if d_model % nhead != 0:
            raise optuna.exceptions.TrialPruned()

        params = dict(alpha=alpha, lr=lr, d_model=d_model, nhead=nhead,
                      num_layers=num_layers, dropout=dropout,
                      patch_len=16, stride=8)

        model = _build_model(params, num_features, context_length, pred_length).to(device)
        criterion = DistillationLoss(alpha=alpha)
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        scaler = torch.amp.GradScaler("cuda",
                     enabled=(use_amp and amp_dtype == "fp16" and device.type == "cuda"))

        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True, drop_last=False, **loader_kw)
        val_loader   = DataLoader(val_dataset,   batch_size=batch_size * 2,
                                  shuffle=False, drop_last=False, **loader_kw)

        best_val = float("inf")
        for epoch in range(hpo_epochs):
            train_one_epoch(model, train_loader, criterion, optimizer, device,
                            scaler, use_amp, amp_dtype)
            val_m = evaluate(model, val_loader, device, use_amp, amp_dtype)
            val_mae = val_m["MAE"]

            if not np.isfinite(val_mae):
                raise optuna.exceptions.TrialPruned()

            best_val = min(best_val, val_mae)

            # Optuna median pruner
            trial.report(val_mae, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return best_val

    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)
    study  = optuna.create_study(direction="minimize", pruner=pruner)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_trial
    print(f"\nHPO 完成：best val MAE = {best.value:.4f}")
    print(f"  best params: {best.params}")
    return best.params


def train_one_alpha(
    train_df, val_df, test_df,
    soft_labels_df,
    alpha: float,
    context_length: int,
    pred_length: int,
    pretrained_path: Path,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    num_workers: int,
    eval_batch_multiplier: int,
    use_amp: bool,
    amp_dtype: str,
    early_stopping_patience: int,
    torch_compile: bool,
    hpo_params: dict | None = None,
    # 自适应蒸馏参数
    adaptive: bool = False,
    horizon_hours: int = 24,
    warmup_epochs: int = 10,
    lambda_freq: float = 0.0,
):
    """训练单个 α 值的蒸馏模型"""
    print(f"\n{'='*70}")
    print(f"训练蒸馏模型：α={alpha:.2f}")
    print(f"  α={alpha:.2f} 表示：{alpha*100:.0f}% 真实标签 + {(1-alpha)*100:.0f}% Chronos 软标签")
    print(f"{'='*70}")

    # 如果传入了 HPO 参数，覆盖默认值
    if hpo_params:
        print(f"使用 HPO 搜索到的超参数：{hpo_params}")
        alpha = hpo_params.get("alpha", alpha)
        lr = hpo_params.get("lr", lr)
        batch_size = hpo_params.get("batch_size", batch_size)
        d_model = hpo_params.get("d_model", 128)
        nhead = hpo_params.get("nhead", 8)
        num_layers = hpo_params.get("num_layers", 5)
        dropout = hpo_params.get("dropout", 0.13)
        patch_len = hpo_params.get("patch_len", 16)
        stride = hpo_params.get("stride", 8)
    else:
        d_model = 128
        nhead = 8
        num_layers = 5
        dropout = 0.13
        patch_len = 16
        stride = 8

    # 创建数据集
    train_dataset = TimeSeriesDataset(train_df, soft_labels_df, context_length, pred_length, split="train")
    val_dataset = TimeSeriesDataset(val_df, soft_labels_df, context_length, pred_length, split="val")
    test_dataset = TimeSeriesDataset(test_df, soft_labels_df, context_length, pred_length, split="test")

    eval_batch_size = max(batch_size, batch_size * eval_batch_multiplier)
    loader_common = {
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        loader_common["prefetch_factor"] = 4

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              drop_last=False, **loader_common)
    val_loader = DataLoader(val_dataset, batch_size=eval_batch_size, shuffle=False,
                            drop_last=False, **loader_common)
    test_loader = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False,
                             drop_last=False, **loader_common)

    print(f"数据集大小：")
    print(f"  Train: {len(train_dataset)} 样本")
    print(f"  Val:   {len(val_dataset)} 样本")
    print(f"  Test:  {len(test_dataset)} 样本")
    print(f"  特征数：{train_dataset.num_features} ({', '.join(train_dataset.feat_cols)})")

    # 创建模型，使用全部动态特征
    model = PatchTST(
        context_length=context_length,
        prediction_length=pred_length,
        num_features=train_dataset.num_features,  # 全部动态特征
        num_known_covariates=len(KNOWN_COVARIATES),
        patch_len=patch_len,
        stride=stride,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=d_model * 4,
        dropout=dropout,
    ).to(device)

    if torch_compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
            print("  已启用 torch.compile")
        except Exception as e:
            print(f"  torch.compile 启用失败，继续使用 eager mode：{e}")

    # 加载预训练权重（跳过，因为结构不兼容）
    # GluonTS 的 PatchTST 实现与标准 PyTorch 不同，无法直接迁移权重
    # 我们从头训练，但使用相同的超参数配置
    print("  从头开始训练（GluonTS 权重结构不兼容）")

    # Horizon-aware alpha：不同 horizon 对应不同的教师可信度
    HORIZON_GAMMA = {6: 0.4, 12: 0.8, 24: 1.0, 48: 0.7, 72: 0.6, 120: 0.5}
    horizon_gamma = HORIZON_GAMMA.get(horizon_hours, 1.0)
    alpha_max = alpha * horizon_gamma
    if adaptive:
        print(f"  自适应蒸馏：horizon={horizon_hours}h, gamma={horizon_gamma:.2f}, alpha_max={alpha_max:.3f}, warmup={warmup_epochs}")

    # 损失函数和优化器
    criterion = DistillationLoss(alpha=alpha, lambda_freq=lambda_freq)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == "fp16" and device.type == "cuda"))
    # Warmup + Cosine decay
    def lr_lambda(epoch):
        warmup = 5
        if epoch < warmup:
            return (epoch + 1) / warmup
        progress = (epoch - warmup) / max(epochs - warmup, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # 训练循环
    best_val_mae = float("inf")
    best_epoch = 0
    no_improve = 0
    history = []
    early_stopping_enabled = early_stopping_patience > 0

    # 先保存一次初始权重，确保 best_model.pt 一定存在
    torch.save(model.state_dict(), output_dir / "best_model.pt")

    print(
        f"\n开始训练（{epochs} epochs，early_stopping="
        f"{'off' if not early_stopping_enabled else early_stopping_patience}，"
        f"batch_size={batch_size}，eval_batch_size={eval_batch_size}，"
        f"num_workers={num_workers}，amp={'on' if use_amp else 'off'}:{amp_dtype}）..."
    )

    # 计算训练集 naive MAE（seasonal naive，步长=1），用作 MASE 分母
    train_wvht = np.concatenate([
        train_dataset.groups[iid][0] for iid in train_dataset.groups
    ])
    train_naive_mae = float(np.mean(np.abs(train_wvht[1:] - train_wvht[:-1])))
    print(f"训练集 naive MAE（MASE 分母）：{train_naive_mae:.4f}")

    t0 = time.time()

    for epoch in range(epochs):
        # 计算动态 alpha（curriculum × horizon-aware）
        if adaptive:
            curriculum = min(1.0, (epoch + 1) / max(warmup_epochs, 1))
            alpha_dynamic = alpha_max * curriculum
        else:
            alpha_dynamic = None  # 使用 criterion 里的固定 alpha

        # 训练
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler, use_amp, amp_dtype,
            alpha_dynamic=alpha_dynamic
        )

        # 验证
        val_metrics = evaluate(model, val_loader, device, use_amp, amp_dtype)

        # 更新学习率
        scheduler.step()

        # 记录
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_metrics["loss"],
            "train_loss_hard": train_metrics["loss_hard"],
            "train_loss_soft": train_metrics["loss_soft"],
            "train_loss_freq": train_metrics["loss_freq"],
            "alpha_dynamic": alpha_dynamic if alpha_dynamic is not None else alpha,
            "val_mae": val_metrics["MAE"],
            "val_rmse": val_metrics["RMSE"],
            "val_mase": val_metrics["MASE"],
            "lr": optimizer.param_groups[0]["lr"],
        })

        if not np.isfinite(train_metrics["loss"]) or not np.isfinite(val_metrics["MAE"]):
            raise ValueError(
                f"训练在 epoch {epoch + 1} 出现非有限值："
                f"train_loss={train_metrics['loss']}, val_mae={val_metrics['MAE']}"
            )

        # 保存最佳模型（用 MAE 判断）
        if val_metrics["MAE"] < best_val_mae:
            best_val_mae = val_metrics["MAE"]
            best_epoch = epoch + 1
            no_improve = 0
            sd = model.state_dict()
            sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
            torch.save(sd, output_dir / "best_model.pt")
        else:
            no_improve += 1

        # 打印进度
        alpha_show = alpha_dynamic if alpha_dynamic is not None else alpha
        print(f"Epoch {epoch+1:3d}/{epochs} | "
              f"Loss: {train_metrics['loss']:.4f} "
              f"(hard={train_metrics['loss_hard']:.4f} soft={train_metrics['loss_soft']:.4f} freq={train_metrics['loss_freq']:.4f}) | "
              f"α={alpha_show:.3f} | "
              f"Val MAE: {val_metrics['MAE']:.4f} | "
              f"Best: {best_val_mae:.4f} @ {best_epoch} | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        # Early stopping
        if early_stopping_enabled and no_improve >= early_stopping_patience:
            print(
                f"\nEarly stopping at epoch {epoch+1} "
                f"(no improvement for {early_stopping_patience} epochs)"
            )
            break

    elapsed = time.time() - t0
    print(f"训练完成，耗时：{elapsed:.1f}秒，最佳 epoch: {best_epoch}")

    # 加载最佳模型并在测试集上评估
    best_model_path = output_dir / "best_model.pt"
    if best_model_path.exists():
        state_dict = torch.load(best_model_path, map_location=device)
        # 如果是 torch.compile 后的模型，key 可能带 _orig_mod. 前缀，统一去掉
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        # 如果当前 model 是 compiled，需要用 _orig_mod 访问原始模型
        if hasattr(model, "_orig_mod"):
            model._orig_mod.load_state_dict(state_dict)
        else:
            model.load_state_dict(state_dict)
    else:
        print("  ⚠ best_model.pt 不存在，使用最后一个 epoch 的权重")

    test_metrics = evaluate(model, test_loader, device, use_amp, amp_dtype, naive_mae=train_naive_mae)

    print(f"\n测试集指标：")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    # 保存结果
    result = {
        "alpha": alpha,
        "status": "success",
        "metrics": test_metrics,
        "best_val_mae": best_val_mae,
        "best_epoch": best_epoch,
        "total_epochs": epoch + 1,
        "early_stopping_patience": early_stopping_patience,
        "batch_size": batch_size,
        "eval_batch_size": eval_batch_size,
        "num_workers": num_workers,
        "amp": use_amp,
        "amp_dtype": amp_dtype,
        "elapsed_seconds": elapsed,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "test_samples": len(test_dataset),
    }

    result_path = output_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    history_df = pd.DataFrame(history)
    history_df.to_csv(output_dir / "history.csv", index=False)

    return result


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

    num_cols = [c for c in df.columns if c not in {"item_id", "timestamp", "datetime"}]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
    df[num_cols] = df[num_cols].ffill().bfill()
    # 全 NaN 列（ffill/bfill 无效）直接填 0
    df[num_cols] = df[num_cols].fillna(0.0)

    df.insert(0, "item_id", item_id)
    df = df.rename(columns={"datetime": "timestamp"})
    return df


def load_soft_labels(soft_labels_path: Path) -> pd.DataFrame:
    """加载 Chronos 软标签"""
    df = pd.read_csv(soft_labels_path, parse_dates=["timestamp"])
    return df


def align_common_feature_columns(frames: list[pd.DataFrame]) -> list[pd.DataFrame]:
    """仅保留所有站点共同拥有的特征，避免 concat 后某些站点整列为 NaN。"""
    if not frames:
        return frames

    common_dynamic = set(DYNAMIC_COLUMNS)
    common_known = set(KNOWN_COVARIATES)
    for df in frames:
        cols = set(df.columns)
        common_dynamic &= cols
        common_known &= cols

    ordered_dynamic = [c for c in DYNAMIC_COLUMNS if c in common_dynamic]
    ordered_known = [c for c in KNOWN_COVARIATES if c in common_known]
    keep_cols = ["item_id", "timestamp"] + ordered_dynamic + ordered_known

    dropped_dynamic = [c for c in DYNAMIC_COLUMNS if c not in ordered_dynamic]
    dropped_known = [c for c in KNOWN_COVARIATES if c not in ordered_known]
    if dropped_dynamic:
        print(f"  由于并非所有站点都包含这些动态特征，已自动丢弃：{dropped_dynamic}")
    if dropped_known:
        print(f"  由于并非所有站点都包含这些已知协变量，已自动丢弃：{dropped_known}")

    aligned = []
    for df in frames:
        sub = df[[c for c in keep_cols if c in df.columns]].copy()
        num_cols = [c for c in sub.columns if c not in {"item_id", "timestamp"}]
        if num_cols:
            sub[num_cols] = sub[num_cols].ffill().bfill()
        aligned.append(sub)
    return aligned


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--soft-labels-dir", type=Path, required=True)
    p.add_argument("--pretrained-model", type=Path, default=None)
    p.add_argument("--metadata-path", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--horizon-hours", type=int, default=24)
    p.add_argument("--horizons", default=None,
                   help="多个 horizon，逗号分隔，如 1,3,6,12,24,48,72,120（覆盖 --horizon-hours）")
    p.add_argument("--alphas", default="0.3,0.5,0.7")
    p.add_argument("--context-hours", type=int, default=168)
    p.add_argument("--freq", default="10min")
    p.add_argument("--train-ratio", type=float, default=0.70)
    p.add_argument("--val-ratio", type=float, default=0.20)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    p.add_argument("--eval-batch-multiplier", type=int, default=2)
    p.add_argument("--early-stopping-patience", type=int, default=0,
                   help="<=0 表示关闭早停，完整训练所有 epochs")
    p.add_argument("--no-amp", action="store_true",
                   help="关闭 CUDA 混合精度训练；默认在 CUDA 下自动开启")
    p.add_argument("--amp-dtype", choices=["bf16", "fp16"], default="bf16")
    p.add_argument("--torch-compile", action="store_true",
                   help="启用 torch.compile 进一步提升吞吐（首次编译会更慢）")
    p.add_argument("--gpu-id", default="0")
    p.add_argument("--limit-stations", type=int, default=0)
    # ── HPO 参数 ──────────────────────────────────────────────────
    p.add_argument("--hpo", action="store_true",
                   help="启用 Optuna 自动调参，搜索完后用最优参数做完整训练")
    p.add_argument("--hpo-trials", type=int, default=30,
                   help="HPO 搜索的 trial 数量（默认 30）")
    p.add_argument("--hpo-epochs", type=int, default=30,
                   help="每个 HPO trial 训练的 epoch 数（默认 30，越少越快）")
    p.add_argument("--hpo-batch-size", type=int, default=128,
                   help="HPO 阶段使用的 batch size（默认 128）")
    p.add_argument("--hpo-storage", default=None,
                   help="Optuna study 持久化路径，如 sqlite:///hpo.db（可选）")
    # ── 自适应蒸馏参数 ──────────────────────────────────────────────
    p.add_argument("--adaptive", action="store_true",
                   help="启用 horizon-aware + curriculum 自适应蒸馏权重")
    p.add_argument("--warmup-epochs", type=int, default=10,
                   help="curriculum warmup epoch 数（默认 10）")
    p.add_argument("--lambda-freq", type=float, default=0.0,
                   help="频域一致性损失系数（默认 0，即关闭）")
    return p


def main():
    args = build_parser().parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备：{device}")
    use_amp = bool(device.type == "cuda" and not args.no_amp)
    if args.no_amp:
        print("已按参数要求关闭 AMP")
    elif device.type == "cuda":
        print(f"已自动开启 AMP（{args.amp_dtype}）")
    else:
        print("检测到当前不是 CUDA 环境，AMP 不可用")

    # 解析 horizon 列表
    if args.horizons:
        horizon_list = [int(h.strip()) for h in args.horizons.split(",")]
    else:
        horizon_list = [args.horizon_hours]

    # 解析 alpha 列表
    alphas = [float(a.strip()) for a in args.alphas.split(",")]
    print(f"将测试 {len(horizon_list)} 个 horizon：{horizon_list}")
    print(f"将测试 {len(alphas)} 个 α 值：{alphas}")

    steps_per_hour = {"10min": 6, "1h": 1, "30min": 2}.get(args.freq, 6)
    context_length = args.context_hours * steps_per_hour

    # 加载 metadata
    if args.metadata_path and args.metadata_path.exists():
        meta = json.loads(args.metadata_path.read_text(encoding="utf-8"))
        shared_start = pd.Timestamp(meta["shared_start"])
        shared_end = pd.Timestamp(meta["shared_end"])
    else:
        shared_start = None
        shared_end = None

    # 加载所有站点数据（只加载一次）
    files = sorted(args.input_dir.glob("*.csv"))
    if args.limit_stations > 0:
        files = files[:args.limit_stations]
    print(f"\n找到 {len(files)} 个站点文件")

    all_data = []
    for path in files:
        item_id = path.stem.split("_")[0]
        print(f"  加载 {item_id}...")
        df = load_station(path, shared_start, shared_end)
        all_data.append(df)

    all_data = align_common_feature_columns(all_data)

    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"总数据量：{len(combined_df)} 行，{combined_df['item_id'].nunique()} 个站点")

    # 切分数据（只切分一次）
    train_dfs, val_dfs, test_dfs = [], [], []
    for item_id in combined_df["item_id"].unique():
        item_df = combined_df[combined_df["item_id"] == item_id].sort_values("timestamp").reset_index(drop=True)
        n = len(item_df)
        train_end = int(n * args.train_ratio)
        val_end = int(n * (args.train_ratio + args.val_ratio))
        train_dfs.append(item_df.iloc[:train_end])
        val_dfs.append(item_df.iloc[train_end:val_end])
        test_dfs.append(item_df.iloc[val_end:])

    train_df = pd.concat(train_dfs, ignore_index=True)
    val_df = pd.concat(val_dfs, ignore_index=True)
    test_df = pd.concat(test_dfs, ignore_index=True)
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # 遍历所有 horizon
    for horizon_hours in horizon_list:
        pred_length = horizon_hours * steps_per_hour
        print(f"\n{'#'*70}")
        print(f"# Horizon: {horizon_hours}h ({pred_length} steps)")
        print(f"{'#'*70}")

        # 加载软标签
        soft_labels_path = args.soft_labels_dir / f"labels_{horizon_hours:03d}h.csv"
        if not soft_labels_path.exists():
            soft_labels_path = args.soft_labels_dir / f"labels_{horizon_hours}h.csv"
        if not soft_labels_path.exists():
            print(f"  ⚠ 软标签不存在，跳过：{soft_labels_path}")
            continue

        print(f"加载软标签：{soft_labels_path}")
        soft_labels_df = load_soft_labels(soft_labels_path)

        # ── HPO 阶段（可选）──────────────────────────────────────
        best_hpo_params = None
        if args.hpo:
            if not HAS_OPTUNA:
                print("⚠ 未安装 optuna，跳过 HPO，使用默认参数")
            else:
                print(f"\n开始 HPO：{args.hpo_trials} trials × {args.hpo_epochs} epochs")
                best_hpo_params = run_hpo(
                    train_df, val_df,
                    soft_labels_df,
                    context_length=context_length,
                    pred_length=pred_length,
                    device=device,
                    use_amp=use_amp,
                    amp_dtype=args.amp_dtype,
                    num_workers=args.num_workers,
                    n_trials=args.hpo_trials,
                    hpo_epochs=args.hpo_epochs,
                    hpo_batch_size=args.hpo_batch_size,
                )
                # 保存 HPO 结果
                hpo_result_path = args.output_dir / f"horizon_{horizon_hours:03d}h" / "hpo_best_params.json"
                hpo_result_path.parent.mkdir(parents=True, exist_ok=True)
                hpo_result_path.write_text(json.dumps(best_hpo_params, indent=2), encoding="utf-8")
                print(f"HPO 最优参数已保存：{hpo_result_path}")

        # ── 完整训练阶段 ──────────────────────────────────────────
        all_results = []
        if args.hpo and best_hpo_params:
            # HPO 模式：只用最优参数训练一次
            print(f"\n使用 HPO 最优参数进行完整训练...")
            output_dir = args.output_dir / f"horizon_{horizon_hours:03d}h" / "best_hpo"
            output_dir.mkdir(parents=True, exist_ok=True)

            result = train_one_alpha(
                train_df, val_df, test_df,
                soft_labels_df,
                alpha=best_hpo_params.get("alpha", 0.5),
                context_length=context_length,
                pred_length=pred_length,
                pretrained_path=args.pretrained_model,
                output_dir=output_dir,
                epochs=args.epochs,
                batch_size=best_hpo_params.get("batch_size", args.batch_size),
                lr=best_hpo_params.get("lr", args.lr),
                device=device,
                num_workers=args.num_workers,
                eval_batch_multiplier=args.eval_batch_multiplier,
                use_amp=use_amp,
                amp_dtype=args.amp_dtype,
                early_stopping_patience=args.early_stopping_patience,
                torch_compile=args.torch_compile,
                hpo_params=best_hpo_params,
                adaptive=args.adaptive,
                horizon_hours=horizon_hours,
                warmup_epochs=args.warmup_epochs,
                lambda_freq=args.lambda_freq,
            )
            all_results.append(result)
        else:
            # 普通模式：遍历所有 alpha
            for alpha in alphas:
                output_dir = args.output_dir / f"horizon_{horizon_hours:03d}h" / f"alpha_{alpha:.2f}"
                output_dir.mkdir(parents=True, exist_ok=True)

                result = train_one_alpha(
                    train_df, val_df, test_df,
                    soft_labels_df,
                    alpha=alpha,
                    context_length=context_length,
                    pred_length=pred_length,
                    pretrained_path=args.pretrained_model,
                    output_dir=output_dir,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    device=device,
                    num_workers=args.num_workers,
                    eval_batch_multiplier=args.eval_batch_multiplier,
                    use_amp=use_amp,
                    amp_dtype=args.amp_dtype,
                    early_stopping_patience=args.early_stopping_patience,
                    torch_compile=args.torch_compile,
                    hpo_params=None,
                    adaptive=args.adaptive,
                    horizon_hours=horizon_hours,
                    warmup_epochs=args.warmup_epochs,
                    lambda_freq=args.lambda_freq,
                )
                all_results.append(result)

        # 保存汇总
        summary_path = args.output_dir / f"horizon_{horizon_hours:03d}h" / "summary.json"
        summary_path.write_text(json.dumps({
            "horizon_hours": horizon_hours,
            "alphas_tested": alphas,
            "results": all_results,
        }, indent=2, ensure_ascii=False), encoding="utf-8")

        print(f"\n{'='*70}")
        print(f"Horizon {horizon_hours}h 汇总")
        print(f"{'='*70}")
        print(f"{'Alpha':<8} {'MAE':<10} {'RMSE':<10} {'Status':<10}")
        print(f"{'-'*70}")
        for r in all_results:
            if r["status"] == "success":
                print(f"{r['alpha']:<8.2f} {r['metrics']['MAE']:<10.4f} {r['metrics']['RMSE']:<10.4f} success")
            else:
                print(f"{r['alpha']:<8.2f} {'N/A':<10} {'N/A':<10} failed")

        print(f"\n汇总结果已保存：{summary_path}")

    print(f"\n全部完成，输出目录：{args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

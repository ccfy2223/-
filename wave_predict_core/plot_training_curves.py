"""
plot_training_curves.py
生成训练过程图表：
- 每个 horizon 的训练曲线（train_loss + val_loss）
- 综合对比图（所有模型×所有 horizon）
- 代表性 horizon 对比（1h/24h/120h）
- Early stopping 统计
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.family': 'DejaVu Sans',
                            'axes.unicode_minus': False,
                            'figure.dpi': 150})

from pathlib import Path

# 配置
run_dir = Path('autogluon_runs_local_3060')
out_dir = Path('thesis_assets/figures')
out_dir.mkdir(parents=True, exist_ok=True)

HORIZONS = [1, 3, 6, 12, 24, 48, 72, 120]
MODELS = ['TemporalFusionTransformer', 'PatchTST', 'DLinear', 'DeepAR']
MODEL_LABELS = {'TemporalFusionTransformer': 'TFT', 'PatchTST': 'PatchTST',
                'DLinear': 'DLinear', 'DeepAR': 'DeepAR'}
COLORS = {'TemporalFusionTransformer': '#1f77b4', 'PatchTST': '#ff7f0e',
          'DLinear': '#2ca02c', 'DeepAR': '#d62728'}

def load_training_curve(horizon_h: int, model: str) -> pd.DataFrame | None:
    """加载单个模型的训练曲线"""
    path = run_dir / f'horizon_{horizon_h:03d}h' / 'training_curves' / model / 'version_0' / 'metrics.csv'
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df

# ── 图A: 每个 horizon 的训练曲线（4 模型对比）──
print('[1/4] 生成每个 horizon 的训练曲线...')
for h in HORIZONS:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, model in enumerate(MODELS):
        ax = axes[idx]
        df = load_training_curve(h, model)

        if df is not None and 'epoch' in df.columns:
            # Train loss
            if 'train_loss' in df.columns:
                train = df.dropna(subset=['train_loss'])
                # 平滑处理：移动平均
                if len(train) > 5:
                    train_smooth = train['train_loss'].rolling(window=5, center=True).mean()
                else:
                    train_smooth = train['train_loss']
                ax.plot(train['epoch'], train_smooth,
                       label='Train Loss', color=COLORS[model], linewidth=2.5, alpha=0.8, linestyle='-')

            # Val loss
            if 'val_loss' in df.columns:
                val = df.dropna(subset=['val_loss'])
                # 平滑处理：移动平均
                if len(val) > 5:
                    val_smooth = val['val_loss'].rolling(window=5, center=True).mean()
                else:
                    val_smooth = val['val_loss']
                ax.plot(val['epoch'], val_smooth,
                       label='Validation Loss', color=COLORS[model], linewidth=2.5, alpha=0.95, linestyle='--')

            ax.set_xlabel('Epoch', fontsize=10)
            ax.set_ylabel('Loss', fontsize=10)
            ax.set_title(f'{MODEL_LABELS[model]} — {h}h Horizon', fontsize=11, fontweight='bold')
            ax.legend(fontsize=9, loc='best')
            ax.grid(linestyle='--', alpha=0.3)
        else:
            ax.text(0.5, 0.5, f'{MODEL_LABELS[model]}\nNo data',
                   ha='center', va='center', fontsize=10, transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])

    fig.tight_layout()
    fig.savefig(out_dir / f'training_curves_{h:03d}h.png', bbox_inches='tight', dpi=200)
    print(f'  已保存: training_curves_{h:03d}h.png')
    plt.close(fig)

# ── 图B: 综合网格（4×8 子图）──
print('[2/4] 生成综合训练曲线网格...')
fig, axes = plt.subplots(4, 8, figsize=(28, 14), squeeze=False)

for row, model in enumerate(MODELS):
    for col, h in enumerate(HORIZONS):
        ax = axes[row][col]
        df = load_training_curve(h, model)

        if df is not None and 'epoch' in df.columns:
            if 'train_loss' in df.columns:
                train = df.dropna(subset=['train_loss'])
                if len(train) > 5:
                    train_smooth = train['train_loss'].rolling(window=5, center=True).mean()
                else:
                    train_smooth = train['train_loss']
                ax.plot(train['epoch'], train_smooth,
                       color=COLORS[model], linewidth=1.5, alpha=0.7, label='Train')
            if 'val_loss' in df.columns:
                val = df.dropna(subset=['val_loss'])
                if len(val) > 5:
                    val_smooth = val['val_loss'].rolling(window=5, center=True).mean()
                else:
                    val_smooth = val['val_loss']
                ax.plot(val['epoch'], val_smooth,
                       color=COLORS[model], linewidth=2, linestyle='--', alpha=0.9, label='Val')

            ax.set_title(f'{MODEL_LABELS[model]} {h}h', fontsize=9, fontweight='bold')
            ax.tick_params(labelsize=7)
            ax.grid(linestyle='--', alpha=0.25)
            if col == 0:
                ax.set_ylabel('Loss', fontsize=8)
            if row == 3:
                ax.set_xlabel('Epoch', fontsize=8)
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])

fig.suptitle('Training Curves Overview — All Models × All Horizons',
             fontsize=16, fontweight='bold', y=0.995)
fig.tight_layout()
fig.savefig(out_dir / 'training_curves_overview.png', bbox_inches='tight', dpi=200)
print('  已保存: training_curves_overview.png')
plt.close(fig)

# ── 图C: 代表性 horizon 对比（1h/24h/120h）──
print('[3/4] 生成代表性 horizon 对比图...')
rep_horizons = [1, 24, 120]
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, h in enumerate(rep_horizons):
    ax = axes[idx]
    for model in MODELS:
        df = load_training_curve(h, model)
        if df is not None and 'val_loss' in df.columns:
            val = df.dropna(subset=['val_loss'])
            # 平滑处理
            if len(val) > 5:
                val_smooth = val['val_loss'].rolling(window=5, center=True).mean()
            else:
                val_smooth = val['val_loss']
            ax.plot(val['epoch'], val_smooth,
                   label=MODEL_LABELS[model], color=COLORS[model], linewidth=2.5, alpha=0.9)

    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Validation Loss', fontsize=11)
    ax.set_title(f'{h}h Horizon', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(linestyle='--', alpha=0.3)

fig.suptitle('Representative Horizons — Validation Loss Comparison',
             fontsize=14, fontweight='bold', y=1.01)
fig.tight_layout()
fig.savefig(out_dir / 'training_curves_representative.png', bbox_inches='tight', dpi=200)
print('  已保存: training_curves_representative.png')
plt.close(fig)

# ── 图D: Early stopping 统计（实际训练轮数）──
print('[4/4] 生成 early stopping 统计图...')
epoch_data = []
for model in MODELS:
    for h in HORIZONS:
        df = load_training_curve(h, model)
        if df is not None and 'epoch' in df.columns:
            max_epoch = df['epoch'].max()
            epoch_data.append({'model': model, 'horizon': h, 'epochs': max_epoch})

if epoch_data:
    epoch_df = pd.DataFrame(epoch_data)
    pivot = epoch_df.pivot(index='model', columns='horizon', values='epochs')

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(HORIZONS))
    width = 0.2

    for i, model in enumerate(MODELS):
        if model in pivot.index:
            vals = [pivot.loc[model, h] if h in pivot.columns else 0 for h in HORIZONS]
            ax.bar(x + i * width, vals, width, label=MODEL_LABELS[model], color=COLORS[model], alpha=0.85)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([f'{h}h' for h in HORIZONS], fontsize=9)
    ax.set_xlabel('Prediction Horizon', fontsize=11)
    ax.set_ylabel('Training Epochs', fontsize=11)
    ax.set_title('Actual Training Epochs (Early Stopping)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / 'training_epochs_early_stopping.png', bbox_inches='tight')
    print('  已保存: training_epochs_early_stopping.png')
    plt.close(fig)

print(f'\n全部完成，输出目录：{out_dir.resolve()}')

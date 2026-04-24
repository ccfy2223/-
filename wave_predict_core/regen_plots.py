"""
regen_plots.py — 去除 WeightedEnsemble，重新生成所有论文图表
输出到 thesis_assets/figures/
"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, '.')

import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.family': 'DejaVu Sans',
                            'axes.unicode_minus': False,
                            'figure.dpi': 150})

from pathlib import Path
from plot_results import (
    load_metrics_summary, load_all_leaderboards, ensure_positive,
    get_available_horizons, savefig,
    METRICS, METRIC_LABELS, PALETTE,
)

EXCLUDE = {'WeightedEnsemble'}

run_dir = Path('autogluon_runs')
out_dir = Path('thesis_assets/figures')
out_dir.mkdir(parents=True, exist_ok=True)
fmt = 'png'

available_horizons = get_available_horizons(run_dir)
metrics_df = load_metrics_summary(run_dir)
test_lb = load_all_leaderboards(run_dir, split='test')
test_lb = ensure_positive(test_lb, METRICS + ['score_test'])
val_lb  = load_all_leaderboards(run_dir, split='validation')
val_lb  = ensure_positive(val_lb, METRICS + ['score_val'])

# 过滤 WeightedEnsemble
test_lb = test_lb[~test_lb['model'].isin(EXCLUDE)].copy()
val_lb  = val_lb[~val_lb['model'].isin(EXCLUDE)].copy()

models = test_lb['model'].unique().tolist()
color_map = {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(models)}
marker_styles = ['o', 's', '^', 'D', 'v', 'P', '*', 'X', 'h', '+', 'x', '1', '2']

# ── 图1: 数据集概览（不涉及模型，直接复制原图逻辑，这里跳过重新生成）──
# fig1_dataset_overview 不含模型对比，保持原图即可

# ── 图2: 各模型×指标折线图 ──
print('[1/6] 各模型指标折线对比图...')
for metric in METRICS:
    if metric not in test_lb.columns:
        continue
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, model in enumerate(models):
        sub = test_lb[test_lb['model'] == model].sort_values('horizon_hours')
        sub = ensure_positive(sub, [metric])
        ax.plot(sub['horizon_hours'], sub[metric],
                marker=marker_styles[i % len(marker_styles)],
                color=color_map[model], label=model, linewidth=1.8, markersize=6)
    ax.set_xlabel('Prediction Horizon (hours)', fontsize=12)
    ax.set_ylabel(METRIC_LABELS.get(metric, metric), fontsize=12)
    ax.set_title(f'Model Comparison — {metric} across Horizons', fontsize=13, fontweight='bold')
    ax.set_xticks(available_horizons)
    ax.legend(fontsize=9, loc='upper left', framealpha=0.8)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    fig.tight_layout()
    savefig(fig, out_dir / f'fig2_comparison_{metric.lower()}', fmt)

# ── 图3: 最优模型综合误差曲线 ──
print('[2/6] 最优模型综合误差曲线...')
df = metrics_df[metrics_df['horizon_hours'].isin(available_horizons)].sort_values('horizon_hours')
fig, ax1 = plt.subplots(figsize=(11, 5))
ax2 = ax1.twinx()
lines = []
xs = df['horizon_hours'].tolist()
specs = [
    ('MAE',   ax1, '-',  'o', '#1f77b4', 'MAE'),
    ('RMSE',  ax1, '--', 's', '#ff7f0e', 'RMSE'),
    ('RMSLE', ax1, ':',  '^', '#2ca02c', 'RMSLE'),
    ('MASE',  ax2, '-.', 'D', '#d62728', 'MASE'),
    ('SMAPE', ax2, '--', 'v', '#9467bd', 'SMAPE'),
]
for col, ax, ls, mk, clr, lbl in specs:
    if col in df.columns:
        l, = ax.plot(xs, df[col].abs(), ls + mk, color=clr, linewidth=2, markersize=6, label=lbl)
        lines.append(l)
ax1.set_xlabel('Prediction Horizon (hours)', fontsize=12)
ax1.set_ylabel('MAE / RMSE / RMSLE (m)', fontsize=12, color='#1f77b4')
ax2.set_ylabel('MASE / SMAPE', fontsize=12, color='#d62728')
ax1.set_xticks(available_horizons)
ax1.legend(lines, [l.get_label() for l in lines], fontsize=9, loc='upper left', framealpha=0.85)
ax1.set_title('Best Model — All Metrics across Prediction Horizons', fontsize=13, fontweight='bold')
ax1.grid(axis='y', linestyle='--', alpha=0.3)
fig.tight_layout()
savefig(fig, out_dir / 'fig3_best_model_all_metrics', fmt)

# ── 图4: MASE 热力图 ──
print('[3/6] MASE 热力图...')
pivot = test_lb.pivot_table(index='model', columns='horizon_hours', values='MASE', aggfunc='mean').abs()
pivot = pivot.reindex(columns=[h for h in available_horizons if h in pivot.columns])
pivot = pivot.sort_values(by=pivot.columns.tolist(), na_position='last')
n_models, n_horizons = pivot.shape
fig, ax = plt.subplots(figsize=(max(10, n_horizons * 0.9), max(3, n_models * 0.6)))
im = ax.imshow(pivot.values, aspect='auto', cmap='YlOrRd', interpolation='nearest')
plt.colorbar(im, ax=ax, label='MASE')
ax.set_xticks(range(n_horizons))
ax.set_xticklabels([f'{h}h' for h in pivot.columns], fontsize=9)
ax.set_yticks(range(n_models))
ax.set_yticklabels(pivot.index.tolist(), fontsize=9)
ax.set_xlabel('Prediction Horizon', fontsize=11)
ax.set_title('Model MASE Heatmap (lower = better)', fontsize=13, fontweight='bold')
for r in range(n_models):
    for c in range(n_horizons):
        val = pivot.values[r, c]
        if not np.isnan(val):
            ax.text(c, r, f'{val:.3f}', ha='center', va='center', fontsize=7,
                    color='black' if val < pivot.values.max() * 0.6 else 'white')
fig.tight_layout()
savefig(fig, out_dir / 'fig4_mase_heatmap', fmt)

# ── 图5: 最优单模型 Val vs Test MASE 汇总 ──
print('[4/6] Val vs Test MASE 汇总图...')
rows = []
for h in available_horizons:
    t = test_lb[test_lb['horizon_hours'] == h].sort_values('MASE')
    v = val_lb[val_lb['horizon_hours'] == h]
    if t.empty:
        continue
    best = t.iloc[0]
    val_row = v[v['model'] == best['model']]
    val_mase = float(val_row['MASE'].iloc[0]) if not val_row.empty else np.nan
    short_name = best['model'][:12] + '..' if len(best['model']) > 14 else best['model']
    rows.append({'horizon': f"{h}h", 'model': short_name,
                 'val_MASE': val_mase, 'test_MASE': best['MASE']})
cmp_df = pd.DataFrame(rows)
x = np.arange(len(cmp_df))
w = 0.35
fig, ax = plt.subplots(figsize=(12, 5))
b1 = ax.bar(x - w / 2, cmp_df['val_MASE'], w, label='Validation MASE', color='#5b9bd5', alpha=0.85)
b2 = ax.bar(x + w / 2, cmp_df['test_MASE'], w, label='Test MASE', color='#ed7d31', alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels([f"{r['horizon']}\n({r['model']})" for _, r in cmp_df.iterrows()], fontsize=8)
ax.set_ylabel('MASE', fontsize=11)
ax.set_title('Best Single Model: Validation vs Test MASE per Horizon', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(axis='y', linestyle='--', alpha=0.4)
for bar in list(b1) + list(b2):
    h_val = bar.get_height()
    if not np.isnan(h_val):
        ax.text(bar.get_x() + bar.get_width() / 2, h_val + 0.005,
                f'{h_val:.4f}', ha='center', va='bottom', fontsize=6)
fig.tight_layout()
savefig(fig, out_dir / 'fig5_val_vs_test_mase', fmt)

# ── 图6: 综合子图矩阵 ──
print('[5/6] 综合子图矩阵...')
df2 = metrics_df[metrics_df['horizon_hours'].isin(available_horizons)].sort_values('horizon_hours')
available_m = [m for m in METRICS if m in df2.columns]
ncols = 3
nrows = (len(available_m) + ncols - 1) // ncols
colors2 = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 3.8), squeeze=False)
for idx, metric in enumerate(available_m):
    ax = axes[idx // ncols][idx % ncols]
    ys = df2[metric].abs().tolist()
    xs2 = df2['horizon_hours'].tolist()
    ax.plot(xs2, ys, 'o-', color=colors2[idx % len(colors2)], linewidth=2, markersize=6)
    ax.fill_between(xs2, ys, alpha=0.12, color=colors2[idx % len(colors2)])
    ax.set_xticks(available_horizons)
    ax.tick_params(axis='x', labelsize=7, rotation=45)
    ax.set_xlabel('Horizon (h)', fontsize=9)
    ax.set_ylabel(METRIC_LABELS.get(metric, metric), fontsize=9)
    ax.set_title(metric, fontsize=11, fontweight='bold')
    ax.grid(linestyle='--', alpha=0.35)
    for xv, yv in zip(xs2, ys):
        ax.annotate(f'{yv:.4f}', (xv, yv), textcoords='offset points',
                    xytext=(0, 6), ha='center', fontsize=6, color=colors2[idx % len(colors2)])
for idx in range(len(available_m), nrows * ncols):
    axes[idx // ncols][idx % ncols].set_visible(False)
fig.suptitle('Best Model — Forecast Error across All Horizons', fontsize=14, fontweight='bold', y=1.01)
fig.tight_layout()
savefig(fig, out_dir / 'fig6_summary_grid', fmt)

# ── 图7: 训练耗时 ──
print('[6/6] 训练耗时图...')
if 'elapsed_seconds' in metrics_df.columns:
    df3 = metrics_df[metrics_df['horizon_hours'].isin(available_horizons)].sort_values('horizon_hours')
    xs3 = df3['horizon_hours'].tolist()
    ys3 = (df3['elapsed_seconds'] / 60).tolist()
    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar([str(x) + 'h' for x in xs3], ys3, color='#5b9bd5', alpha=0.85, edgecolor='white')
    for bar, yv in zip(bars, ys3):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f'{yv:.1f}m', ha='center', va='bottom', fontsize=8)
    ax.set_xlabel('Prediction Horizon', fontsize=11)
    ax.set_ylabel('Training Time (minutes)', fontsize=11)
    ax.set_title('Training Time per Horizon', fontsize=12, fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    fig.tight_layout()
    savefig(fig, out_dir / 'fig7_training_time', fmt)

print(f'\n全部完成，输出目录：{out_dir.resolve()}')
print('生成图片：')
for f in sorted(out_dir.glob('fig*.png')):
    print(f'  {f.name}')

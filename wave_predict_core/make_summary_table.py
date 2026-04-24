"""
make_summary_table.py
生成所有模型 × 所有 horizon 的汇总表格（MAE / MASE / RMSE）
输出到 thesis_assets/figures/
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

run_dir = Path('autogluon_runs')
out_dir = Path('thesis_assets/figures')
out_dir.mkdir(parents=True, exist_ok=True)

HORIZONS = [1, 3, 6, 12, 24, 48, 72, 120]

MODEL_MAP = {
    'Chronos[amazon__chronos-t5-large]': 'Chronos',
    'PatchTST':                          'PatchTST',
    'TemporalFusionTransformer':         'TFT',
    'DLinear':                           'DLinear',
    'DeepAR':                            'DeepAR',
    'WeightedEnsemble':                  'Ensemble',
    'RecursiveTabular':                  'RecursiveTabular',
    'DirectTabular':                     'DirectTabular',
}

MODEL_ORDER = ['Chronos', 'PatchTST', 'TFT', 'DLinear',
               'Ensemble', 'RecursiveTabular', 'DirectTabular', 'DeepAR']

# ── 加载所有数据 ───────────────────────────────────────────────────────────────
records = []
for h in HORIZONS:
    path = run_dir / f'horizon_{h:03d}h' / 'leaderboard_test.csv'
    if not path.exists():
        continue
    df = pd.read_csv(path)
    for _, row in df.iterrows():
        key = row['model']
        if key not in MODEL_MAP:
            continue
        records.append({
            'horizon': h,
            'model': MODEL_MAP[key],
            'MAE':   abs(float(row['MAE'])),
            'MASE':  abs(float(row['MASE'])),
            'RMSE':  abs(float(row['RMSE'])),
            'SMAPE': abs(float(row['SMAPE'])),
        })

data = pd.DataFrame(records)

# ── 生成 CSV 汇总表 ────────────────────────────────────────────────────────────
for metric in ['MAE', 'MASE', 'RMSE', 'SMAPE']:
    pivot = data.pivot_table(index='model', columns='horizon', values=metric)
    pivot = pivot.reindex(index=[m for m in MODEL_ORDER if m in pivot.index])
    pivot.columns = [f'{h}h' for h in pivot.columns]
    pivot = pivot.round(4)
    pivot.to_csv(out_dir / f'summary_{metric.lower()}.csv')
    print(f'\n=== {metric} ===')
    print(pivot.to_string())

# ── 图1：MAE 完整热力图 ────────────────────────────────────────────────────────
print('\n[1/3] 绘制 MAE 完整热力图...')
pivot_mae = data.pivot_table(index='model', columns='horizon', values='MAE')
pivot_mae = pivot_mae.reindex(index=[m for m in MODEL_ORDER if m in pivot_mae.index])

fig, ax = plt.subplots(figsize=(16, 6))
im = ax.imshow(pivot_mae.values, aspect='auto', cmap='RdYlGn_r',
               vmin=0, vmax=0.6)

ax.set_xticks(range(len(pivot_mae.columns)))
ax.set_xticklabels([f'{h}h' for h in pivot_mae.columns], fontsize=11)
ax.set_yticks(range(len(pivot_mae.index)))
ax.set_yticklabels(pivot_mae.index, fontsize=11)
ax.set_xlabel('Prediction Horizon', fontsize=12)
ax.set_title('MAE — All Models × All Horizons', fontsize=14, fontweight='bold')

for i in range(len(pivot_mae.index)):
    for j in range(len(pivot_mae.columns)):
        val = pivot_mae.values[i, j]
        if not np.isnan(val):
            color = 'white' if val > 0.35 else 'black'
            ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                    fontsize=9, color=color, fontweight='bold')

plt.colorbar(im, ax=ax, label='MAE (m)', shrink=0.85)
fig.tight_layout()
fig.savefig(out_dir / 'full_mae_heatmap.png', bbox_inches='tight', dpi=200)
print('  已保存: full_mae_heatmap.png')
plt.close(fig)

# ── 图2：MASE 完整热力图 ───────────────────────────────────────────────────────
print('[2/3] 绘制 MASE 完整热力图...')
pivot_mase = data.pivot_table(index='model', columns='horizon', values='MASE')
pivot_mase = pivot_mase.reindex(index=[m for m in MODEL_ORDER if m in pivot_mase.index])

fig, ax = plt.subplots(figsize=(16, 6))
im = ax.imshow(pivot_mase.values, aspect='auto', cmap='RdYlGn_r',
               vmin=0, vmax=1.5)

ax.set_xticks(range(len(pivot_mase.columns)))
ax.set_xticklabels([f'{h}h' for h in pivot_mase.columns], fontsize=11)
ax.set_yticks(range(len(pivot_mase.index)))
ax.set_yticklabels(pivot_mase.index, fontsize=11)
ax.set_xlabel('Prediction Horizon', fontsize=12)
ax.set_title('MASE — All Models × All Horizons', fontsize=14, fontweight='bold')

for i in range(len(pivot_mase.index)):
    for j in range(len(pivot_mase.columns)):
        val = pivot_mase.values[i, j]
        if not np.isnan(val):
            color = 'white' if val > 0.9 else 'black'
            ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                    fontsize=9, color=color, fontweight='bold')

plt.colorbar(im, ax=ax, label='MASE', shrink=0.85)
fig.tight_layout()
fig.savefig(out_dir / 'full_mase_heatmap.png', bbox_inches='tight', dpi=200)
print('  已保存: full_mase_heatmap.png')
plt.close(fig)

# ── 图3：各模型 MAE 折线图（完整版）──────────────────────────────────────────
print('[3/3] 绘制完整折线图...')
COLORS = {
    'Chronos':          '#e6194b',
    'PatchTST':         '#3cb44b',
    'TFT':              '#4363d8',
    'DLinear':          '#f58231',
    'DeepAR':           '#911eb4',
    'Ensemble':         '#42d4f4',
    'RecursiveTabular': '#808000',
    'DirectTabular':    '#a9a9a9',
}
MARKERS = {
    'Chronos': 'o', 'PatchTST': 's', 'TFT': '^', 'DLinear': 'D',
    'DeepAR': 'v', 'Ensemble': 'P', 'RecursiveTabular': 'x', 'DirectTabular': '+',
}
LINESTYLES = {
    'Chronos': '-', 'PatchTST': '-', 'TFT': '-', 'DLinear': '-',
    'DeepAR': '--', 'Ensemble': '-', 'RecursiveTabular': '--', 'DirectTabular': '--',
}

# 主要模型（实线）和次要模型（虚线）分开
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

for ax_idx, (ax, models_to_show, title) in enumerate(zip(
    axes,
    [['Chronos', 'PatchTST', 'TFT', 'DLinear', 'Ensemble'],
     ['Chronos', 'RecursiveTabular', 'DirectTabular', 'DeepAR', 'Ensemble']],
    ['Deep Learning Models', 'Tabular & Baseline Models'],
)):
    for model in models_to_show:
        sub = data[data['model'] == model].sort_values('horizon')
        if sub.empty:
            continue
        ax.plot(sub['horizon'], sub['MAE'],
                label=model,
                color=COLORS[model],
                marker=MARKERS[model],
                linestyle=LINESTYLES[model],
                linewidth=2.5, markersize=8, alpha=0.9)

    ax.set_xlabel('Prediction Horizon (h)', fontsize=12)
    ax.set_ylabel('MAE (m)', fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xticks(HORIZONS)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(linestyle='--', alpha=0.35)

fig.suptitle('MAE vs Prediction Horizon — All Models',
             fontsize=14, fontweight='bold', y=1.02)
fig.tight_layout()
fig.savefig(out_dir / 'full_mae_lines.png', bbox_inches='tight', dpi=200)
print('  已保存: full_mae_lines.png')
plt.close(fig)

print(f'\n全部完成，输出目录：{out_dir.resolve()}')

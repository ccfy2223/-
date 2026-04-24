"""
plot_predictions.py
===================
从 autogluon_runs 的 leaderboard_test.csv 提取各模型指标
绘制多维度对比图，保存到 thesis_assets/figures/
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.unicode_minus': False,
    'figure.dpi': 150,
})

from pathlib import Path

# ── 配置 ──────────────────────────────────────────────────────────────────────
run_dir = Path('autogluon_runs')
out_dir = Path('thesis_assets/figures')
out_dir.mkdir(parents=True, exist_ok=True)

HORIZONS = [1, 3, 6, 12, 24, 48, 72, 120]

# 只展示主要模型
MODELS = {
    'Chronos[amazon__chronos-t5-large]': 'Chronos',
    'PatchTST':                          'PatchTST',
    'TemporalFusionTransformer':         'TFT',
    'DLinear':                           'DLinear',
    'DeepAR':                            'DeepAR',
    'WeightedEnsemble':                  'Ensemble',
    'RecursiveTabular':                  'RecursiveTabular',
    'DirectTabular':                     'DirectTabular',
}

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
    'Chronos':          'o',
    'PatchTST':         's',
    'TFT':              '^',
    'DLinear':          'D',
    'DeepAR':           'v',
    'Ensemble':         'P',
    'RecursiveTabular': 'x',
    'DirectTabular':    '+',
}

METRICS = ['MAE', 'RMSE', 'MASE', 'SMAPE']
METRIC_LABELS = {
    'MAE':   'MAE (m)',
    'RMSE':  'RMSE (m)',
    'MASE':  'MASE',
    'SMAPE': 'SMAPE (%)',
}

# ── 加载数据 ──────────────────────────────────────────────────────────────────
def load_all():
    records = []
    for h in HORIZONS:
        path = run_dir / f'horizon_{h:03d}h' / 'leaderboard_test.csv'
        if not path.exists():
            continue
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            model_key = row['model']
            if model_key not in MODELS:
                continue
            label = MODELS[model_key]
            rec = {'horizon': h, 'model': label}
            for m in METRICS:
                rec[m] = abs(float(row[m])) if m in row and pd.notna(row[m]) else np.nan
            records.append(rec)
    return pd.DataFrame(records)

data = load_all()
print(f"加载数据：{len(data)} 条记录")
print(data.groupby('model')['horizon'].count())

# ── 图1：各指标随 Horizon 变化折线图（4×1）────────────────────────────────────
print('[1/4] 绘制各指标折线图...')
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes = axes.flatten()

for idx, metric in enumerate(METRICS):
    ax = axes[idx]
    for model_label in MODELS.values():
        sub = data[data['model'] == model_label].sort_values('horizon')
        if sub.empty:
            continue
        ax.plot(sub['horizon'], sub[metric],
                label=model_label,
                color=COLORS[model_label],
                marker=MARKERS[model_label],
                linewidth=2, markersize=7, alpha=0.9)

    ax.set_xlabel('Prediction Horizon (h)', fontsize=11)
    ax.set_ylabel(METRIC_LABELS[metric], fontsize=11)
    ax.set_title(f'{metric} vs Horizon', fontsize=12, fontweight='bold')
    ax.set_xticks(HORIZONS)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(linestyle='--', alpha=0.35)

fig.suptitle('Model Performance Across Prediction Horizons',
             fontsize=14, fontweight='bold', y=1.01)
fig.tight_layout()
fig.savefig(out_dir / 'metrics_vs_horizon.png', bbox_inches='tight', dpi=200)
print('  已保存: metrics_vs_horizon.png')
plt.close(fig)

# ── 图2：MAE 热力图（模型 × Horizon）────────────────────────────────────────
print('[2/4] 绘制 MAE 热力图...')
model_order = list(MODELS.values())
pivot = data.pivot_table(index='model', columns='horizon', values='MAE')
pivot = pivot.reindex(index=[m for m in model_order if m in pivot.index])

fig, ax = plt.subplots(figsize=(14, 5))
im = ax.imshow(pivot.values, aspect='auto', cmap='RdYlGn_r', vmin=0)

ax.set_xticks(range(len(pivot.columns)))
ax.set_xticklabels([f'{h}h' for h in pivot.columns], fontsize=10)
ax.set_yticks(range(len(pivot.index)))
ax.set_yticklabels(pivot.index, fontsize=11)
ax.set_xlabel('Prediction Horizon', fontsize=11)
ax.set_title('MAE Heatmap — Model × Horizon', fontsize=13, fontweight='bold')

# 在格子里写数值
for i in range(len(pivot.index)):
    for j in range(len(pivot.columns)):
        val = pivot.values[i, j]
        if not np.isnan(val):
            ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                    fontsize=9, color='black' if val < 0.5 else 'white')

plt.colorbar(im, ax=ax, label='MAE (m)', shrink=0.8)
fig.tight_layout()
fig.savefig(out_dir / 'mae_heatmap.png', bbox_inches='tight', dpi=200)
print('  已保存: mae_heatmap.png')
plt.close(fig)

# ── 图3：代表性 Horizon 柱状图对比（1h / 24h / 120h）────────────────────────
print('[3/4] 绘制代表性 Horizon 柱状图...')
rep_horizons = [1, 24, 120]
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for idx, h in enumerate(rep_horizons):
    ax = axes[idx]
    sub = data[data['horizon'] == h].copy()
    sub = sub[sub['model'].isin(model_order)].copy()
    sub['_order'] = sub['model'].map({m: i for i, m in enumerate(model_order)})
    sub = sub.sort_values('_order')

    bars = ax.bar(
        sub['model'], sub['MAE'],
        color=[COLORS[m] for m in sub['model']],
        edgecolor='white', linewidth=0.8, alpha=0.9,
    )

    # 数值标注
    for bar, val in zip(bars, sub['MAE']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_title(f'{h}h Horizon', fontsize=12, fontweight='bold')
    ax.set_ylabel('MAE (m)', fontsize=10)
    ax.set_ylim(0, sub['MAE'].max() * 1.2)
    ax.tick_params(axis='x', rotation=30, labelsize=9)
    ax.grid(axis='y', linestyle='--', alpha=0.35)

fig.suptitle('MAE Comparison at Representative Horizons',
             fontsize=14, fontweight='bold', y=1.02)
fig.tight_layout()
fig.savefig(out_dir / 'mae_bar_representative.png', bbox_inches='tight', dpi=200)
print('  已保存: mae_bar_representative.png')
plt.close(fig)

# ── 图4：雷达图（24h Horizon 多指标对比）────────────────────────────────────
print('[4/4] 绘制雷达图...')
radar_metrics = ['MAE', 'RMSE', 'MASE', 'SMAPE']
sub24 = data[data['horizon'] == 24].copy()

# 归一化到 0-1（越小越好，所以取反）
norm = sub24[radar_metrics].copy()
for m in radar_metrics:
    mn, mx = norm[m].min(), norm[m].max()
    norm[m] = 1 - (norm[m] - mn) / (mx - mn + 1e-8)

N = len(radar_metrics)
angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

for _, row in sub24.iterrows():
    model = row['model']
    if model not in COLORS:
        continue
    vals = norm[norm.index == row.name][radar_metrics].values.flatten().tolist()
    vals += vals[:1]
    ax.plot(angles, vals, color=COLORS[model], linewidth=2,
            marker=MARKERS[model], markersize=6, label=model)
    ax.fill(angles, vals, color=COLORS[model], alpha=0.08)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(radar_metrics, fontsize=12)
ax.set_ylim(0, 1)
ax.set_title('24h Horizon — Multi-Metric Radar\n(higher = better)',
             fontsize=12, fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=10)
ax.grid(linestyle='--', alpha=0.4)

fig.tight_layout()
fig.savefig(out_dir / 'radar_24h.png', bbox_inches='tight', dpi=200)
print('  已保存: radar_24h.png')
plt.close(fig)

# ── 汇总表 ────────────────────────────────────────────────────────────────────
print('\n生成汇总表...')
summary = data.pivot_table(index='model', columns='horizon', values='MAE').round(4)
summary = summary.reindex(index=[m for m in model_order if m in summary.index])
summary.to_csv(out_dir / 'mae_summary_table.csv')
print(summary.to_string())

print(f'\n全部完成，输出目录：{out_dir.resolve()}')

"""
plot_results.py
===============
读取 AutoGluon 训练产出，生成论文所需全部图表：

1. 各模型在 13 个时间尺度下的 MAE / RMSE / SMAPE / MASE / RMSLE 折线对比图
2. 最优模型在所有时间尺度下的综合误差曲线（含双 y 轴）
3. 各时间尺度下模型排行热力图（MASE 越小越好）
4. 每个时间尺度的模型排行柱状图（Validation MASE vs Test MASE）
5. 每个 horizon 的完整 leaderboard 保存为表格图（可选）

用法：
    python plot_results.py                          # 使用默认 autogluon_runs/
    python plot_results.py --run-dir my_run_dir     # 指定输出目录
    python plot_results.py --format pdf             # 保存为 PDF（默认 png）
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

matplotlib.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.unicode_minus": False,
        "figure.dpi": 150,
    }
)

HORIZONS = [1, 3, 6, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120]
METRICS = ["MAE", "RMSE", "SMAPE", "MASE", "RMSLE"]
METRIC_LABELS = {
    "MAE": "MAE (m)",
    "RMSE": "RMSE (m)",
    "SMAPE": "SMAPE",
    "MASE": "MASE",
    "RMSLE": "RMSLE",
}

# 色板
PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a",
]


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------

def load_metrics_summary(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "metrics_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"未找到 metrics_summary.csv：{path}")
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def load_leaderboard(run_dir: Path, horizon_hours: int, split: str = "test") -> pd.DataFrame | None:
    horizon_dir = run_dir / f"horizon_{horizon_hours:03d}h"
    path = horizon_dir / f"leaderboard_{split}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df["horizon_hours"] = horizon_hours
    return df


def load_all_leaderboards(run_dir: Path, split: str = "test") -> pd.DataFrame:
    frames = []
    for h in HORIZONS:
        df = load_leaderboard(run_dir, h, split)
        if df is not None:
            frames.append(df)
    if not frames:
        raise FileNotFoundError(f"在 {run_dir} 中未找到任何 leaderboard_{split}.csv")
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------

def ensure_positive(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """AutoGluon 的 score 列是负数，转正。"""
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = out[c].abs()
    return out


def get_available_horizons(run_dir: Path) -> list[int]:
    pattern = re.compile(r"horizon_(\d+)h")
    found = []
    for d in sorted(run_dir.iterdir()):
        m = pattern.fullmatch(d.name)
        if m:
            found.append(int(m.group(1)))
    return sorted(found)


def savefig(fig: plt.Figure, path: Path, fmt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(f".{fmt}"), bbox_inches="tight")
    print(f"  已保存：{path.with_suffix(f'.{fmt}')}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 图 1：各模型 × 5 指标 折线图（每个指标一张子图）
# ---------------------------------------------------------------------------

def plot_metrics_by_model(
    lb_all: pd.DataFrame,
    horizons: list[int],
    out_dir: Path,
    fmt: str,
) -> None:
    """对每个 metric 画一张图，X 轴为预测时间尺度，每条线代表一个模型。"""
    models = lb_all["model"].unique().tolist()
    color_map = {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(models)}
    marker_styles = ["o", "s", "^", "D", "v", "P", "*", "X", "h", "+", "x", "1", "2"]

    for metric in METRICS:
        if metric not in lb_all.columns:
            print(f"  [警告] leaderboard 中无 {metric} 列，跳过")
            continue

        fig, ax = plt.subplots(figsize=(10, 5))
        for i, model in enumerate(models):
            sub = lb_all[lb_all["model"] == model].copy()
            sub = ensure_positive(sub, [metric])
            sub = sub.sort_values("horizon_hours")
            xs = sub["horizon_hours"].tolist()
            ys = sub[metric].tolist()
            ax.plot(
                xs, ys,
                marker=marker_styles[i % len(marker_styles)],
                color=color_map[model],
                label=model,
                linewidth=1.8,
                markersize=6,
            )

        ax.set_xlabel("Prediction Horizon (hours)", fontsize=12)
        ax.set_ylabel(METRIC_LABELS.get(metric, metric), fontsize=12)
        ax.set_title(f"Model Comparison — {metric} across Horizons", fontsize=13, fontweight="bold")
        ax.set_xticks(horizons)
        ax.xaxis.set_tick_params(labelsize=9)
        ax.legend(fontsize=9, loc="upper left", framealpha=0.8)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        fig.tight_layout()
        savefig(fig, out_dir / f"model_comparison_{metric.lower()}", fmt)


# ---------------------------------------------------------------------------
# 图 2：最优模型综合误差曲线（双 y 轴：MAE/RMSE 左，SMAPE/MASE 右）
# ---------------------------------------------------------------------------

def plot_best_model_metrics(
    metrics_df: pd.DataFrame,
    horizons: list[int],
    out_dir: Path,
    fmt: str,
) -> None:
    df = metrics_df[metrics_df["horizon_hours"].isin(horizons)].copy().sort_values("horizon_hours")

    fig, ax1 = plt.subplots(figsize=(11, 5))
    ax2 = ax1.twinx()

    xs = df["horizon_hours"].tolist()

    lines = []
    if "MAE" in df.columns:
        l, = ax1.plot(xs, df["MAE"].abs(), "o-", color="#1f77b4", linewidth=2, markersize=6, label="MAE")
        lines.append(l)
    if "RMSE" in df.columns:
        l, = ax1.plot(xs, df["RMSE"].abs(), "s--", color="#ff7f0e", linewidth=2, markersize=6, label="RMSE")
        lines.append(l)
    if "RMSLE" in df.columns:
        l, = ax1.plot(xs, df["RMSLE"].abs(), "^:", color="#2ca02c", linewidth=2, markersize=6, label="RMSLE")
        lines.append(l)

    if "MASE" in df.columns:
        l, = ax2.plot(xs, df["MASE"].abs(), "D-.", color="#d62728", linewidth=2, markersize=6, label="MASE")
        lines.append(l)
    if "SMAPE" in df.columns:
        l, = ax2.plot(xs, df["SMAPE"].abs(), "v--", color="#9467bd", linewidth=2, markersize=6, label="SMAPE")
        lines.append(l)

    ax1.set_xlabel("Prediction Horizon (hours)", fontsize=12)
    ax1.set_ylabel("MAE / RMSE / RMSLE (m)", fontsize=12, color="#1f77b4")
    ax2.set_ylabel("MASE / SMAPE", fontsize=12, color="#d62728")
    ax1.set_xticks(horizons)
    ax1.tick_params(axis="x", labelsize=9)
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    ax1.grid(axis="y", linestyle="--", alpha=0.3)

    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, fontsize=9, loc="upper left", framealpha=0.85)
    ax1.set_title("Best Model — All Metrics across Prediction Horizons", fontsize=13, fontweight="bold")
    fig.tight_layout()
    savefig(fig, out_dir / "best_model_all_metrics", fmt)


# ---------------------------------------------------------------------------
# 图 3：模型 MASE 排行热力图（行=模型，列=时间尺度）
# ---------------------------------------------------------------------------

def plot_heatmap(
    lb_all: pd.DataFrame,
    horizons: list[int],
    out_dir: Path,
    fmt: str,
    metric: str = "MASE",
) -> None:
    if metric not in lb_all.columns:
        print(f"  [警告] leaderboard 中无 {metric} 列，跳过热力图")
        return

    pivot = lb_all.pivot_table(index="model", columns="horizon_hours", values=metric, aggfunc="mean")
    pivot = pivot.abs()
    pivot = pivot.reindex(columns=[h for h in horizons if h in pivot.columns])
    pivot = pivot.sort_values(by=pivot.columns.tolist(), na_position="last")

    n_models, n_horizons = pivot.shape
    fig_w = max(10, n_horizons * 0.9)
    fig_h = max(3, n_models * 0.6)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    plt.colorbar(im, ax=ax, label=metric)

    ax.set_xticks(range(n_horizons))
    ax.set_xticklabels([f"{h}h" for h in pivot.columns], fontsize=9)
    ax.set_yticks(range(n_models))
    ax.set_yticklabels(pivot.index.tolist(), fontsize=9)
    ax.set_xlabel("Prediction Horizon", fontsize=11)
    ax.set_title(f"Model {metric} Heatmap (lower = better)", fontsize=13, fontweight="bold")

    # 在格子里标注数值
    for r in range(n_models):
        for c in range(n_horizons):
            val = pivot.values[r, c]
            if not np.isnan(val):
                ax.text(c, r, f"{val:.3f}", ha="center", va="center", fontsize=7,
                        color="black" if val < pivot.values.max() * 0.6 else "white")

    fig.tight_layout()
    savefig(fig, out_dir / f"heatmap_{metric.lower()}", fmt)


# ---------------------------------------------------------------------------
# 图 4：Validation MASE vs Test MASE 分组柱状图（每个 horizon）
# ---------------------------------------------------------------------------

def plot_val_vs_test_mase(
    val_lb: pd.DataFrame,
    test_lb: pd.DataFrame,
    horizons: list[int],
    out_dir: Path,
    fmt: str,
) -> None:
    for h in horizons:
        v = val_lb[val_lb["horizon_hours"] == h].copy()
        t = test_lb[test_lb["horizon_hours"] == h].copy()
        if v.empty or t.empty:
            continue

        v = ensure_positive(v, ["MASE", "score_val"])
        t = ensure_positive(t, ["MASE", "score_test"])

        # 取 validation_MASE 列（如存在）或 MASE
        val_col = "validation_MASE" if "validation_MASE" in v.columns else "MASE"
        test_col = "MASE"

        models = t["model"].tolist()
        val_vals = []
        for m in models:
            row = v[v["model"] == m]
            val_vals.append(float(row[val_col].iloc[0]) if not row.empty else np.nan)
        test_vals = t[test_col].tolist()

        x = np.arange(len(models))
        width = 0.35

        fig, ax = plt.subplots(figsize=(max(6, len(models) * 1.2), 5))
        bars1 = ax.bar(x - width / 2, val_vals, width, label="Validation MASE", color="#5b9bd5", alpha=0.85)
        bars2 = ax.bar(x + width / 2, test_vals, width, label="Test MASE", color="#ed7d31", alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel("MASE", fontsize=11)
        ax.set_title(f"Validation vs Test MASE — {h}h Horizon", fontsize=12, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

        # 标注数值
        for bar in bars1:
            h_val = bar.get_height()
            if not np.isnan(h_val):
                ax.text(bar.get_x() + bar.get_width() / 2, h_val + 0.002,
                        f"{h_val:.4f}", ha="center", va="bottom", fontsize=7)
        for bar in bars2:
            h_val = bar.get_height()
            if not np.isnan(h_val):
                ax.text(bar.get_x() + bar.get_width() / 2, h_val + 0.002,
                        f"{h_val:.4f}", ha="center", va="bottom", fontsize=7)

        fig.tight_layout()
        savefig(fig, out_dir / f"val_vs_test_mase_{h:03d}h", fmt)


# ---------------------------------------------------------------------------
# 图 5：所有指标在所有时间尺度的子图矩阵（论文总览图）
# ---------------------------------------------------------------------------

def plot_summary_grid(
    metrics_df: pd.DataFrame,
    horizons: list[int],
    out_dir: Path,
    fmt: str,
) -> None:
    df = metrics_df[metrics_df["horizon_hours"].isin(horizons)].copy().sort_values("horizon_hours")
    available = [m for m in METRICS if m in df.columns]
    n = len(available)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 3.8), squeeze=False)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for idx, metric in enumerate(available):
        ax = axes[idx // ncols][idx % ncols]
        ys = df[metric].abs().tolist()
        xs = df["horizon_hours"].tolist()
        ax.plot(xs, ys, "o-", color=colors[idx % len(colors)], linewidth=2, markersize=6)
        ax.fill_between(xs, ys, alpha=0.12, color=colors[idx % len(colors)])
        ax.set_xticks(horizons)
        ax.tick_params(axis="x", labelsize=7, rotation=45)
        ax.set_xlabel("Horizon (h)", fontsize=9)
        ax.set_ylabel(METRIC_LABELS.get(metric, metric), fontsize=9)
        ax.set_title(metric, fontsize=11, fontweight="bold")
        ax.grid(linestyle="--", alpha=0.35)
        # 标注每个点的数值
        for x_val, y_val in zip(xs, ys):
            ax.annotate(f"{y_val:.4f}", (x_val, y_val),
                        textcoords="offset points", xytext=(0, 6),
                        ha="center", fontsize=6, color=colors[idx % len(colors)])

    # 隐藏多余子图
    for idx in range(len(available), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle("Best Model — Forecast Error across All Horizons", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    savefig(fig, out_dir / "summary_grid_all_metrics", fmt)


# ---------------------------------------------------------------------------
# 图 6：训练耗时条形图
# ---------------------------------------------------------------------------

def plot_elapsed_time(
    metrics_df: pd.DataFrame,
    horizons: list[int],
    out_dir: Path,
    fmt: str,
) -> None:
    if "elapsed_seconds" not in metrics_df.columns:
        return
    df = metrics_df[metrics_df["horizon_hours"].isin(horizons)].copy().sort_values("horizon_hours")
    xs = df["horizon_hours"].tolist()
    ys = (df["elapsed_seconds"] / 60).tolist()  # 转分钟

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar([str(x) + "h" for x in xs], ys, color="#5b9bd5", alpha=0.85, edgecolor="white")
    for bar, y_val in zip(bars, ys):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{y_val:.1f}m", ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("Prediction Horizon", fontsize=11)
    ax.set_ylabel("Training Time (minutes)", fontsize=11)
    ax.set_title("Training Time per Horizon", fontsize=12, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    savefig(fig, out_dir / "training_elapsed_time", fmt)


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot AutoGluon wave prediction results")
    parser.add_argument("--run-dir", type=Path, default=Path("autogluon_runs"),
                        help="AutoGluon 训练输出目录（含 metrics_summary.csv 和各 horizon 子目录）")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="图表保存目录（默认为 run-dir/plots/）")
    parser.add_argument("--format", default="png", choices=["png", "pdf", "svg"],
                        help="图片格式")
    parser.add_argument("--skip-per-horizon", action="store_true",
                        help="跳过每个 horizon 单独的 Val vs Test 柱状图（节省时间）")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir: Path = args.run_dir
    out_dir: Path = args.out_dir if args.out_dir else run_dir / "plots"
    fmt: str = args.format

    print(f"读取训练结果：{run_dir.resolve()}")
    print(f"图表输出目录：{out_dir.resolve()}")

    # ---- 检测实际可用的 horizon ----
    available_horizons = get_available_horizons(run_dir)
    if not available_horizons:
        raise FileNotFoundError(f"在 {run_dir} 中未找到任何 horizon_XXXh 子目录。")
    print(f"检测到 {len(available_horizons)} 个 horizon：{available_horizons}")

    # ---- 加载数据 ----
    metrics_df = load_metrics_summary(run_dir)
    print(f"  metrics_summary 加载完成，共 {len(metrics_df)} 行")

    test_lb = load_all_leaderboards(run_dir, split="test")
    test_lb = ensure_positive(test_lb, METRICS + ["score_test"])
    print(f"  test leaderboard 加载完成，共 {len(test_lb)} 行，模型：{test_lb['model'].unique().tolist()}")

    val_lb = load_all_leaderboards(run_dir, split="validation")
    val_lb = ensure_positive(val_lb, METRICS + ["score_val"])
    print(f"  validation leaderboard 加载完成，共 {len(val_lb)} 行")

    # ---- 生成图表 ----
    print("\n[1/6] 各模型指标折线对比图 ...")
    plot_metrics_by_model(test_lb, available_horizons, out_dir, fmt)

    print("[2/6] 最优模型综合误差曲线（双 y 轴）...")
    plot_best_model_metrics(metrics_df, available_horizons, out_dir, fmt)

    print("[3/6] 模型 MASE 热力图 ...")
    plot_heatmap(test_lb, available_horizons, out_dir, fmt)

    if not args.skip_per_horizon:
        print("[4/6] 每个 horizon 的 Val vs Test MASE 柱状图 ...")
        plot_val_vs_test_mase(val_lb, test_lb, available_horizons, out_dir, fmt)
    else:
        print("[4/6] 跳过 per-horizon 柱状图")

    print("[5/6] 综合指标子图矩阵 ...")
    plot_summary_grid(metrics_df, available_horizons, out_dir, fmt)

    print("[6/6] 训练耗时条形图 ...")
    plot_elapsed_time(metrics_df, available_horizons, out_dir, fmt)

    print(f"\n全部图表已保存至：{out_dir.resolve()}")


if __name__ == "__main__":
    main()

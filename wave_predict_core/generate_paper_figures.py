from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.lines import Line2D
from matplotlib import font_manager
import pandas as pd
import plotly.express as px


ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "thesis_assets" / "figures"
TABLE_DIR = ROOT / "thesis_assets" / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)

STATION_CSV = TABLE_DIR / "station_overview.csv"
MAP_HTML = FIG_DIR / "fig_station_world_map.html"
MAP_PNG = FIG_DIR / "fig_station_world_map.png"
FRAMEWORK_PNG = FIG_DIR / "fig_experiment_framework_cn.png"


def find_chinese_font() -> str | None:
    candidates = [
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\simsun.ttc",
    ]
    for path in candidates:
        if Path(path).exists():
            return font_manager.FontProperties(fname=path).get_name()
    return None


FONT_NAME = find_chinese_font()
if FONT_NAME:
    matplotlib.rcParams["font.family"] = FONT_NAME
matplotlib.rcParams["axes.unicode_minus"] = False


def load_station_df() -> pd.DataFrame:
    df = pd.read_csv(STATION_CSV, encoding="utf-8")
    rename = {
        "subtropical_atlantic": "亚热带大西洋",
        "caribbean_tropical": "加勒比热带",
        "gulf_of_mexico": "墨西哥湾",
        "north_atlantic_shelf": "北大西洋陆架",
        "california_current": "加利福尼亚流",
        "oregon_washington": "俄勒冈-华盛顿海岸",
        "alaska_high_latitude": "阿拉斯加高纬",
        "hawaii_open_ocean": "夏威夷开阔洋",
    }
    clean_names = {
        "41010": "卡纳维拉尔角东侧",
        "41043": "圣胡安东北",
        "42040": "多芬岛南侧",
        "44025": "长岛南侧",
        "46026": "旧金山外海",
        "46050": "纽波特外海",
        "46061": "威廉王子湾入口",
        "51001": "夏威夷西北",
    }
    df["station_id"] = df["station_id"].astype(str)
    df["region_cn"] = df["region_key"].map(rename).fillna(df["region_key"])
    df["station_label"] = df["station_id"] + " " + df["station_id"].map(clean_names).fillna("")
    return df


def export_station_map(df: pd.DataFrame) -> None:
    color_map = {
        "亚热带大西洋": "#1f77b4",
        "加勒比热带": "#ff7f0e",
        "墨西哥湾": "#2ca02c",
        "北大西洋陆架": "#9467bd",
        "加利福尼亚流": "#d62728",
        "俄勒冈-华盛顿海岸": "#17becf",
        "阿拉斯加高纬": "#8c564b",
        "夏威夷开阔洋": "#e377c2",
    }

    fig = px.scatter_geo(
        df,
        lat="latitude",
        lon="longitude",
        color="region_cn",
        color_discrete_map=color_map,
        text="station_id",
        hover_name="station_label",
        hover_data={
            "latitude": ":.3f",
            "longitude": ":.3f",
            "depth_m": True,
            "median_swh_m": True,
            "max_swh_m": True,
            "region_cn": True,
            "station_id": False,
            "station_label": False,
        },
        projection="natural earth",
    )
    fig.update_traces(
        marker=dict(size=13, line=dict(width=1.6, color="white"), opacity=0.95),
        textposition="top center",
        textfont=dict(size=13, color="#111111"),
    )
    fig.update_geos(
        showland=True,
        landcolor="#efe7db",
        showcountries=True,
        countrycolor="#9a8f84",
        showocean=True,
        oceancolor="#dceaf7",
        coastlinecolor="#6c7a89",
        showcoastlines=True,
        lataxis=dict(showgrid=True, gridcolor="#c9d7e6", dtick=20),
        lonaxis=dict(showgrid=True, gridcolor="#c9d7e6", dtick=30),
    )
    fig.update_layout(
        title=dict(
            text="NDBC 8个代表性浮标站点全球位置分布图",
            x=0.5,
            xanchor="center",
            font=dict(size=26),
        ),
        legend_title_text="海洋动力环境",
        template="plotly_white",
        width=1600,
        height=900,
        margin=dict(l=20, r=20, t=70, b=20),
        font=dict(size=16),
    )
    fig.write_html(MAP_HTML, include_plotlyjs="cdn")


def export_station_map_matplotlib(df: pd.DataFrame) -> None:
    colors = {
        "亚热带大西洋": "#1f77b4",
        "加勒比热带": "#ff7f0e",
        "墨西哥湾": "#2ca02c",
        "北大西洋陆架": "#9467bd",
        "加利福尼亚流": "#d62728",
        "俄勒冈-华盛顿海岸": "#17becf",
        "阿拉斯加高纬": "#8c564b",
        "夏威夷开阔洋": "#e377c2",
    }

    fig, ax = plt.subplots(figsize=(16, 8.8))
    ax.set_facecolor("#dceaf7")
    fig.patch.set_facecolor("white")

    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 85)
    ax.set_xticks(range(-180, 181, 30))
    ax.set_yticks(range(-60, 91, 15))
    ax.grid(True, linestyle="--", linewidth=0.7, color="#b6cadc", alpha=0.9)

    # 简化的世界陆地区域底色
    land_blocks = [
        (-170, 15, 115, 57),   # 北美
        (-82, -56, 50, 68),    # 南美
        (-20, -35, 70, 72),    # 非洲
        (-10, 35, 50, 35),     # 欧洲
        (35, 5, 145, 65),      # 亚洲
        (110, -45, 45, 35),    # 澳大利亚
        (-55, 60, 38, 20),     # 格陵兰
    ]
    for x, y, w, h in land_blocks:
        rect = Rectangle((x, y), w, h, facecolor="#efe7db", edgecolor="#c2b7a8", linewidth=1.0, zorder=0)
        ax.add_patch(rect)

    for _, row in df.iterrows():
        c = colors[row["region_cn"]]
        ax.scatter(row["longitude"], row["latitude"], s=120, color=c, edgecolors="white", linewidths=1.8, zorder=3)
        dx = 3 if row["longitude"] < -100 else 2
        dy = 2.5 if row["latitude"] < 45 else -3
        ax.text(
            row["longitude"] + dx,
            row["latitude"] + dy,
            row["station_id"],
            fontsize=11,
            fontweight="bold",
            color="#1f2937",
            zorder=4,
        )

    legend_handles = []
    for name, color in colors.items():
        if name in set(df["region_cn"]):
            legend_handles.append(Line2D([0], [0], marker="o", color="w", label=name,
                                         markerfacecolor=color, markeredgecolor="white",
                                         markeredgewidth=1.5, markersize=10))
    ax.legend(handles=legend_handles, title="海洋动力环境", loc="lower left", frameon=True, fontsize=11, title_fontsize=12)

    ax.set_title("NDBC 8个代表性浮标站点全球位置分布图", fontsize=22, fontweight="bold", pad=16)
    ax.set_xlabel("经度 / °", fontsize=13)
    ax.set_ylabel("纬度 / °", fontsize=13)
    for spine in ax.spines.values():
        spine.set_color("#7b8794")
        spine.set_linewidth(1.0)

    fig.tight_layout()
    fig.savefig(MAP_PNG, dpi=220, bbox_inches="tight")
    plt.close(fig)


def draw_box(ax, xy, w, h, text, fc="#ffffff", ec="#5a6472", lw=1.6, ls="-", fontsize=13, bold=False):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
        linestyle=ls,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="bold" if bold else "normal",
        color="#222222",
        wrap=True,
    )
    return patch


def arrow(ax, x1, y1, x2, y2):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", lw=1.8, color="#4b5563", shrinkA=0, shrinkB=0),
    )


def export_framework_diagram() -> None:
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    outer = Rectangle((0.03, 0.04), 0.94, 0.92, linewidth=1.2, edgecolor="#666666", facecolor="#fcfcfc")
    ax.add_patch(outer)

    draw_box(ax, (0.39, 0.90), 0.22, 0.055, "NDBC多站点SWH与气象协变量数据", fc="#f8fbff", fontsize=14, bold=True)

    draw_box(ax, (0.32, 0.74), 0.36, 0.12, "数据预处理", fc="#fffaf0", ec="#8b8b8b", ls="--", fontsize=16, bold=True)
    draw_box(ax, (0.41, 0.80), 0.18, 0.04, "缺失值处理与异常值清洗", fc="#ffffff", fontsize=12)
    draw_box(ax, (0.41, 0.75), 0.18, 0.04, "重采样、时间对齐与特征构造", fc="#ffffff", fontsize=12)

    draw_box(ax, (0.18, 0.58), 0.64, 0.09, "", fc="#f8fbff", ec="#8b8b8b", ls="--", fontsize=16, bold=True)
    ax.text(0.50, 0.635, "数据集划分", ha="center", va="center", fontsize=18, fontweight="bold", color="#30343b")
    draw_box(ax, (0.24, 0.595), 0.15, 0.045, "训练集\n前70%", fontsize=12)
    draw_box(ax, (0.425, 0.595), 0.15, 0.045, "验证集\n中间20%", fontsize=12)
    draw_box(ax, (0.61, 0.595), 0.15, 0.045, "测试集\n最后10%", fontsize=12)

    draw_box(ax, (0.10, 0.34), 0.80, 0.18, "模型训练与预测", fc="#f9fcff", ec="#8b8b8b", ls="--", fontsize=18, bold=True)
    model_boxes = [
        (0.17, 0.43, "PatchTST"),
        (0.32, 0.43, "TFT"),
        (0.47, 0.43, "DLinear"),
        (0.62, 0.43, "DeepAR"),
        (0.77, 0.43, "Chronos"),
    ]
    for x, y, name in model_boxes:
        draw_box(ax, (x, y), 0.11, 0.055, name, fc="#ffffff", fontsize=13, bold=True)

    draw_box(ax, (0.17, 0.355), 0.16, 0.05, "AutoGluon-TimeSeries统一训练接口", fc="#ffffff", fontsize=12)
    draw_box(ax, (0.39, 0.355), 0.16, 0.05, "多时域预测：1h–120h", fc="#ffffff", fontsize=12)
    draw_box(ax, (0.61, 0.355), 0.19, 0.05, "验证集早停与最优模型选择", fc="#ffffff", fontsize=12)

    draw_box(ax, (0.10, 0.11), 0.28, 0.15, "结果分析", fc="#fffaf0", ec="#8b8b8b", ls="--", fontsize=18, bold=True)
    draw_box(ax, (0.16, 0.19), 0.16, 0.04, "各时域预测结果可视化", fc="#ffffff", fontsize=12)
    draw_box(ax, (0.16, 0.145), 0.16, 0.04, "不同模型横向性能比较", fc="#ffffff", fontsize=12)

    draw_box(ax, (0.56, 0.10), 0.32, 0.16, "结果评价", fc="#f8fbff", ec="#8b8b8b", ls="--", fontsize=18, bold=True)
    eval_boxes = [
        (0.62, 0.19, "MASE"),
        (0.72, 0.19, "MAE"),
        (0.80, 0.19, "RMSE"),
        (0.62, 0.145, "SMAPE"),
        (0.72, 0.145, "训练时间"),
        (0.80, 0.145, "预测时间"),
    ]
    for x, y, text in eval_boxes:
        draw_box(ax, (x, y), 0.07, 0.035, text, fc="#ffffff", fontsize=11)

    arrow(ax, 0.50, 0.90, 0.50, 0.86)
    arrow(ax, 0.50, 0.74, 0.50, 0.67)
    arrow(ax, 0.50, 0.58, 0.50, 0.52)
    arrow(ax, 0.50, 0.34, 0.50, 0.26)
    arrow(ax, 0.50, 0.26, 0.72, 0.26)
    arrow(ax, 0.56, 0.18, 0.38, 0.18)

    ax.text(0.5, 0.985, "海浪有效波高预测实验框架图", ha="center", va="top", fontsize=22, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FRAMEWORK_PNG, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df = load_station_df()
    export_station_map(df)
    export_station_map_matplotlib(df)
    export_framework_diagram()
    print(MAP_HTML)
    print(FRAMEWORK_PNG)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()

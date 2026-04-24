from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "thesis_assets" / "figures"
LATEX_FIG_DIR = ROOT / "-2026-LaTeX--main" / "figures" / "from_111md"
OUT_NAME = "training_curves_overview.png"


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = []
    if bold:
        candidates.extend(
            [
                Path(r"C:\Windows\Fonts\msyhbd.ttc"),
                Path(r"C:\Windows\Fonts\simhei.ttf"),
                Path(r"C:\Windows\Fonts\arialbd.ttf"),
            ]
        )
    else:
        candidates.extend(
            [
                Path(r"C:\Windows\Fonts\msyh.ttc"),
                Path(r"C:\Windows\Fonts\arial.ttf"),
            ]
        )
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def nice_horizon_label(path: Path) -> str:
    stem = path.stem
    suffix = stem.split("_")[-1]
    if suffix.endswith("h"):
        try:
            return f"{int(suffix[:-1])}h"
        except ValueError:
            return suffix
    return suffix


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    LATEX_FIG_DIR.mkdir(parents=True, exist_ok=True)

    curve_paths = sorted(
        [
            p
            for p in FIG_DIR.glob("training_curves_*h.png")
            if p.name != OUT_NAME and "overview" not in p.name and "representative" not in p.name
        ],
        key=lambda p: int(p.stem.split("_")[-1][:-1]),
    )
    if not curve_paths:
        raise FileNotFoundError("No training_curves_*h.png files found in thesis_assets/figures")

    images = [Image.open(path).convert("RGB") for path in curve_paths]

    cols = 2
    rows = math.ceil(len(images) / cols)
    tile_w = max(img.width for img in images)
    tile_h = max(img.height for img in images)

    title_h = 170
    gap_x = 26
    gap_y = 34
    margin = 40
    label_h = 44

    canvas_w = margin * 2 + cols * tile_w + (cols - 1) * gap_x
    canvas_h = margin * 2 + title_h + rows * (tile_h + label_h) + (rows - 1) * gap_y
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)

    title_font = load_font(34, bold=True)
    subtitle_font = load_font(18, bold=False)
    panel_font = load_font(22, bold=True)

    title = "图3-11 训练曲线综合网格"
    subtitle = "基于各预测时域独立训练曲线图自动拼版生成，汇总展示 1h 至 120h 的训练与验证损失变化"
    title_box = draw.textbbox((0, 0), title, font=title_font)
    subtitle_box = draw.textbbox((0, 0), subtitle, font=subtitle_font)
    draw.text(((canvas_w - (title_box[2] - title_box[0])) / 2, 28), title, fill="#17324D", font=title_font)
    draw.text(
        ((canvas_w - (subtitle_box[2] - subtitle_box[0])) / 2, 86),
        subtitle,
        fill="#516B84",
        font=subtitle_font,
    )

    for idx, (path, img) in enumerate(zip(curve_paths, images, strict=True)):
        row = idx // cols
        col = idx % cols
        x = margin + col * (tile_w + gap_x)
        y = margin + title_h + row * (tile_h + label_h + gap_y)

        label = f"({chr(ord('a') + idx)}) {nice_horizon_label(path)}"
        label_box = draw.textbbox((0, 0), label, font=panel_font)
        label_x = x + (tile_w - (label_box[2] - label_box[0])) / 2
        draw.text((label_x, y), label, fill="#1F3A56", font=panel_font)

        img_y = y + label_h
        if img.width != tile_w or img.height != tile_h:
            paste_x = x + (tile_w - img.width) // 2
            paste_y = img_y + (tile_h - img.height) // 2
            canvas.paste(img, (paste_x, paste_y))
        else:
            canvas.paste(img, (x, img_y))

        draw.rounded_rectangle(
            [x, img_y, x + tile_w, img_y + tile_h],
            radius=8,
            outline="#D4DFE8",
            width=2,
        )

    out_path = FIG_DIR / OUT_NAME
    latex_out = LATEX_FIG_DIR / OUT_NAME
    canvas.save(out_path, quality=95)
    canvas.save(latex_out, quality=95)
    print(f"[done] {out_path}")
    print(f"[done] {latex_out}")


if __name__ == "__main__":
    main()

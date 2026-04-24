from __future__ import annotations

import re
import shutil
from pathlib import Path


CHAPTER_FILE_MAP = {
    "第一章": "ch01_intro.tex",
    "第二章": "ch02_data.tex",
    "第三章": "ch03_baselines.tex",
    "第四章": "ch04_distill.tex",
    "第五章": "ch05_conclusion.tex",
}

SECTION_RE = re.compile(r"^(#{2,4})\s+(.*)$")
IMAGE_RE = re.compile(r"^!\[(.*?)\]\((.*?)\)\s*$")
TABLE_CAPTION_RE = re.compile(r"^\*\*(表[^*]+)\*\*\s*$")
FIG_CAPTION_RE = re.compile(r"^\*\*(图[^*]+)\*\*\s*$")
NUMBER_PREFIX_RE = re.compile(r"^\d+(?:\.\d+)*\s+")


def escape_tex(text: str) -> str:
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "_": r"\_",
        "#": r"\#",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


def convert_inline(text: str) -> str:
    parts = re.split(r"(\$[^$]+\$)", text)
    converted: list[str] = []
    for part in parts:
        if not part:
            continue
        if part.startswith("$") and part.endswith("$"):
            converted.append(part)
            continue
        converted.append(convert_plain_text(part))
    return "".join(converted)


def convert_plain_text(text: str) -> str:
    text = text.replace("–", "--").replace("—", "--")
    pattern = re.compile(r"(`[^`]+`|\*\*[^*]+\*\*)")
    pos = 0
    out: list[str] = []
    for match in pattern.finditer(text):
        if match.start() > pos:
            out.append(escape_tex(text[pos:match.start()]))
        token = match.group(0)
        if token.startswith("`"):
            out.append(r"\texttt{" + escape_tex(token[1:-1]) + "}")
        else:
            out.append(r"\textbf{" + escape_tex(token[2:-2]) + "}")
        pos = match.end()
    if pos < len(text):
        out.append(escape_tex(text[pos:]))
    return "".join(out)


def strip_number_prefix(title: str) -> str:
    return NUMBER_PREFIX_RE.sub("", title).strip()


def clean_caption(text: str) -> str:
    return text.replace("（", "(").replace("）", ")").strip()


def parse_markdown(source_text: str) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {}
    current_key = "_preamble"
    sections[current_key] = []

    for line in source_text.splitlines():
        match = SECTION_RE.match(line.strip())
        if match and match.group(1) == "##":
            current_key = match.group(2).strip()
            sections.setdefault(current_key, [])
            continue
        sections.setdefault(current_key, []).append(line.rstrip("\n"))
    return sections


def split_chapters(sections: dict[str, list[str]]) -> dict[str, list[str]]:
    chapter_blocks: dict[str, list[str]] = {}
    for key, value in sections.items():
        for chapter_name in CHAPTER_FILE_MAP:
            if key.startswith(chapter_name):
                chapter_blocks[chapter_name] = value
    return chapter_blocks


def emit_paragraph(paragraph_lines: list[str], out: list[str]) -> None:
    if not paragraph_lines:
        return
    text = " ".join(line.strip() for line in paragraph_lines if line.strip())
    if text:
        out.append(convert_inline(text))
        out.append("")
    paragraph_lines.clear()


def convert_math_block(lines: list[str]) -> list[str]:
    out = [r"\begin{equation}"]
    out.extend(lines)
    out.append(r"\end{equation}")
    out.append("")
    return out


def parse_table_rows(table_lines: list[str]) -> list[list[str]]:
    rows: list[list[str]] = []
    for line in table_lines:
        stripped = line.strip().strip("|")
        cells = [cell.strip() for cell in stripped.split("|")]
        if all(set(cell) <= {"-", ":"} for cell in cells):
            continue
        rows.append(cells)
    return rows


def emit_table(rows: list[list[str]], caption: str | None) -> list[str]:
    if not rows:
        return []
    col_count = len(rows[0])
    col_spec = "|".join(["c"] * col_count)
    out = [r"\begin{table}[htbp]", r"\centering"]
    if caption:
        out.append(r"\caption{" + escape_tex(caption) + "}")
    out.extend(
        [
            r"\resizebox{\textwidth}{!}{%",
            r"\begin{tabular}{" + "|" + col_spec + "|" + "}",
            r"\hline",
        ]
    )
    header = " & ".join(convert_inline(cell) for cell in rows[0]) + r" \\"
    out.append(header)
    out.append(r"\hline")
    for row in rows[1:]:
        out.append(" & ".join(convert_inline(cell) for cell in row) + r" \\")
        out.append(r"\hline")
    out.extend([r"\end{tabular}%", r"}", r"\end{table}", ""])
    return out


def copy_image(image_rel_path: str, source_root: Path, figure_root: Path) -> str:
    source_path = (source_root / image_rel_path).resolve()
    figure_root.mkdir(parents=True, exist_ok=True)
    target_path = figure_root / source_path.name
    if source_path.exists():
        shutil.copy2(source_path, target_path)
    return f"from_111md/{source_path.name}"


def emit_figure(caption: str, image_path: str) -> list[str]:
    return [
        r"\begin{figure}[htbp]",
        r"\centering",
        r"\includegraphics[width=0.9\textwidth]{" + image_path.replace("\\", "/") + "}",
        r"\caption{" + escape_tex(caption) + "}",
        r"\end{figure}",
        "",
    ]


def convert_chapter(lines: list[str], source_root: Path, template_root: Path) -> list[str]:
    out: list[str] = ['% !TeX root = ../hainumain.tex', '% !Mode:: "TeX:UTF-8"', ""]
    paragraph_lines: list[str] = []
    pending_table_caption: str | None = None
    figure_root = template_root / "figures" / "from_111md"

    i = 0
    while i < len(lines):
        raw_line = lines[i]
        stripped = raw_line.strip()

        if not stripped or stripped == "---":
            emit_paragraph(paragraph_lines, out)
            i += 1
            continue

        heading_match = SECTION_RE.match(stripped)
        if heading_match:
            emit_paragraph(paragraph_lines, out)
            level = len(heading_match.group(1))
            title = strip_number_prefix(heading_match.group(2))
            if level == 3:
                out.append(r"\section{" + escape_tex(title) + "}")
                out.append("")
            elif level == 4:
                out.append(r"\subsection{" + escape_tex(title) + "}")
                out.append("")
            i += 1
            continue

        table_caption_match = TABLE_CAPTION_RE.match(stripped)
        if table_caption_match:
            emit_paragraph(paragraph_lines, out)
            pending_table_caption = clean_caption(table_caption_match.group(1))
            i += 1
            continue

        if FIG_CAPTION_RE.match(stripped):
            emit_paragraph(paragraph_lines, out)
            i += 1
            continue

        image_match = IMAGE_RE.match(stripped)
        if image_match:
            emit_paragraph(paragraph_lines, out)
            caption = clean_caption(image_match.group(1))
            copied_image = copy_image(image_match.group(2), source_root, figure_root)
            out.extend(emit_figure(caption, copied_image))
            i += 1
            continue

        if stripped == "$$":
            emit_paragraph(paragraph_lines, out)
            math_lines: list[str] = []
            i += 1
            while i < len(lines) and lines[i].strip() != "$$":
                math_lines.append(lines[i])
                i += 1
            out.extend(convert_math_block(math_lines))
            i += 1
            continue

        if stripped.startswith("|"):
            emit_paragraph(paragraph_lines, out)
            table_lines: list[str] = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i])
                i += 1
            rows = parse_table_rows(table_lines)
            out.extend(emit_table(rows, pending_table_caption))
            pending_table_caption = None
            continue

        paragraph_lines.append(raw_line)
        i += 1

    emit_paragraph(paragraph_lines, out)
    return out


def write_text(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    workspace_root = Path(__file__).resolve().parents[1]
    source_md = workspace_root / "111.md"
    template_root = workspace_root / "-2026-LaTeX--main"
    body_root = template_root / "body"

    sections = parse_markdown(source_md.read_text(encoding="utf-8"))
    chapters = split_chapters(sections)

    for chapter_name, file_name in CHAPTER_FILE_MAP.items():
        lines = chapters.get(chapter_name, [])
        if not lines:
            continue
        converted = convert_chapter(lines, workspace_root, template_root)
        chapter_title = strip_number_prefix(chapter_name)
        chapter_heading = next(
            (key for key in sections.keys() if key.startswith(chapter_name)),
            chapter_name,
        )
        chapter_title = chapter_heading.split(" ", 1)[1].strip() if " " in chapter_heading else chapter_title
        converted.insert(3, r"\chapter{" + escape_tex(chapter_title) + "}")
        converted.insert(4, "")
        write_text(body_root / file_name, converted)

    print("Converted chapters:")
    for file_name in CHAPTER_FILE_MAP.values():
        path = body_root / file_name
        print(path)


if __name__ == "__main__":
    main()

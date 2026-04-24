from __future__ import annotations

import gzip
import itertools
from pathlib import Path


def show_gzip_head(path: Path, n: int = 12) -> None:
    print(f"=== {path} ===")
    with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as handle:
        print("".join(itertools.islice(handle, n)))


def show_lines(path: Path, start: int, end: int) -> None:
    print(f"=== {path}:{start}-{end} ===")
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    for idx in range(start - 1, min(end, len(lines))):
        print(f"{idx + 1}: {lines[idx]}")


def main() -> None:
    root = Path(__file__).resolve().parent
    show_gzip_head(root / "data" / "42040" / "2004" / "42040h2004.txt.gz")
    show_gzip_head(root / "data" / "41010" / "2008" / "41010h2008.txt.gz")
    show_lines(Path(r"C:\Users\cccfy\Desktop\毕业论文\海浪预测项目\data_preprocessing.py"), 280, 360)


if __name__ == "__main__":
    main()

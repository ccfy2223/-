from __future__ import annotations

import gzip
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROCESSED_ROOT = ROOT / "processed"
OUTPUT_ROOT = ROOT / "processed_csv"


def decompress_gzip_csv(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(src, "rb") as f_in, dst.open("wb") as f_out:
        shutil.copyfileobj(f_in, f_out)


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    files = sorted(PROCESSED_ROOT.rglob("*.csv.gz"))
    if not files:
        raise SystemExit("No .csv.gz files found under processed/")

    for src in files:
        relative = src.relative_to(PROCESSED_ROOT)
        dst = OUTPUT_ROOT / relative.with_suffix("")
        decompress_gzip_csv(src, dst)
        print(f"Exported: {dst}")

    metadata_src = PROCESSED_ROOT / "shared_timeline_metadata.json"
    if metadata_src.exists():
        metadata_dst = OUTPUT_ROOT / metadata_src.name
        shutil.copy2(metadata_src, metadata_dst)
        print(f"Copied:   {metadata_dst}")


if __name__ == "__main__":
    main()

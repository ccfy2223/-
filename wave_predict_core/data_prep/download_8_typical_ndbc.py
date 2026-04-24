from __future__ import annotations

import csv
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

INDEX_URL = "https://www.ndbc.noaa.gov/data/historical/stdmet/"
USER_AGENT = "Mozilla/5.0 (compatible; NDBC-Downloader/1.0)"
OUTPUT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = OUTPUT_ROOT / "data"
MANIFEST_PATH = OUTPUT_ROOT / "selected_stations.csv"
CATALOG_CANDIDATES = [
    Path(r"C:\Users\cccfy\Desktop\毕业论文\海浪预测项目\catalogs\classic_ndbc_stations.csv"),
]

FILE_RE = re.compile(r'href="([A-Za-z0-9]{5})h((?:19|20)\d{2})\.txt\.gz"', re.IGNORECASE)

REGION_CANDIDATES = [
    (
        "north_atlantic_shelf",
        "North Atlantic shelf, winter extratropical storms",
        ["44025", "44065", "44009", "44011"],
    ),
    (
        "subtropical_atlantic",
        "Subtropical western Atlantic, hurricane exposure",
        ["41010", "41009", "41013", "41002"],
    ),
    (
        "gulf_of_mexico",
        "Gulf of Mexico, semi-enclosed basin with tropical cyclones",
        ["42040", "42019", "42020", "42001"],
    ),
    (
        "caribbean_tropical",
        "Caribbean tropical waters and trade-wind sea state",
        ["41043", "41044", "42059", "41053"],
    ),
    (
        "california_current",
        "California coast, swell plus coastal wind sea",
        ["46026", "46059", "46237", "46042"],
    ),
    (
        "oregon_washington",
        "U.S. Pacific Northwest, energetic winter wave climate",
        ["46050", "46041", "46029", "46005"],
    ),
    (
        "alaska_high_latitude",
        "High-latitude North Pacific and Gulf of Alaska",
        ["46061", "46035", "46001", "46066"],
    ),
    (
        "hawaii_open_ocean",
        "Open-ocean central Pacific swell regime near Hawaii",
        ["51001", "51000", "51002", "51003"],
    ),
]

FALLBACK_CANDIDATES = [
    "45007",
    "45008",
    "45003",
    "45161",
    "46050",
    "46026",
    "44025",
    "41010",
]


def fetch_text(url: str, timeout: int = 120) -> str:
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="ignore")


def parse_available_years(index_html: str) -> Dict[str, List[int]]:
    available: Dict[str, set[int]] = {}
    for station_id, year in FILE_RE.findall(index_html):
        station_id = station_id.upper()
        available.setdefault(station_id, set()).add(int(year))
    return {station: sorted(years) for station, years in available.items()}


def normalize_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", key.lower())


def load_station_catalog() -> Dict[str, Dict[str, str]]:
    for path in CATALOG_CANDIDATES:
        if not path.exists():
            continue
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                if not reader.fieldnames:
                    continue
                key_map = {normalize_key(name): name for name in reader.fieldnames}
                id_key = next(
                    (
                        key_map[k]
                        for k in (
                            "stationid",
                            "station",
                            "stid",
                            "id",
                            "stationid5",
                        )
                        if k in key_map
                    ),
                    None,
                )
                if not id_key:
                    continue
                name_key = next((key_map[k] for k in ("stationname", "name", "title") if k in key_map), None)
                lat_key = next((key_map[k] for k in ("latitude", "lat") if k in key_map), None)
                lon_key = next((key_map[k] for k in ("longitude", "lon", "lng") if k in key_map), None)
                catalog: Dict[str, Dict[str, str]] = {}
                for row in reader:
                    station_id = (row.get(id_key) or "").strip().upper()
                    if not station_id:
                        continue
                    catalog[station_id] = {
                        "name": (row.get(name_key) or "").strip() if name_key else "",
                        "latitude": (row.get(lat_key) or "").strip() if lat_key else "",
                        "longitude": (row.get(lon_key) or "").strip() if lon_key else "",
                    }
                if catalog:
                    return catalog
        except OSError:
            continue
    return {}


def choose_station(candidates: Iterable[str], available: Dict[str, List[int]], used: set[str], min_years: int = 10):
    best = None
    for station_id in candidates:
        years = available.get(station_id)
        if not years or station_id in used:
            continue
        candidate = (station_id, years)
        if len(years) >= min_years:
            return candidate
        if best is None or len(years) > len(best[1]):
            best = candidate
    return best


def select_stations(available: Dict[str, List[int]]) -> List[Dict[str, object]]:
    selected: List[Dict[str, object]] = []
    used: set[str] = set()
    for region_key, profile, candidates in REGION_CANDIDATES:
        chosen = choose_station(candidates, available, used)
        if not chosen:
            continue
        station_id, years = chosen
        used.add(station_id)
        selected.append(
            {
                "region_key": region_key,
                "profile": profile,
                "station_id": station_id,
                "years": years,
            }
        )

    if len(selected) < 8:
        for station_id in FALLBACK_CANDIDATES:
            years = available.get(station_id)
            if not years or station_id in used:
                continue
            used.add(station_id)
            selected.append(
                {
                    "region_key": f"fallback_{len(selected) + 1}",
                    "profile": "Fallback long-record station",
                    "station_id": station_id,
                    "years": years,
                }
            )
            if len(selected) == 8:
                break

    if len(selected) < 8:
        for station_id, years in sorted(available.items(), key=lambda item: (-len(item[1]), item[0])):
            if station_id in used:
                continue
            used.add(station_id)
            selected.append(
                {
                    "region_key": f"fallback_{len(selected) + 1}",
                    "profile": "Fallback long-record station",
                    "station_id": station_id,
                    "years": years,
                }
            )
            if len(selected) == 8:
                break

    return selected[:8]


def download_file(url: str, destination: Path, retries: int = 3) -> str:
    if destination.exists() and destination.stat().st_size > 0:
        return "skipped"

    destination.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, retries + 1):
        try:
            req = Request(url, headers={"User-Agent": USER_AGENT})
            with urlopen(req, timeout=180) as resp:
                with destination.open("wb") as handle:
                    shutil.copyfileobj(resp, handle)
            if destination.stat().st_size == 0:
                raise OSError("Downloaded empty file")
            return "downloaded"
        except (HTTPError, URLError, OSError) as exc:
            if destination.exists():
                destination.unlink(missing_ok=True)
            if attempt == retries:
                raise RuntimeError(f"Failed to download {url}: {exc}") from exc
            time.sleep(2 * attempt)
    raise RuntimeError(f"Failed to download {url}")


def write_manifest(selected: List[Dict[str, object]], catalog: Dict[str, Dict[str, str]]) -> None:
    with MANIFEST_PATH.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "station_id",
                "station_name",
                "latitude",
                "longitude",
                "region_key",
                "profile",
                "year_start",
                "year_end",
                "year_count",
                "local_folder",
            ]
        )
        for item in selected:
            station_id = str(item["station_id"])
            years = list(item["years"])
            meta = catalog.get(station_id, {})
            writer.writerow(
                [
                    station_id,
                    meta.get("name", ""),
                    meta.get("latitude", ""),
                    meta.get("longitude", ""),
                    item["region_key"],
                    item["profile"],
                    years[0],
                    years[-1],
                    len(years),
                    str((DATA_ROOT / station_id).resolve()),
                ]
            )


def main() -> int:
    print("Fetching NDBC historical stdmet index...")
    index_html = fetch_text(INDEX_URL)
    available = parse_available_years(index_html)
    if not available:
        print("No station files found in the NDBC historical index.", file=sys.stderr)
        return 1

    selected = select_stations(available)
    if len(selected) < 8:
        print(f"Only found {len(selected)} stations with downloadable data.", file=sys.stderr)
        return 1

    catalog = load_station_catalog()
    print("Selected stations:")
    for item in selected:
        years = list(item["years"])
        station_id = str(item["station_id"])
        meta = catalog.get(station_id, {})
        display_name = meta.get("name") or "unknown"
        print(
            f"  {station_id} | {display_name} | {item['profile']} | "
            f"{years[0]}-{years[-1]} ({len(years)} files)"
        )

    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    total_downloaded = 0
    total_skipped = 0
    total_files = 0

    for item in selected:
        station_id = str(item["station_id"])
        years = list(item["years"])
        downloaded = 0
        skipped = 0
        for year in years:
            filename = f"{station_id}h{year}.txt.gz"
            url = INDEX_URL + filename
            destination = DATA_ROOT / station_id / str(year) / filename
            status = download_file(url, destination)
            total_files += 1
            if status == "downloaded":
                downloaded += 1
                total_downloaded += 1
            else:
                skipped += 1
                total_skipped += 1
        print(f"  Finished {station_id}: {downloaded} downloaded, {skipped} skipped, {len(years)} total.")

    write_manifest(selected, catalog)
    print()
    print(f"Done. Files available under: {DATA_ROOT}")
    print(f"Manifest written to: {MANIFEST_PATH}")
    print(f"Summary: {total_files} files total, {total_downloaded} downloaded, {total_skipped} skipped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from align_ndbc_timelines import read_station_table

ROOT = Path(__file__).resolve().parent
SELECTED_PATH = ROOT / "selected_stations.csv"
PROCESSED_METADATA_PATH = ROOT / "processed" / "shared_timeline_metadata.json"
REPORT_ROOT = ROOT / "reports"
SUMMARY_CSV_PATH = REPORT_ROOT / "station_summary.csv"
SUMMARY_MD_PATH = REPORT_ROOT / "station_summary.md"
SUMMARY_HTML_PATH = REPORT_ROOT / "station_overview.html"
MAP_HTML_PATH = REPORT_ROOT / "station_map.html"
MAP_PNG_PATH = REPORT_ROOT / "station_map.png"

USER_AGENT = "Mozilla/5.0 (compatible; StationOverview/1.0)"

MONTH_ABBR = {
    1: "Jan.",
    2: "Feb.",
    3: "Mar.",
    4: "Apr.",
    5: "May",
    6: "Jun.",
    7: "Jul.",
    8: "Aug.",
    9: "Sep.",
    10: "Oct.",
    11: "Nov.",
    12: "Dec.",
}


def fetch_text(url: str) -> str:
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=120) as resp:
        return resp.read().decode("utf-8", errors="ignore")


def station_page_url(station_id: str) -> str:
    return f"https://www.ndbc.noaa.gov/station_page.php?station={station_id}"


def parse_station_page(station_id: str) -> dict[str, object]:
    html = fetch_text(station_page_url(station_id))
    cleaned = re.sub(r"<[^>]+>", " ", html)
    cleaned = re.sub(r"\s+", " ", cleaned)

    title_match = re.search(rf"Station {station_id} .*? - (.*?) Owned and maintained by", cleaned, re.IGNORECASE)
    coord_match = re.search(
        r"([0-9]+(?:\.[0-9]+)?)\s+N\s+([0-9]+(?:\.[0-9]+)?)\s+W\s+\(",
        cleaned,
        re.IGNORECASE,
    )
    depth_match = re.search(r"Water depth:\s*([0-9]+(?:\.[0-9]+)?)\s*m", cleaned, re.IGNORECASE)

    if not title_match or not coord_match or not depth_match:
        raise ValueError(f"Failed to parse station page for {station_id}")

    station_name = re.sub(r"\s+", " ", title_match.group(1)).strip().strip("-").strip()
    latitude = float(coord_match.group(1))
    longitude = -float(coord_match.group(2))
    depth_m = float(depth_match.group(1))
    return {
        "station_name": station_name,
        "latitude": latitude,
        "longitude": longitude,
        "depth_m": depth_m,
        "source_url": station_page_url(station_id),
    }


def format_coord(lat: float, lon: float) -> str:
    lat_dir = "N" if lat >= 0 else "S"
    lon_dir = "E" if lon >= 0 else "W"
    return f"{abs(lat):.3f}°{lat_dir}, {abs(lon):.3f}°{lon_dir}"


def format_date(ts: pd.Timestamp) -> str:
    return f"{MONTH_ABBR[ts.month]} {ts.day}, {ts.year}"


def format_record_span(start: pd.Timestamp, end: pd.Timestamp) -> str:
    return f"{format_date(start)} - {format_date(end)}"


def load_station_summary() -> pd.DataFrame:
    selected = pd.read_csv(SELECTED_PATH, encoding="utf-8-sig", dtype={"station_id": str})
    with PROCESSED_METADATA_PATH.open("r", encoding="utf-8") as handle:
        processed_meta = json.load(handle)
    region_lookup = {item["station_id"]: item["region_key"] for item in processed_meta["stations"]}

    rows: list[dict[str, object]] = []
    for station_id in selected["station_id"]:
        station_meta = parse_station_page(station_id)
        raw_frame = read_station_table(station_id)

        raw_start = pd.Timestamp(raw_frame["datetime"].min())
        raw_end = pd.Timestamp(raw_frame["datetime"].max())
        wvht = raw_frame["WVHT"].dropna()
        rows.append(
            {
                "station_id": station_id,
                "station_name": station_meta["station_name"],
                "coordinates": format_coord(station_meta["latitude"], station_meta["longitude"]),
                "latitude": station_meta["latitude"],
                "longitude": station_meta["longitude"],
                "depth_m": station_meta["depth_m"],
                "record_span": format_record_span(raw_start, raw_end),
                "record_start": raw_start,
                "record_end": raw_end,
                "median_swh_m": round(float(wvht.median()), 2),
                "max_swh_m": round(float(wvht.max()), 2),
                "data_volume": int(raw_frame.shape[0]),
                "region_key": region_lookup.get(station_id, ""),
                "source_url": station_meta["source_url"],
            }
        )

    summary = pd.DataFrame(rows).sort_values("station_id").reset_index(drop=True)
    return summary


def build_markdown_table(summary: pd.DataFrame) -> str:
    display = summary[
        [
            "station_id",
            "station_name",
            "coordinates",
            "depth_m",
            "record_span",
            "median_swh_m",
            "max_swh_m",
            "data_volume",
            "region_key",
        ]
    ].copy()
    display.columns = [
        "Station ID",
        "Station Name",
        "Coordinates",
        "Depth (m)",
        "Record Span",
        "Median SWH (m)",
        "Max SWH (m)",
        "Data Volume",
        "Region",
    ]
    headers = list(display.columns)
    rows = [[str(value) for value in row] for row in display.itertuples(index=False, name=None)]
    table = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        table.append("| " + " | ".join(row) + " |")
    return "\n".join(table)


def build_map(summary: pd.DataFrame) -> go.Figure:
    fig = px.scatter_geo(
        summary,
        lat="latitude",
        lon="longitude",
        color="region_key",
        hover_name="station_id",
        hover_data={
            "station_name": True,
            "depth_m": True,
            "median_swh_m": True,
            "max_swh_m": True,
            "latitude": ":.3f",
            "longitude": ":.3f",
            "region_key": False,
        },
        projection="natural earth",
    )
    fig.update_traces(
        marker=dict(size=10, line=dict(width=1.2, color="white")),
        text=summary["station_id"],
        textposition="top center",
    )
    fig.update_geos(
        showland=True,
        landcolor="#f4efe6",
        showcountries=True,
        countrycolor="#909090",
        showocean=True,
        oceancolor="#d7e8f7",
        coastlinecolor="#6f7d8c",
        showcoastlines=True,
        lataxis=dict(showgrid=True, gridcolor="#c7d3df"),
        lonaxis=dict(showgrid=True, gridcolor="#c7d3df"),
    )
    fig.update_layout(
        title="Locations of the Selected NDBC Stations",
        template="plotly_white",
        legend_title_text="Region",
        width=1400,
        height=760,
        margin=dict(l=30, r=30, t=70, b=20),
    )
    return fig


def write_html_report(summary: pd.DataFrame, map_fig: go.Figure) -> None:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    table_html = summary[
        [
            "station_id",
            "station_name",
            "coordinates",
            "depth_m",
            "record_span",
            "median_swh_m",
            "max_swh_m",
            "data_volume",
        ]
    ].rename(
        columns={
            "station_id": "Station ID",
            "station_name": "Station Name",
            "coordinates": "Coordinates",
            "depth_m": "Depth (m)",
            "record_span": "Record Span",
            "median_swh_m": "Median SWH (m)",
            "max_swh_m": "Max SWH (m)",
            "data_volume": "Data Volume",
        }
    ).to_html(index=False, classes="station-table", border=0)

    map_div = map_fig.to_html(full_html=False, include_plotlyjs="cdn")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>NDBC Station Overview</title>
  <style>
    body {{
      margin: 32px;
      font-family: "Times New Roman", "Noto Serif", serif;
      color: #1f1f1f;
      background: #fcfbf8;
    }}
    h1, h2 {{
      font-weight: 600;
      margin-bottom: 12px;
    }}
    p {{
      font-size: 17px;
      line-height: 1.6;
      max-width: 1100px;
    }}
    .station-table {{
      border-collapse: collapse;
      width: 100%;
      max-width: 1200px;
      margin-top: 18px;
      margin-bottom: 36px;
      font-size: 18px;
    }}
    .station-table thead th {{
      border-top: 2px solid #333;
      border-bottom: 1.5px solid #333;
      padding: 10px 8px;
      text-align: left;
    }}
    .station-table tbody td {{
      border-bottom: 1px solid #bbb;
      padding: 10px 8px;
      vertical-align: top;
    }}
    .footnote {{
      font-size: 14px;
      color: #555;
      margin-top: 10px;
    }}
  </style>
</head>
<body>
  <h1>Selected NDBC Station Overview</h1>
  <p>
    Table columns are organized in a paper-ready style, including station identifier,
    geographic coordinates, water depth, record span, significant wave height statistics,
    and data volume. The map below marks the eight selected stations on a world map.
  </p>
  {table_html}
  <div class="footnote">
    Data volume is the number of raw historical meteorological records parsed from the downloaded NDBC archive.
    Metadata such as station name, coordinates, and water depth were taken from official NOAA NDBC station pages.
  </div>
  <h2>World Map of Station Locations</h2>
  {map_div}
</body>
</html>
"""
    SUMMARY_HTML_PATH.write_text(html, encoding="utf-8")


def main() -> None:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    summary = load_station_summary()
    summary.to_csv(SUMMARY_CSV_PATH, index=False, encoding="utf-8-sig")

    markdown = "# Selected NDBC Stations\n\n" + build_markdown_table(summary) + "\n"
    SUMMARY_MD_PATH.write_text(markdown, encoding="utf-8")

    map_fig = build_map(summary)
    map_fig.write_html(MAP_HTML_PATH)
    try:
        map_fig.write_image(MAP_PNG_PATH, scale=2)
        png_status = f"PNG saved to {MAP_PNG_PATH}"
    except Exception as exc:
        png_status = f"PNG export skipped: {exc}"

    write_html_report(summary, map_fig)

    print(f"Summary CSV saved to: {SUMMARY_CSV_PATH}")
    print(f"Summary Markdown saved to: {SUMMARY_MD_PATH}")
    print(f"Map HTML saved to: {MAP_HTML_PATH}")
    print(f"Overview HTML saved to: {SUMMARY_HTML_PATH}")
    print(png_status)
    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

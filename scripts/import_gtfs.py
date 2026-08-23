from __future__ import annotations

import argparse
import io
from pathlib import Path
from urllib.parse import urlparse
from zipfile import ZipFile

import pandas as pd
import requests

from thermalos.config import available_cities, city_config


def _read_source(source: str) -> bytes:
    parsed = urlparse(source)
    if parsed.scheme in {"http", "https"}:
        r = requests.get(source, timeout=90)
        r.raise_for_status()
        return r.content
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(path)
    return path.read_bytes()


def main() -> None:
    p = argparse.ArgumentParser(
        description="Import a verified static GTFS ZIP and extract stops for ThermalOS."
    )
    p.add_argument("--city", choices=available_cities(), required=True)
    p.add_argument(
        "--source",
        required=True,
        help="Local GTFS .zip path or direct HTTPS download URL. For Houston, obtain the current static feed from the official METRO data portal.",
    )
    args = p.parse_args()

    cfg = city_config(args.city)
    payload = _read_source(args.source)
    with ZipFile(io.BytesIO(payload)) as zf:
        names = {name.lower(): name for name in zf.namelist()}
        if "stops.txt" not in names:
            # Some publishers place GTFS files inside a single folder.
            match = next((name for name in zf.namelist() if name.lower().endswith("/stops.txt")), None)
            if not match:
                raise RuntimeError("GTFS ZIP does not contain stops.txt")
            stop_name = match
        else:
            stop_name = names["stops.txt"]
        with zf.open(stop_name) as f:
            stops = pd.read_csv(f)

    required = {"stop_lat", "stop_lon"}
    missing = required - set(stops.columns)
    if missing:
        raise RuntimeError(f"stops.txt missing required columns: {sorted(missing)}")

    keep = [c for c in ["stop_id", "stop_code", "stop_name", "stop_desc", "stop_lat", "stop_lon", "location_type", "parent_station"] if c in stops]
    stops = stops[keep].copy()
    stops["stop_lat"] = pd.to_numeric(stops["stop_lat"], errors="coerce")
    stops["stop_lon"] = pd.to_numeric(stops["stop_lon"], errors="coerce")
    stops = stops.dropna(subset=["stop_lat", "stop_lon"]).drop_duplicates().reset_index(drop=True)

    out = Path("data/raw") / cfg["city_key"] / "gtfs_stops.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    stops.to_csv(out, index=False)
    print(f"Saved {len(stops):,} GTFS stops -> {out}")
    print("Source was explicitly supplied by the user/operator; ThermalOS does not infer that an arbitrary feed is current.")


if __name__ == "__main__":
    main()

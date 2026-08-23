from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd
import requests

from thermalos.data.arcgis import query_feature_layer_geojson


TREE_CANOPY_LAYER = "https://gisweb.miamidade.gov/arcgis/rest/services/MD_TreeCanopy/MapServer/11"
COOLING_CENTERS_LAYER = "https://gis.miami.gov/gis/rest/services/EOC/Cooling_Centers/FeatureServer/0"
GTFS_URL = "http://www.miamidade.gov/transit/googletransit/current/google_transit.zip"


def fetch_tree_canopy_geojson(layer_url: str = TREE_CANOPY_LAYER) -> dict:
    return query_feature_layer_geojson(layer_url)


def fetch_cooling_centers_geojson(layer_url: str = COOLING_CENTERS_LAYER) -> dict:
    return query_feature_layer_geojson(layer_url)


def download_gtfs_stops(url: str = GTFS_URL, timeout_s: float = 90.0) -> pd.DataFrame:
    r = requests.get(url, timeout=timeout_s)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        with zf.open("stops.txt") as f:
            stops = pd.read_csv(f)
    keep = [c for c in ["stop_id", "stop_name", "stop_lat", "stop_lon", "location_type"] if c in stops]
    return stops[keep].copy()

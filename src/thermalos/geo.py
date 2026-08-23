from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def square_aoi(lat: float, lon: float, width_km: float) -> dict:
    """Return a WGS84 GeoJSON FeatureCollection square centered at lat/lon."""
    half_lat = (width_km / 2.0) / 111.32
    lon_scale = 111.32 * math.cos(math.radians(lat))
    half_lon = (width_km / 2.0) / lon_scale
    ring = [
        [lon - half_lon, lat - half_lat],
        [lon + half_lon, lat - half_lat],
        [lon + half_lon, lat + half_lat],
        [lon - half_lon, lat + half_lat],
        [lon - half_lon, lat - half_lat],
    ]
    return {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "properties": {},
            "geometry": {"type": "Polygon", "coordinates": [ring]},
        }],
    }


def feature_centroid(feature: dict) -> tuple[float, float]:
    geom = feature.get("geometry") or {}
    coords = geom.get("coordinates") or []
    if not coords:
        return float("nan"), float("nan")
    if geom.get("type") == "Polygon":
        ring = coords[0]
    elif geom.get("type") == "MultiPolygon":
        ring = coords[0][0]
    else:
        return float("nan"), float("nan")
    if len(ring) > 1 and ring[0] == ring[-1]:
        ring = ring[:-1]
    if not ring:
        return float("nan"), float("nan")
    lon = float(np.mean([p[0] for p in ring]))
    lat = float(np.mean([p[1] for p in ring]))
    return lat, lon


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6_371_000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def gaussian_kernel(distance_m: float, sigma_m: float) -> float:
    sigma_m = max(float(sigma_m), 1e-9)
    return math.exp(-(distance_m**2) / (2.0 * sigma_m**2))

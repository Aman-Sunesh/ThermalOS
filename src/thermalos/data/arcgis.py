from __future__ import annotations

from typing import Any

import requests


class ArcGISError(RuntimeError):
    pass


def query_feature_layer_geojson(
    layer_url: str,
    *,
    where: str = "1=1",
    out_fields: str = "*",
    out_sr: int = 4326,
    page_size: int = 1000,
    timeout_s: float = 60.0,
    max_pages: int = 100,
) -> dict[str, Any]:
    """Query an ArcGIS Feature/MapServer layer and return a GeoJSON FeatureCollection.

    The function paginates by `resultOffset`, which is supported by the Miami-Dade
    feature layers used in this repo. If the server does not support pagination,
    it still returns the first page cleanly.
    """
    query_url = layer_url.rstrip("/") + "/query"
    features: list[dict] = []
    offset = 0
    for _ in range(max_pages):
        params = {
            "where": where,
            "outFields": out_fields,
            "returnGeometry": "true",
            "outSR": str(out_sr),
            "f": "geojson",
            "resultOffset": str(offset),
            "resultRecordCount": str(page_size),
        }
        r = requests.get(query_url, params=params, timeout=timeout_s)
        if not r.ok:
            raise ArcGISError(f"{query_url} -> {r.status_code}: {r.text[:1000]}")
        body = r.json()
        if "error" in body:
            raise ArcGISError(str(body["error"]))
        batch = body.get("features", [])
        features.extend(batch)
        if len(batch) < page_size:
            break
        offset += len(batch)
    return {"type": "FeatureCollection", "features": features}


def layer_metadata(layer_url: str, timeout_s: float = 30.0) -> dict[str, Any]:
    r = requests.get(layer_url, params={"f": "json"}, timeout=timeout_s)
    if not r.ok:
        raise ArcGISError(f"{layer_url} -> {r.status_code}: {r.text[:1000]}")
    body = r.json()
    if "error" in body:
        raise ArcGISError(str(body["error"]))
    return body

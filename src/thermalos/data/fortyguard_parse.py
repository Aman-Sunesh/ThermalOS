from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from thermalos.geo import feature_centroid


def heatmap_to_frame(result: dict, *, city: str, area: str, suffix: str = "") -> pd.DataFrame:
    features = result.get("map_data", {}).get("features", [])
    rows = []
    for f in features:
        row = dict(f.get("properties") or {})
        lat, lon = feature_centroid(f)
        row.update({"lat": lat, "lon": lon, "city": city, "area": area})
        rows.append(row)
    df = pd.DataFrame(rows)
    if suffix:
        protected = {"tile_id", "lat", "lon", "city", "area"}
        df = df.rename(columns={c: f"{c}_{suffix}" for c in df.columns if c not in protected})
    return df


def environmental_to_row(result: dict) -> dict:
    locations = result.get("locations") or []
    if not locations:
        return {}
    loc = locations[0]
    out = {
        "lat": pd.to_numeric(loc.get("lat", loc.get("latitude")), errors="coerce"),
        "lon": pd.to_numeric(loc.get("lon", loc.get("longitude")), errors="coerce"),
        "elevation": pd.to_numeric(loc.get("elevation"), errors="coerce"),
        "temperature_c": pd.to_numeric(loc.get("temperature"), errors="coerce"),
    }
    for key, value in (loc.get("parameters") or {}).items():
        vals = value if isinstance(value, list) else [value]
        nums = pd.to_numeric(pd.Series(vals), errors="coerce").dropna()
        if len(nums):
            out[key] = float(nums.mean())
    clear = (loc.get("solar_irradiance") or {}).get("clear_sky") or {}
    for key, value in clear.items():
        num = pd.to_numeric(value, errors="coerce")
        if pd.notna(num):
            out[f"solar_{key}"] = float(num)
    return out


def segmentation_to_row(result: dict, kind: str = "satellite") -> dict:
    if kind == "satellite":
        segments = ((result.get("segmentation") or {}).get("segments") or {})
        coords = result.get("coordinates") or {}
    else:
        front = result.get("front") or {}
        segments = front.get("segments") or {}
        coords = result.get("coordinates") or {}
    out = {
        "lat": pd.to_numeric(coords.get("latitude"), errors="coerce"),
        "lon": pd.to_numeric(coords.get("longitude"), errors="coerce"),
    }
    clean = {}
    for k, v in segments.items():
        try:
            x = float(str(v).replace("%", "").strip())
        except Exception:
            continue
        if x > 1.0:
            x /= 100.0
        clean[k.lower().replace(" ", "_")] = float(np.clip(x, 0, 1))
    out.update({f"seg_{k}": v for k, v in clean.items()})
    return out

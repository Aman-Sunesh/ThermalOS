from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from shapely.geometry import Point, shape

from thermalos.features.heatlens import add_heatlens_features


def nearest_sample_impute(tiles: pd.DataFrame, samples: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Nearest-sample planning proxy, constrained to the same AOI when possible."""
    out = tiles.copy()
    if samples.empty or not {"lat", "lon"}.issubset(samples.columns):
        return out

    sample_lat = pd.to_numeric(samples["lat"], errors="coerce")
    sample_lon = pd.to_numeric(samples["lon"], errors="coerce")
    same_area_available = "area" in out.columns and "area" in samples.columns

    for col in columns:
        if col not in samples.columns:
            continue
        values = pd.to_numeric(samples[col], errors="coerce")
        base_valid = sample_lat.notna() & sample_lon.notna() & values.notna()
        if not base_valid.any():
            continue

        for idx, row in out.iterrows():
            valid = base_valid.copy()
            if same_area_available:
                area_valid = samples["area"].astype(str) == str(row["area"])
                if (valid & area_valid).any():
                    valid &= area_valid
            lat = sample_lat.loc[valid].to_numpy(dtype=float)
            lon = sample_lon.loc[valid].to_numpy(dtype=float)
            vals = values.loc[valid].to_numpy(dtype=float)
            d2 = (lat - float(row["lat"])) ** 2 + (lon - float(row["lon"])) ** 2
            out.loc[idx, col] = vals[int(np.argmin(d2))]

    return out


def spatial_join_polygon_properties(tiles: pd.DataFrame, geojson: dict, property_map: dict[str, str]) -> pd.DataFrame:
    out = tiles.copy()
    features = []
    for f in geojson.get("features", []):
        try:
            geom = shape(f["geometry"])
        except Exception:
            continue
        features.append((geom, f.get("properties") or {}))
    if not features:
        return out
    for idx, row in out.iterrows():
        p = Point(float(row["lon"]), float(row["lat"]))
        match = None
        for geom, props in features:
            if geom.contains(p) or geom.touches(p):
                match = props
                break
        if match is None:
            continue
        for target, source in property_map.items():
            if source in match:
                out.loc[idx, target] = match[source]
    return out


def neutral_fill_morphology(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    defaults = {
        "canopy_fraction": 0.20,
        "pervious_fraction": 0.25,
        "impervious_fraction": 0.55,
        "building_fraction": 0.25,
        "road_fraction": 0.25,
    }
    imputed = []
    for col, default in defaults.items():
        if col not in out:
            out[col] = default
            imputed.append(col)
        else:
            missing = out[col].isna()
            if missing.any():
                out.loc[missing, col] = default
                imputed.append(col)
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(default).clip(0, 1)
    return out, sorted(set(imputed))


def build_processed_table(raw_tiles: pd.DataFrame, *, vulnerable_threshold: float = 0.60) -> tuple[pd.DataFrame, dict]:
    out, imputed = neutral_fill_morphology(raw_tiles)

    # Raw harvests can number tiles independently inside each study area.
    # Preserve the source-local identifier, but make canonical tile_id unique
    # city-wide for optimization, robustness, verification, and explainability.
    if "tile_id" in out.columns:
        source_tile_id = out["tile_id"].astype(str)
    else:
        source_tile_id = pd.Series(np.arange(len(out)), index=out.index).astype(str)
    out["source_tile_id"] = source_tile_id
    city_key = out["city"].fillna("unknown_city").astype(str) if "city" in out.columns else pd.Series("unknown_city", index=out.index, dtype=str)
    area_key = out["area"].fillna("unknown_area").astype(str) if "area" in out.columns else pd.Series("unknown_area", index=out.index, dtype=str)
    out["tile_id"] = city_key + "::" + area_key + "::" + source_tile_id
    for col, default in {
        "population": 25.0,
        "exceedance_h": 4.0,
        "temperature_c": 31.0,
    }.items():
        if col not in out:
            out[col] = default
            imputed.append(col)
    # Preserve the upstream duration exactly once, then sanitize only the
    # processed planning field. This keeps impossible negative upstream values
    # auditable without allowing them to create negative exposure burden.
    if "source_exceedance_h" not in out.columns:
        out["source_exceedance_h"] = pd.to_numeric(out["exceedance_h"], errors="coerce")
    else:
        out["source_exceedance_h"] = pd.to_numeric(out["source_exceedance_h"], errors="coerce")
    out["exceedance_h"] = pd.to_numeric(out["exceedance_h"], errors="coerce").clip(lower=0.0)
    out = add_heatlens_features(out, vulnerable_threshold=vulnerable_threshold)
    provenance = {
        "rows": int(len(out)),
        "neutral_imputed_fields": sorted(set(imputed)),
        "warning": ("Neutral-filled fields must be replaced before making real municipal claims." if imputed else "No neutral morphology fields were imputed; satellite-derived morphology remains a planning proxy."),
    }
    return out, provenance

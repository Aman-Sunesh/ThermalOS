from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from shapely.geometry import Point, shape
from shapely.strtree import STRtree

from thermalos.config import available_cities, city_config
from thermalos.features.assemble import build_processed_table, nearest_sample_impute


def _load(path: Path) -> pd.DataFrame:
    if not path.exists() or not path.stat().st_size:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _load_geojson(path: Path) -> dict:
    if not path.exists() or not path.stat().st_size:
        return {"type": "FeatureCollection", "features": []}
    return json.loads(path.read_text(encoding="utf-8"))


def _approx_area_km2(geom) -> float:
    """Approximate WGS84 polygon area for neighborhood-scale density estimates."""
    if geom.is_empty:
        return float("nan")
    lat = float(geom.centroid.y)
    km_per_deg_lat = 111.32
    km_per_deg_lon = 111.32 * math.cos(math.radians(lat))
    return float(abs(geom.area) * km_per_deg_lat * km_per_deg_lon)


def _join_miami_tree_canopy_and_geoid(tiles: pd.DataFrame, geojson: dict) -> pd.DataFrame:
    out = tiles.copy()
    polygons = []
    geoms = []
    for feat in geojson.get("features", []):
        try:
            geom = shape(feat["geometry"])
        except Exception:
            continue
        props = feat.get("properties") or {}
        geoid = str(props.get("GEOID") or "").strip()
        canopy = pd.to_numeric(pd.Series([props.get("percUTC")]), errors="coerce").iloc[0]
        geoms.append(geom)
        polygons.append((geoid, canopy, _approx_area_km2(geom)))

    if not geoms:
        return out
    tree = STRtree(geoms)

    out["canopy_geoid_legacy"] = out.get("canopy_geoid_legacy", pd.Series(index=out.index, dtype=object))
    out["canopy_fraction"] = pd.to_numeric(out.get("canopy_fraction"), errors="coerce")
    out["blockgroup_area_km2"] = pd.to_numeric(out.get("blockgroup_area_km2"), errors="coerce")

    for idx, row in out.iterrows():
        pnt = Point(float(row["lon"]), float(row["lat"]))
        hits = tree.query(pnt, predicate="intersects")
        if len(hits) == 0:
            continue
        j = int(hits[0])
        geoid, canopy_pct, area_km2 = polygons[j]
        out.loc[idx, "canopy_geoid_legacy"] = geoid
        if pd.notna(canopy_pct):
            out.loc[idx, "canopy_fraction"] = float(canopy_pct) / 100.0
        out.loc[idx, "blockgroup_area_km2"] = area_km2
    return out


def _join_census_blockgroup_geography(tiles: pd.DataFrame, geojson: dict) -> pd.DataFrame:
    """Attach ACS-2024 Census block-group GEOIDs using a spatial index."""
    out = tiles.copy()
    records = []
    geoms = []
    for feat in geojson.get("features", []):
        try:
            geom = shape(feat["geometry"])
        except Exception:
            continue
        props = feat.get("properties") or {}
        geoid = str(props.get("GEOID") or "").strip()
        if len(geoid) != 12:
            continue
        land_m2 = pd.to_numeric(pd.Series([props.get("AREALAND")]), errors="coerce").iloc[0]
        area_km2 = (
            float(land_m2) / 1_000_000.0
            if pd.notna(land_m2) and float(land_m2) > 0
            else _approx_area_km2(geom)
        )
        geoms.append(geom)
        records.append((geoid, geoid[:11], area_km2))

    if not geoms:
        return out
    tree = STRtree(geoms)

    out["geoid_blockgroup"] = pd.Series(index=out.index, dtype=object)
    out["geoid_tract"] = pd.Series(index=out.index, dtype=object)
    out["blockgroup_area_km2"] = pd.to_numeric(out.get("blockgroup_area_km2"), errors="coerce")

    for idx, row in out.iterrows():
        pnt = Point(float(row["lon"]), float(row["lat"]))
        hits = tree.query(pnt, predicate="intersects")
        if len(hits) == 0:
            continue
        j = int(hits[0])
        geoid_bg, geoid_tract, area_km2 = records[j]
        out.loc[idx, "geoid_blockgroup"] = geoid_bg
        out.loc[idx, "geoid_tract"] = geoid_tract
        out.loc[idx, "blockgroup_area_km2"] = area_km2
    return out


def _join_acs_tracts(tiles: pd.DataFrame, acs: pd.DataFrame) -> pd.DataFrame:
    """Join tract-level vulnerability variables from ACS 2024.

    Poverty and vehicle-availability variables requested by ThermalOS are not
    populated in the block-group response used by this project, while the tract
    response is nearly complete. Keep population at block-group resolution and
    vulnerability at tract resolution rather than silently fabricating values.
    """
    out = tiles.copy()
    if acs.empty or "geoid_tract" not in out or "geoid_tract" not in acs:
        return out

    acs = acs.copy()
    acs["geoid_tract"] = acs["geoid_tract"].astype(str).str.replace(".0", "", regex=False).str.zfill(11)
    out["geoid_tract"] = out["geoid_tract"].astype(str).str.replace(".0", "", regex=False).str.zfill(11)
    keep = [
        c
        for c in [
            "geoid_tract",
            "poverty_fraction",
            "no_vehicle_fraction",
            "poverty_count",
            "poverty_universe",
            "households_total",
            "no_vehicle_households",
        ]
        if c in acs.columns
    ]
    return out.merge(acs[keep].drop_duplicates("geoid_tract"), on="geoid_tract", how="left", suffixes=("", "_tract"))


def _join_acs_blockgroups(tiles: pd.DataFrame, acs: pd.DataFrame, granularity_m: float) -> pd.DataFrame:
    out = tiles.copy()
    if acs.empty or 'geoid_blockgroup' not in out or 'geoid_blockgroup' not in acs:
        return out

    acs = acs.copy()
    acs['geoid_blockgroup'] = acs['geoid_blockgroup'].astype(str).str.replace('.0', '', regex=False).str.zfill(12)
    out['geoid_blockgroup'] = out['geoid_blockgroup'].astype(str).str.replace('.0', '', regex=False).str.zfill(12)

    if 'population' not in acs.columns:
        return out

    if 'population' in out.columns:
        existing_population = pd.to_numeric(out['population'], errors='coerce').copy()
    else:
        existing_population = pd.Series(np.nan, index=out.index, dtype=float)

    acs_join = (
        acs[['geoid_blockgroup', 'population']]
        .drop_duplicates('geoid_blockgroup')
        .rename(columns={'population': 'population_acs'})
    )

    out = out.merge(acs_join, on='geoid_blockgroup', how='left')

    source_pop = pd.to_numeric(out['population_acs'], errors='coerce')
    area = pd.to_numeric(out.get('blockgroup_area_km2'), errors='coerce')
    tile_area_km2 = (float(granularity_m) ** 2) / 1_000_000.0
    allocated = (source_pop / area.replace(0, np.nan)) * tile_area_km2

    out['population_acs_matched'] = allocated.notna()
    existing_population = existing_population.reset_index(drop=True).reindex(out.index)
    out['population'] = allocated.where(allocated.notna(), existing_population)

    out = out.drop(columns=['population_acs'])
    return out

def _haversine_m(lat1: float, lon1: float, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    r = 6_371_000.0
    p1 = np.radians(lat1)
    p2 = np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.minimum(1.0, np.sqrt(a)))


def _add_point_count(
    tiles: pd.DataFrame,
    points: pd.DataFrame,
    *,
    lat_col: str,
    lon_col: str,
    output_col: str,
    radius_m: float,
) -> pd.DataFrame:
    out = tiles.copy()
    if points.empty or lat_col not in points or lon_col not in points:
        return out
    plat = pd.to_numeric(points[lat_col], errors="coerce").to_numpy(dtype=float)
    plon = pd.to_numeric(points[lon_col], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(plat) & np.isfinite(plon)
    plat, plon = plat[valid], plon[valid]
    if len(plat) == 0:
        return out
    counts = []
    for _, row in out.iterrows():
        d = _haversine_m(float(row["lat"]), float(row["lon"]), plat, plon)
        counts.append(int(np.sum(d <= radius_m)))
    out[output_col] = counts
    return out


def _geojson_points_to_frame(geojson: dict) -> pd.DataFrame:
    rows = []
    for feat in geojson.get("features", []):
        geom = feat.get("geometry") or {}
        if geom.get("type") != "Point":
            continue
        coords = geom.get("coordinates") or []
        if len(coords) < 2:
            continue
        rows.append({"lon": coords[0], "lat": coords[1], **(feat.get("properties") or {})})
    return pd.DataFrame(rows)


def _add_cooling_access(tiles: pd.DataFrame, centers: pd.DataFrame) -> pd.DataFrame:
    out = tiles.copy()
    if centers.empty or "lat" not in centers or "lon" not in centers:
        return out
    clat = pd.to_numeric(centers["lat"], errors="coerce").to_numpy(dtype=float)
    clon = pd.to_numeric(centers["lon"], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(clat) & np.isfinite(clon)
    clat, clon = clat[valid], clon[valid]
    if len(clat) == 0:
        return out
    access = []
    for _, row in out.iterrows():
        d = _haversine_m(float(row["lat"]), float(row["lon"]), clat, clon)
        nearest = float(np.min(d))
        # 1 near a center, smoothly decaying with ~3 km scale.
        access.append(float(np.exp(-nearest / 3000.0)))
    out["cooling_center_access"] = access
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--city", choices=available_cities(), default="miami")
    p.add_argument("--input", type=Path, default=None)
    args = p.parse_args()

    cfg = city_config(args.city)
    root = Path("data")
    raw_path = args.input or root / "interim" / f"{cfg['city_key']}_fortyguard_tiles.csv"
    if not raw_path.exists():
        fallback = root / "sample" / f"{cfg['city_key']}_demo_tiles.csv"
        if not fallback.exists():
            raise FileNotFoundError(f"No raw/live data at {raw_path} and no sample at {fallback}")
        print(f"Live file absent; using clearly labeled synthetic demo data: {fallback}")
        raw_path = fallback

    tiles = pd.read_csv(raw_path)
    # Do not sanitize exceedance here. build_processed_table preserves the raw
    # upstream value in source_exceedance_h and clips only the processed field.
    interim = root / "interim"
    raw_city = root / "raw" / cfg["city_key"]
    enrichment = {
        "environmental_samples": 0,
        "satellite_samples": 0,
        "tree_canopy": False,
        "census_blockgroup_geography_2024": False,
        "acs_blockgroups": False,
        "acs_tracts": False,
        "gtfs": False,
        "cooling_centers": False,
        "schools": False,
    }

    env = _load(interim / f"{cfg['city_key']}_env_samples.csv")
    if not env.empty and {"lat", "lon"}.issubset(env.columns):
        # Normalize the actual FortyGuard environmental column names into a
        # stable cross-city schema.  Earlier builds only propagated solar
        # fields because the returned API names use *_celsius / *_percent.
        env = env.copy()
        env_aliases = {
            "apparent_temperature_celsius": "env_apparent_temperature_c",
            "wet_bulb_temperature_celsius": "env_wet_bulb_c",
            "relative_humidity_percent": "env_relative_humidity_pct",
            "heat_index_celsius": "env_heat_index_c",
            "cloud_cover_octas": "env_cloud_cover_context",
        }
        for source_col, canonical_col in env_aliases.items():
            if canonical_col not in env.columns and source_col in env.columns:
                env[canonical_col] = pd.to_numeric(env[source_col], errors="coerce")

        # FortyGuard's environmental endpoint supplies one or more observed
        # weather-context samples per AOI.  Use the AOI median as an
        # *independent event anchor* for Universal ThermalTwin v2 rather than
        # computing the anchor from the held-out tile-temperature labels.
        if "temperature_c" in env.columns and "area" in env.columns:
            area_anchor = (
                env.assign(temperature_c=pd.to_numeric(env["temperature_c"], errors="coerce"))
                .groupby("area", dropna=False)["temperature_c"]
                .median()
                .rename("area_reference_temperature_c")
            )
            tiles = tiles.merge(area_anchor, left_on="area", right_index=True, how="left")

        cols = [
            c
            for c in [
                "env_apparent_temperature_c",
                "env_wet_bulb_c",
                "env_relative_humidity_pct",
                "env_heat_index_c",
                "precipitation_mm",
                "env_cloud_cover_context",
                "solar_ghi",
                "solar_dni",
                "solar_dhi",
            ]
            if c in env.columns
        ]
        tiles = nearest_sample_impute(tiles, env, cols)
        enrichment["environmental_samples"] = int(len(env))
        enrichment["independent_area_temperature_anchor"] = bool(
            "area_reference_temperature_c" in tiles.columns
            and pd.to_numeric(tiles["area_reference_temperature_c"], errors="coerce").notna().all()
        )

    sat = _load(interim / f"{cfg['city_key']}_satellite_samples.csv")
    if not sat.empty and {'lat', 'lon'}.issubset(sat.columns):
        def _coverage_max(frame: pd.DataFrame, options: list[str]) -> pd.Series:
            available = [c for c in options if c in frame.columns]
            if not available:
                return pd.Series(np.nan, index=frame.index, dtype=float)
            numeric = frame[available].apply(pd.to_numeric, errors='coerce')
            return numeric.max(axis=1, skipna=True)

        # FortyGuard returns per-class coverage metrics. Some class labels contain
        # synonym groups (for example road,route and sidewalk,pavement), so use
        # the exact returned column names. Composite fields intentionally use
        # max rather than sum to avoid double-counting potentially overlapping
        # segmentation classes.
        # Keep a dedicated, source-consistent morphology schema for the
        # cross-city observational model.  This prevents Miami's official
        # canopy GIS (which is excellent for local planning) from being mixed
        # with FortyGuard satellite canopy proxies in other cities during
        # transfer learning.
        sat['fg_building_fraction'] = _coverage_max(
            sat, ['building_fraction', 'seg_building', 'seg_house']
        )
        sat['fg_canopy_fraction'] = _coverage_max(
            sat, ['canopy_fraction', 'seg_tree']
        )
        sat['fg_road_fraction'] = _coverage_max(
            sat, ['road_fraction', 'seg_road', 'seg_road,_route']
        )
        sat['fg_vegetation_fraction'] = _coverage_max(
            sat, ['vegetation_fraction', 'seg_vegetation', 'seg_tree', 'seg_grass', 'seg_plant']
        )
        sat['fg_impervious_fraction'] = _coverage_max(
            sat,
            [
                'impervious_fraction',
                'seg_building',
                'seg_house',
                'seg_road',
                'seg_road,_route',
                'seg_pavement',
                'seg_sidewalk,_pavement',
                'seg_path',
            ],
        )
        sat['fg_pervious_fraction'] = _coverage_max(
            sat,
            [
                'pervious_fraction',
                'seg_tree',
                'seg_grass',
                'seg_plant',
                'seg_earth,_ground',
            ],
        )

        portable_cols = [
            'fg_canopy_fraction',
            'fg_vegetation_fraction',
            'fg_building_fraction',
            'fg_road_fraction',
            'fg_impervious_fraction',
            'fg_pervious_fraction',
        ]
        portable_cols = [
            c for c in portable_cols
            if c in sat.columns and pd.to_numeric(sat[c], errors='coerce').notna().any()
        ]
        tiles = nearest_sample_impute(tiles, sat, portable_cols)

        # The decision engine continues to use canonical planning columns. Use
        # the portable FortyGuard proxy as the default source, while allowing a
        # city-specific authoritative layer below (Miami canopy) to override
        # the relevant planning field only.
        canonical_map = {
            'fg_canopy_fraction': 'canopy_fraction',
            'fg_vegetation_fraction': 'vegetation_fraction',
            'fg_building_fraction': 'building_fraction',
            'fg_road_fraction': 'road_fraction',
            'fg_impervious_fraction': 'impervious_fraction',
            'fg_pervious_fraction': 'pervious_fraction',
        }
        for portable_col, canonical_col in canonical_map.items():
            if portable_col not in tiles.columns:
                continue
            portable_values = pd.to_numeric(tiles[portable_col], errors='coerce')
            if canonical_col not in tiles.columns:
                tiles[canonical_col] = portable_values
            else:
                existing_values = pd.to_numeric(tiles[canonical_col], errors='coerce')
                tiles[canonical_col] = existing_values.where(existing_values.notna(), portable_values)

        enrichment['satellite_samples'] = int(len(sat))
        enrichment['portable_morphology_source'] = (
            'FortyGuard satellite segmentation; same feature definitions in every city'
        )
        enrichment['satellite_morphology_method'] = (
            'same-AOI nearest valid sample; temperature+spatial maximin sampling; '
            'non-additive max of relevant FortyGuard class-cover metrics'
        )

    # Miami-specific official enrichment. These joins are optional and fail open;
    # provenance records exactly which layers were present.
    if cfg["city_key"] == "miami":
        canopy_geo = _load_geojson(raw_city / "tree_canopy_blockgroups.geojson")
        if canopy_geo.get("features"):
            tiles = _join_miami_tree_canopy_and_geoid(tiles, canopy_geo)
            enrichment["tree_canopy"] = True

        census_geo = _load_geojson(raw_city / "census_blockgroups_2024.geojson")
        if census_geo.get("features"):
            tiles = _join_census_blockgroup_geography(tiles, census_geo)
            enrichment["census_blockgroup_geography_2024"] = True

        acs = _load(raw_city / "acs_blockgroups_2024.csv")
        if not acs.empty:
            tiles = _join_acs_blockgroups(
                tiles,
                acs,
                granularity_m=float(cfg.get("granularity_m", 100)),
            )
            enrichment["acs_blockgroups"] = True

        acs_tracts = _load(raw_city / "acs_tracts_2024.csv")
        if not acs_tracts.empty:
            tiles = _join_acs_tracts(tiles, acs_tracts)
            enrichment["acs_tracts"] = True

        gtfs = _load(raw_city / "gtfs_stops.csv")
        if not gtfs.empty:
            tiles = _add_point_count(
                tiles,
                gtfs,
                lat_col="stop_lat",
                lon_col="stop_lon",
                output_col="transit_stop_count",
                radius_m=400.0,
            )
            enrichment["gtfs"] = True

        cooling = _geojson_points_to_frame(_load_geojson(raw_city / "cooling_centers.geojson"))
        if not cooling.empty:
            tiles = _add_cooling_access(tiles, cooling)
            enrichment["cooling_centers"] = True

        schools = _geojson_points_to_frame(_load_geojson(raw_city / "public_schools.geojson"))
        if not schools.empty:
            tiles = _add_point_count(
                tiles,
                schools,
                lat_col="lat",
                lon_col="lon",
                output_col="school_count",
                radius_m=600.0,
            )
            enrichment["schools"] = True

    # Transfer-city enrichment uses the same Census/ACS/GTFS contracts as Miami.
    # City-specific official canopy or facility layers can be added without
    # changing the canonical tile schema or the downstream decision engine.
    if cfg["city_key"] != "miami":
        census_geo = _load_geojson(raw_city / "census_blockgroups_2024.geojson")
        if census_geo.get("features"):
            tiles = _join_census_blockgroup_geography(tiles, census_geo)
            enrichment["census_blockgroup_geography_2024"] = True

        acs = _load(raw_city / "acs_blockgroups_2024.csv")
        if not acs.empty:
            tiles = _join_acs_blockgroups(
                tiles,
                acs,
                granularity_m=float(cfg.get("granularity_m", 100)),
            )
            enrichment["acs_blockgroups"] = True

        acs_tracts = _load(raw_city / "acs_tracts_2024.csv")
        if not acs_tracts.empty:
            tiles = _join_acs_tracts(tiles, acs_tracts)
            enrichment["acs_tracts"] = True

        gtfs = _load(raw_city / "gtfs_stops.csv")
        if not gtfs.empty and {"stop_lat", "stop_lon"}.issubset(gtfs.columns):
            tiles = _add_point_count(
                tiles,
                gtfs,
                lat_col="stop_lat",
                lon_col="stop_lon",
                output_col="transit_stop_count",
                radius_m=400.0,
            )
            enrichment["gtfs"] = True

        cooling = _geojson_points_to_frame(_load_geojson(raw_city / "cooling_centers.geojson"))
        if not cooling.empty:
            tiles = _add_cooling_access(tiles, cooling)
            enrichment["cooling_centers"] = True

        schools = _geojson_points_to_frame(_load_geojson(raw_city / "public_schools.geojson"))
        if not schools.empty:
            tiles = _add_point_count(
                tiles,
                schools,
                lat_col="lat",
                lon_col="lon",
                output_col="school_count",
                radius_m=600.0,
            )
            enrichment["schools"] = True

    coverage = {
        "canopy_fraction_non_null_pct": round(float(pd.to_numeric(tiles.get("canopy_fraction"), errors="coerce").notna().mean() * 100), 1) if "canopy_fraction" in tiles else 0.0,
        "census_blockgroup_geoid_non_null_pct": round(float(tiles.get("geoid_blockgroup", pd.Series(index=tiles.index, dtype=object)).notna().mean() * 100), 1),
        "real_blockgroup_population_pct": round(float(tiles.get("population_acs_matched", pd.Series(False, index=tiles.index)).fillna(False).astype(bool).mean() * 100), 1),
        "poverty_fraction_non_null_pct": round(float(pd.to_numeric(tiles.get("poverty_fraction"), errors="coerce").notna().mean() * 100), 1) if "poverty_fraction" in tiles else 0.0,
        "no_vehicle_fraction_non_null_pct": round(float(pd.to_numeric(tiles.get("no_vehicle_fraction"), errors="coerce").notna().mean() * 100), 1) if "no_vehicle_fraction" in tiles else 0.0,
    }

    processed, provenance = build_processed_table(
        tiles,
        vulnerable_threshold=float(cfg.get("vulnerable_threshold", 0.60)),
    )

    out = root / "processed"
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / f"{cfg['city_key']}_tiles.csv"
    processed.to_csv(csv_path, index=False)
    provenance.update(
        {
            "city": cfg["name"],
            "source_file": str(raw_path),
            "synthetic_demo": "sample" in raw_path.parts,
            "enrichment": enrichment,
            "coverage": coverage,
            "transfer_role": str(cfg.get("transfer_role", "unspecified")),
        }
    )
    (out / f"{cfg['city_key']}_provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    print(f"Saved {len(processed):,} processed tiles -> {csv_path}")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()

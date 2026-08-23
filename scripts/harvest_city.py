from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from thermalos.api.fortyguard import FortyGuardClient
from thermalos.config import available_cities, city_config
from thermalos.data.fortyguard_parse import environmental_to_row, heatmap_to_frame, segmentation_to_row
from thermalos.geo import square_aoi


def _key(df: pd.DataFrame) -> pd.Series:
    return df["lat"].round(5).astype(str) + ":" + df["lon"].round(5).astype(str)


def _join_spatial_same_grid(base: pd.DataFrame, other: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    b = base.copy()
    o = other.copy()
    b["_grid_key"] = _key(b)
    o["_grid_key"] = _key(o)
    use = ["_grid_key"] + [c for c in cols if c in o]
    b = b.merge(o[use], on="_grid_key", how="left")
    return b.drop(columns="_grid_key")


def quantile_sample(df: pd.DataFrame, n: int, col: str = "temperature_c") -> pd.DataFrame:
    if n <= 0 or df.empty:
        return df.iloc[0:0]
    work = df.sort_values(col).reset_index(drop=True)
    idx = np.unique(np.linspace(0, len(work) - 1, min(n, len(work))).round().astype(int))
    return work.iloc[idx]




def temperature_spatial_sample(df: pd.DataFrame, n: int, col: str = "temperature_c") -> pd.DataFrame:
    """Deterministic maximin sample spanning both heat and geography.

    Starts with cold/median/hot representatives, then greedily chooses points
    that are far from the already-selected set in a combined spatial/thermal
    metric. This is preferable to temperature-only quantiles for sparse
    satellite morphology because it avoids clustering all samples in one part
    of an AOI.
    """
    if n <= 0 or df.empty:
        return df.iloc[0:0].copy()
    work = df.dropna(subset=["lat", "lon", col]).copy().reset_index(drop=True)
    if len(work) <= n:
        return work

    lat = pd.to_numeric(work["lat"], errors="coerce").to_numpy(float)
    lon = pd.to_numeric(work["lon"], errors="coerce").to_numpy(float)
    temp = pd.to_numeric(work[col], errors="coerce").to_numpy(float)
    lat0 = float(np.nanmedian(lat))
    y = (lat - lat0) * 111.32
    x = (lon - float(np.nanmedian(lon))) * 111.32 * np.cos(np.radians(lat0))
    tr = float(np.nanmax(temp) - np.nanmin(temp))
    tnorm = (temp - float(np.nanmin(temp))) / (tr if tr > 1e-9 else 1.0)

    seed_candidates = [
        int(np.nanargmin(temp)),
        int(np.nanargmin(np.abs(temp - np.nanmedian(temp)))),
        int(np.nanargmax(temp)),
    ]
    selected: list[int] = []
    for idx in seed_candidates:
        if idx not in selected and len(selected) < n:
            selected.append(idx)

    spatial_scale = max(float(np.hypot(x.max() - x.min(), y.max() - y.min())), 1e-6)
    while len(selected) < n:
        remaining = [i for i in range(len(work)) if i not in selected]
        best_idx = remaining[0]
        best_score = -1.0
        for i in remaining:
            ds = min(np.hypot(x[i] - x[j], y[i] - y[j]) / spatial_scale for j in selected)
            dt = min(abs(tnorm[i] - tnorm[j]) for j in selected)
            score = 0.65 * ds + 0.35 * dt
            if score > best_score:
                best_score = score
                best_idx = i
        selected.append(best_idx)

    out = work.iloc[selected].copy().reset_index(drop=True)
    out["sampling_strategy"] = "temperature_spatial_maximin"
    out["sampling_rank"] = np.arange(1, len(out) + 1)
    return out


def harvest(city: str, mode: str, refresh: bool) -> None:
    cfg = city_config(city)
    client = FortyGuardClient(verbose=True)
    out = Path("data/interim")
    out.mkdir(parents=True, exist_ok=True)
    env_rows = []
    sat_rows = []
    street_rows = []
    tile_frames = []
    study_date = cfg["study_date"]
    threshold = float(cfg["threshold_c"])
    hcfg = cfg.get("harvest", {})

    for area_name, area in cfg["areas"].items():
        print(f"\n=== {cfg['name']} / {area_name} ===")
        aoi = square_aoi(area["lat"], area["lon"], float(area.get("width_km", 3.0)))
        snapshots = {}
        snapshot_times = ["14:00"] if mode == "minimal" else hcfg.get("snapshot_times", ["14:00"])
        for time_str in snapshot_times:
            print("TCM", time_str)
            rr = client.heatmap(
                polygon_aoi=aoi,
                start_date=study_date,
                start_time=time_str,
                filter_type=1,
                granularity=int(cfg.get("granularity_m", 100)),
                refresh=refresh,
            )
            df = heatmap_to_frame(rr.result, city=cfg["city_key"], area=area_name)
            df = df.rename(columns={
                "average_temperature": f"temperature_{time_str.replace(':','')}_c",
                "min_temperature": f"min_temperature_{time_str.replace(':','')}_c",
                "max_temperature": f"max_temperature_{time_str.replace(':','')}_c",
            })
            snapshots[time_str] = df

        base_time = "14:00" if "14:00" in snapshots else next(iter(snapshots))
        base = snapshots[base_time].copy()
        # Rename canonical afternoon field.
        if "temperature_1400_c" in base:
            base["temperature_c"] = pd.to_numeric(base["temperature_1400_c"], errors="coerce")
        else:
            temp_col = [c for c in base.columns if c.startswith("temperature_")][0]
            base["temperature_c"] = pd.to_numeric(base[temp_col], errors="coerce")

        for t, df in snapshots.items():
            if t == base_time:
                continue
            cols = [c for c in df.columns if c.startswith("temperature_")]
            base = _join_spatial_same_grid(base, df, cols)

        if mode != "minimal" and hcfg.get("include_daily", True):
            print("Daily TCM")
            rr = client.heatmap(
                polygon_aoi=aoi,
                start_date=study_date,
                filter_type=3,
                granularity=int(cfg.get("granularity_m", 100)),
                refresh=refresh,
            )
            daily = heatmap_to_frame(rr.result, city=cfg["city_key"], area=area_name)
            daily = daily.rename(columns={
                "average_temperature": "daily_mean_temperature_c",
                "min_temperature": "daily_min_temperature_c",
                "max_temperature": "daily_max_temperature_c",
            })
            base = _join_spatial_same_grid(base, daily, ["daily_mean_temperature_c", "daily_min_temperature_c", "daily_max_temperature_c"])

        analytics = [("exceedance", "exceedance_h")] if mode == "minimal" else [("exceedance", "exceedance_h"), ("persistence", "persistence_h")]
        for analytic, target in analytics:
            if not hcfg.get(f"include_{analytic}", True):
                continue
            print(analytic, "threshold", threshold)
            rr = client.heatmap(
                polygon_aoi=aoi,
                start_date=study_date,
                filter_type=3,
                granularity=int(cfg.get("granularity_m", 100)),
                analytic_type=analytic,
                threshold=threshold,
                direction="above",
                refresh=refresh,
            )
            adf = heatmap_to_frame(rr.result, city=cfg["city_key"], area=area_name)
            adf = adf.rename(columns={"value": target})
            base = _join_spatial_same_grid(base, adf, [target])

        # Point-based Premium/context endpoints from temperature-stratified locations.
        sample = quantile_sample(base.dropna(subset=["temperature_c"]), 3)
        if hcfg.get("include_environmental", True):
            env_sample = quantile_sample(base.dropna(subset=["temperature_c"]), 1 if mode == "minimal" else 3)
            for _, row in env_sample.iterrows():
                try:
                    rr = client.environmental_parameters(
                        latitude=float(row["lat"]), longitude=float(row["lon"]),
                        temperature_c=float(row["temperature_c"]),
                        start_date=study_date, start_time="14:00", refresh=refresh,
                    )
                    erow = environmental_to_row(rr.result)
                    erow["lat"] = float(row["lat"])
                    erow["lon"] = float(row["lon"])
                    erow.update({"city": cfg["city_key"], "area": area_name})
                    env_rows.append(erow)
                except Exception as exc:
                    print("ENV failed:", exc)

        sat_n = 0 if mode == "minimal" else int(hcfg.get("satellite_samples_per_area", 5))
        for _, row in temperature_spatial_sample(base.dropna(subset=["temperature_c"]), sat_n).iterrows():
            try:
                rr = client.satellite(
                    latitude=float(row["lat"]), longitude=float(row["lon"]),
                    start_date=study_date, start_time="14:00",
                    granularity=int(cfg.get("granularity_m", 100)), refresh=refresh,
                )
                srow = segmentation_to_row(rr.result, "satellite")
                srow["lat"] = float(row["lat"])
                srow["lon"] = float(row["lon"])
                srow["source_temperature_c"] = float(row["temperature_c"])
                srow["sampling_strategy"] = str(row.get("sampling_strategy", "temperature_spatial_maximin"))
                srow["sampling_rank"] = int(row.get("sampling_rank", len(sat_rows) + 1))
                srow.update({"city": cfg["city_key"], "area": area_name})
                sat_rows.append(srow)
            except Exception as exc:
                print("SAT failed:", exc)

        street_n = int(hcfg.get("streetview_samples_per_area", 0))
        for _, row in quantile_sample(base.dropna(subset=["temperature_c"]), street_n).iterrows():
            try:
                rr = client.streetview(latitude=float(row["lat"]), longitude=float(row["lon"]), refresh=refresh)
                srow = segmentation_to_row(rr.result, "streetview")
                srow["lat"] = float(row["lat"])
                srow["lon"] = float(row["lon"])
                srow.update({"city": cfg["city_key"], "area": area_name})
                street_rows.append(srow)
            except Exception as exc:
                print("STREET failed (optional):", exc)

        base["threshold_c"] = threshold
        tile_frames.append(base)

    tiles = pd.concat(tile_frames, ignore_index=True)
    tiles.to_csv(out / f"{cfg['city_key']}_fortyguard_tiles.csv", index=False)
    pd.DataFrame(env_rows).to_csv(out / f"{cfg['city_key']}_env_samples.csv", index=False)
    pd.DataFrame(sat_rows).to_csv(out / f"{cfg['city_key']}_satellite_samples.csv", index=False)
    pd.DataFrame(street_rows).to_csv(out / f"{cfg['city_key']}_streetview_samples.csv", index=False)
    print("\nSaved", len(tiles), "tiles to", out)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--city", choices=available_cities(), default="miami")
    p.add_argument("--mode", choices=["minimal", "mvp"], default="mvp")
    p.add_argument("--refresh", action="store_true")
    args = p.parse_args()
    load_dotenv()
    harvest(args.city, args.mode, args.refresh)


if __name__ == "__main__":
    main()

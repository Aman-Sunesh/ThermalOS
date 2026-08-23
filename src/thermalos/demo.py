from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from thermalos.features.heatlens import add_heatlens_features


# Calibrated only to summary statistics observed during the geography diagnostics.
# This does NOT reconstruct FortyGuard tile values and must never be described as real API data.
PROFILES = {
    "miami": {
        "Little_Havana": {
            "lat": 25.7650, "lon": -80.2190,
            "temp_mean": 31.5, "temp_p95p05": 1.786,
            "exceed_mean": 10.5, "exceed_p95p05": 8.20,
            "daily_amp": 6.20,
            "canopy": 0.10, "impervious": 0.70, "building": 0.34, "road": 0.24,
            "population": 48, "vulnerability": 0.72, "transit": 2.5,
        },
        "Hialeah": {
            "lat": 25.8576, "lon": -80.2781,
            "temp_mean": 31.0, "temp_p95p05": 0.665,
            "exceed_mean": 8.0, "exceed_p95p05": 4.45,
            "daily_amp": 6.05,
            "canopy": 0.14, "impervious": 0.68, "building": 0.31, "road": 0.26,
            "population": 52, "vulnerability": 0.66, "transit": 1.8,
        },
        "Kendall": {
            "lat": 25.6793, "lon": -80.3173,
            "temp_mean": 31.6, "temp_p95p05": 1.614,
            "exceed_mean": 7.5, "exceed_p95p05": 4.36,
            "daily_amp": 7.35,
            "canopy": 0.26, "impervious": 0.53, "building": 0.24, "road": 0.22,
            "population": 31, "vulnerability": 0.42, "transit": 1.0,
        },
    },
    "houston": {
        "Downtown": {
            "lat": 29.7604, "lon": -95.3698,
            "temp_mean": 34.5, "temp_p95p05": 1.704,
            "exceed_mean": 8.0, "exceed_p95p05": 4.35,
            "daily_amp": 8.56,
            "canopy": 0.12, "impervious": 0.76, "building": 0.42, "road": 0.24,
            "population": 25, "vulnerability": 0.48, "transit": 3.0,
        },
        "East_End": {
            "lat": 29.7480, "lon": -95.3160,
            "temp_mean": 35.0, "temp_p95p05": 1.750,
            "exceed_mean": 9.0, "exceed_p95p05": 6.97,
            "daily_amp": 9.47,
            "canopy": 0.15, "impervious": 0.68, "building": 0.30, "road": 0.27,
            "population": 37, "vulnerability": 0.69, "transit": 1.7,
        },
        "Gulfton": {
            "lat": 29.7160, "lon": -95.4840,
            "temp_mean": 36.3, "temp_p95p05": 0.451,
            "exceed_mean": 7.0, "exceed_p95p05": 1.21,
            "daily_amp": 9.53,
            "canopy": 0.11, "impervious": 0.71, "building": 0.36, "road": 0.21,
            "population": 60, "vulnerability": 0.78, "transit": 2.0,
        },
    },
}


def _sigma_from_p95p05(span: float) -> float:
    return max(span / 3.289707, 1e-4)


def generate_demo_city(city: str, *, n_side: int = 8, seed: int = 42) -> pd.DataFrame:
    city = city.lower()
    if city not in PROFILES:
        raise KeyError(city)
    rng = np.random.default_rng(seed + (0 if city == "miami" else 1000))
    rows = []
    tile_num = 0
    for area, p in PROFILES[city].items():
        temp_sigma = _sigma_from_p95p05(p["temp_p95p05"])
        dur_sigma = _sigma_from_p95p05(p["exceed_p95p05"])
        # ~250 m spacing in an ~2 km square for a compact demo.
        lat_step = 0.0022
        lon_step = 0.0024
        for iy in range(n_side):
            for ix in range(n_side):
                dx = ix - (n_side - 1) / 2
                dy = iy - (n_side - 1) / 2
                # Smooth spatial field + small noise; not independent random pixels.
                wave = 0.55 * math.sin(ix / 1.7) + 0.45 * math.cos(iy / 1.9)
                noise = rng.normal(0, 0.35)
                z = (wave + noise) / 1.4
                temp = p["temp_mean"] + temp_sigma * z
                exceed = max(0.2, p["exceed_mean"] + dur_sigma * (0.9 * z + rng.normal(0, 0.25)))
                canopy = float(np.clip(p["canopy"] + rng.normal(0, 0.045) - 0.015 * z, 0.01, 0.75))
                impervious = float(np.clip(p["impervious"] + rng.normal(0, 0.05) + 0.02 * z, 0.10, 0.95))
                building = float(np.clip(p["building"] + rng.normal(0, 0.04), 0.03, 0.80))
                road = float(np.clip(p["road"] + rng.normal(0, 0.03), 0.03, 0.65))
                pervious = float(np.clip(1.0 - impervious - 0.15 * building, 0.03, 0.75))
                vuln = float(np.clip(p["vulnerability"] + rng.normal(0, 0.09), 0.05, 0.98))
                pop = max(3.0, p["population"] * rng.lognormal(0, 0.20))
                transit = max(0.0, p["transit"] + rng.normal(0, 0.8))
                humidity = 69 + (4 if city == "miami" else 0) + rng.normal(0, 3)
                wet_bulb = temp - (5.0 if city == "miami" else 6.0) + rng.normal(0, 0.4)
                apparent = temp + (4.0 if city == "miami" else 4.5) + rng.normal(0, 0.5)
                solar = 760 + rng.normal(0, 70)
                rows.append({
                    "tile_id": f"{city[:3]}_{tile_num:04d}",
                    "city": city,
                    "area": area,
                    "lat": p["lat"] + dy * lat_step,
                    "lon": p["lon"] + dx * lon_step,
                    "temperature_c": temp,
                    "daily_max_temperature_c": temp + 0.45 * p["daily_amp"],
                    "exceedance_h": exceed,
                    "persistence_h": max(0.1, exceed * rng.uniform(0.75, 1.0)),
                    "apparent_temperature_c": apparent,
                    "wet_bulb_c": wet_bulb,
                    "relative_humidity_pct": float(np.clip(humidity, 30, 100)),
                    "solar_ghi": max(0.0, solar),
                    "population": pop,
                    "vulnerability": vuln,
                    "canopy_fraction": canopy,
                    "vegetation_fraction": min(0.95, canopy + 0.35 * pervious),
                    "pervious_fraction": pervious,
                    "impervious_fraction": impervious,
                    "building_fraction": building,
                    "road_fraction": road,
                    "transit_stop_count": transit,
                    "school_count": max(0.0, rng.poisson(0.20)),
                    "park_count": max(0.0, rng.poisson(0.10)),
                    "cooling_center_access": float(np.clip(rng.beta(2, 5), 0, 1)),
                    "data_provenance": "synthetic_demo_calibrated_to_diagnostic_summaries",
                })
                tile_num += 1
    df = pd.DataFrame(rows)
    return add_heatlens_features(df)


def write_demo_data(root: str | Path = "data/sample") -> list[Path]:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    paths = []
    for city in ["miami", "houston"]:
        df = generate_demo_city(city)
        path = root / f"{city}_demo_tiles.csv"
        df.to_csv(path, index=False)
        paths.append(path)
    (root / "README.md").write_text(
        "# Sample data\n\n"
        "The CSV files in this directory are deterministic **synthetic demo data** calibrated only to the summary statistics observed during the ThermalOS geography diagnostics. They are not raw FortyGuard tiles and must not be presented as measured or predicted city data. Use the live harvesting pipeline for real-data results.\n",
        encoding="utf-8",
    )
    return paths

from __future__ import annotations

import numpy as np
import pandas as pd


def robust_unit_interval(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce").astype(float)
    if x.notna().sum() == 0:
        return pd.Series(0.5, index=series.index, dtype=float)
    med = float(x.median())
    x = x.fillna(med)
    lo = float(x.quantile(0.05))
    hi = float(x.quantile(0.95))
    if hi <= lo + 1e-12:
        return pd.Series(0.5, index=series.index, dtype=float)
    return ((x - lo) / (hi - lo)).clip(0, 1)


def compute_vulnerability(df: pd.DataFrame) -> pd.Series:
    """Transparent 0–1 planning vulnerability index.

    Uses available columns only; this is intentionally a relative planning index,
    not a clinical or demographic-risk classification.
    """
    parts = []
    for col, weight in [
        ("poverty_fraction", 0.45),
        ("no_vehicle_fraction", 0.30),
        ("age_vulnerability_fraction", 0.25),
    ]:
        if col in df and pd.to_numeric(df[col], errors="coerce").notna().any():
            parts.append((robust_unit_interval(df[col]), weight))
    if not parts:
        if "vulnerability" in df:
            return pd.to_numeric(df["vulnerability"], errors="coerce").fillna(0.5).clip(0, 1)
        return pd.Series(0.5, index=df.index, dtype=float)
    denom = sum(w for _, w in parts)
    out = sum(s * w for s, w in parts) / denom
    return out.clip(0, 1)


def compute_exposure_multiplier(df: pd.DataFrame) -> pd.Series:
    """Simple transparent activity proxy from optional POI counts."""
    mult = pd.Series(1.0, index=df.index, dtype=float)
    if "transit_stop_count" in df:
        mult += 0.18 * robust_unit_interval(df["transit_stop_count"])
    if "school_count" in df:
        mult += 0.16 * robust_unit_interval(df["school_count"])
    if "park_count" in df:
        mult += 0.08 * robust_unit_interval(df["park_count"])
    return mult.clip(1.0, 1.6)


def add_heatlens_features(df: pd.DataFrame, *, vulnerable_threshold: float = 0.60) -> pd.DataFrame:
    out = df.copy()
    required_defaults = {
        "temperature_c": 30.0,
        "exceedance_h": 1.0,
        "population": 1.0,
    }
    for col, default in required_defaults.items():
        if col not in out:
            out[col] = default
    temp = robust_unit_interval(out["temperature_c"])
    dur = robust_unit_interval(out["exceedance_h"])
    components = [(temp, 0.30), (dur, 0.35)]
    if "apparent_temperature_c" in out:
        components.append((robust_unit_interval(out["apparent_temperature_c"]), 0.20))
    if "wet_bulb_c" in out:
        components.append((robust_unit_interval(out["wet_bulb_c"]), 0.15))
    denom = sum(w for _, w in components)
    out["thermal_stress_index"] = sum(s * w for s, w in components) / denom
    out["vulnerability"] = compute_vulnerability(out)
    out["exposure_multiplier"] = compute_exposure_multiplier(out)
    pop = pd.to_numeric(out["population"], errors="coerce").fillna(0).clip(lower=0)
    duration = pd.to_numeric(out["exceedance_h"], errors="coerce").fillna(0).clip(lower=0)
    out["exposed_population"] = pop * out["exposure_multiplier"]
    # Relative planning burden: person-hours scaled by a 1..2 stress multiplier.
    out["baseline_person_hours"] = out["exposed_population"] * duration * (1.0 + out["thermal_stress_index"])
    out["equity_weighted_person_hours"] = out["baseline_person_hours"] * (1.0 + 0.75 * out["vulnerability"])
    out["high_vulnerability"] = out["vulnerability"] >= vulnerable_threshold
    return out

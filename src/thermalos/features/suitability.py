from __future__ import annotations

import numpy as np
import pandas as pd


def _col(df: pd.DataFrame, name: str, default: float) -> pd.Series:
    if name not in df:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[name], errors="coerce").fillna(default).clip(0, 1)


def intervention_suitability(df: pd.DataFrame) -> pd.DataFrame:
    """Return one 0–1 suitability score per intervention and tile.

    These are transparent heuristics designed for rapid planning iteration.
    They should be replaced or calibrated with municipal engineering rules for
    production use.
    """
    canopy = _col(df, "canopy_fraction", 0.20)
    pervious = _col(df, "pervious_fraction", 0.25)
    impervious = _col(df, "impervious_fraction", 0.55)
    building = _col(df, "building_fraction", 0.25)
    road = _col(df, "road_fraction", 0.25)
    exposure = pd.to_numeric(df.get("exposure_multiplier", 1.0), errors="coerce")
    if not isinstance(exposure, pd.Series):
        exposure = pd.Series(float(exposure), index=df.index)
    exposure = ((exposure.fillna(1.0) - 1.0) / 0.6).clip(0, 1)
    vuln = _col(df, "vulnerability", 0.5)

    out = pd.DataFrame(index=df.index)
    # Trees: canopy gap + plantable/pervious space + some human relevance.
    out["tree_canopy"] = (0.45 * (1 - canopy) + 0.35 * pervious + 0.20 * exposure).clip(0, 1)
    # Shade: especially valuable where people are outside and canopy is low.
    out["shade_structure"] = (0.48 * exposure + 0.32 * (1 - canopy) + 0.20 * np.maximum(road, impervious)).clip(0, 1)
    # Cool pavement: only makes sense where impervious/road surface exists.
    out["cool_pavement"] = (0.55 * impervious + 0.35 * road + 0.10 * (1 - canopy)).clip(0, 1)
    # Cool roofs: building share is the dominant physical requirement.
    out["cool_roof"] = (0.80 * building + 0.20 * vuln).clip(0, 1)
    # Cooling nodes: high exposure + vulnerability + low existing cooling-center availability.
    existing = _col(df, "cooling_center_access", 0.2)
    out["cooling_node"] = (0.45 * exposure + 0.35 * vuln + 0.20 * (1 - existing)).clip(0, 1)
    return out

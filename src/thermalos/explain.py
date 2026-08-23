from __future__ import annotations

import pandas as pd


def explain_project(project: pd.Series, tile: pd.Series | None = None) -> str:
    reasons = []
    intervention = str(project.get("intervention", "project"))
    feas = float(project.get("feasibility", 0.0))
    vuln = float(project.get("vulnerability", 0.0))
    if feas >= 0.7:
        reasons.append("high physical/site suitability")
    elif feas >= 0.5:
        reasons.append("good physical/site suitability")
    if vuln >= 0.7:
        reasons.append("high planning vulnerability")
    if tile is not None:
        if float(tile.get("exceedance_h", 0.0)) >= 5:
            reasons.append("long hot-duration burden")
        if float(tile.get("canopy_fraction", 0.2)) <= 0.15 and intervention in {"tree_canopy", "shade_structure"}:
            reasons.append("low canopy")
        if float(tile.get("impervious_fraction", 0.5)) >= 0.6 and intervention == "cool_pavement":
            reasons.append("high impervious cover")
        if float(tile.get("building_fraction", 0.2)) >= 0.35 and intervention == "cool_roof":
            reasons.append("substantial building footprint")
        if float(tile.get("exposure_multiplier", 1.0)) >= 1.25:
            reasons.append("elevated human-activity proxy")
    if not reasons:
        reasons.append("positive modeled benefit per budget under current constraints")
    return ", ".join(reasons)


def explain_selected(selected: pd.DataFrame, tiles: pd.DataFrame) -> pd.DataFrame:
    """Attach compact human-readable reasons to selected projects."""
    if selected.empty:
        out = selected.copy()
        out["reason"] = pd.Series(dtype=str)
        return out
    lookup = tiles.set_index("tile_id", drop=False) if "tile_id" in tiles else pd.DataFrame()
    out = selected.copy()
    reasons = []
    for _, p in out.iterrows():
        tile = None
        if not lookup.empty and p.get("tile_id") in lookup.index:
            t = lookup.loc[p.get("tile_id")]
            tile = t.iloc[0] if isinstance(t, pd.DataFrame) else t
        reasons.append(explain_project(p, tile))
    out["reason"] = reasons
    return out

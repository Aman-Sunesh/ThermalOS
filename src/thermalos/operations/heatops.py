from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from thermalos.features.heatlens import robust_unit_interval
from thermalos.optimization.portfolio import optimize_portfolio


@dataclass
class HeatOpsResult:
    access: pd.DataFrame
    candidates: pd.DataFrame
    selected: pd.DataFrame
    summary: dict


def operations_config(path: str | Path | None = None) -> dict:
    if path is None:
        path = Path(__file__).resolve().parents[3] / "configs" / "operations.yaml"
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def compute_cooling_access_gap(tiles: pd.DataFrame) -> pd.DataFrame:
    """Compute an operational heat-access priority without inventing facility access.

    If a real ``cooling_center_access`` layer exists, it is incorporated. Otherwise
    the score is explicitly a transit/mobility access proxy driven by heat,
    vulnerability, no-vehicle prevalence, and transit availability.
    """
    out = tiles.copy()
    def series(name: str, default: float) -> pd.Series:
        raw = out[name] if name in out.columns else pd.Series(default, index=out.index, dtype=float)
        return pd.to_numeric(raw, errors="coerce").fillna(default)

    heat = series("thermal_stress_index", 0.5).clip(0, 1)
    vuln = series("vulnerability", 0.5).clip(0, 1)
    transit_raw = series("transit_stop_count", 0.0)
    transit_access = robust_unit_interval(transit_raw)
    no_vehicle = series("no_vehicle_fraction", 0.0).clip(0, 1)
    mobility_friction = no_vehicle * (1.0 - transit_access)

    base_gap = (0.52 * heat + 0.30 * vuln + 0.18 * mobility_friction).clip(0, 1)
    if "cooling_center_access" in out.columns and pd.to_numeric(out["cooling_center_access"], errors="coerce").notna().any():
        center_access = pd.to_numeric(out["cooling_center_access"], errors="coerce").fillna(0.0).clip(0, 1)
        gap = (base_gap * (0.65 + 0.35 * (1.0 - center_access))).clip(0, 1)
        basis = "thermal + vulnerability + transit/no-vehicle + cooling-center access"
        out["cooling_service_access"] = center_access
    else:
        gap = base_gap
        basis = "thermal + vulnerability + transit/no-vehicle proxy; cooling-center inventory unavailable"
        out["cooling_service_access"] = np.nan

    exposed = series("exposed_population", 0.0) if "exposed_population" in out.columns else series("population", 0.0)
    exposed = exposed.clip(lower=0)
    hours = series("exceedance_h", 0.0).clip(lower=0)
    out["transit_access_score"] = transit_access
    out["mobility_friction"] = mobility_friction
    out["cooling_access_gap"] = gap
    out["heatops_priority_person_hours"] = exposed * hours * (0.5 + gap)
    out["cooling_access_basis"] = basis
    out["access_gap_band"] = pd.cut(
        gap,
        bins=[-np.inf, 0.35, 0.55, 0.75, np.inf],
        labels=["Lower", "Moderate", "High", "Critical"],
    ).astype(str)
    return out


def build_operational_candidates(tiles: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    cfg = config or operations_config()
    actions = cfg.get("actions", {})
    access = compute_cooling_access_gap(tiles)
    rows: list[dict] = []
    for _, tile in access.iterrows():
        baseline = float(tile.get("baseline_person_hours", 0.0))
        if baseline <= 0:
            continue
        gap = float(tile.get("cooling_access_gap", 0.0))
        transit = float(tile.get("transit_access_score", 0.0))
        canopy = float(tile.get("canopy_fraction", 0.2))
        vuln = float(tile.get("vulnerability", 0.5))
        for name, spec in actions.items():
            if name == "hydration_station":
                feasibility = np.clip(0.35 + 0.35 * transit + 0.30 * gap, 0, 1)
            elif name == "temporary_shade":
                feasibility = np.clip(0.25 + 0.45 * gap + 0.30 * (1.0 - canopy), 0, 1)
            else:  # mobile cooling
                feasibility = np.clip(0.30 + 0.45 * gap + 0.25 * vuln, 0, 1)
            relief_fraction = float(spec.get("direct_exposure_relief_fraction", 0.1)) * float(feasibility)
            expected = min(baseline, baseline * relief_fraction)
            low = expected * float(spec.get("low_multiplier", 0.75))
            high = min(baseline, expected * float(spec.get("high_multiplier", 1.15)))
            rows.append(
                {
                    "candidate_id": f"ops::{tile['tile_id']}::{name}",
                    "tile_id": tile["tile_id"],
                    "city": tile.get("city", ""),
                    "area": tile.get("area", ""),
                    "lat": float(tile["lat"]),
                    "lon": float(tile["lon"]),
                    "intervention": name,
                    "label": spec.get("label", name),
                    "mechanism": spec.get("mechanism", ""),
                    "feasibility": float(feasibility),
                    "cost_usd": float(spec.get("operating_cost_usd", 0.0)),
                    "temp_relief_c_expected": 0.0,
                    "direct_exposure_relief_fraction": relief_fraction,
                    "benefit_expected_person_hours": expected,
                    "benefit_low_person_hours": low,
                    "benefit_high_person_hours": high,
                    "vulnerability": vuln,
                    "high_vulnerability": bool(tile.get("high_vulnerability", vuln >= 0.60)),
                    "baseline_person_hours": baseline,
                    "spillover_sigma_m": float(spec.get("spillover_sigma_m", 120.0)),
                    "spillover_strength": float(spec.get("spillover_strength", 0.05)),
                    "cooling_access_gap": gap,
                    "access_gap_band": tile.get("access_gap_band", ""),
                    "transit_access_score": transit,
                    "mobility_friction": float(tile.get("mobility_friction", 0.0)),
                    "cooling_access_basis": tile.get("cooling_access_basis", ""),
                }
            )
    # Stable schema is required for legitimate no-action events. In particular,
    # zero heat burden should flow through the optimizer as an empty candidate
    # set rather than fail because columns such as cost_usd do not exist.
    columns = [
        "candidate_id",
        "tile_id",
        "city",
        "area",
        "lat",
        "lon",
        "intervention",
        "label",
        "mechanism",
        "feasibility",
        "cost_usd",
        "temp_relief_c_expected",
        "direct_exposure_relief_fraction",
        "benefit_expected_person_hours",
        "benefit_low_person_hours",
        "benefit_high_person_hours",
        "vulnerability",
        "high_vulnerability",
        "baseline_person_hours",
        "spillover_sigma_m",
        "spillover_strength",
        "cooling_access_gap",
        "access_gap_band",
        "transit_access_score",
        "mobility_friction",
        "cooling_access_basis",
    ]
    return pd.DataFrame(rows, columns=columns)


def plan_heatops(
    tiles: pd.DataFrame,
    *,
    operating_budget_usd: float = 60_000,
    equity_min_fraction: float = 0.35,
    max_neighborhood_spend_fraction: float = 0.60,
    min_neighborhoods_served: int = 1,
    config: dict | None = None,
) -> HeatOpsResult:
    cfg = config or operations_config()
    access = compute_cooling_access_gap(tiles)
    candidates = build_operational_candidates(access, cfg)
    res = optimize_portfolio(
        candidates,
        budget_usd=operating_budget_usd,
        objective="balanced",
        impact_basis="expected",
        equity_min_fraction=equity_min_fraction,
        max_neighborhood_spend_fraction=max_neighborhood_spend_fraction,
        max_intervention_spend_fraction=1.0,
        min_neighborhoods_served=min_neighborhoods_served,
    )
    selected = res.selected.copy()
    if not selected.empty:
        selected = selected.sort_values("benefit_expected_person_hours", ascending=False).reset_index(drop=True)
    summary = {
        **res.summary,
        "operating_window": cfg.get("planning", {}).get("operating_window", "13:30-18:30"),
        "critical_access_gap_tiles": int((access["access_gap_band"] == "Critical").sum()),
        "high_or_critical_access_gap_tiles": int(access["access_gap_band"].isin(["High", "Critical"]).sum()),
        "access_basis": str(access["cooling_access_basis"].iloc[0]) if len(access) else "unavailable",
        "scientific_status": "operational planning scenario; not emergency or medical advice",
    }
    return HeatOpsResult(access, candidates, selected, summary)

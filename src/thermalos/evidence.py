from __future__ import annotations

import pandas as pd


def _coverage(df: pd.DataFrame, column: str) -> float:
    if column not in df.columns or len(df) == 0:
        return 0.0
    return float(pd.to_numeric(df[column], errors="coerce").notna().mean())


def build_evidence_ledger(tiles: pd.DataFrame, provenance: dict, intervention_config: dict) -> pd.DataFrame:
    """Create an auditable lineage table for the values visible in ThermalOS."""
    enrichment = provenance.get("enrichment", {}) if isinstance(provenance.get("enrichment"), dict) else {}
    synthetic = bool(provenance.get("synthetic_demo", False))
    source_file = str(provenance.get("source_file", ""))
    rows = [
        {
            "layer": "Thermal field",
            "evidence_class": "Synthetic" if synthetic else "Observed/model-derived upstream",
            "source": "Synthetic demo generator" if synthetic else "FortyGuard Temperature API / validated cache",
            "coverage_pct": 100 * _coverage(tiles, "temperature_c"),
            "used_for": "HeatLens temperature and burden",
            "uncertainty_or_limit": "FortyGuard output is an input to ThermalOS; it is not a causal intervention label." if not synthetic else "Not suitable for real municipal claims.",
            "source_detail": source_file,
        },
        {
            "layer": "Satellite morphology",
            "evidence_class": "Derived",
            "source": "FortyGuard satellite segmentation" if enrichment.get("satellite_samples") else "Neutral/demo morphology",
            "coverage_pct": 100 * min(_coverage(tiles, "building_fraction"), _coverage(tiles, "impervious_fraction"), _coverage(tiles, "pervious_fraction"), _coverage(tiles, "road_fraction")),
            "used_for": "Intervention suitability and observational context",
            "uncertainty_or_limit": enrichment.get("satellite_morphology_method", "Planning proxy; spatial sampling may be sparse."),
            "source_detail": f"samples={enrichment.get('satellite_samples', 0)}",
        },
        {
            "layer": "Tree canopy",
            "evidence_class": "Observed GIS" if enrichment.get("tree_canopy") else "Derived / proxy",
            "source": "Miami-Dade GIS" if enrichment.get("tree_canopy") else "FortyGuard satellite tree class / fallback",
            "coverage_pct": 100 * _coverage(tiles, "canopy_fraction"),
            "used_for": "Shade opportunity, suitability, context",
            "uncertainty_or_limit": "Canopy fraction is spatial context; observed canopy-temperature association is not a causal intervention effect.",
            "source_detail": "",
        },
        {
            "layer": "Population exposure",
            "evidence_class": "Derived",
            "source": "2024 ACS block-group population" if enrichment.get("acs_blockgroups") else "Demo/fallback population",
            "coverage_pct": 100 * _coverage(tiles, "population"),
            "used_for": "Exposure-weighted burden",
            "uncertainty_or_limit": "Residential population density is a planning exposure proxy, not individual mobility or protected people.",
            "source_detail": "Block-group density allocated to tile area where ACS geography is available.",
        },
        {
            "layer": "Vulnerability",
            "evidence_class": "Derived",
            "source": "2024 ACS tract indicators" if enrichment.get("acs_tracts") else "Demo/fallback vulnerability",
            "coverage_pct": 100 * _coverage(tiles, "vulnerability"),
            "used_for": "Equity weighting and policy constraints",
            "uncertainty_or_limit": "Composite planning index; not an individual clinical-risk score.",
            "source_detail": "Poverty and no-vehicle indicators where available.",
        },
        {
            "layer": "Transit access",
            "evidence_class": "Observed schedule proxy" if enrichment.get("gtfs") else "Proxy / unavailable",
            "source": "GTFS stops" if enrichment.get("gtfs") else "No real GTFS enrichment detected",
            "coverage_pct": 100 * _coverage(tiles, "transit_stop_count"),
            "used_for": "Exposure context and HeatOps mobility access",
            "uncertainty_or_limit": "Stop proximity does not measure actual ridership, service quality, wait conditions, or individual travel.",
            "source_detail": "",
        },
        {
            "layer": "Intervention effects",
            "evidence_class": "Modeled",
            "source": "Configurable literature-bounded priors + Monte Carlo",
            "coverage_pct": 100.0,
            "used_for": "Counterfactual benefit distributions",
            "uncertainty_or_limit": "Scenario estimates, not measured causal effects. ThermalVerify is designed to collect post-deployment evidence.",
            "source_detail": f"families={len(intervention_config.get('interventions', {}))}",
        },
        {
            "layer": "Project costs",
            "evidence_class": "Policy assumption",
            "source": "Planning bundles",
            "coverage_pct": 100.0,
            "used_for": "Budget-constrained optimization",
            "uncertainty_or_limit": "Must be replaced with current local procurement estimates before a real capital recommendation.",
            "source_detail": "configs/interventions.yaml",
        },
        {
            "layer": "Portfolio decision",
            "evidence_class": "Derived",
            "source": "SciPy mixed-integer linear optimization",
            "coverage_pct": 100.0,
            "used_for": "Budget, equity, concentration, and geographic policy constraints",
            "uncertainty_or_limit": "Optimal/feasible relative to the candidate set, assumptions, objective, and constraints supplied.",
            "source_detail": "Binary MILP with at-most-one intervention per tile.",
        },
        {
            "layer": "System-level modeled relief",
            "evidence_class": "Modeled",
            "source": "Distance-decay spillover + multiplicative overlap accounting",
            "coverage_pct": 100.0,
            "used_for": "Digital twin scenario outcome",
            "uncertainty_or_limit": "Not a causal evaluation. Distinct from first-order project benefit used in the budget frontier.",
            "source_detail": "",
        },
    ]
    return pd.DataFrame(rows)


def evidence_summary(ledger: pd.DataFrame) -> dict:
    if ledger.empty:
        return {"layers": 0, "real_or_derived_layers": 0, "modeled_layers": 0, "policy_assumptions": 0}
    cls = ledger["evidence_class"].astype(str)
    return {
        "layers": int(len(ledger)),
        "real_or_derived_layers": int(cls.isin(["Observed/model-derived upstream", "Observed GIS", "Observed schedule proxy", "Derived"]).sum()),
        "modeled_layers": int((cls == "Modeled").sum()),
        "policy_assumptions": int((cls == "Policy assumption").sum()),
    }

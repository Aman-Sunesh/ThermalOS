import pandas as pd

from thermalos.config import city_config, generalization_config
from thermalos.system_transfer import apply_system_gates, audit_city_contract


def _minimal_real_tiles(area="Little_Havana"):
    rows = []
    for i in range(4):
        row = {
            "tile_id": f"miami::{area}::{i}",
            "area": area,
            "lat": 25.75 + i * 0.001,
            "lon": -80.2 - i * 0.001,
            "temperature_c": 35.0 + i * 0.1,
            "exceedance_h": 1.0,
            "population": 20.0,
            "vulnerability": 0.7,
            "baseline_person_hours": 20.0,
            "area_reference_temperature_c": 34.5,
            "fg_canopy_fraction": 0.2,
            "fg_vegetation_fraction": 0.25,
            "fg_pervious_fraction": 0.3,
            "fg_impervious_fraction": 0.6,
            "fg_building_fraction": 0.4,
            "fg_road_fraction": 0.2,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def test_v3_protocol_retires_morphology_only_forecast_claim():
    cfg = generalization_config()
    assert cfg["protocol_version"].startswith("ThermalOS-v3")
    assert cfg["thermal_context"]["universal_temperature_prediction_claim"] is False
    assert cfg["development_cities"] == ["miami", "houston"]
    assert cfg["blind_cities"] == ["phoenix", "atlanta", "los_angeles"]


def test_contract_audit_enforces_exact_registered_satellite_set():
    cfg = city_config("miami")
    # Limit this unit-test config to one AOI while retaining production semantics.
    cfg = {**cfg, "areas": {"Little_Havana": cfg["areas"]["Little_Havana"]}}
    tiles = _minimal_real_tiles()
    sat = pd.DataFrame({"area": ["Little_Havana"] * 15, "lat": range(15), "lon": range(15)})
    audit = audit_city_contract(
        city="miami",
        tiles=tiles,
        provenance={"synthetic_demo": False, "transfer_role": "development"},
        city_cfg=cfg,
        satellite_samples=sat,
        target_samples_per_area=15,
        expected_role="development",
        exact_satellite_set=True,
    )
    assert audit.passed
    sat_extra = pd.concat([sat, sat.iloc[[0]]], ignore_index=True)
    audit_extra = audit_city_contract(
        city="miami",
        tiles=tiles,
        provenance={"synthetic_demo": False, "transfer_role": "development"},
        city_cfg=cfg,
        satellite_samples=sat_extra,
        target_samples_per_area=15,
        expected_role="development",
        exact_satellite_set=True,
    )
    assert not audit_extra.checks["registered_satellite_sampling"]


def test_system_gates_do_not_include_temperature_prediction_metric():
    summary = {
        "contract_fraction": 1.0,
        "candidate_generation_success": True,
        "milp_feasible": True,
        "budget_constraint_satisfied": True,
        "equity_spend_fraction": 0.50,
        "robustness_feasible_fraction": 1.0,
        "policy_scenarios_feasible_fraction": 1.0,
        "equity_frontier_feasible_fraction": 1.0,
        "evidence_ledger_layers": 10,
        "evidence_min_coverage_pct": 100.0,
    }
    gates = generalization_config()["gates"]
    out = apply_system_gates(summary, gates, equity_min_fraction=0.35)
    assert out["overall_system_pass"]
    assert not any("temperature" in k or "spearman" in k or "hotspot" in k for k in out)

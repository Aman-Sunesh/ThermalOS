import inspect

import pandas as pd

from thermalos.config import interventions_config
from thermalos.demo import generate_demo_city
from thermalos.models.interventions import build_candidates
from thermalos.optimization.portfolio import optimize_portfolio
from thermalos.operations.heatops import plan_heatops
from thermalos.reporting.dossier import build_dossier_pdf
from thermalos.verification import build_verification_registry, evaluate_post_deployment


def _selected():
    tiles = generate_demo_city("miami", n_side=5, seed=21)
    candidates = build_candidates(tiles, interventions_config(), seed=4).candidates
    selected = optimize_portfolio(
        candidates, budget_usd=350_000, equity_min_fraction=0.0
    ).selected
    return tiles, selected


def test_thermalverify_empty_controls_returns_insufficient_not_crash():
    tiles, selected = _selected()
    registry = build_verification_registry(
        tiles,
        selected,
        controls_per_project=5,
        min_control_distance_m=1_000_000_000.0,
    )
    assert len(registry.projects) > 0
    assert "candidate_id" in registry.controls.columns
    assert registry.controls.empty

    post = tiles.copy()
    post.loc[
        post["tile_id"].astype(str).isin(registry.projects["tile_id"].astype(str)),
        "temperature_c",
    ] -= 1.0

    out = evaluate_post_deployment(registry, post)
    assert len(out) == len(registry.projects)
    assert (out["verification_status"] == "insufficient_controls").all()
    assert out["observed_cooling_c"].isna().all()


def test_heatops_zero_burden_is_clean_no_action_state():
    tiles = generate_demo_city("miami", n_side=5, seed=22)
    tiles["baseline_person_hours"] = 0.0
    tiles["exceedance_h"] = 0.0
    result = plan_heatops(
        tiles,
        operating_budget_usd=60_000,
        equity_min_fraction=0.0,
    )
    assert result.candidates.empty
    assert "cost_usd" in result.candidates.columns
    assert result.selected.empty
    assert float(result.summary.get("spent_usd", 0.0)) == 0.0


def test_dossier_has_explicit_no_action_semantics_and_builds():
    source = inspect.getsource(build_dossier_pdf)
    assert "no_action_triggered" in source
    assert "no_action_reason" in source

    pdf = build_dossier_pdf(
        city_name="No Action Test City",
        summary={
            "decision_state": "no_action_triggered",
            "no_action_reason": "No positive baseline heat burden under the frozen event threshold.",
            "budget_usd": 2_000_000,
            "spent_usd": 0.0,
            "projects": 0,
        },
        selected=pd.DataFrame(),
        provenance={"synthetic_demo": True, "source_file": "test"},
        evidence_ledger=None,
        robustness_summary=None,
        verification_projects=pd.DataFrame(),
    )
    assert pdf.startswith(b"%PDF")
    assert len(pdf) > 1000

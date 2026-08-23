import pandas as pd

from thermalos.analytics.policy_lab import run_policy_stress_lab
from thermalos.analytics.robustness import run_robustness
from thermalos.config import interventions_config
from thermalos.copilot import execute_copilot, parse_copilot_command
from thermalos.demo import generate_demo_city
from thermalos.evidence import build_evidence_ledger
from thermalos.generalization import run_cross_city_comparison
from thermalos.learning import update_intervention_priors
from thermalos.models.interventions import build_candidates
from thermalos.operations.heatops import compute_cooling_access_gap, plan_heatops
from thermalos.optimization.portfolio import optimize_portfolio
from thermalos.reporting.dossier import build_dossier_pdf
from thermalos.verification import build_verification_registry


def _small_city(city="miami"):
    return generate_demo_city(city, n_side=4, seed=12 if city == "miami" else 13)


def test_robustness_engine_bounds_and_baseline_tracking():
    tiles = _small_city("miami")
    candidates = build_candidates(tiles, interventions_config(), seed=4).candidates
    out = run_robustness(
        candidates,
        budget_usd=500_000,
        equity_min_fraction=0.0,
        scenarios=4,
        pool_size=250,
        seed=3,
    )
    assert out.scenarios == 4
    assert 0 <= out.portfolio_stability <= 1
    assert 0 <= out.median_jaccard <= 1
    assert out.direct_benefit_p10 <= out.direct_benefit_p50 <= out.direct_benefit_p90
    if len(out.project_stability):
        assert out.project_stability["selection_frequency"].between(0, 1).all()
        assert out.project_stability["baseline_selected"].any()


def test_policy_stress_lab_runs_named_scenarios_and_frontier():
    tiles = _small_city("miami")
    candidates = build_candidates(tiles, interventions_config(), seed=5).candidates
    out = run_policy_stress_lab(candidates, budget_usd=500_000, area_count=3)
    assert {"Impact-first", "Balanced", "Equity-first", "Distributed", "Conservative"}.issubset(set(out.scenarios["scenario"]))
    assert len(out.equity_frontier) >= 5
    assert (out.scenarios["spent_usd"] <= 500_000 + 1e-6).all()


def test_thermalverify_freezes_baseline_without_fake_observed_effects():
    tiles = _small_city("miami")
    candidates = build_candidates(tiles, interventions_config(), seed=2).candidates
    selected = optimize_portfolio(candidates, budget_usd=500_000, equity_min_fraction=0.0).selected
    registry = build_verification_registry(tiles, selected, controls_per_project=3, min_control_distance_m=100)
    assert len(registry.projects) == len(selected)
    assert (registry.projects["verification_status"] == "baseline_captured").all()
    assert registry.projects["observed_temp_change_c"].isna().all()
    assert (registry.projects["control_count"] <= 3).all()
    assert "causal" in registry.protocol["claim_boundary"].lower()


def test_heatops_access_and_operating_budget():
    tiles = _small_city("miami")
    access = compute_cooling_access_gap(tiles)
    assert access["cooling_access_gap"].between(0, 1).all()
    ops = plan_heatops(tiles, operating_budget_usd=40_000, equity_min_fraction=0.0)
    assert float(ops.summary.get("spent_usd", 0)) <= 40_000 + 1e-6
    assert set(ops.selected.get("intervention", pd.Series(dtype=str))).issubset({"hydration_station", "temporary_shade", "mobile_cooling"})


def test_copilot_parses_and_executes_constraints():
    cmd = parse_copilot_command("Give me a $3M balanced plan with at least 50% equity and no neighborhood over 40%")
    assert cmd.updates["budget_usd"] == 3_000_000
    assert cmd.updates["equity_min_fraction"] == 0.50
    assert cmd.updates["max_neighborhood_spend_fraction"] == 0.40
    tiles = _small_city("miami")
    candidates = build_candidates(tiles, interventions_config(), seed=7).candidates
    response = execute_copilot(
        "Use a $500k conservative plan",
        candidates,
        {"budget_usd": 700_000, "equity_min_fraction": 0.0, "enabled_interventions": None},
        robustness_scenarios=2,
    )
    assert response.summary["budget_usd"] == 500_000
    assert response.summary["impact_basis"] == "conservative"
    assert response.selected["cost_usd"].sum() <= 500_000 + 1e-6


def test_evidence_dossier_and_adaptive_learning_hook():
    tiles = _small_city("miami")
    cfg = interventions_config()
    candidates = build_candidates(tiles, cfg, seed=1).candidates
    res = optimize_portfolio(candidates, budget_usd=500_000, equity_min_fraction=0.0)
    provenance = {"synthetic_demo": True, "source_file": "demo", "enrichment": {}}
    ledger = build_evidence_ledger(tiles, provenance, cfg)
    assert {"Modeled", "Policy assumption"}.issubset(set(ledger["evidence_class"]))
    pdf = build_dossier_pdf(
        city_name="Test City",
        summary={**res.summary, "modeled_person_hours_avoided_with_spillover": 1000, "modeled_reduction_fraction": 0.05},
        selected=res.selected,
        provenance=provenance,
        evidence_ledger=ledger,
    )
    assert pdf.startswith(b"%PDF")
    verified = pd.DataFrame({
        "intervention": ["tree_canopy"] * 4,
        "observed_cooling_c": [0.50, 0.60, 0.55, 0.58],
    })
    cal = update_intervention_priors(cfg, verified, minimum_verified=3)
    tree = cal.audit.loc[cal.audit["intervention"] == "tree_canopy"].iloc[0]
    assert bool(tree["updated"])
    assert tree["posterior_mean_c"] > 0

    # Reviewed files may use the human-facing intervention label rather than the
    # internal key; the adaptive hook should normalize both forms.
    verified_label = pd.DataFrame({
        "intervention": ["Tree canopy package"] * 3,
        "observed_cooling_c": [0.50, 0.60, 0.55],
    })
    cal_label = update_intervention_priors(cfg, verified_label, minimum_verified=3)
    assert bool(cal_label.audit.loc[cal_label.audit["intervention"] == "tree_canopy", "updated"].iloc[0])


def test_cross_city_engine_uses_same_pipeline_without_houston_refit():
    miami = _small_city("miami")
    houston = _small_city("houston")
    out = run_cross_city_comparison(
        miami,
        houston,
        miami_provenance={"synthetic_demo": True},
        houston_provenance={"synthetic_demo": True},
        intervention_config=interventions_config(),
        budget_usd=500_000,
    )
    assert set(out.comparison["city"]) == {"Miami-Dade", "Houston"}
    assert out.transfer_metrics["no_houston_refit"] is True
    assert len(out.portfolios["Houston"]) > 0

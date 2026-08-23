from thermalos.config import interventions_config
from thermalos.demo import generate_demo_city
from thermalos.models.interventions import build_candidates
from thermalos.optimization.portfolio import apply_portfolio_to_tiles, optimize_portfolio


def test_optimizer_budget_and_unique_tile():
    tiles = generate_demo_city("miami", n_side=3)
    c = build_candidates(tiles, interventions_config(), seed=2).candidates
    res = optimize_portfolio(c, budget_usd=500_000, objective="balanced", equity_min_fraction=0.0)
    assert res.selected["cost_usd"].sum() <= 500_000 + 1e-6
    assert res.selected["tile_id"].nunique() == len(res.selected)


def test_counterfactual_never_increases_burden():
    tiles = generate_demo_city("miami", n_side=3)
    c = build_candidates(tiles, interventions_config(), seed=3).candidates
    res = optimize_portfolio(c, budget_usd=500_000, objective="maximum_impact")
    out = apply_portfolio_to_tiles(tiles, res.selected)
    assert (out["counterfactual_person_hours"] <= out["baseline_person_hours"] + 1e-9).all()
    assert (out["modeled_relief_fraction"].between(0, 0.85)).all()


def test_policy_caps_and_minimum_neighborhoods_are_enforced():
    tiles = generate_demo_city("miami", n_side=4)
    c = build_candidates(tiles, interventions_config(), seed=5).candidates
    budget = 500_000
    res = optimize_portfolio(
        c,
        budget_usd=budget,
        objective="balanced",
        impact_basis="expected",
        equity_min_fraction=0.0,
        max_neighborhood_spend_fraction=0.60,
        max_intervention_spend_fraction=0.70,
        min_neighborhoods_served=2,
    )
    assert len(res.selected) > 0
    assert res.selected["area"].nunique() >= 2
    assert res.selected.groupby("area")["cost_usd"].sum().max() <= 0.60 * budget + 1e-6
    assert res.selected.groupby("intervention")["cost_usd"].sum().max() <= 0.70 * budget + 1e-6
    assert res.summary["max_intervention_spend_fraction_achieved"] <= 0.70 + 1e-9
    assert 0.0 <= res.summary["max_intervention_spend_fraction_of_deployed"] <= 1.0 + 1e-9


def test_conservative_basis_drives_digital_twin_with_lower_bound():
    tiles = generate_demo_city("miami", n_side=3)
    c = build_candidates(tiles, interventions_config(), seed=6).candidates
    res = optimize_portfolio(
        c,
        budget_usd=500_000,
        objective="balanced",
        impact_basis="conservative",
        equity_min_fraction=0.0,
    )
    assert res.summary["impact_basis"] == "conservative"
    assert abs(
        res.summary["planning_person_hours_avoided_first_order"]
        - res.summary["low_person_hours_avoided_first_order"]
    ) < 1e-9
    out_low = apply_portfolio_to_tiles(tiles, res.selected, impact_basis="conservative")
    out_expected = apply_portfolio_to_tiles(tiles, res.selected, impact_basis="expected")
    assert out_low["person_hours_avoided"].sum() <= out_expected["person_hours_avoided"].sum() + 1e-9

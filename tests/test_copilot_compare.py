import pandas as pd

from thermalos.config import interventions_config
from thermalos.copilot import execute_copilot, parse_copilot_command
from thermalos.demo import generate_demo_city
from thermalos.models.interventions import build_candidates


def test_copilot_executes_two_budget_comparison():
    cmd = parse_copilot_command("Compare a $300k plan with a $500k plan")
    assert cmd.intent == "compare"
    assert cmd.updates["budget_usd"] == 300_000
    assert cmd.comparison_budget_usd == 500_000

    tiles = generate_demo_city("miami", n_side=4, seed=12)
    candidates = build_candidates(tiles, interventions_config(), seed=42).candidates
    state = {
        "budget_usd": 300_000,
        "objective": "balanced",
        "impact_basis": "expected",
        "equity_min_fraction": 0.0,
        "enabled_interventions": None,
        "max_neighborhood_spend_fraction": 1.0,
        "max_intervention_spend_fraction": 1.0,
        "min_neighborhoods_served": 1,
    }
    result = execute_copilot("Compare a $300k plan with a $500k plan", candidates, state)
    assert result.supplemental is not None
    assert len(result.supplemental) == 2
    assert result.supplemental["budget_usd"].tolist() == [300_000.0, 500_000.0]
    assert result.summary["comparison_budget_usd"] == 500_000
    assert result.supplemental["status"].astype(str).str.contains("feasible|optimal", case=False).all()

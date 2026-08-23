from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from thermalos.optimization.portfolio import optimize_portfolio


@dataclass
class PolicyLabResult:
    scenarios: pd.DataFrame
    equity_frontier: pd.DataFrame


def _summary_row(name: str, res, budget_usd: float, settings: dict) -> dict:
    s = res.summary
    return {
        "scenario": name,
        "budget_usd": float(budget_usd),
        "spent_usd": float(s.get("spent_usd", 0.0)),
        "projects": int(s.get("projects", 0)),
        "direct_person_hours_avoided": float(s.get("planning_person_hours_avoided_first_order", 0.0)),
        "equity_spend_fraction": float(s.get("vulnerable_spend_fraction", 0.0)),
        "neighborhoods_served": int(s.get("neighborhoods_served", 0)),
        "largest_neighborhood_budget_share": float(s.get("max_neighborhood_spend_fraction_achieved", 0.0)),
        "largest_intervention_budget_share": float(s.get("max_intervention_spend_fraction_achieved", 0.0)),
        "objective": settings["objective"],
        "impact_basis": settings["impact_basis"],
        "equity_floor": settings["equity_min_fraction"],
        "neighborhood_cap": settings["max_neighborhood_spend_fraction"],
        "intervention_cap": settings["max_intervention_spend_fraction"],
        "minimum_neighborhoods": settings["min_neighborhoods_served"],
        "status": s.get("status", "unknown"),
    }


def run_policy_stress_lab(
    candidates: pd.DataFrame,
    *,
    budget_usd: float,
    enabled_interventions: list[str] | None = None,
    area_count: int | None = None,
    equity_frontier_points: tuple[float, ...] = (0.0, 0.20, 0.35, 0.50, 0.60, 0.70),
) -> PolicyLabResult:
    """Compare materially different policy philosophies under the same budget."""
    if area_count is None:
        area_count = int(candidates["area"].nunique()) if "area" in candidates else 1
    distributed_k = max(1, min(3, area_count))
    scenarios = {
        "Impact-first": dict(
            objective="maximum_impact", impact_basis="expected", equity_min_fraction=0.0,
            max_neighborhood_spend_fraction=1.0, max_intervention_spend_fraction=1.0,
            min_neighborhoods_served=1,
        ),
        "Balanced": dict(
            objective="balanced", impact_basis="expected", equity_min_fraction=0.35,
            max_neighborhood_spend_fraction=1.0, max_intervention_spend_fraction=1.0,
            min_neighborhoods_served=1,
        ),
        "Equity-first": dict(
            objective="maximum_equity", impact_basis="expected", equity_min_fraction=0.60,
            max_neighborhood_spend_fraction=0.60, max_intervention_spend_fraction=1.0,
            min_neighborhoods_served=min(2, area_count),
        ),
        "Distributed": dict(
            objective="balanced", impact_basis="expected", equity_min_fraction=0.35,
            max_neighborhood_spend_fraction=0.45, max_intervention_spend_fraction=0.55,
            min_neighborhoods_served=distributed_k,
        ),
        "Conservative": dict(
            objective="balanced", impact_basis="conservative", equity_min_fraction=0.35,
            max_neighborhood_spend_fraction=1.0, max_intervention_spend_fraction=1.0,
            min_neighborhoods_served=1,
        ),
    }
    rows = []
    for name, settings in scenarios.items():
        res = optimize_portfolio(candidates, budget_usd=budget_usd, enabled_interventions=enabled_interventions, **settings)
        rows.append(_summary_row(name, res, budget_usd, settings))

    frontier_rows = []
    for floor in equity_frontier_points:
        settings = dict(
            objective="balanced",
            impact_basis="expected",
            equity_min_fraction=float(floor),
            max_neighborhood_spend_fraction=1.0,
            max_intervention_spend_fraction=1.0,
            min_neighborhoods_served=1,
        )
        res = optimize_portfolio(candidates, budget_usd=budget_usd, enabled_interventions=enabled_interventions, **settings)
        frontier_rows.append(_summary_row(f"Equity {int(round(100 * floor))}%", res, budget_usd, settings))

    return PolicyLabResult(pd.DataFrame(rows), pd.DataFrame(frontier_rows))

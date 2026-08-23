from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from thermalos.optimization.portfolio import PortfolioResult, apply_portfolio_to_tiles, optimize_portfolio


@dataclass
class PlanEvaluation:
    portfolio: PortfolioResult
    counterfactual_tiles: pd.DataFrame
    metrics: dict


def evaluate_plan(
    tiles: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    budget_usd: float,
    objective: str = "balanced",
    impact_basis: str = "expected",
    equity_min_fraction: float = 0.35,
    enabled_interventions: list[str] | None = None,
    max_neighborhood_spend_fraction: float = 1.0,
    max_intervention_spend_fraction: float = 1.0,
    min_neighborhoods_served: int = 1,
    max_relief_fraction: float = 0.85,
) -> PlanEvaluation:
    """Run one complete ThermalOS planning scenario.

    This helper keeps the decision engine, downstream digital-twin accounting, and
    KPI semantics identical across the UI, robustness studies, the policy lab,
    cross-city evaluation, and the Copilot.
    """
    portfolio = optimize_portfolio(
        candidates,
        budget_usd=budget_usd,
        objective=objective,
        impact_basis=impact_basis,
        equity_min_fraction=equity_min_fraction,
        enabled_interventions=enabled_interventions,
        max_neighborhood_spend_fraction=max_neighborhood_spend_fraction,
        max_intervention_spend_fraction=max_intervention_spend_fraction,
        min_neighborhoods_served=min_neighborhoods_served,
    )
    counter = apply_portfolio_to_tiles(
        tiles,
        portfolio.selected,
        max_relief_fraction=max_relief_fraction,
        impact_basis=impact_basis,
    )
    baseline = float(pd.to_numeric(counter["baseline_person_hours"], errors="coerce").fillna(0).sum())
    after = float(pd.to_numeric(counter["counterfactual_person_hours"], errors="coerce").fillna(0).sum())
    avoided = max(0.0, baseline - after)
    selected = portfolio.selected
    spent = float(pd.to_numeric(selected.get("cost_usd", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if len(selected) else 0.0
    vuln_spend = (
        float(pd.to_numeric(selected.loc[selected["high_vulnerability"], "cost_usd"], errors="coerce").fillna(0).sum())
        if len(selected) and "high_vulnerability" in selected
        else 0.0
    )
    metrics = {
        **portfolio.summary,
        "baseline_person_hours": baseline,
        "counterfactual_person_hours": after,
        "modeled_person_hours_avoided_with_spillover": avoided,
        "modeled_reduction_fraction": avoided / baseline if baseline else 0.0,
        "vulnerable_spend_fraction": vuln_spend / spent if spent else 0.0,
        "scientific_status": "scenario estimate; not a causal impact evaluation",
    }
    return PlanEvaluation(portfolio=portfolio, counterfactual_tiles=counter, metrics=metrics)

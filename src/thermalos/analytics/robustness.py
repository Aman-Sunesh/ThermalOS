from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from thermalos.optimization.portfolio import optimize_portfolio


@dataclass
class RobustnessResult:
    baseline_selected: pd.DataFrame
    project_stability: pd.DataFrame
    scenario_summary: pd.DataFrame
    portfolio_stability: float
    median_jaccard: float
    direct_benefit_p10: float
    direct_benefit_p50: float
    direct_benefit_p90: float
    scenarios: int
    candidate_pool_size: int
    method_note: str


def _competitive_pool(candidates: pd.DataFrame, baseline_ids: set[str], pool_size: int) -> pd.DataFrame:
    if len(candidates) <= pool_size:
        return candidates.copy().reset_index(drop=True)
    c = candidates.copy()
    expected = pd.to_numeric(c["benefit_expected_person_hours"], errors="coerce").fillna(0.0)
    low = pd.to_numeric(c["benefit_low_person_hours"], errors="coerce").fillna(0.0)
    cost = pd.to_numeric(c["cost_usd"], errors="coerce").fillna(np.inf).clip(lower=1.0)
    # Score rewards both expected value and downside protection, preserving projects
    # that remain competitive under uncertainty rather than only mean-optimal ones.
    c["_robust_pool_score"] = (0.65 * expected + 0.35 * low) / cost
    top = c.nlargest(pool_size, "_robust_pool_score")
    baseline = c[c["candidate_id"].astype(str).isin(baseline_ids)]
    return (
        pd.concat([top, baseline], ignore_index=True)
        .drop_duplicates("candidate_id")
        .drop(columns=["_robust_pool_score"], errors="ignore")
        .reset_index(drop=True)
    )


def run_robustness(
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
    scenarios: int = 32,
    seed: int = 20260822,
    pool_size: int = 1800,
    cost_sigma_fraction: float = 0.08,
) -> RobustnessResult:
    """Stress-test portfolio selection under effect and cost uncertainty.

    Candidate benefits are sampled from triangular distributions bounded by each
    project's configured lower/expected/upper benefit estimates. Project costs are
    independently perturbed with clipped log-normal noise. Each sampled world is
    re-optimized under the *same policy constraints*.

    For interactive latency, the repeated solves use a competitive subset selected
    by expected/downside benefit-per-dollar plus every baseline-funded project.
    The baseline portfolio itself is always solved on the full candidate universe.
    """
    scenarios = int(max(1, scenarios))
    baseline = optimize_portfolio(
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
    baseline_ids = set(baseline.selected.get("candidate_id", pd.Series(dtype=str)).astype(str))
    pool = _competitive_pool(candidates, baseline_ids, max(250, int(pool_size)))
    rng = np.random.default_rng(seed)

    counts: dict[str, int] = {}
    scenario_rows: list[dict] = []
    baseline_denom = max(len(baseline_ids), 1)

    for scenario_id in range(scenarios):
        c = pool.copy()
        lo = pd.to_numeric(c["benefit_low_person_hours"], errors="coerce").fillna(0).to_numpy(float)
        mid = pd.to_numeric(c["benefit_expected_person_hours"], errors="coerce").fillna(0).to_numpy(float)
        hi = pd.to_numeric(c["benefit_high_person_hours"], errors="coerce").fillna(0).to_numpy(float)
        lo = np.minimum(lo, mid)
        hi = np.maximum(hi, mid)
        sampled_benefit = mid.copy()
        variable = hi > lo + 1e-12
        if variable.any():
            sampled_benefit[variable] = rng.triangular(lo[variable], mid[variable], hi[variable])

        # Mean-one log-normal multiplier, clipped to prevent unrealistic tails in
        # an interactive planning stress test.
        sigma = max(float(cost_sigma_fraction), 0.0)
        mu = -0.5 * sigma * sigma
        cost_mult = np.exp(rng.normal(mu, sigma, len(c)))
        cost_mult = np.clip(cost_mult, 0.75, 1.30)
        c["cost_usd"] = pd.to_numeric(c["cost_usd"], errors="coerce").fillna(0).to_numpy(float) * cost_mult

        # A sampled scenario represents one plausible realization, so the chosen
        # planning-basis column is replaced by that realization. The other basis is
        # retained unless conservative planning is explicitly requested.
        if str(impact_basis).lower().startswith("conserv"):
            c["benefit_low_person_hours"] = sampled_benefit
        else:
            c["benefit_expected_person_hours"] = sampled_benefit

        res = optimize_portfolio(
            c,
            budget_usd=budget_usd,
            objective=objective,
            impact_basis=impact_basis,
            equity_min_fraction=equity_min_fraction,
            enabled_interventions=enabled_interventions,
            max_neighborhood_spend_fraction=max_neighborhood_spend_fraction,
            max_intervention_spend_fraction=max_intervention_spend_fraction,
            min_neighborhoods_served=min_neighborhoods_served,
        )
        ids = set(res.selected.get("candidate_id", pd.Series(dtype=str)).astype(str))
        for cid in ids:
            counts[cid] = counts.get(cid, 0) + 1
        union = baseline_ids | ids
        jaccard = len(baseline_ids & ids) / len(union) if union else 1.0
        scenario_rows.append(
            {
                "scenario": scenario_id + 1,
                "projects": int(len(res.selected)),
                "spent_usd": float(res.summary.get("spent_usd", 0.0)),
                "direct_person_hours_avoided": float(res.summary.get("planning_person_hours_avoided_first_order", 0.0)),
                "vulnerable_spend_fraction": float(res.summary.get("vulnerable_spend_fraction", 0.0)),
                "jaccard_vs_baseline": float(jaccard),
                "status": str(res.summary.get("status", "unknown")),
            }
        )

    scenario_summary = pd.DataFrame(scenario_rows)
    all_ids = set(counts) | baseline_ids
    lookup = candidates.drop_duplicates("candidate_id").copy()
    lookup["candidate_id"] = lookup["candidate_id"].astype(str)
    lookup = lookup.set_index("candidate_id")
    rows = []
    for cid in all_ids:
        if cid not in lookup.index:
            continue
        row = lookup.loc[cid]
        rows.append(
            {
                "candidate_id": cid,
                "tile_id": row.get("tile_id"),
                "area": row.get("area", ""),
                "intervention": row.get("intervention", ""),
                "label": row.get("label", ""),
                "cost_usd": float(row.get("cost_usd", 0.0)),
                "benefit_expected_person_hours": float(row.get("benefit_expected_person_hours", 0.0)),
                "baseline_selected": cid in baseline_ids,
                "selection_frequency": counts.get(cid, 0) / scenarios,
            }
        )
    stability = pd.DataFrame(rows)
    if not stability.empty:
        stability = stability.sort_values(["baseline_selected", "selection_frequency"], ascending=[False, False]).reset_index(drop=True)

    baseline_stability = stability[stability["baseline_selected"]].copy() if not stability.empty else pd.DataFrame()
    if len(baseline_stability):
        weights = pd.to_numeric(baseline_stability["cost_usd"], errors="coerce").fillna(0).to_numpy(float)
        freq = pd.to_numeric(baseline_stability["selection_frequency"], errors="coerce").fillna(0).to_numpy(float)
        portfolio_stability = float(np.average(freq, weights=weights)) if weights.sum() > 0 else float(freq.mean())
    else:
        portfolio_stability = 0.0

    benefits = pd.to_numeric(scenario_summary["direct_person_hours_avoided"], errors="coerce").fillna(0).to_numpy(float)
    return RobustnessResult(
        baseline_selected=baseline.selected.copy(),
        project_stability=stability,
        scenario_summary=scenario_summary,
        portfolio_stability=portfolio_stability,
        median_jaccard=float(scenario_summary["jaccard_vs_baseline"].median()) if len(scenario_summary) else 0.0,
        direct_benefit_p10=float(np.quantile(benefits, 0.10)) if len(benefits) else 0.0,
        direct_benefit_p50=float(np.quantile(benefits, 0.50)) if len(benefits) else 0.0,
        direct_benefit_p90=float(np.quantile(benefits, 0.90)) if len(benefits) else 0.0,
        scenarios=scenarios,
        candidate_pool_size=int(len(pool)),
        method_note=(
            "Repeated constrained MILP over a competitive candidate subset; effects sampled from project lower/expected/upper bounds and costs perturbed with clipped log-normal noise. "
            "This is a decision-stability stress test, not a probabilistic guarantee."
        ),
    )

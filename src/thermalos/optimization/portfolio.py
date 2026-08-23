from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

from thermalos.geo import gaussian_kernel, haversine_m


@dataclass
class PortfolioResult:
    selected: pd.DataFrame
    summary: dict
    solver: str


def _normalize_impact_basis(impact_basis: str) -> str:
    basis = str(impact_basis).lower().replace(" ", "_")
    if basis in {"conservative", "low", "lower_bound", "lower-bound"}:
        return "conservative"
    return "expected"


def _benefit_column(impact_basis: str) -> str:
    return (
        "benefit_low_person_hours"
        if _normalize_impact_basis(impact_basis) == "conservative"
        else "benefit_expected_person_hours"
    )


def _objective_values(
    c: pd.DataFrame,
    objective: str,
    impact_basis: str = "expected",
) -> np.ndarray:
    objective = str(objective).lower().replace(" ", "_")

    # Backward compatibility with the original CLI/UI where "conservative" was
    # an objective rather than an orthogonal planning basis.
    if objective in {"conservative", "robust"}:
        impact_basis = "conservative"
        objective = "balanced"

    primary = c[_benefit_column(impact_basis)].to_numpy(float)
    vuln = c["vulnerability"].to_numpy(float)
    cost = c["cost_usd"].to_numpy(float)

    if objective in {"maximum_impact", "impact"}:
        return primary
    if objective in {"maximum_equity", "equity"}:
        return primary * (0.6 + 1.4 * vuln)
    if objective in {"minimum_cost", "cost_efficiency", "efficiency"}:
        return primary / np.maximum(cost / 100_000.0, 0.05)

    # Balanced: impact remains dominant, with a transparent vulnerability and
    # benefit-per-dollar preference. The chosen impact basis controls which
    # intervention-effect estimate is optimized.
    efficiency = primary / np.maximum(cost / 100_000.0, 0.05)
    efficiency = efficiency / max(np.nanmedian(efficiency), 1e-6)
    return primary * (0.85 + 0.45 * vuln) + 0.08 * np.nanmedian(primary) * efficiency


def _empty_result(
    c: pd.DataFrame,
    *,
    budget_usd: float,
    objective: str,
    impact_basis: str,
    equity_min_fraction: float,
    max_neighborhood_spend_fraction: float,
    max_intervention_spend_fraction: float,
    min_neighborhoods_served: int,
    solver: str,
    status: str,
) -> PortfolioResult:
    summary = {
        "budget_usd": float(budget_usd),
        "spent_usd": 0.0,
        "budget_utilization": 0.0,
        "projects": 0,
        "expected_person_hours_avoided_first_order": 0.0,
        "low_person_hours_avoided_first_order": 0.0,
        "planning_person_hours_avoided_first_order": 0.0,
        "vulnerable_spend_fraction": 0.0,
        "objective": objective,
        "impact_basis": _normalize_impact_basis(impact_basis),
        "equity_min_fraction": float(np.clip(equity_min_fraction, 0, 1)),
        "max_neighborhood_spend_fraction": float(np.clip(max_neighborhood_spend_fraction, 0, 1)),
        "max_intervention_spend_fraction": float(np.clip(max_intervention_spend_fraction, 0, 1)),
        "min_neighborhoods_served": int(max(1, min_neighborhoods_served)),
        "neighborhoods_served": 0,
        "max_neighborhood_spend_fraction_achieved": 0.0,
        "max_intervention_spend_fraction_achieved": 0.0,
        "max_neighborhood_spend_fraction_of_deployed": 0.0,
        "max_intervention_spend_fraction_of_deployed": 0.0,
        "status": status,
    }
    return PortfolioResult(c.iloc[0:0].copy(), summary, solver)


def optimize_portfolio(
    candidates: pd.DataFrame,
    *,
    budget_usd: float,
    objective: str = "balanced",
    impact_basis: str = "expected",
    equity_min_fraction: float = 0.0,
    enabled_interventions: list[str] | None = None,
    max_neighborhood_spend_fraction: float = 1.0,
    max_intervention_spend_fraction: float = 1.0,
    min_neighborhoods_served: int = 1,
) -> PortfolioResult:
    """Solve the city intervention portfolio as a binary MILP.

    Policy controls are explicit constraints rather than hidden diversification:
    - ``impact_basis`` selects expected vs lower-bound intervention benefits.
    - neighborhood/intervention caps are fractions of the selected capital budget.
    - ``min_neighborhoods_served`` requires funded projects in at least K areas.
    """
    c = candidates.copy()
    if enabled_interventions is not None:
        c = c[c["intervention"].isin(enabled_interventions)].copy()
    c = c[c["cost_usd"] <= budget_usd].reset_index(drop=True)

    basis = _normalize_impact_basis(impact_basis)
    alpha = float(np.clip(equity_min_fraction, 0, 1))
    area_cap = float(np.clip(max_neighborhood_spend_fraction, 0, 1))
    intervention_cap = float(np.clip(max_intervention_spend_fraction, 0, 1))

    if c.empty or budget_usd <= 0:
        return _empty_result(
            c,
            budget_usd=budget_usd,
            objective=objective,
            impact_basis=basis,
            equity_min_fraction=alpha,
            max_neighborhood_spend_fraction=area_cap,
            max_intervention_spend_fraction=intervention_cap,
            min_neighborhoods_served=min_neighborhoods_served,
            solver="none",
            status="no_candidates",
        )

    areas = sorted(c["area"].fillna("Unknown").astype(str).unique().tolist())
    requested_min_areas = int(max(1, min_neighborhoods_served))
    min_areas = min(requested_min_areas, len(areas))

    benefit = _objective_values(c, objective, basis)
    n = len(c)

    # Auxiliary y_a binaries make "minimum neighborhoods served" an exact MILP
    # constraint instead of a post-hoc diversity heuristic.
    use_area_indicators = min_areas > 0
    area_to_aux = {area: n + j for j, area in enumerate(areas)} if use_area_indicators else {}
    total_vars = n + len(area_to_aux)

    rows: list[np.ndarray] = []
    lbs: list[float] = []
    ubs: list[float] = []

    def add_constraint(candidate_coeffs: np.ndarray, lb: float, ub: float, aux: dict[int, float] | None = None) -> None:
        row = np.zeros(total_vars, dtype=float)
        row[:n] = candidate_coeffs
        if aux:
            for pos, val in aux.items():
                row[pos] = val
        rows.append(row)
        lbs.append(lb)
        ubs.append(ub)

    costs = c["cost_usd"].to_numpy(float)

    # Capital budget.
    add_constraint(costs, -np.inf, float(budget_usd))

    # At most one intervention bundle per tile in the MVP.
    for _, idxs in c.groupby("tile_id").groups.items():
        row = np.zeros(n)
        row[list(idxs)] = 1.0
        add_constraint(row, -np.inf, 1.0)

    # Equity: vulnerable spend / total spend >= alpha. This remains linear:
    # sum(cost*vulnerable*x) - alpha*sum(cost*x) >= 0.
    if alpha > 0:
        vulnerable = c["high_vulnerability"].astype(float).to_numpy()
        add_constraint(costs * (vulnerable - alpha), 0.0, np.inf)

    # Maximum dollar share of the *capital budget* that may be assigned to any
    # one neighborhood. 100% is exactly the original unconstrained behavior.
    if area_cap < 1.0 - 1e-12:
        for area, idxs in c.groupby("area", dropna=False).groups.items():
            row = np.zeros(n)
            row[list(idxs)] = costs[list(idxs)]
            add_constraint(row, -np.inf, area_cap * float(budget_usd))

    # Same transparent cap for intervention-type concentration.
    if intervention_cap < 1.0 - 1e-12:
        for _, idxs in c.groupby("intervention").groups.items():
            row = np.zeros(n)
            row[list(idxs)] = costs[list(idxs)]
            add_constraint(row, -np.inf, intervention_cap * float(budget_usd))

    # Link each area indicator y_a to whether any candidate in that area is
    # selected: y_a <= sum(x_i) <= M_a*y_a. Then require sum(y_a) >= K.
    if use_area_indicators:
        normalized_area = c["area"].fillna("Unknown").astype(str)
        for area in areas:
            idxs = np.flatnonzero(normalized_area.to_numpy() == area)
            row = np.zeros(n)
            row[idxs] = 1.0
            y_pos = area_to_aux[area]
            add_constraint(row, 0.0, np.inf, aux={y_pos: -1.0})
            add_constraint(row, -np.inf, 0.0, aux={y_pos: -float(len(idxs))})

        row = np.zeros(n)
        aux = {pos: 1.0 for pos in area_to_aux.values()}
        add_constraint(row, float(min_areas), np.inf, aux=aux)

    A = np.vstack(rows)
    constraints = LinearConstraint(A, np.array(lbs), np.array(ubs))
    full_benefit = np.concatenate([benefit, np.zeros(total_vars - n)])

    res = milp(
        c=-full_benefit,
        integrality=np.ones(total_vars, dtype=int),
        bounds=Bounds(np.zeros(total_vars), np.ones(total_vars)),
        constraints=constraints,
        options={"time_limit": 12.0},
    )

    if res.x is None:
        # Do not silently violate explicit planner constraints with a greedy
        # fallback. An infeasible policy combination is itself useful feedback.
        return _empty_result(
            c,
            budget_usd=budget_usd,
            objective=objective,
            impact_basis=basis,
            equity_min_fraction=alpha,
            max_neighborhood_spend_fraction=area_cap,
            max_intervention_spend_fraction=intervention_cap,
            min_neighborhoods_served=min_areas,
            solver="scipy_milp",
            status="infeasible_or_no_solution",
        )

    selected = c.loc[np.asarray(res.x[:n]) > 0.5].copy()
    solver = "scipy_milp"

    spent = float(selected["cost_usd"].sum()) if len(selected) else 0.0
    vuln_spend = float(selected.loc[selected["high_vulnerability"], "cost_usd"].sum()) if len(selected) else 0.0
    expected_total = float(selected["benefit_expected_person_hours"].sum()) if len(selected) else 0.0
    low_total = float(selected["benefit_low_person_hours"].sum()) if len(selected) else 0.0
    planning_total = low_total if basis == "conservative" else expected_total

    if spent and len(selected):
        area_spend = selected.groupby("area", dropna=False)["cost_usd"].sum()
        intervention_spend = selected.groupby("intervention")["cost_usd"].sum()
        max_area_share = float(area_spend.max() / float(budget_usd)) if budget_usd else 0.0
        max_intervention_share = float(intervention_spend.max() / float(budget_usd)) if budget_usd else 0.0
        max_area_share_deployed = float(area_spend.max() / spent) if spent else 0.0
        max_intervention_share_deployed = float(intervention_spend.max() / spent) if spent else 0.0
        neighborhoods_served = int(selected["area"].fillna("Unknown").nunique())
    else:
        max_area_share = 0.0
        max_intervention_share = 0.0
        max_area_share_deployed = 0.0
        max_intervention_share_deployed = 0.0
        neighborhoods_served = 0

    summary = {
        "budget_usd": float(budget_usd),
        "spent_usd": spent,
        "budget_utilization": spent / float(budget_usd) if budget_usd else 0.0,
        "projects": int(len(selected)),
        "expected_person_hours_avoided_first_order": expected_total,
        "low_person_hours_avoided_first_order": low_total,
        "planning_person_hours_avoided_first_order": planning_total,
        "vulnerable_spend_fraction": vuln_spend / spent if spent else 0.0,
        "objective": objective,
        "impact_basis": basis,
        "equity_min_fraction": alpha,
        "max_neighborhood_spend_fraction": area_cap,
        "max_intervention_spend_fraction": intervention_cap,
        "min_neighborhoods_served": int(min_areas),
        "neighborhoods_served": neighborhoods_served,
        "max_neighborhood_spend_fraction_achieved": max_area_share,
        "max_intervention_spend_fraction_achieved": max_intervention_share,
        "max_neighborhood_spend_fraction_of_deployed": max_area_share_deployed,
        "max_intervention_spend_fraction_of_deployed": max_intervention_share_deployed,
        "status": "optimal_or_feasible",
    }
    return PortfolioResult(selected.reset_index(drop=True), summary, solver)


def apply_portfolio_to_tiles(
    tiles: pd.DataFrame,
    selected: pd.DataFrame,
    *,
    max_relief_fraction: float = 0.85,
    impact_basis: str = "expected",
) -> pd.DataFrame:
    """Apply selected projects with distance spillover and overlap correction.

    ``impact_basis='conservative'`` uses each project's lower-bound benefit for
    the displayed digital twin, keeping the UI consistent with robust planning.
    """
    out = tiles.copy()
    base = pd.to_numeric(out["baseline_person_hours"], errors="coerce").fillna(0).to_numpy(float)
    residual = np.ones(len(out), dtype=float)
    benefit_col = _benefit_column(impact_basis)

    for _, p in selected.iterrows():
        local_base = max(float(p.get("baseline_person_hours", 0.0)), 1e-9)
        local_relief = float(p.get(benefit_col, p.get("benefit_expected_person_hours", 0.0))) / local_base
        local_relief = float(np.clip(local_relief, 0, max_relief_fraction))
        sigma = float(p.get("spillover_sigma_m", 100.0))
        strength = float(p.get("spillover_strength", 0.1))
        for j, tile in out.iterrows():
            d = haversine_m(float(p["lat"]), float(p["lon"]), float(tile["lat"]), float(tile["lon"]))
            kernel = gaussian_kernel(d, sigma)
            contribution = local_relief * ((1.0 - strength) if d > 1 else 1.0) * kernel
            contribution = float(np.clip(contribution, 0, max_relief_fraction))
            residual[j] *= 1.0 - contribution
    relief = np.clip(1.0 - residual, 0, max_relief_fraction)
    out["modeled_relief_fraction"] = relief
    out["counterfactual_person_hours"] = base * (1.0 - relief)
    out["person_hours_avoided"] = base - out["counterfactual_person_hours"].to_numpy(float)
    out["impact_basis"] = _normalize_impact_basis(impact_basis)
    return out


def marginal_value_curve(
    candidates: pd.DataFrame,
    budgets: list[float],
    *,
    objective: str = "balanced",
    impact_basis: str = "expected",
    equity_min_fraction: float = 0.0,
    enabled_interventions: list[str] | None = None,
    max_neighborhood_spend_fraction: float = 1.0,
    max_intervention_spend_fraction: float = 1.0,
    min_neighborhoods_served: int = 1,
) -> pd.DataFrame:
    rows = []
    basis = _normalize_impact_basis(impact_basis)
    for budget in budgets:
        res = optimize_portfolio(
            candidates,
            budget_usd=budget,
            objective=objective,
            impact_basis=basis,
            equity_min_fraction=equity_min_fraction,
            enabled_interventions=enabled_interventions,
            max_neighborhood_spend_fraction=max_neighborhood_spend_fraction,
            max_intervention_spend_fraction=max_intervention_spend_fraction,
            min_neighborhoods_served=min_neighborhoods_served,
        )
        direct = res.summary.get("planning_person_hours_avoided_first_order", 0)
        rows.append({
            "budget_usd": budget,
            "spent_usd": res.summary.get("spent_usd", 0),
            "projects": res.summary.get("projects", 0),
            "neighborhoods_served": res.summary.get("neighborhoods_served", 0),
            "first_order_person_hours_avoided": direct,
            # Backward-compatible alias used by earlier scripts.
            "person_hours_avoided": direct,
        })
    return pd.DataFrame(rows)

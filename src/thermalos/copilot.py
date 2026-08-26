from __future__ import annotations

from dataclasses import dataclass, field
import re

import pandas as pd

from thermalos.analytics.robustness import run_robustness
from thermalos.optimization.portfolio import optimize_portfolio


INTERVENTION_ALIASES = {
    "tree": "tree_canopy",
    "trees": "tree_canopy",
    "canopy": "tree_canopy",
    "shade": "shade_structure",
    "pavement": "cool_pavement",
    "cool pavement": "cool_pavement",
    "roof": "cool_roof",
    "roofs": "cool_roof",
    "cool roof": "cool_roof",
    "hydration": "cooling_node",
    "cooling": "cooling_node",
}


@dataclass
class CopilotCommand:
    intent: str = "optimize"
    updates: dict = field(default_factory=dict)
    target_intervention: str | None = None
    comparison_budget_usd: float | None = None
    raw_text: str = ""


@dataclass
class CopilotResponse:
    command: CopilotCommand
    narrative: str
    summary: dict
    selected: pd.DataFrame
    supplemental: pd.DataFrame | None = None


def _parse_money_values(text: str) -> list[float]:
    patterns = [
        r"\$\s*([0-9]+(?:\.[0-9]+)?)\s*(m|million)\b",
        r"\b([0-9]+(?:\.[0-9]+)?)\s*(m|million)\s*(?:budget|dollars?)?\b",
        r"\$\s*([0-9]+(?:\.[0-9]+)?)\s*(k|thousand)\b",
        r"\b([0-9]+(?:\.[0-9]+)?)\s*(k|thousand)\s*(?:budget|dollars?)?\b",
    ]
    hits: list[tuple[int, int, float]] = []
    for pat in patterns:
        for m in re.finditer(pat, text, re.I):
            value = float(m.group(1))
            unit = m.group(2).lower()
            amount = value * (1_000_000 if unit in {"m", "million"} else 1_000)
            hits.append((m.start(), m.end(), amount))
    hits.sort(key=lambda x: (x[0], -(x[1] - x[0])))
    values: list[float] = []
    occupied: list[tuple[int, int]] = []
    for start, end, amount in hits:
        if any(not (end <= a or start >= b) for a, b in occupied):
            continue
        occupied.append((start, end))
        values.append(amount)
    return values


def _parse_money(text: str) -> float | None:
    values = _parse_money_values(text)
    return values[0] if values else None

def _parse_percent(text: str, anchors: tuple[str, ...]) -> float | None:
    for anchor in anchors:
        patterns = [
            rf"(?:at least|minimum|min|floor)?\s*(\d{{1,3}})\s*%\s*(?:of\s+)?{anchor}",
            rf"{anchor}.{{0,18}}?(?:at least|minimum|min|floor|to|at)?\s*(\d{{1,3}})\s*%",
        ]
        for pat in patterns:
            m = re.search(pat, text, re.I)
            if m:
                return max(0.0, min(100.0, float(m.group(1)))) / 100.0
    return None


def parse_copilot_command(text: str, current_state: dict | None = None) -> CopilotCommand:
    state = dict(current_state or {})
    raw = text.strip()
    lower = raw.lower()
    cmd = CopilotCommand(raw_text=raw)

    if any(k in lower for k in ["robust", "stability", "uncertainty", "assumptions wrong"]):
        cmd.intent = "robustness"
    elif any(k in lower for k in ["why wasn't", "why was not", "why no ", "why didn\'t", "why didn't"]):
        cmd.intent = "explain_exclusion"
    elif any(k in lower for k in ["compare", "trade-off", "tradeoff", "versus", " vs "]):
        cmd.intent = "compare"
    elif any(k in lower for k in ["brief", "dossier", "council", "report"]):
        cmd.intent = "brief"

    money_values = _parse_money_values(lower)
    if cmd.intent == "compare":
        if len(money_values) >= 2:
            cmd.updates["budget_usd"] = money_values[0]
            cmd.comparison_budget_usd = money_values[1]
        elif len(money_values) == 1:
            # "Compare this with a $4M plan" keeps the current plan as baseline.
            cmd.comparison_budget_usd = money_values[0]
    elif money_values:
        cmd.updates["budget_usd"] = money_values[0]

    equity = _parse_percent(lower, ("equity", "vulnerable"))
    if equity is not None:
        cmd.updates["equity_min_fraction"] = equity

    area_cap = None
    m = re.search(r"(?:no\s+)?(?:neighborhood|area)[^.\n]{0,24}?(?:over|above|max(?:imum)?|cap(?:ped)?(?: at)?)\s*(\d{1,3})\s*%", lower)
    if m:
        area_cap = max(0.0, min(100.0, float(m.group(1)))) / 100.0
    if area_cap is not None:
        cmd.updates["max_neighborhood_spend_fraction"] = area_cap

    intervention_cap = None
    m = re.search(r"(?:intervention|type)[^.\n]{0,24}?(?:over|above|max(?:imum)?|cap(?:ped)?(?: at)?)\s*(\d{1,3})\s*%", lower)
    if m:
        intervention_cap = max(0.0, min(100.0, float(m.group(1)))) / 100.0
    if intervention_cap is not None:
        cmd.updates["max_intervention_spend_fraction"] = intervention_cap

    m = re.search(r"(?:at least|minimum|min)\s*(\d+)\s*(?:neighborhoods?|areas?)", lower)
    if m:
        cmd.updates["min_neighborhoods_served"] = max(1, int(m.group(1)))

    if "conservative" in lower or "lower bound" in lower or "downside" in lower:
        cmd.updates["impact_basis"] = "conservative"
    elif "expected" in lower or "mean effect" in lower:
        cmd.updates["impact_basis"] = "expected"

    if "maximum equity" in lower or "equity-first" in lower or "equity first" in lower:
        cmd.updates["objective"] = "maximum_equity"
    elif (
        "maximum direct impact" in lower
        or "maximum impact" in lower
        or "direct-impact-first" in lower
        or "direct impact first" in lower
        or "impact-first" in lower
        or "impact first" in lower
    ):
        cmd.updates["objective"] = "maximum_impact"
    elif "cost efficiency" in lower or "cost-efficient" in lower or "cheapest" in lower:
        cmd.updates["objective"] = "cost_efficiency"
    elif "balanced" in lower:
        cmd.updates["objective"] = "balanced"

    for alias, canonical in INTERVENTION_ALIASES.items():
        if alias in lower:
            cmd.target_intervention = canonical
            if cmd.intent == "explain_exclusion":
                break

    # Preserve only explicit updates; the executor merges these into current state.
    _ = state
    return cmd


def _fmt_money(value: float) -> str:
    return f"${value / 1_000_000:.2f}M" if value >= 1_000_000 else f"${value / 1_000:.0f}k"


def explain_intervention_exclusion(candidates: pd.DataFrame, selected: pd.DataFrame, intervention: str | None) -> str:
    if not intervention:
        return "Specify an intervention family (for example shade, trees, cool pavement, cool roofs, or cooling) to audit why it did or did not enter the portfolio."
    pool = candidates[candidates["intervention"].astype(str) == intervention].copy()
    if pool.empty:
        return f"No feasible {intervention.replace('_', ' ')} candidates were generated under the current site-suitability rules."
    chosen = selected[selected["intervention"].astype(str) == intervention]
    pool["value_per_100k"] = pd.to_numeric(pool["benefit_expected_person_hours"], errors="coerce").fillna(0) / pd.to_numeric(pool["cost_usd"], errors="coerce").replace(0, pd.NA) * 100_000
    top_value = float(pool["value_per_100k"].max())
    if len(chosen):
        return (
            f"{len(chosen)} {intervention.replace('_', ' ')} project(s) are funded. The strongest feasible candidate delivers about "
            f"{top_value:,.0f} expected person-hours per $100k. Selection reflects that value together with budget, equity, per-tile exclusivity, and concentration constraints."
        )
    selected_value = 0.0
    if len(selected):
        selected_value = float(
            (pd.to_numeric(selected["benefit_expected_person_hours"], errors="coerce") / pd.to_numeric(selected["cost_usd"], errors="coerce").replace(0, pd.NA) * 100_000).median()
        )
    return (
        f"No {intervention.replace('_', ' ')} project is funded in this scenario even though feasible options exist. Its best expected value is about "
        f"{top_value:,.0f} person-hours per $100k versus a median of {selected_value:,.0f} among funded projects. The MILP also weighs vulnerability and must satisfy the active policy constraints, so benefit-per-dollar alone does not determine selection."
    )


def execute_copilot(
    text: str,
    candidates: pd.DataFrame,
    current_state: dict,
    *,
    robustness_scenarios: int = 20,
) -> CopilotResponse:
    cmd = parse_copilot_command(text, current_state)
    state = dict(current_state)
    state.update(cmd.updates)
    defaults = {
        "budget_usd": 2_000_000.0,
        "objective": "balanced",
        "impact_basis": "expected",
        "equity_min_fraction": 0.35,
        "enabled_interventions": None,
        "max_neighborhood_spend_fraction": 1.0,
        "max_intervention_spend_fraction": 1.0,
        "min_neighborhoods_served": 1,
    }
    for k, v in defaults.items():
        state.setdefault(k, v)

    res = optimize_portfolio(
        candidates,
        budget_usd=float(state["budget_usd"]),
        objective=str(state["objective"]),
        impact_basis=str(state["impact_basis"]),
        equity_min_fraction=float(state["equity_min_fraction"]),
        enabled_interventions=state.get("enabled_interventions"),
        max_neighborhood_spend_fraction=float(state["max_neighborhood_spend_fraction"]),
        max_intervention_spend_fraction=float(state["max_intervention_spend_fraction"]),
        min_neighborhoods_served=int(state["min_neighborhoods_served"]),
    )

    if cmd.intent == "compare":
        compare_budget = cmd.comparison_budget_usd
        if compare_budget is None:
            narrative = (
                "A comparison request was recognized, but a second capital budget is required. "
                "For example: Compare a $2M plan with a $4M plan."
            )
            return CopilotResponse(cmd, narrative, {**state, **res.summary}, res.selected)

        alt_state = dict(state)
        alt_state["budget_usd"] = float(compare_budget)
        alt = optimize_portfolio(
            candidates,
            budget_usd=float(alt_state["budget_usd"]),
            objective=str(alt_state["objective"]),
            impact_basis=str(alt_state["impact_basis"]),
            equity_min_fraction=float(alt_state["equity_min_fraction"]),
            enabled_interventions=alt_state.get("enabled_interventions"),
            max_neighborhood_spend_fraction=float(alt_state["max_neighborhood_spend_fraction"]),
            max_intervention_spend_fraction=float(alt_state["max_intervention_spend_fraction"]),
            min_neighborhoods_served=int(alt_state["min_neighborhoods_served"]),
        )

        base_direct = float(res.summary.get("planning_person_hours_avoided_first_order", 0.0))
        alt_direct = float(alt.summary.get("planning_person_hours_avoided_first_order", 0.0))
        delta_direct = alt_direct - base_direct
        delta_pct = (100.0 * delta_direct / base_direct) if base_direct else 0.0
        base_equity = float(res.summary.get("vulnerable_spend_fraction", 0.0))
        alt_equity = float(alt.summary.get("vulnerable_spend_fraction", 0.0))

        comparison = pd.DataFrame([
            {
                "scenario": "Baseline",
                "budget_usd": float(state["budget_usd"]),
                "spent_usd": float(res.summary.get("spent_usd", 0.0)),
                "projects": len(res.selected),
                "direct_person_hours_avoided": base_direct,
                "equity_spend_fraction": base_equity,
                "neighborhoods_served": int(res.summary.get("neighborhoods_served", 0)),
                "status": res.summary.get("status", "unknown"),
            },
            {
                "scenario": "Comparison",
                "budget_usd": float(alt_state["budget_usd"]),
                "spent_usd": float(alt.summary.get("spent_usd", 0.0)),
                "projects": len(alt.selected),
                "direct_person_hours_avoided": alt_direct,
                "equity_spend_fraction": alt_equity,
                "neighborhoods_served": int(alt.summary.get("neighborhoods_served", 0)),
                "status": alt.summary.get("status", "unknown"),
            },
        ])

        narrative = (
            f"Compared two actual ThermalOS MILP solutions under identical policy constraints. "
            f"Changing the capital envelope from {_fmt_money(float(state['budget_usd']))} to {_fmt_money(float(alt_state['budget_usd']))} "
            f"changes direct modeled relief from {base_direct:,.0f} to {alt_direct:,.0f} person-hours "
            f"({delta_pct:+.1f}%), and funded projects from {len(res.selected)} to {len(alt.selected)}. "
            f"Equity-aligned spend changes from {100 * base_equity:.0f}% to {100 * alt_equity:.0f}%."
        )
        summary = {
            **state,
            **res.summary,
            "comparison_budget_usd": float(alt_state["budget_usd"]),
            "comparison_projects": len(alt.selected),
            "comparison_direct_person_hours_avoided": alt_direct,
            "comparison_equity_spend_fraction": alt_equity,
            "direct_benefit_delta": delta_direct,
            "direct_benefit_delta_pct": delta_pct,
        }
        return CopilotResponse(cmd, narrative, summary, res.selected, comparison)

    if cmd.intent == "explain_exclusion":
        narrative = explain_intervention_exclusion(candidates, res.selected, cmd.target_intervention)
        return CopilotResponse(cmd, narrative, {**state, **res.summary}, res.selected)

    if cmd.intent == "robustness":
        rob = run_robustness(
            candidates,
            budget_usd=float(state["budget_usd"]),
            objective=str(state["objective"]),
            impact_basis=str(state["impact_basis"]),
            equity_min_fraction=float(state["equity_min_fraction"]),
            enabled_interventions=state.get("enabled_interventions"),
            max_neighborhood_spend_fraction=float(state["max_neighborhood_spend_fraction"]),
            max_intervention_spend_fraction=float(state["max_intervention_spend_fraction"]),
            min_neighborhoods_served=int(state["min_neighborhoods_served"]),
            scenarios=robustness_scenarios,
        )
        narrative = (
            f"ThermalOS re-optimized {rob.scenarios} plausible effect/cost worlds under the same policy constraints. "
            f"The cost-weighted stability of the baseline-funded portfolio is {100 * rob.portfolio_stability:.0f}%, with median selection-set overlap of {100 * rob.median_jaccard:.0f}%. "
            f"Direct modeled benefit spans roughly {rob.direct_benefit_p10:,.0f}-{rob.direct_benefit_p90:,.0f} person-hours across the central 80% of stress-test scenarios."
        )
        summary = {**state, **res.summary, "portfolio_stability": rob.portfolio_stability, "median_jaccard": rob.median_jaccard}
        return CopilotResponse(cmd, narrative, summary, res.selected, rob.project_stability)

    if cmd.intent == "brief":
        narrative = (
            f"The current decision dossier would cover a {_fmt_money(float(state['budget_usd']))} portfolio with {len(res.selected)} funded projects, "
            f"{100 * float(res.summary.get('vulnerable_spend_fraction', 0)):.0f}% equity-aligned spend, and {float(res.summary.get('planning_person_hours_avoided_first_order', 0)):,.0f} direct modeled person-hours avoided. "
            "Use the Decision Dossier download to export the auditable PDF with assumptions, provenance, and verification protocol."
        )
    else:
        narrative = (
            f"Re-optimized the actual ThermalOS MILP at {_fmt_money(float(state['budget_usd']))}: {len(res.selected)} projects, "
            f"{100 * float(res.summary.get('vulnerable_spend_fraction', 0)):.0f}% equity-aligned spend, "
            f"{float(res.summary.get('planning_person_hours_avoided_first_order', 0)):,.0f} direct modeled person-hours avoided, "
            f"using the {state['impact_basis']} impact basis."
        )
    return CopilotResponse(cmd, narrative, {**state, **res.summary}, res.selected)

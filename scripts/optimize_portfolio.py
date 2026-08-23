from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from thermalos.config import available_cities, interventions_config
from thermalos.explain import explain_selected
from thermalos.models.interventions import build_candidates
from thermalos.models.thermal_twin import ObservationalThermalTwin
from thermalos.optimization.portfolio import apply_portfolio_to_tiles, marginal_value_curve, optimize_portfolio


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--city", choices=available_cities(), default="miami")
    p.add_argument("--budget", type=float, default=2_000_000)
    p.add_argument(
        "--objective",
        default="balanced",
        choices=["balanced", "maximum_impact", "maximum_equity", "conservative", "cost_efficiency"],
    )
    p.add_argument(
        "--impact-basis",
        choices=["expected", "conservative"],
        default="expected",
        help="Expected = optimize mean effects; conservative = optimize lower-bound effects.",
    )
    p.add_argument("--equity-min", type=float, default=0.35)
    p.add_argument(
        "--max-neighborhood-spend",
        type=float,
        default=1.0,
        help="Maximum fraction of the capital budget assignable to any one neighborhood (e.g. 0.50).",
    )
    p.add_argument(
        "--max-intervention-spend",
        type=float,
        default=1.0,
        help="Maximum fraction of the capital budget assignable to any one intervention type (e.g. 0.60).",
    )
    p.add_argument("--min-neighborhoods", type=int, default=1)
    args = p.parse_args()

    # Preserve compatibility with the original --objective conservative flag.
    impact_basis = "conservative" if args.objective == "conservative" else args.impact_basis
    objective = "balanced" if args.objective == "conservative" else args.objective

    tile_path = Path("data/processed") / f"{args.city}_tiles.csv"
    if not tile_path.exists():
        raise FileNotFoundError(f"Run build_city_features.py first: {tile_path}")
    tiles = pd.read_csv(tile_path)
    cfg = interventions_config()

    tree_cal = None
    model_path = Path("models") / f"thermal_twin_{args.city}.joblib"
    if model_path.exists():
        model = ObservationalThermalTwin.load(model_path)
        tree_cal = model.canopy_sensitivity(tiles)

    candidates = build_candidates(tiles, cfg, observational_tree_cooling=tree_cal).candidates
    result = optimize_portfolio(
        candidates,
        budget_usd=args.budget,
        objective=objective,
        impact_basis=impact_basis,
        equity_min_fraction=args.equity_min,
        max_neighborhood_spend_fraction=args.max_neighborhood_spend,
        max_intervention_spend_fraction=args.max_intervention_spend,
        min_neighborhoods_served=args.min_neighborhoods,
    )
    counter = apply_portfolio_to_tiles(
        tiles,
        result.selected,
        max_relief_fraction=float(cfg["model"].get("max_tile_relief_fraction", 0.85)),
        impact_basis=impact_basis,
    )

    baseline = float(counter["baseline_person_hours"].sum())
    after = float(counter["counterfactual_person_hours"].sum())
    summary = dict(result.summary)
    summary.update({
        "solver": result.solver,
        "baseline_person_hours": baseline,
        "counterfactual_person_hours": after,
        "modeled_person_hours_avoided_with_spillover": baseline - after,
        "modeled_reduction_fraction": (baseline - after) / baseline if baseline else 0.0,
        "scientific_status": "scenario estimate; not a causal impact evaluation",
    })

    out = Path("outputs") / f"{args.city}_portfolio"
    out.mkdir(parents=True, exist_ok=True)
    selected = explain_selected(result.selected, tiles)
    selected.to_csv(out / "selected_projects.csv", index=False)
    candidates.to_csv(out / "candidate_projects.csv", index=False)
    counter.to_csv(out / "counterfactual_tiles.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    budgets = sorted(set([0.5e6, 1e6, 2e6, 3e6, 5e6, args.budget]))
    marginal_value_curve(
        candidates,
        budgets,
        objective=objective,
        impact_basis=impact_basis,
        equity_min_fraction=args.equity_min,
        max_neighborhood_spend_fraction=args.max_neighborhood_spend,
        max_intervention_spend_fraction=args.max_intervention_spend,
        min_neighborhoods_served=args.min_neighborhoods,
    ).to_csv(out / "marginal_value_curve.csv", index=False)

    print(json.dumps(summary, indent=2))
    if len(selected):
        print("\nTop projects:")
        cols = ["label", "area", "cost_usd", "benefit_expected_person_hours", "benefit_low_person_hours", "reason"]
        print(selected[cols].sort_values(
            "benefit_low_person_hours" if impact_basis == "conservative" else "benefit_expected_person_hours",
            ascending=False,
        ).head(12).to_string(index=False))
    else:
        print("\nNo feasible projects under the requested policy constraints.")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()

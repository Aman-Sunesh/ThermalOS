from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from thermalos.config import available_cities, interventions_config
from thermalos.copilot import execute_copilot
from thermalos.models.interventions import build_candidates


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--city", choices=available_cities(), default="miami")
    p.add_argument("--prompt", required=True)
    p.add_argument("--budget", type=float, default=2_000_000)
    args = p.parse_args()
    tiles = pd.read_csv(Path("data/processed") / f"{args.city}_tiles.csv")
    candidates = build_candidates(tiles, interventions_config(), seed=42).candidates
    state = {
        "budget_usd": args.budget,
        "objective": "balanced",
        "impact_basis": "expected",
        "equity_min_fraction": 0.35,
        "enabled_interventions": None,
        "max_neighborhood_spend_fraction": 1.0,
        "max_intervention_spend_fraction": 1.0,
        "min_neighborhoods_served": 1,
    }
    result = execute_copilot(args.prompt, candidates, state)
    print(result.narrative)
    if result.command.updates:
        print("Parsed updates:", result.command.updates)
    print(result.selected[["label", "area", "cost_usd", "benefit_expected_person_hours"]].head(15).to_string(index=False))


if __name__ == "__main__":
    main()

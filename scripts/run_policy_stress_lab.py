from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from thermalos.analytics.policy_lab import run_policy_stress_lab
from thermalos.config import available_cities, interventions_config
from thermalos.models.interventions import build_candidates


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--city", choices=available_cities(), default="miami")
    p.add_argument("--budget", type=float, default=2_000_000)
    args = p.parse_args()
    tiles = pd.read_csv(Path("data/processed") / f"{args.city}_tiles.csv")
    candidates = build_candidates(tiles, interventions_config(), seed=42).candidates
    result = run_policy_stress_lab(candidates, budget_usd=args.budget, area_count=int(tiles["area"].nunique()))
    out = Path("outputs") / args.city / "policy_stress_lab"
    out.mkdir(parents=True, exist_ok=True)
    result.scenarios.to_csv(out / "policy_scenarios.csv", index=False)
    result.equity_frontier.to_csv(out / "equity_frontier.csv", index=False)
    print(result.scenarios.to_string(index=False))
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from thermalos.analytics.robustness import run_robustness
from thermalos.config import available_cities, interventions_config
from thermalos.models.interventions import build_candidates


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--city", choices=available_cities(), default="miami")
    p.add_argument("--budget", type=float, default=2_000_000)
    p.add_argument("--equity-min", type=float, default=0.35)
    p.add_argument("--scenarios", type=int, default=32)
    p.add_argument("--pool-size", type=int, default=1800)
    args = p.parse_args()

    tiles = pd.read_csv(Path("data/processed") / f"{args.city}_tiles.csv")
    candidates = build_candidates(tiles, interventions_config(), seed=42).candidates
    result = run_robustness(
        candidates,
        budget_usd=args.budget,
        equity_min_fraction=args.equity_min,
        scenarios=args.scenarios,
        pool_size=args.pool_size,
    )
    out = Path("outputs") / args.city / "robustness"
    out.mkdir(parents=True, exist_ok=True)
    result.project_stability.to_csv(out / "project_stability.csv", index=False)
    result.scenario_summary.to_csv(out / "scenario_summary.csv", index=False)
    summary = {
        "portfolio_stability": result.portfolio_stability,
        "median_jaccard": result.median_jaccard,
        "direct_benefit_p10": result.direct_benefit_p10,
        "direct_benefit_p50": result.direct_benefit_p50,
        "direct_benefit_p90": result.direct_benefit_p90,
        "scenarios": result.scenarios,
        "candidate_pool_size": result.candidate_pool_size,
        "method_note": result.method_note,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from thermalos.config import available_cities, interventions_config
from thermalos.models.interventions import build_candidates
from thermalos.optimization.portfolio import optimize_portfolio
from thermalos.verification import build_verification_registry


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--city", choices=available_cities(), default="miami")
    p.add_argument("--budget", type=float, default=2_000_000)
    p.add_argument("--equity-min", type=float, default=0.35)
    args = p.parse_args()
    tiles = pd.read_csv(Path("data/processed") / f"{args.city}_tiles.csv")
    candidates = build_candidates(tiles, interventions_config(), seed=42).candidates
    selected = optimize_portfolio(candidates, budget_usd=args.budget, equity_min_fraction=args.equity_min).selected
    registry = build_verification_registry(tiles, selected)
    out = Path("outputs") / args.city / "thermalverify"
    out.mkdir(parents=True, exist_ok=True)
    registry.projects.to_csv(out / "verification_registry.csv", index=False)
    registry.controls.to_csv(out / "matched_controls.csv", index=False)
    (out / "protocol.json").write_text(json.dumps(registry.protocol, indent=2), encoding="utf-8")
    print(f"Baselines captured: {len(registry.projects)}")
    print(f"Matched controls: {len(registry.controls)}")
    print(registry.protocol["claim_boundary"])
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()

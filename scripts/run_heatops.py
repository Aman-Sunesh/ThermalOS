from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from thermalos.config import available_cities
from thermalos.operations.heatops import plan_heatops


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--city", choices=available_cities(), default="miami")
    p.add_argument("--budget", type=float, default=60_000)
    p.add_argument("--equity-min", type=float, default=0.35)
    args = p.parse_args()
    tiles = pd.read_csv(Path("data/processed") / f"{args.city}_tiles.csv")
    result = plan_heatops(tiles, operating_budget_usd=args.budget, equity_min_fraction=args.equity_min)
    out = Path("outputs") / args.city / "heatops"
    out.mkdir(parents=True, exist_ok=True)
    result.selected.to_csv(out / "selected_actions.csv", index=False)
    result.access.to_csv(out / "cooling_access_gap.csv", index=False)
    (out / "summary.json").write_text(json.dumps(result.summary, indent=2), encoding="utf-8")
    print(json.dumps(result.summary, indent=2))
    if len(result.selected):
        print(result.selected[["label", "area", "cost_usd", "benefit_expected_person_hours", "cooling_access_gap"]].to_string(index=False))
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()

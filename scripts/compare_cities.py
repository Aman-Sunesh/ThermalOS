from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from thermalos.config import interventions_config
from thermalos.generalization import run_cross_city_comparison


def load(city: str):
    processed = Path("data/processed") / f"{city}_tiles.csv"
    provenance = Path("data/processed") / f"{city}_provenance.json"
    sample = Path("data/sample") / f"{city}_demo_tiles.csv"
    path = processed if processed.exists() else sample
    meta = json.loads(provenance.read_text(encoding="utf-8")) if provenance.exists() else {"synthetic_demo": path == sample, "source_file": str(path), "enrichment": {}}
    return pd.read_csv(path), meta


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--budget", type=float, default=2_000_000)
    args = p.parse_args()
    miami, mprov = load("miami")
    houston, hprov = load("houston")
    result = run_cross_city_comparison(
        miami,
        houston,
        miami_provenance=mprov,
        houston_provenance=hprov,
        intervention_config=interventions_config(),
        budget_usd=args.budget,
    )
    out = Path("outputs/generalization")
    out.mkdir(parents=True, exist_ok=True)
    result.comparison.to_csv(out / "cross_city_comparison.csv", index=False)
    (out / "transfer_metrics.json").write_text(json.dumps(result.transfer_metrics, indent=2), encoding="utf-8")
    for city, portfolio in result.portfolios.items():
        portfolio.to_csv(out / f"{city.lower().replace('-', '_').replace(' ', '_')}_portfolio.csv", index=False)
    print(result.comparison.to_string(index=False))
    print(json.dumps(result.transfer_metrics, indent=2))
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()

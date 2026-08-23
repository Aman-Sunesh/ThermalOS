from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from thermalos.config import available_cities, city_config, interventions_config
from thermalos.evidence import build_evidence_ledger
from thermalos.models.interventions import build_candidates
from thermalos.reporting.dossier import build_dossier_pdf
from thermalos.scenario import evaluate_plan
from thermalos.verification import build_verification_registry


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--city", choices=available_cities(), default="miami")
    p.add_argument("--budget", type=float, default=2_000_000)
    p.add_argument("--equity-min", type=float, default=0.35)
    args = p.parse_args()
    tiles_path = Path("data/processed") / f"{args.city}_tiles.csv"
    if not tiles_path.exists():
        tiles_path = Path("data/sample") / f"{args.city}_demo_tiles.csv"
    tiles = pd.read_csv(tiles_path)
    prov_path = Path("data/processed") / f"{args.city}_provenance.json"
    provenance = json.loads(prov_path.read_text(encoding="utf-8")) if prov_path.exists() else {
        "synthetic_demo": "sample" in tiles_path.parts,
        "source_file": str(tiles_path),
        "enrichment": {},
    }
    cfg = interventions_config()
    candidates = build_candidates(tiles, cfg, seed=42).candidates
    plan = evaluate_plan(
        tiles,
        candidates,
        budget_usd=args.budget,
        equity_min_fraction=args.equity_min,
        max_relief_fraction=float(cfg.get("model", {}).get("max_tile_relief_fraction", 0.85)),
    )
    registry = build_verification_registry(tiles, plan.portfolio.selected)
    ledger = build_evidence_ledger(tiles, provenance, cfg)
    robustness_path = Path("outputs") / args.city / "robustness" / "summary.json"
    robustness_summary = (
        json.loads(robustness_path.read_text(encoding="utf-8"))
        if robustness_path.exists()
        else None
    )
    city_name = city_config(args.city).get("name", args.city)
    pdf = build_dossier_pdf(
        city_name=city_name,
        summary=plan.metrics,
        selected=plan.portfolio.selected,
        provenance=provenance,
        evidence_ledger=ledger,
        robustness_summary=robustness_summary,
        verification_projects=registry.projects,
    )
    out = Path("outputs") / args.city / "dossier"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "thermalos_decision_dossier.pdf"
    path.write_bytes(pdf)
    print(f"Saved {len(pdf):,} bytes -> {path}")


if __name__ == "__main__":
    main()

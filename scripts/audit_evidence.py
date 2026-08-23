from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from thermalos.config import available_cities, interventions_config
from thermalos.evidence import build_evidence_ledger, evidence_summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--city", choices=available_cities(), default="miami")
    args = p.parse_args()
    tiles = pd.read_csv(Path("data/processed") / f"{args.city}_tiles.csv")
    prov_path = Path("data/processed") / f"{args.city}_provenance.json"
    provenance = json.loads(prov_path.read_text(encoding="utf-8")) if prov_path.exists() else {"synthetic_demo": True, "enrichment": {}}
    ledger = build_evidence_ledger(tiles, provenance, interventions_config())
    out = Path("outputs") / args.city / "trust"
    out.mkdir(parents=True, exist_ok=True)
    ledger.to_csv(out / "evidence_ledger.csv", index=False)
    print(json.dumps(evidence_summary(ledger), indent=2))
    print(ledger.to_string(index=False))
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()

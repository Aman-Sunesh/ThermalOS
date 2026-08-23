from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from thermalos.config import city_config, generalization_config, interventions_config
from thermalos.system_transfer import (
    apply_system_gates,
    audit_city_contract,
    pipeline_hash_manifest,
    run_system_city,
    sha256_file,
)


def _load_city(city: str) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    tiles_path = Path("data/processed") / f"{city}_tiles.csv"
    prov_path = Path("data/processed") / f"{city}_provenance.json"
    sat_path = Path("data/interim") / f"{city}_satellite_samples.csv"
    missing = [str(p) for p in (tiles_path, prov_path, sat_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Build development data first for {city}. Missing: {missing}")
    return (
        pd.read_csv(tiles_path),
        json.loads(prov_path.read_text(encoding="utf-8")),
        pd.read_csv(sat_path),
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Freeze the full ThermalOS decision-system transfer protocol on development cities only. "
            "This does not train a morphology-only temperature predictor."
        )
    )
    p.add_argument("--budget", type=float, default=2_000_000)
    p.add_argument("--equity-min", type=float, default=0.35)
    p.add_argument("--robustness-scenarios", type=int, default=32)
    p.add_argument("--robustness-pool-size", type=int, default=1200)
    args = p.parse_args()

    protocol = generalization_config()
    dev_cities = [str(x) for x in protocol.get("development_cities", [])]
    blind_cities = [str(x) for x in protocol.get("blind_cities", [])]
    target_n = int(protocol.get("satellite_sampling", {}).get("samples_per_area", 15))
    exact_set = bool(protocol.get("satellite_sampling", {}).get("active_set_exact_per_area", True))
    gates = protocol.get("gates", {})
    cfg_interventions = interventions_config()

    if len(dev_cities) < 2:
        raise RuntimeError("System-transfer freeze requires at least two development cities")
    if not blind_cities:
        raise RuntimeError("No prospective blind cities are registered")

    out = Path("outputs/generalization/freeze")
    out.mkdir(parents=True, exist_ok=True)
    replay_dir = out / "development_system_replay"
    replay_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    data_hashes: dict[str, dict[str, str]] = {}
    contracts: dict[str, dict] = {}

    for city in dev_cities:
        tiles, prov, sat = _load_city(city)
        if bool(prov.get("synthetic_demo", False)):
            raise RuntimeError(f"Refusing production freeze on synthetic development data: {city}")
        if str(prov.get("transfer_role")) != "development":
            raise RuntimeError(f"{city} transfer_role={prov.get('transfer_role')!r}; expected 'development'")

        contract = audit_city_contract(
            city=city,
            tiles=tiles,
            provenance=prov,
            city_cfg=city_config(city),
            satellite_samples=sat,
            target_samples_per_area=target_n,
            expected_role="development",
            exact_satellite_set=exact_set,
        )
        contracts[city] = contract.to_dict()
        if not contract.passed:
            failed = [k for k, v in contract.checks.items() if not v]
            raise RuntimeError(f"Development data-contract failure for {city}: {failed}; details={contract.details}")

        print(f"\n=== DEVELOPMENT SYSTEM REPLAY: {city.upper()} ===")
        result = run_system_city(
            city=city,
            tiles=tiles,
            provenance=prov,
            intervention_config=cfg_interventions,
            contract=contract,
            budget_usd=args.budget,
            equity_min_fraction=args.equity_min,
            robustness_scenarios=args.robustness_scenarios,
            robustness_pool_size=args.robustness_pool_size,
        )
        gate_result = apply_system_gates(result.summary, gates, equity_min_fraction=args.equity_min)
        row = {**result.summary, **gate_result}
        rows.append(row)

        result.portfolio.to_csv(replay_dir / f"{city}_portfolio.csv", index=False)
        result.robustness_scenarios.to_csv(replay_dir / f"{city}_robustness_scenarios.csv", index=False)
        result.project_stability.to_csv(replay_dir / f"{city}_project_stability.csv", index=False)
        result.policy_scenarios.to_csv(replay_dir / f"{city}_policy_scenarios.csv", index=False)
        result.equity_frontier.to_csv(replay_dir / f"{city}_equity_frontier.csv", index=False)
        result.evidence_ledger.to_csv(replay_dir / f"{city}_evidence_ledger.csv", index=False)

        data_hashes[city] = {
            "processed_tiles_sha256": sha256_file(Path("data/processed") / f"{city}_tiles.csv"),
            "provenance_sha256": sha256_file(Path("data/processed") / f"{city}_provenance.json"),
            "satellite_samples_sha256": sha256_file(Path("data/interim") / f"{city}_satellite_samples.csv"),
        }

    summary = pd.DataFrame(rows)
    summary.to_csv(replay_dir / "development_system_summary.csv", index=False)
    print("\n=== DEVELOPMENT SYSTEM SUMMARY ===")
    cols = [
        "city_key", "tiles", "candidate_count", "projects", "spent_usd",
        "equity_spend_fraction", "robustness_feasible_fraction", "portfolio_stability",
        "policy_scenarios_feasible_fraction", "equity_frontier_feasible_fraction",
        "evidence_min_coverage_pct", "overall_system_pass",
    ]
    print(summary[[c for c in cols if c in summary.columns]].to_string(index=False))

    if not bool(summary["overall_system_pass"].all()):
        failed = summary.loc[~summary["overall_system_pass"].astype(bool), ["city_key"] + [c for c in summary if c.startswith("passes_")]]
        raise RuntimeError(
            "Development system replay failed predeclared gates; blind-city opening remains blocked.\n"
            + failed.to_string(index=False)
        )

    pipeline_files, pipeline_sha = pipeline_hash_manifest(".")
    city_cfg_hashes = {
        city: sha256_file(Path("configs") / f"{city}.yaml")
        for city in [*dev_cities, *blind_cities]
    }

    diagnostics_path = Path("outputs/generalization/comprehensive_diagnostics/recommendation.json")
    morphology_diagnostic = None
    if diagnostics_path.exists():
        try:
            morphology_diagnostic = json.loads(diagnostics_path.read_text(encoding="utf-8"))
        except Exception:
            morphology_diagnostic = {"warning": "diagnostic recommendation file could not be parsed"}

    manifest = {
        "protocol_version": protocol.get("protocol_version"),
        "freeze_status": "production_system_frozen",
        "production_blind_evaluation_allowed": True,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "development_cities": dev_cities,
        "prospective_blind_cities": blind_cities,
        "budget_usd": float(args.budget),
        "equity_min_fraction": float(args.equity_min),
        "robustness_scenarios": int(args.robustness_scenarios),
        "robustness_pool_size": int(args.robustness_pool_size),
        "satellite_sampling": protocol.get("satellite_sampling", {}),
        "thermal_context": protocol.get("thermal_context", {}),
        "gates": gates,
        "development_contracts": contracts,
        "development_data_sha256": data_hashes,
        "development_system_summary": summary.to_dict(orient="records"),
        "pipeline_files_sha256": pipeline_files,
        "pipeline_snapshot_sha256": pipeline_sha,
        "city_config_sha256": city_cfg_hashes,
        "generalization_config_sha256": sha256_file("configs/generalization.yaml"),
        "interventions_config_sha256": sha256_file("configs/interventions.yaml"),
        "morphology_only_transfer_diagnostic": morphology_diagnostic,
        "temperature_prediction_claim": False,
        "no_blind_city_refit_or_retuning": True,
        "claim_boundary": protocol.get("claim_boundary"),
    }
    manifest_path = out / "thermalos_system_transfer_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\nFREEZE STATUS: production_system_frozen")
    print("PIPELINE SHA256:", pipeline_sha)
    print("MANIFEST:", manifest_path)
    print("TEMPERATURE PREDICTION CLAIM: false")
    print("BLIND CITIES REMAIN UNOPENED:", ", ".join(blind_cities))


if __name__ == "__main__":
    main()

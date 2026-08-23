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
    run_system_city,
    sha256_file,
    verify_pipeline_hashes,
)


def _load_blind_city(city: str) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    tiles_path = Path("data/processed") / f"{city}_tiles.csv"
    prov_path = Path("data/processed") / f"{city}_provenance.json"
    sat_path = Path("data/interim") / f"{city}_satellite_samples.csv"
    missing = [str(p) for p in (tiles_path, prov_path, sat_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Blind city not fully built: {city}. Missing: {missing}")
    prov = json.loads(prov_path.read_text(encoding="utf-8"))
    if bool(prov.get("synthetic_demo", False)):
        raise RuntimeError(f"Refusing blind evaluation on synthetic data: {city}")
    if str(prov.get("transfer_role")) != "blind_test":
        raise RuntimeError(f"{city} transfer_role={prov.get('transfer_role')!r}; expected 'blind_test'")
    return pd.read_csv(tiles_path), prov, pd.read_csv(sat_path)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Open prospective blind cities once with the frozen ThermalOS v3 system-transfer protocol."
    )
    p.add_argument("--cities", nargs="+", default=None)
    p.add_argument("--manifest", default="outputs/generalization/freeze/thermalos_system_transfer_manifest.json")
    p.add_argument("--allow-repeat", action="store_true", help="For debugging only; repeated cities are no longer prospective blind results.")
    args = p.parse_args()

    protocol = generalization_config()
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError("Freeze the system first with scripts/freeze_system_transfer.py")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("freeze_status") != "production_system_frozen" or not manifest.get("production_blind_evaluation_allowed", False):
        raise RuntimeError("System manifest is not a valid production freeze; refusing blind evaluation")
    if manifest.get("temperature_prediction_claim") is not False:
        raise RuntimeError("Unexpected manifest semantics: v3 must not claim morphology-only temperature prediction")

    if sha256_file("configs/generalization.yaml") != manifest.get("generalization_config_sha256"):
        raise RuntimeError("Generalization protocol changed after freeze")
    if sha256_file("configs/interventions.yaml") != manifest.get("interventions_config_sha256"):
        raise RuntimeError("Intervention assumptions changed after freeze")
    problems = verify_pipeline_hashes(manifest.get("pipeline_files_sha256", {}), ".")
    if problems:
        raise RuntimeError("Frozen pipeline changed after freeze: " + ", ".join(problems[:12]))

    dev = set(str(x) for x in manifest.get("development_cities", []))
    allowed = [str(x) for x in manifest.get("prospective_blind_cities", [])]
    cities = args.cities or allowed
    unknown = [c for c in cities if c not in allowed]
    if unknown:
        raise RuntimeError(f"Not in preregistered blind set: {unknown}")
    if set(cities) & dev:
        raise RuntimeError("Development cities cannot be opened as blind")

    for city in cities:
        cfg_hash = sha256_file(Path("configs") / f"{city}.yaml")
        if cfg_hash != manifest.get("city_config_sha256", {}).get(city):
            raise RuntimeError(f"Blind city config changed after freeze: {city}")

    out = Path("outputs/generalization/blind_system_transfer")
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "blind_open_log.json"
    prior_log: list[dict] = []
    if log_path.exists():
        loaded = json.loads(log_path.read_text(encoding="utf-8"))
        prior_log = loaded if isinstance(loaded, list) else [loaded]
    already_opened = {c for e in prior_log for c in e.get("cities_opened", [])}
    repeated = sorted(set(cities) & already_opened)
    if repeated and not args.allow_repeat:
        raise RuntimeError(
            f"These cities were already opened: {repeated}. Refusing to relabel a repeat as prospective blind. "
            "Use --allow-repeat only for debugging and report it as a repeat."
        )

    target_n = int(manifest.get("satellite_sampling", {}).get("samples_per_area", 15))
    exact_set = bool(manifest.get("satellite_sampling", {}).get("active_set_exact_per_area", True))
    budget = float(manifest.get("budget_usd", 2_000_000))
    equity_min = float(manifest.get("equity_min_fraction", 0.35))
    scenarios = int(manifest.get("robustness_scenarios", 32))
    pool_size = int(manifest.get("robustness_pool_size", 1200))
    gates = manifest.get("gates", protocol.get("gates", {}))
    cfg_interventions = interventions_config()

    rows: list[dict] = []
    contract_rows: list[dict] = []
    successful_cities: list[str] = []

    for city in cities:
        print(f"\n=== PROSPECTIVE SYSTEM TRANSFER: {city.upper()} ===")
        tiles, prov, sat = _load_blind_city(city)
        contract = audit_city_contract(
            city=city,
            tiles=tiles,
            provenance=prov,
            city_cfg=city_config(city),
            satellite_samples=sat,
            target_samples_per_area=target_n,
            expected_role="blind_test",
            exact_satellite_set=exact_set,
        )
        contract_rows.append(contract.to_dict())
        if not contract.passed:
            failed = [k for k, v in contract.checks.items() if not v]
            raise RuntimeError(f"Blind data-contract failure for {city}: {failed}; no result logged")

        result = run_system_city(
            city=city,
            tiles=tiles,
            provenance=prov,
            intervention_config=cfg_interventions,
            contract=contract,
            budget_usd=budget,
            equity_min_fraction=equity_min,
            robustness_scenarios=scenarios,
            robustness_pool_size=pool_size,
        )
        gate_result = apply_system_gates(result.summary, gates, equity_min_fraction=equity_min)
        row = {**result.summary, **gate_result}
        rows.append(row)
        successful_cities.append(city)

        result.portfolio.to_csv(out / f"{city}_portfolio.csv", index=False)
        result.robustness_scenarios.to_csv(out / f"{city}_robustness_scenarios.csv", index=False)
        result.project_stability.to_csv(out / f"{city}_project_stability.csv", index=False)
        result.policy_scenarios.to_csv(out / f"{city}_policy_scenarios.csv", index=False)
        result.equity_frontier.to_csv(out / f"{city}_equity_frontier.csv", index=False)
        result.evidence_ledger.to_csv(out / f"{city}_evidence_ledger.csv", index=False)

    summary = pd.DataFrame(rows)
    contracts = pd.DataFrame(contract_rows)
    summary.to_csv(out / "blind_system_transfer_summary.csv", index=False)
    contracts.to_csv(out / "blind_data_contract_audit.csv", index=False)

    print("\n=== BLIND SYSTEM TRANSFER SUMMARY ===")
    cols = [
        "city_key", "tiles", "candidate_count", "projects", "spent_usd",
        "equity_spend_fraction", "robustness_feasible_fraction", "portfolio_stability",
        "policy_scenarios_feasible_fraction", "equity_frontier_feasible_fraction",
        "evidence_min_coverage_pct", "overall_system_pass",
    ]
    print(summary[[c for c in cols if c in summary.columns]].to_string(index=False))

    event = {
        "opened_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": str(manifest_path),
        "pipeline_snapshot_sha256": manifest.get("pipeline_snapshot_sha256"),
        "cities_opened": successful_cities,
        "repeat_debug_run": bool(repeated),
        "no_city_model_refit": True,
        "temperature_prediction_claim": False,
        "all_system_gates_passed": bool(summary["overall_system_pass"].all()) if len(summary) else False,
        "claim_boundary": manifest.get("claim_boundary"),
    }
    prior_log.append(event)
    log_path.write_text(json.dumps(prior_log, indent=2), encoding="utf-8")

    print("\nSaved ->", out)
    print("Blind-open audit log ->", log_path)
    print("ALL SYSTEM GATES PASSED:", bool(summary["overall_system_pass"].all()) if len(summary) else False)


if __name__ == "__main__":
    main()

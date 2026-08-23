from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from thermalos.analytics.policy_lab import run_policy_stress_lab
from thermalos.analytics.robustness import run_robustness
from thermalos.evidence import build_evidence_ledger
from thermalos.generalization import city_plan_row
from thermalos.models.interventions import build_candidates
from thermalos.models.thermal_twin import PORTABLE_MORPHOLOGY_FEATURES


BASE_REQUIRED_COLUMNS = [
    "tile_id",
    "area",
    "lat",
    "lon",
    "temperature_c",
    "exceedance_h",
    "population",
    "vulnerability",
    "baseline_person_hours",
    "area_reference_temperature_c",
]


@dataclass
class CityContractAudit:
    city: str
    checks: dict[str, bool]
    details: dict[str, object]

    @property
    def fraction(self) -> float:
        if not self.checks:
            return 0.0
        return float(sum(bool(v) for v in self.checks.values()) / len(self.checks))

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(bool(v) for v in self.checks.values())

    def to_dict(self) -> dict:
        return {
            "city": self.city,
            "contract_fraction": self.fraction,
            "contract_passed": self.passed,
            **{f"check_{k}": bool(v) for k, v in self.checks.items()},
            **self.details,
        }


@dataclass
class SystemCityResult:
    summary: dict
    portfolio: pd.DataFrame
    candidates: pd.DataFrame
    robustness_scenarios: pd.DataFrame
    project_stability: pd.DataFrame
    policy_scenarios: pd.DataFrame
    equity_frontier: pd.DataFrame
    evidence_ledger: pd.DataFrame
    contract: CityContractAudit


def _complete(df: pd.DataFrame, column: str, threshold: float = 0.999) -> bool:
    if column not in df.columns or len(df) == 0:
        return False
    return bool(pd.to_numeric(df[column], errors="coerce").notna().mean() >= threshold)


def _status_ok(value: object) -> bool:
    s = str(value).strip().lower()
    return "optimal" in s or "feasible" in s


def audit_city_contract(
    *,
    city: str,
    tiles: pd.DataFrame,
    provenance: dict,
    city_cfg: dict,
    satellite_samples: pd.DataFrame,
    target_samples_per_area: int,
    expected_role: str | None = None,
    exact_satellite_set: bool = True,
) -> CityContractAudit:
    """Audit the frozen cross-city data contract without fitting any model."""

    expected_areas = [str(x) for x in city_cfg.get("areas", {}).keys()]
    sat_counts = (
        {str(k): int(v) for k, v in satellite_samples["area"].astype(str).value_counts().to_dict().items()}
        if "area" in satellite_samples.columns
        else {}
    )
    sat_unique_ok = bool(
        {"area", "lat", "lon"}.issubset(satellite_samples.columns)
        and not satellite_samples.duplicated(["area", "lat", "lon"]).any()
    )
    if exact_satellite_set:
        sat_ok = bool(expected_areas) and sat_unique_ok and all(
            sat_counts.get(a, 0) == int(target_samples_per_area) for a in expected_areas
        )
    else:
        sat_ok = bool(expected_areas) and sat_unique_ok and all(
            sat_counts.get(a, 0) >= int(target_samples_per_area) for a in expected_areas
        )

    portable_ok = all(_complete(tiles, c) for c in PORTABLE_MORPHOLOGY_FEATURES)
    required_ok = all(_complete(tiles, c) for c in BASE_REQUIRED_COLUMNS if c not in {"tile_id", "area"})
    role = str(provenance.get("transfer_role", ""))
    role_ok = expected_role is None or role == expected_role
    enrichment = provenance.get("enrichment", {}) if isinstance(provenance.get("enrichment"), dict) else {}
    gtfs_required = bool(city_cfg.get("external", {}).get("transit_reference"))
    gtfs_ok = bool(enrichment.get("gtfs")) and _complete(tiles, "transit_stop_count")
    unique_tile_ok = bool("tile_id" in tiles.columns and tiles["tile_id"].astype(str).is_unique)
    areas_ok = bool(expected_areas) and set(tiles.get("area", pd.Series(dtype=str)).astype(str).unique()) == set(expected_areas)

    checks = {
        "real_non_synthetic_data": not bool(provenance.get("synthetic_demo", False)),
        "expected_transfer_role": role_ok,
        "required_columns_complete": required_ok,
        "unique_tile_identity": unique_tile_ok,
        "configured_areas_match": areas_ok,
        "portable_morphology_complete": portable_ok,
        "satellite_coordinates_unique": sat_unique_ok,
        "registered_satellite_sampling": sat_ok,
        "gtfs_enrichment_complete": (not gtfs_required) or gtfs_ok,
    }
    details = {
        "rows": int(len(tiles)),
        "areas": int(tiles["area"].nunique()) if "area" in tiles.columns else 0,
        "expected_areas": expected_areas,
        "satellite_samples_per_area": sat_counts,
        "satellite_target_per_area": int(target_samples_per_area),
        "portable_morphology_features": list(PORTABLE_MORPHOLOGY_FEATURES),
        "transfer_role": role,
    }
    return CityContractAudit(city=city, checks=checks, details=details)


def run_system_city(
    *,
    city: str,
    tiles: pd.DataFrame,
    provenance: dict,
    intervention_config: dict,
    contract: CityContractAudit,
    budget_usd: float,
    equity_min_fraction: float,
    robustness_scenarios: int,
    robustness_pool_size: int,
) -> SystemCityResult:
    """Run the unchanged decision, robustness, policy, and trust stack for one city."""

    if not 0.0 <= float(equity_min_fraction) <= 1.0:
        raise ValueError("equity_min_fraction must be between 0 and 1 inclusive.")

    candidate_result = build_candidates(tiles, intervention_config, seed=42)
    candidates = candidate_result.candidates
    row, selected = city_plan_row(
        city,
        tiles,
        provenance,
        intervention_config,
        budget_usd,
        equity_min_fraction=equity_min_fraction,
    )

    robust = run_robustness(
        candidates,
        budget_usd=budget_usd,
        equity_min_fraction=equity_min_fraction,
        scenarios=robustness_scenarios,
        pool_size=robustness_pool_size,
        seed=42,
    )
    robust_feasible = int(robust.scenario_summary["status"].map(_status_ok).sum()) if len(robust.scenario_summary) else 0
    robust_fraction = robust_feasible / max(1, len(robust.scenario_summary))

    policy = run_policy_stress_lab(
        candidates,
        budget_usd=budget_usd,
        area_count=int(tiles["area"].nunique()) if "area" in tiles.columns else 1,
    )
    policy_fraction = float(policy.scenarios["status"].map(_status_ok).mean()) if len(policy.scenarios) else 0.0
    frontier_fraction = float(policy.equity_frontier["status"].map(_status_ok).mean()) if len(policy.equity_frontier) else 0.0

    ledger = build_evidence_ledger(tiles, provenance, intervention_config)
    evidence_min = float(pd.to_numeric(ledger["coverage_pct"], errors="coerce").fillna(0.0).min()) if len(ledger) else 0.0

    spent = float(row.get("spent_usd", 0.0))
    equity = float(row.get("equity_spend_fraction", 0.0))
    no_action_triggered = bool(
        len(candidates) == 0 and float(row.get("baseline_person_hours", 0.0)) <= 1e-9
    )
    status = {
        **row,
        "city_key": city,
        "contract_fraction": contract.fraction,
        "contract_passed": contract.passed,
        "candidate_count": int(len(candidates)),
        "intervention_families": int(candidates["intervention"].nunique()) if "intervention" in candidates.columns else 0,
        "candidate_generation_success": bool(len(candidates) > 0),
        "decision_state": "no_action_triggered" if no_action_triggered else "action_plan",
        "no_action_triggered": no_action_triggered,
        "no_action_reason": (
            "No positive baseline heat burden under the frozen event threshold."
            if no_action_triggered else ""
        ),
        "milp_feasible": bool(len(selected) > 0 and spent <= float(budget_usd) + 1.0),
        "budget_constraint_satisfied": bool(spent <= float(budget_usd) + 1.0),
        "equity_constraint_satisfied": bool(equity + 1e-9 >= float(equity_min_fraction)),
        "robustness_scenarios": int(len(robust.scenario_summary)),
        "robustness_feasible_worlds": int(robust_feasible),
        "robustness_feasible_fraction": float(robust_fraction),
        "portfolio_stability": float(robust.portfolio_stability),
        "median_jaccard": float(robust.median_jaccard),
        "direct_benefit_p10": float(robust.direct_benefit_p10),
        "direct_benefit_p50": float(robust.direct_benefit_p50),
        "direct_benefit_p90": float(robust.direct_benefit_p90),
        "policy_scenarios": int(len(policy.scenarios)),
        "policy_scenarios_feasible_fraction": float(policy_fraction),
        "equity_frontier_points": int(len(policy.equity_frontier)),
        "equity_frontier_feasible_fraction": float(frontier_fraction),
        "evidence_ledger_layers": int(len(ledger)),
        "evidence_min_coverage_pct": evidence_min,
        "temperature_prediction_claim": False,
        "no_city_model_refit": True,
    }
    return SystemCityResult(
        summary=status,
        portfolio=selected,
        candidates=candidates,
        robustness_scenarios=robust.scenario_summary,
        project_stability=robust.project_stability,
        policy_scenarios=policy.scenarios,
        equity_frontier=policy.equity_frontier,
        evidence_ledger=ledger,
        contract=contract,
    )


def apply_system_gates(summary: dict, gates: dict, *, equity_min_fraction: float) -> dict:
    """Apply only the predeclared system-transfer gates."""

    if not 0.0 <= float(equity_min_fraction) <= 1.0:
        raise ValueError("equity_min_fraction must be between 0 and 1 inclusive.")

    no_action = bool(summary.get("no_action_triggered", False))
    checks = {
        "contract": float(summary.get("contract_fraction", 0.0)) >= float(gates.get("required_data_contract_fraction_min", 1.0)),
        "candidate_generation": no_action or bool(summary.get("candidate_generation_success", False)),
        "milp": no_action or bool(summary.get("milp_feasible", False)),
        "budget": bool(summary.get("budget_constraint_satisfied", False)),
        "equity": no_action or float(summary.get("equity_spend_fraction", 0.0)) + 1e-9 >= float(
            gates.get("equity_spend_fraction_min", equity_min_fraction)
        ),
        "robustness": no_action or float(summary.get("robustness_feasible_fraction", 0.0)) >= float(
            gates.get("robustness_feasible_fraction_min", 1.0)
        ),
        "policy_scenarios": no_action or float(summary.get("policy_scenarios_feasible_fraction", 0.0)) >= float(
            gates.get("policy_scenarios_feasible_fraction_min", 1.0)
        ),
        "equity_frontier": no_action or float(summary.get("equity_frontier_feasible_fraction", 0.0)) >= float(
            gates.get("equity_frontier_feasible_fraction_min", 1.0)
        ),
        "evidence_layers": int(summary.get("evidence_ledger_layers", 0)) >= int(gates.get("evidence_ledger_min_layers", 10)),
        "evidence_coverage": float(summary.get("evidence_min_coverage_pct", 0.0)) >= float(
            gates.get("evidence_min_coverage_pct", 99.0)
        ),
    }
    return {
        **{f"passes_{k}_gate": bool(v) for k, v in checks.items()},
        "overall_system_pass": all(checks.values()),
    }


def sha256_file(path: str | Path) -> str:
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def frozen_pipeline_files(root: str | Path = ".") -> list[Path]:
    """Files whose behavior/data contract must not change after the freeze."""

    root = Path(root)
    files: list[Path] = []
    files.extend(sorted((root / "src" / "thermalos").rglob("*.py")))
    for rel in [
        "scripts/build_city_features.py",
        "scripts/harvest_city.py",
        "scripts/harvest_satellite_only.py",
        "scripts/fetch_census_geography.py",
        "scripts/fetch_acs.py",
        "scripts/import_gtfs.py",
        "configs/generalization.yaml",
        "configs/interventions.yaml",
        "configs/operations.yaml",
    ]:
        p = root / rel
        if p.exists():
            files.append(p)
    # De-duplicate while preserving deterministic relative-path ordering.
    unique = {p.resolve(): p for p in files}
    return sorted(unique.values(), key=lambda p: p.relative_to(root).as_posix())


def pipeline_hash_manifest(root: str | Path = ".") -> tuple[dict[str, str], str]:
    root = Path(root).resolve()
    per_file: dict[str, str] = {}
    aggregate = hashlib.sha256()
    for path in frozen_pipeline_files(root):
        rel = path.resolve().relative_to(root).as_posix()
        digest = sha256_file(path)
        per_file[rel] = digest
        aggregate.update(rel.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(digest.encode("ascii"))
        aggregate.update(b"\n")
    return per_file, aggregate.hexdigest()


def verify_pipeline_hashes(expected: dict[str, str], root: str | Path = ".") -> list[str]:
    root = Path(root)
    problems: list[str] = []
    for rel, digest in expected.items():
        p = root / rel
        if not p.exists():
            problems.append(f"missing:{rel}")
        elif sha256_file(p) != digest:
            problems.append(f"changed:{rel}")
    return problems


def read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))

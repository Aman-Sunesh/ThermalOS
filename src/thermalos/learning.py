from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class CalibrationResult:
    updated_config: dict
    audit: pd.DataFrame
    verified_projects: int
    status: str


def _norm_intervention(value: object) -> str:
    return " ".join(
        str(value or "")
        .strip()
        .lower()
        .replace("/", " ")
        .replace("_", " ")
        .replace("-", " ")
        .split()
    )


def adaptive_calibration_status(verification_projects: pd.DataFrame, *, minimum_verified: int = 5) -> dict:
    if verification_projects.empty:
        n = 0
    else:
        status = verification_projects.get("verification_status", pd.Series(dtype=str)).astype(str)
        n = int(status.str.startswith("verified").sum())
    return {
        "verified_projects": n,
        "minimum_verified_for_local_calibration": int(minimum_verified),
        "ready": n >= minimum_verified,
        "status": "local calibration ready" if n >= minimum_verified else "literature-calibrated; awaiting verified local projects",
    }


def update_intervention_priors(
    base_config: dict,
    verified_effects: pd.DataFrame,
    *,
    prior_strength_projects: float = 8.0,
    minimum_verified: int = 3,
) -> CalibrationResult:
    """Empirical-Bayes style shrinkage update for local intervention cooling priors.

    ``verified_effects`` must contain ``intervention`` and ``observed_cooling_c``.
    Updates are deliberately conservative: local observations are shrunk toward the
    literature/config prior and no update occurs with fewer than ``minimum_verified``
    valid observations for an intervention.
    """
    cfg = deepcopy(base_config)
    rows: list[dict] = []
    valid_total = 0
    if verified_effects.empty or not {"intervention", "observed_cooling_c"}.issubset(verified_effects.columns):
        return CalibrationResult(cfg, pd.DataFrame(), 0, "no verified local effects")

    effects = verified_effects.copy()
    effects["_intervention_norm"] = effects["intervention"].map(_norm_intervention)

    for name, spec in cfg.get("interventions", {}).items():
        prior_mean = float(spec.get("temp_effect_c_mean", 0.0))
        prior_sd = float(spec.get("temp_effect_c_sd", 0.0))
        aliases = {
            _norm_intervention(name),
            _norm_intervention(spec.get("label", name)),
        }
        # Accept common human-facing shorthand without changing the canonical
        # key written to the audit/config.
        aliases |= {
            a.replace(" package", "").replace(" retrofit", "").strip()
            for a in list(aliases)
        }
        common_aliases = {
            "tree_canopy": {"trees", "tree", "tree canopy"},
            "shade_structure": {"shade", "shade structure", "shade structures"},
            "cool_pavement": {"pavement", "cool pavement", "cool pavements"},
            "cool_roof": {"roof", "roofs", "cool roof", "cool roofs"},
            "cooling_node": {"cooling", "hydration", "cooling hydration", "cooling node"},
        }
        aliases |= {_norm_intervention(v) for v in common_aliases.get(name, set())}
        vals = pd.to_numeric(
            effects.loc[effects["_intervention_norm"].isin(aliases), "observed_cooling_c"],
            errors="coerce",
        ).dropna()
        vals = vals[(vals >= 0) & np.isfinite(vals)]
        n = int(len(vals))
        valid_total += n
        updated = False
        posterior_mean = prior_mean
        posterior_sd = prior_sd
        if n >= minimum_verified:
            observed_mean = float(vals.mean())
            weight_local = n / (n + max(float(prior_strength_projects), 1e-6))
            posterior_mean = (1.0 - weight_local) * prior_mean + weight_local * observed_mean
            sample_sd = float(vals.std(ddof=1)) if n > 1 else prior_sd
            posterior_sd = max(0.02, (1.0 - weight_local) * prior_sd + weight_local * sample_sd)
            lo = float(spec.get("temp_effect_c_min", 0.0))
            hi = float(spec.get("temp_effect_c_max", max(prior_mean, lo)))
            spec["temp_effect_c_mean"] = float(np.clip(posterior_mean, lo, hi))
            spec["temp_effect_c_sd"] = float(posterior_sd)
            updated = True
        rows.append(
            {
                "intervention": name,
                "verified_projects": n,
                "prior_mean_c": prior_mean,
                "observed_mean_c": float(vals.mean()) if n else np.nan,
                "posterior_mean_c": float(spec.get("temp_effect_c_mean", prior_mean)),
                "prior_sd_c": prior_sd,
                "posterior_sd_c": float(spec.get("temp_effect_c_sd", prior_sd)),
                "updated": updated,
            }
        )
    audit = pd.DataFrame(rows)
    updates = int(audit["updated"].sum()) if len(audit) else 0
    status = f"updated {updates} intervention priors" if updates else "insufficient verified local evidence; literature priors retained"
    return CalibrationResult(cfg, audit, valid_total, status)

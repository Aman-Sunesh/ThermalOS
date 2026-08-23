from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from thermalos.features.suitability import intervention_suitability


@dataclass
class CandidateBuildResult:
    candidates: pd.DataFrame
    seed: int


def _bounded_normal(rng: np.random.Generator, mean: float, sd: float, lo: float, hi: float, n: int) -> np.ndarray:
    if sd <= 0:
        return np.full(n, np.clip(mean, lo, hi), dtype=float)
    x = rng.normal(mean, sd, n)
    return np.clip(x, lo, hi)


def build_candidates(
    tiles: pd.DataFrame,
    config: dict,
    *,
    seed: int = 42,
    min_feasibility: float = 0.25,
    observational_tree_cooling: np.ndarray | None = None,
) -> CandidateBuildResult:
    """Create one uncertain project option per viable tile/intervention."""
    rng = np.random.default_rng(seed)
    suit = intervention_suitability(tiles)
    int_cfg = config["interventions"]
    model_cfg = config.get("model", {})
    n_mc = int(model_cfg.get("monte_carlo_samples", 400))
    qlo = float(model_cfg.get("lower_quantile", 0.10))
    qhi = float(model_cfg.get("upper_quantile", 0.90))
    duration_sens = float(model_cfg.get("duration_sensitivity_h_per_c", 1.5))
    rows = []

    for pos, (idx, tile) in enumerate(tiles.iterrows()):
        baseline = float(tile.get("baseline_person_hours", 0.0))
        if baseline <= 0:
            continue
        vulnerability = float(tile.get("vulnerability", 0.5))
        exposed_pop = float(tile.get("exposed_population", tile.get("population", 0.0)))
        exceedance_h = max(float(tile.get("exceedance_h", 0.0)), 0.0)

        for name, spec in int_cfg.items():
            feas = float(suit.loc[idx, name])
            if feas < min_feasibility:
                continue
            base_cost = float(spec["base_cost_usd"])
            jitter = float(spec.get("cost_jitter_fraction", 0.0))
            # deterministic tile-specific cost variation for spatial/project heterogeneity
            cost_factor = 1.0 + jitter * (0.6 * (1 - feas) + 0.4 * ((pos % 7) / 6 - 0.5))
            cost = max(1000.0, base_cost * cost_factor)

            mean = float(spec.get("temp_effect_c_mean", 0.0))
            sd = float(spec.get("temp_effect_c_sd", 0.0))
            lo = float(spec.get("temp_effect_c_min", 0.0))
            hi = float(spec.get("temp_effect_c_max", max(mean, lo)))

            # Observational morphology-temperature models are not causal effect
            # estimators. They therefore do NOT alter intervention priors by
            # default. A legacy research-only opt-in remains for reproducibility,
            # but production/demo planning keeps literature-bounded priors intact.
            allow_obs_cal = bool(model_cfg.get("allow_observational_tree_calibration", False))
            if (
                allow_obs_cal
                and name == "tree_canopy"
                and observational_tree_cooling is not None
                and pos < len(observational_tree_cooling)
            ):
                obs = float(np.clip(observational_tree_cooling[pos], 0, hi))
                mean = 0.70 * mean + 0.30 * obs

            temp_samples = _bounded_normal(rng, mean, sd, lo, hi, n_mc) * feas
            duration_relief = np.minimum(exceedance_h, duration_sens * temp_samples)
            direct = float(spec.get("direct_exposure_relief_fraction", 0.0)) * feas
            radiant_penalty = float(spec.get("radiant_penalty_fraction", 0.0)) * max(float(tile.get("solar_ghi", 0.0)) / 1000.0, 0.0)
            direct = max(0.0, direct - radiant_penalty)

            # Relief has two non-additive pieces: less hot duration + direct protected exposure.
            duration_fraction = duration_relief / max(exceedance_h, 1e-6)
            total_fraction = 1.0 - (1.0 - np.clip(duration_fraction, 0, 1)) * (1.0 - np.clip(direct, 0, 0.95))
            benefit_samples = np.minimum(baseline, baseline * total_fraction)

            expected = float(np.mean(benefit_samples))
            low = float(np.quantile(benefit_samples, qlo))
            high = float(np.quantile(benefit_samples, qhi))
            temp_expected = float(np.mean(temp_samples))

            rows.append({
                "candidate_id": f"{tile['tile_id']}::{name}",
                "tile_id": tile["tile_id"],
                "city": tile.get("city", ""),
                "area": tile.get("area", ""),
                "lat": float(tile["lat"]),
                "lon": float(tile["lon"]),
                "intervention": name,
                "label": spec.get("label", name),
                "mechanism": spec.get("mechanism", ""),
                "feasibility": feas,
                "cost_usd": float(round(cost, 2)),
                "temp_relief_c_expected": temp_expected,
                "direct_exposure_relief_fraction": direct,
                "benefit_expected_person_hours": expected,
                "benefit_low_person_hours": low,
                "benefit_high_person_hours": high,
                "vulnerability": vulnerability,
                "high_vulnerability": bool(tile.get("high_vulnerability", vulnerability >= 0.6)),
                "baseline_person_hours": baseline,
                "spillover_sigma_m": float(spec.get("spillover_sigma_m", 100)),
                "spillover_strength": float(spec.get("spillover_strength", 0.1)),
            })
    candidates = pd.DataFrame(rows)
    if candidates.empty:
        candidates = pd.DataFrame(columns=[
            "candidate_id", "tile_id", "city", "area", "lat", "lon",
            "intervention", "label", "mechanism", "feasibility", "cost_usd",
            "temp_relief_c_expected", "direct_exposure_relief_fraction",
            "benefit_expected_person_hours", "benefit_low_person_hours",
            "benefit_high_person_hours", "vulnerability", "high_vulnerability",
            "baseline_person_hours", "spillover_sigma_m", "spillover_strength",
        ])
    return CandidateBuildResult(candidates, seed)

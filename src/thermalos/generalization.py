from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from thermalos.models.interventions import build_candidates
from thermalos.models.thermal_twin import UniversalThermalTwin
from thermalos.scenario import evaluate_plan


@dataclass
class CrossCityResult:
    comparison: pd.DataFrame
    transfer_metrics: dict
    portfolios: dict[str, pd.DataFrame]


@dataclass
class DevelopmentBenchmark:
    fold_metrics: pd.DataFrame
    estimator_summary: pd.DataFrame
    selected_estimator: str


def city_plan_row(
    city: str,
    tiles: pd.DataFrame,
    provenance: dict,
    config: dict,
    budget_usd: float,
    *,
    equity_min_fraction: float = 0.35,
) -> tuple[dict, pd.DataFrame]:
    """Run one unchanged ThermalOS decision engine on a city tile table."""
    candidates = build_candidates(tiles, config, seed=42).candidates
    evaluation = evaluate_plan(
        tiles,
        candidates,
        budget_usd=budget_usd,
        objective="balanced",
        impact_basis="expected",
        equity_min_fraction=equity_min_fraction,
        max_relief_fraction=float(config.get("model", {}).get("max_tile_relief_fraction", 0.85)),
    )
    selected = evaluation.portfolio.selected
    dominant = "None"
    top_area = "None"
    if len(selected):
        dominant = str(selected.groupby("label").size().idxmax())
        top_area = str(selected.groupby("area").size().idxmax()).replace("_", " ")
    m = evaluation.metrics
    row = {
        "city": city,
        "data_mode": "Synthetic demo" if provenance.get("synthetic_demo", False) else "Real planning data",
        "transfer_role": provenance.get("transfer_role", "unspecified"),
        "tiles": int(len(tiles)),
        "baseline_person_hours": float(m.get("baseline_person_hours", 0.0)),
        "budget_usd": float(budget_usd),
        "projects": int(m.get("projects", 0)),
        "spent_usd": float(m.get("spent_usd", 0.0)),
        "equity_spend_fraction": float(m.get("vulnerable_spend_fraction", 0.0)),
        "direct_person_hours_avoided": float(m.get("planning_person_hours_avoided_first_order", 0.0)),
        "system_person_hours_avoided": float(m.get("modeled_person_hours_avoided_with_spillover", 0.0)),
        "modeled_reduction_fraction": float(m.get("modeled_reduction_fraction", 0.0)),
        "dominant_intervention": dominant,
        "top_area": top_area,
        "decision_claim_boundary": (
            "Decision-support scenario under one frozen candidate/optimization pipeline; "
            "not evidence that intervention causal effects transport unchanged across cities."
        ),
    }
    return row, selected


def benchmark_development_estimators(
    city_frames: Mapping[str, pd.DataFrame],
    *,
    estimators: tuple[str, ...] = ("hist_gradient_boosting", "extra_trees", "ensemble"),
) -> DevelopmentBenchmark:
    """Leave-one-development-city-out benchmark for model-family selection.

    This is allowed *before* the prospective blind cities are opened. Once a
    model family is selected and frozen, blind-city evaluation must not alter it.
    """
    if len(city_frames) < 2:
        raise ValueError("Need at least two development cities for leave-one-city-out selection")

    fold_rows: list[dict] = []
    items = list(city_frames.items())
    for estimator in estimators:
        for held_out, test in items:
            train_parts = []
            for city, frame in items:
                if city == held_out:
                    continue
                part = frame.copy()
                if "city" not in part.columns:
                    part["city"] = city
                train_parts.append(part)
            train = pd.concat(train_parts, ignore_index=True)
            test_df = test.copy()
            if "city" not in test_df.columns:
                test_df["city"] = held_out
            model = UniversalThermalTwin(estimator=estimator).fit(train)
            metrics = model.evaluate(test_df).to_dict()
            fold_rows.append({"estimator": estimator, "held_out_city": held_out, **metrics})

    folds = pd.DataFrame(fold_rows)
    summaries = []
    for estimator, g in folds.groupby("estimator", sort=False):
        mean_mae = float(g["anomaly_mae_c"].mean())
        mean_rho = float(pd.to_numeric(g["anomaly_spearman"], errors="coerce").fillna(0.0).mean())
        mean_recall = float(g["top20_hotspot_recall"].mean())
        # Fixed development-only selection score: lower is better.
        # MAE remains primary; ranking/hotspot terms break near-ties in favor of
        # decision-relevant spatial ordering.
        score = mean_mae + 0.50 * (1.0 - mean_rho) + 0.50 * (1.0 - mean_recall)
        summaries.append(
            {
                "estimator": estimator,
                "mean_anomaly_mae_c": mean_mae,
                "mean_anomaly_rmse_c": float(g["anomaly_rmse_c"].mean()),
                "mean_anomaly_spearman": mean_rho,
                "mean_top20_hotspot_recall": mean_recall,
                "selection_score": float(score),
            }
        )
    summary = pd.DataFrame(summaries).sort_values(
        ["selection_score", "mean_anomaly_mae_c", "estimator"]
    ).reset_index(drop=True)
    selected = str(summary.iloc[0]["estimator"])
    return DevelopmentBenchmark(folds, summary, selected)


def evaluate_frozen_model(
    model: UniversalThermalTwin,
    tiles: pd.DataFrame,
    *,
    city_key: str,
    provenance: dict,
    development_cities: list[str] | tuple[str, ...],
) -> dict:
    """Evaluate a frozen model without fitting or updating it."""
    if city_key in set(development_cities):
        role = "development"
    else:
        role = str(provenance.get("transfer_role", "unspecified"))
    m = model.evaluate(tiles).to_dict()
    return {
        "city": city_key,
        "role": role,
        "synthetic_demo": bool(provenance.get("synthetic_demo", False)),
        "no_city_refit": True,
        "model_target": model.target_definition_,
        "anchor_definition": model.anchor_definition_,
        "raw_lat_lon_used": False,
        **m,
        "claim_boundary": (
            "Observational local thermal-anomaly transfer diagnostic only. "
            "It is not a causal intervention-effect transportability test."
        ),
    }


def run_cross_city_comparison(
    miami_tiles: pd.DataFrame,
    houston_tiles: pd.DataFrame,
    *,
    miami_provenance: dict,
    houston_provenance: dict,
    intervention_config: dict,
    budget_usd: float = 2_000_000,
) -> CrossCityResult:
    """Development-only replay of the unchanged ThermalOS decision pipeline.

    v3 deliberately does *not* score a morphology-only temperature forecaster.
    FortyGuard's observed thermal field is an upstream system input. This replay
    asks whether the same candidate/optimization machinery runs coherently in both
    development cities before the prospective three-city system freeze.
    """
    rows = []
    portfolios: dict[str, pd.DataFrame] = {}
    for key, tiles, prov in [
        ("Miami-Dade", miami_tiles, miami_provenance),
        ("Houston", houston_tiles, houston_provenance),
    ]:
        row, sel = city_plan_row(key, tiles, prov, intervention_config, budget_usd)
        rows.append(row)
        portfolios[key] = sel

    transfer: dict = {
        "development_replay": True,
        "same_decision_pipeline": True,
        "no_houston_refit": True,
        "temperature_prediction_claim": False,
        "morphology_only_predictor_role": "retired from prospective success criteria",
        "model_version": "ThermalOS-v3-system-transfer",
        "claim_boundary": (
            "Development-system replay only. FortyGuard thermal intelligence is an observed upstream input; "
            "intervention effects remain literature-bounded planning priors rather than learned causal effects."
        ),
    }
    return CrossCityResult(pd.DataFrame(rows), transfer, portfolios)

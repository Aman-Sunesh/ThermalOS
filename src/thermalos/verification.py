from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from thermalos.geo import haversine_m


MATCH_FEATURES = [
    "temperature_c",
    "baseline_person_hours",
    "vulnerability",
    "canopy_fraction",
    "impervious_fraction",
    "population",
]


@dataclass
class VerificationRegistry:
    projects: pd.DataFrame
    controls: pd.DataFrame
    protocol: dict


def _standardized_matrix(df: pd.DataFrame, columns: list[str]) -> tuple[np.ndarray, list[str]]:
    used = [c for c in columns if c in df.columns and pd.to_numeric(df[c], errors="coerce").notna().any()]
    if not used:
        return np.zeros((len(df), 0), dtype=float), []
    x = df[used].apply(pd.to_numeric, errors="coerce")
    x = x.fillna(x.median(numeric_only=True)).fillna(0.0)
    mu = x.mean(axis=0)
    sd = x.std(axis=0).replace(0, 1.0).fillna(1.0)
    return ((x - mu) / sd).to_numpy(float), used


def build_verification_registry(
    tiles: pd.DataFrame,
    selected: pd.DataFrame,
    *,
    baseline_date: str | None = None,
    controls_per_project: int = 5,
    min_control_distance_m: float = 300.0,
) -> VerificationRegistry:
    """Create a pre-deployment measurement-and-verification registry.

    No observed post-intervention impact is fabricated. The registry freezes the
    planning baseline, expected effect range, and matched-control set that a future
    FortyGuard refresh can evaluate after implementation.
    """
    if baseline_date is None:
        baseline_date = date.today().isoformat()
    controls_per_project = max(1, int(controls_per_project))
    tile_table = tiles.drop_duplicates("tile_id").reset_index(drop=True).copy()
    x, used = _standardized_matrix(tile_table, MATCH_FEATURES)
    tile_pos = {str(tid): i for i, tid in enumerate(tile_table["tile_id"].astype(str))}
    selected_tile_ids = set(selected.get("tile_id", pd.Series(dtype=str)).astype(str))

    project_rows: list[dict] = []
    control_rows: list[dict] = []

    for _, project in selected.iterrows():
        tid = str(project.get("tile_id"))
        if tid not in tile_pos:
            continue
        i = tile_pos[tid]
        area = str(project.get("area", tile_table.iloc[i].get("area", "")))
        plat = float(project.get("lat", tile_table.iloc[i]["lat"]))
        plon = float(project.get("lon", tile_table.iloc[i]["lon"]))

        eligible: list[tuple[float, int, float]] = []
        for j, row in tile_table.iterrows():
            ctid = str(row["tile_id"])
            if ctid == tid or ctid in selected_tile_ids:
                continue
            d_m = haversine_m(plat, plon, float(row["lat"]), float(row["lon"]))
            if d_m < min_control_distance_m:
                continue
            feature_dist = float(np.linalg.norm(x[i] - x[j])) if x.shape[1] else 0.0
            area_penalty = 0.0 if str(row.get("area", "")) == area else 1.5
            # Prefer same-neighborhood controls with similar baseline state while
            # avoiding sites close enough to plausibly receive intervention spillover.
            score = feature_dist + area_penalty + min(d_m / 10_000.0, 1.0) * 0.15
            eligible.append((score, j, d_m))
        eligible.sort(key=lambda z: z[0])
        chosen = eligible[:controls_per_project]
        control_ids = [str(tile_table.iloc[j]["tile_id"]) for _, j, _ in chosen]

        project_rows.append(
            {
                "candidate_id": str(project.get("candidate_id", f"{tid}::{project.get('intervention', '')}")),
                "tile_id": tid,
                "area": area,
                "intervention": project.get("intervention", ""),
                "label": project.get("label", ""),
                "lat": plat,
                "lon": plon,
                "verification_status": "baseline_captured",
                "baseline_date": baseline_date,
                "baseline_temperature_c": float(tile_table.iloc[i].get("temperature_c", np.nan)),
                "baseline_person_hours": float(tile_table.iloc[i].get("baseline_person_hours", np.nan)),
                "expected_person_hours_avoided": float(project.get("benefit_expected_person_hours", 0.0)),
                "low_person_hours_avoided": float(project.get("benefit_low_person_hours", 0.0)),
                "high_person_hours_avoided": float(project.get("benefit_high_person_hours", 0.0)),
                "expected_temp_relief_c": float(project.get("temp_relief_c_expected", 0.0)),
                "control_tile_ids": "|".join(control_ids),
                "control_count": len(control_ids),
                "verification_schedule": "30 / 90 / 365 days after deployment",
                "observed_temp_change_c": np.nan,
                "weather_normalized_temp_change_c": np.nan,
                "verification_note": "Awaiting implementation and post-deployment FortyGuard observations.",
            }
        )
        for rank, (score, j, d_m) in enumerate(chosen, start=1):
            row = tile_table.iloc[j]
            control_rows.append(
                {
                    "candidate_id": str(project.get("candidate_id", "")),
                    "project_tile_id": tid,
                    "control_tile_id": str(row["tile_id"]),
                    "rank": rank,
                    "match_score": score,
                    "distance_m": d_m,
                    "same_area": str(row.get("area", "")) == area,
                    "baseline_temperature_c": float(row.get("temperature_c", np.nan)),
                    "baseline_person_hours": float(row.get("baseline_person_hours", np.nan)),
                }
            )

    protocol = {
        "design": "pre/post matched-control verification",
        "baseline_state": "frozen at planning time",
        "post_deployment_windows_days": [30, 90, 365],
        "weather_normalization": "required before attributing observed cooling to an intervention",
        "control_matching_features": used,
        "minimum_control_distance_m": float(min_control_distance_m),
        "claim_boundary": (
            "ThermalVerify defines an auditable measurement protocol. Until post-deployment observations exist, "
            "the intervention effect remains a modeled scenario estimate rather than a measured causal impact."
        ),
    }
    # Preserve a stable controls schema even when no eligible matched controls
    # exist. This lets post-deployment evaluation return an auditable
    # "insufficient_controls" result instead of crashing on an empty DataFrame.
    control_columns = [
        "candidate_id",
        "project_tile_id",
        "control_tile_id",
        "rank",
        "match_score",
        "distance_m",
        "same_area",
        "baseline_temperature_c",
        "baseline_person_hours",
    ]
    controls_df = pd.DataFrame(control_rows, columns=control_columns)
    return VerificationRegistry(pd.DataFrame(project_rows), controls_df, protocol)


def evaluate_post_deployment(
    registry: VerificationRegistry,
    post_tiles: pd.DataFrame,
    *,
    temperature_column: str = "temperature_c",
) -> pd.DataFrame:
    """Evaluate post-deployment temperature change with matched controls.

    This is a transparent difference-in-differences style diagnostic, not a full
    causal estimator. Weather normalization or repeated matched days should be
    applied before using the result as evidence of intervention effectiveness.
    """
    current = post_tiles.drop_duplicates("tile_id").copy()
    current["tile_id"] = current["tile_id"].astype(str)
    current = current.set_index("tile_id")
    baseline_controls = registry.controls.copy()
    if "candidate_id" not in baseline_controls.columns:
        # Be defensive for registries produced by older releases or external
        # callers that supply an empty controls table without columns.
        baseline_controls = pd.DataFrame(
            columns=[
                "candidate_id",
                "project_tile_id",
                "control_tile_id",
                "rank",
                "match_score",
                "distance_m",
                "same_area",
                "baseline_temperature_c",
                "baseline_person_hours",
            ]
        )
    rows: list[dict] = []
    for _, p in registry.projects.iterrows():
        tid = str(p["tile_id"])
        if tid not in current.index:
            continue
        project_post = float(pd.to_numeric(pd.Series([current.loc[tid].get(temperature_column)]), errors="coerce").iloc[0])
        project_pre = float(p["baseline_temperature_c"])
        crows = baseline_controls[baseline_controls["candidate_id"].astype(str) == str(p["candidate_id"])]
        control_changes = []
        for _, c in crows.iterrows():
            cid = str(c["control_tile_id"])
            if cid not in current.index:
                continue
            post = pd.to_numeric(pd.Series([current.loc[cid].get(temperature_column)]), errors="coerce").iloc[0]
            pre = c.get("baseline_temperature_c")
            if pd.notna(post) and pd.notna(pre):
                control_changes.append(float(post) - float(pre))
        raw_change = project_post - project_pre
        control_change = float(np.median(control_changes)) if control_changes else np.nan
        normalized = raw_change - control_change if np.isfinite(control_change) else np.nan
        rows.append(
            {
                "candidate_id": p["candidate_id"],
                "tile_id": tid,
                "raw_temp_change_c": raw_change,
                "matched_control_change_c": control_change,
                "weather_normalized_temp_change_c": normalized,
                "observed_cooling_c": -normalized if np.isfinite(normalized) else np.nan,
                "controls_observed": len(control_changes),
                "verification_status": "observed_pending_review" if control_changes else "insufficient_controls",
                "claim_boundary": "Difference-in-differences diagnostic; causal attribution requires repeated/weather-matched validation.",
            }
        )
    return pd.DataFrame(rows)

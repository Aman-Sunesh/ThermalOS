import numpy as np
import pandas as pd

from thermalos.config import generalization_config
from thermalos.demo import generate_demo_city
from thermalos.features.assemble import build_processed_table
from thermalos.models.thermal_twin import (
    PORTABLE_BASE_FEATURES,
    UniversalThermalTwin,
    add_portable_relative_features,
    city_area_balanced_weights,
)


def test_portable_feature_engineering_excludes_coordinates_and_adds_relative_context():
    df = generate_demo_city("miami", n_side=4, seed=4)
    x = add_portable_relative_features(df)
    assert "lat" not in x.columns
    assert "lon" not in x.columns
    assert "exceedance_h" not in x.columns
    assert set(PORTABLE_BASE_FEATURES).intersection(x.columns)
    assert "fg_canopy_fraction__delta_area" in x.columns
    assert "fg_canopy_fraction__rank_area" in x.columns
    assert x["fg_canopy_fraction__rank_area"].between(0, 1).all()


def test_universal_twin_v2_runs_no_refit_transfer_task():
    miami = generate_demo_city("miami", n_side=5, seed=7)
    houston = generate_demo_city("houston", n_side=5, seed=8)
    model = UniversalThermalTwin(estimator="extra_trees").fit(miami)
    metrics = model.evaluate(houston)
    assert metrics.n == len(houston)
    assert np.isfinite(metrics.anomaly_mae_c)
    assert np.isfinite(metrics.anomaly_rmse_c)
    assert 0 <= metrics.top20_hotspot_recall <= 1
    assert 0 <= metrics.top20_hotspot_jaccard <= 1
    assert "lat" not in model.features_
    assert "lon" not in model.features_


def test_city_area_balanced_weights_equalize_city_and_area_mass():
    df = pd.DataFrame(
        {
            "city": ["a"] * 8 + ["b"] * 4,
            "area": ["a1"] * 6 + ["a2"] * 2 + ["b1"] * 2 + ["b2"] * 2,
        }
    )
    w = pd.Series(city_area_balanced_weights(df), index=df.index)
    city_mass = w.groupby(df["city"]).sum()
    assert np.allclose(city_mass.iloc[0], city_mass.iloc[1])
    for city in ["a", "b"]:
        idx = df["city"] == city
        area_mass = w[idx].groupby(df.loc[idx, "area"]).sum()
        assert np.allclose(area_mass.iloc[0], area_mass.iloc[1])


def test_processed_exceedance_preserves_raw_upstream_value_but_clips_planning_field():
    raw = pd.DataFrame(
        {
            "tile_id": [0, 1],
            "city": ["test", "test"],
            "area": ["A", "A"],
            "lat": [25.0, 25.001],
            "lon": [-80.0, -80.001],
            "temperature_c": [35.0, 36.0],
            "exceedance_h": [-0.7, 2.0],
            "population": [10.0, 10.0],
            "canopy_fraction": [0.2, 0.2],
            "pervious_fraction": [0.2, 0.2],
            "impervious_fraction": [0.6, 0.6],
            "building_fraction": [0.3, 0.3],
            "road_fraction": [0.2, 0.2],
        }
    )
    out, _ = build_processed_table(raw)
    assert out.loc[0, "source_exceedance_h"] == -0.7
    assert out.loc[0, "exceedance_h"] == 0.0
    assert out["baseline_person_hours"].min() >= 0


def test_generalization_protocol_preregisters_three_blind_cities():
    cfg = generalization_config()
    assert cfg["development_cities"] == ["miami", "houston"]
    assert cfg["blind_cities"] == ["phoenix", "atlanta", "los_angeles"]
    assert cfg["thermal_context"]["raw_lat_lon_allowed"] is False
    assert cfg["thermal_context"]["universal_temperature_prediction_claim"] is False
    assert cfg["satellite_sampling"]["samples_per_area"] == 15

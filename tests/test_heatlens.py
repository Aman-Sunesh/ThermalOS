import pandas as pd

from thermalos.features.heatlens import add_heatlens_features, robust_unit_interval


def test_robust_unit_interval_bounds():
    x = robust_unit_interval(pd.Series([1, 2, 3, 100]))
    assert x.between(0, 1).all()


def test_heatlens_positive_and_equity():
    df = pd.DataFrame({
        "temperature_c": [30, 34],
        "exceedance_h": [2, 8],
        "population": [20, 20],
        "vulnerability": [0.2, 0.8],
    })
    out = add_heatlens_features(df)
    assert (out["baseline_person_hours"] > 0).all()
    assert out.loc[1, "equity_weighted_person_hours"] > out.loc[1, "baseline_person_hours"]
    assert bool(out.loc[1, "high_vulnerability"])

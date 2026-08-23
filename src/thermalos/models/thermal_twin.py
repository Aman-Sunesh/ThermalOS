from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline


# Legacy absolute-temperature feature set. Kept only for backward compatibility.
# New cross-city work should use UniversalThermalTwin below.
DEFAULT_FEATURES = [
    "canopy_fraction",
    "pervious_fraction",
    "impervious_fraction",
    "building_fraction",
    "road_fraction",
    "relative_humidity_pct",
    "solar_ghi",
    "lat",
    "lon",
]

# v2 deliberately excludes raw latitude/longitude and any tile-level feature that
# is directly derived from the temperature target (for example exceedance_h).
# Use the same FortyGuard satellite-derived morphology schema in every city.
# City-local planning layers (for example Miami-Dade official tree canopy) may
# override the canonical planning columns, but must never silently change the
# feature definition used by the cross-city observational model.
PORTABLE_MORPHOLOGY_FEATURES = [
    "fg_canopy_fraction",
    "fg_vegetation_fraction",
    "fg_pervious_fraction",
    "fg_impervious_fraction",
    "fg_building_fraction",
    "fg_road_fraction",
]

PORTABLE_WEATHER_FEATURES = [
    # Independent AOI/event reference temperature from the environmental
    # endpoint.  This is context, not a tile-level target-derived feature.
    "area_reference_temperature_c",
    "env_relative_humidity_pct",
    "env_apparent_temperature_c",
    "env_wet_bulb_c",
    "env_heat_index_c",
    "precipitation_mm",
    "env_cloud_cover_context",
    "solar_ghi",
    "solar_dni",
    "solar_dhi",
]

PORTABLE_BASE_FEATURES = PORTABLE_MORPHOLOGY_FEATURES + PORTABLE_WEATHER_FEATURES


@dataclass
class ModelMetrics:
    mae: float
    rmse: float
    r2: float
    n: int


@dataclass
class TransferMetrics:
    """Metrics for the portable local-thermal-anomaly task.

    The primary target is temperature relative to the observed area/event
    anchor. This avoids asking a morphology model to memorize city climate.
    """

    anomaly_mae_c: float
    anomaly_rmse_c: float
    anomaly_r2: float
    anomaly_spearman: float | None
    top20_hotspot_recall: float
    top20_hotspot_jaccard: float
    anchored_absolute_mae_c: float
    anchored_absolute_rmse_c: float
    n: int
    areas: int

    def to_dict(self) -> dict:
        return asdict(self)


class ObservationalThermalTwin:
    """Legacy associational morphology -> absolute-temperature model.

    Retained for backward compatibility with old notebooks/tests. It is not the
    model used by the prospective multi-state transfer protocol.
    """

    def __init__(self, features: list[str] | None = None) -> None:
        self.features = features or DEFAULT_FEATURES.copy()
        self.features_: list[str] = []
        self.model: Pipeline | None = None

    def _usable(self, df: pd.DataFrame) -> list[str]:
        return [
            c
            for c in self.features
            if c in df.columns and pd.to_numeric(df[c], errors="coerce").notna().sum() > 0
        ]

    def fit(self, df: pd.DataFrame, target: str = "temperature_c") -> "ObservationalThermalTwin":
        self.features_ = self._usable(df)
        if not self.features_:
            raise ValueError("No usable ThermalTwin features")
        y = pd.to_numeric(df[target], errors="coerce")
        mask = y.notna()
        if mask.sum() < 20:
            raise ValueError("Need at least 20 target rows")
        X = df.loc[mask, self.features_].apply(pd.to_numeric, errors="coerce")
        y = y.loc[mask]
        self.model = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        loss="squared_error",
                        max_iter=250,
                        learning_rate=0.05,
                        max_leaf_nodes=15,
                        l2_regularization=0.5,
                        random_state=42,
                    ),
                ),
            ]
        )
        self.model.fit(X, y)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model is not fitted")
        X = df.reindex(columns=self.features_).apply(pd.to_numeric, errors="coerce")
        return self.model.predict(X)

    def evaluate(self, df: pd.DataFrame, target: str = "temperature_c") -> ModelMetrics:
        y = pd.to_numeric(df[target], errors="coerce")
        mask = y.notna()
        pred = self.predict(df.loc[mask])
        yy = y.loc[mask].to_numpy()
        return ModelMetrics(
            mae=float(mean_absolute_error(yy, pred)),
            rmse=float(mean_squared_error(yy, pred) ** 0.5),
            r2=float(r2_score(yy, pred)),
            n=int(mask.sum()),
        )

    def canopy_sensitivity(self, df: pd.DataFrame, delta_fraction: float = 0.10) -> np.ndarray:
        base = self.predict(df)
        cf = df.copy()
        if "canopy_fraction" in cf:
            cf["canopy_fraction"] = (
                pd.to_numeric(cf["canopy_fraction"], errors="coerce").fillna(0.2)
                + delta_fraction
            ).clip(0, 1)
        if "pervious_fraction" in cf:
            cf["pervious_fraction"] = (
                pd.to_numeric(cf["pervious_fraction"], errors="coerce").fillna(0.25)
                + 0.5 * delta_fraction
            ).clip(0, 1)
        if "impervious_fraction" in cf:
            cf["impervious_fraction"] = (
                pd.to_numeric(cf["impervious_fraction"], errors="coerce").fillna(0.55)
                - delta_fraction
            ).clip(0, 1)
        after = self.predict(cf)
        return np.clip(base - after, 0.0, None)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> "ObservationalThermalTwin":
        return joblib.load(path)


def _group_columns(df: pd.DataFrame) -> list[str]:
    """Return the finest stable event-area grouping available in a tile table."""
    if "city" in df.columns and "area" in df.columns:
        return ["city", "area"]
    if "area" in df.columns:
        return ["area"]
    if "city" in df.columns:
        return ["city"]
    return []


def _group_transform(series: pd.Series, df: pd.DataFrame, fn: str) -> pd.Series:
    groups = _group_columns(df)
    if not groups:
        if fn == "median":
            return pd.Series(float(series.median()), index=series.index, dtype=float)
        if fn == "rank":
            return series.rank(pct=True, method="average")
        raise ValueError(fn)
    temp = df[groups].copy()
    temp["__value"] = series
    gb = temp.groupby(groups, dropna=False)["__value"]
    if fn == "median":
        return gb.transform("median")
    if fn == "rank":
        return gb.rank(pct=True, method="average")
    raise ValueError(fn)


def add_portable_relative_features(
    df: pd.DataFrame,
    *,
    base_features: Iterable[str] = PORTABLE_BASE_FEATURES,
) -> pd.DataFrame:
    """Create geography-relative urban-form features for transfer learning.

    For morphology variables we retain the raw planning proxy and add:
      * delta from the area's median morphology;
      * within-area percentile rank.

    These quantities are much more portable than raw coordinates. No raw
    latitude/longitude is included in the returned feature matrix.
    """

    out = pd.DataFrame(index=df.index)
    requested = list(base_features)
    for col in requested:
        source_col = col
        # Legacy/demo tables predate the dedicated fg_* schema.  Keep them
        # usable for unit tests and diagnostics, but production freeze/blind
        # scripts explicitly require the source-consistent fg_* columns.
        if source_col not in df.columns and col.startswith("fg_"):
            fallback = col[3:]
            if fallback in df.columns:
                source_col = fallback
        if source_col not in df.columns:
            continue
        x = pd.to_numeric(df[source_col], errors="coerce")
        if not x.notna().any():
            continue
        out[col] = x
        if col in PORTABLE_MORPHOLOGY_FEATURES:
            med = _group_transform(x, df, "median")
            out[f"{col}__delta_area"] = x - med
            out[f"{col}__rank_area"] = _group_transform(x, df, "rank")
    return out


def area_temperature_anchor(
    df: pd.DataFrame,
    *,
    target: str = "temperature_c",
    allow_target_median_fallback: bool = True,
) -> pd.Series:
    """Observed area/event anchor used to separate climate from local anomaly.

    The prospective protocol prefers ``area_reference_temperature_c`` derived
    from FortyGuard's environmental endpoint.  That anchor is independent of
    the held-out tile-temperature labels.  ``area_temperature_anchor_c`` is
    accepted as an explicit operator-supplied equivalent.  A target-median
    fallback remains only for legacy diagnostics/tests and is rejected by the
    production freeze/blind scripts.
    """
    for col in ("area_reference_temperature_c", "area_temperature_anchor_c"):
        if col in df.columns:
            supplied = pd.to_numeric(df[col], errors="coerce")
            if supplied.notna().any():
                return supplied
    if not allow_target_median_fallback:
        raise ValueError(
            "Missing independent area/event temperature anchor. Expected "
            "area_reference_temperature_c (preferred) or area_temperature_anchor_c."
        )
    if target not in df.columns:
        raise ValueError(
            "No temperature target or independent area/event anchor supplied."
        )
    y = pd.to_numeric(df[target], errors="coerce")
    return _group_transform(y, df, "median")


def temperature_anomaly(
    df: pd.DataFrame,
    *,
    target: str = "temperature_c",
) -> pd.Series:
    y = pd.to_numeric(df[target], errors="coerce")
    return y - area_temperature_anchor(df, target=target)


def city_area_balanced_weights(df: pd.DataFrame) -> np.ndarray:
    """Give each development city equal total weight and each area equal weight.

    This prevents the largest city/AOI from dominating the global model.
    Returned weights are normalized to mean one for stable estimator behavior.
    """

    n = len(df)
    if n == 0:
        return np.array([], dtype=float)

    if "city" not in df.columns and "area" not in df.columns:
        return np.ones(n, dtype=float)

    city_values = (
        df["city"].fillna("unknown_city").astype(str)
        if "city" in df.columns
        else pd.Series("single_city", index=df.index)
    )
    area_values = (
        df["area"].fillna("unknown_area").astype(str)
        if "area" in df.columns
        else pd.Series("single_area", index=df.index)
    )

    keys = pd.DataFrame({"city": city_values, "area": area_values}, index=df.index)
    weights = pd.Series(0.0, index=df.index, dtype=float)
    cities = list(keys["city"].drop_duplicates())
    for city in cities:
        cidx = keys.index[keys["city"] == city]
        city_areas = list(keys.loc[cidx, "area"].drop_duplicates())
        for area in city_areas:
            aidx = keys.index[(keys["city"] == city) & (keys["area"] == area)]
            if len(aidx):
                weights.loc[aidx] = 1.0 / (len(cities) * len(city_areas) * len(aidx))
    arr = weights.to_numpy(dtype=float)
    mean = float(np.mean(arr))
    return arr / mean if mean > 0 else np.ones(n, dtype=float)


def _hotspot_scores(y_true: np.ndarray, y_pred: np.ndarray, fraction: float = 0.20) -> tuple[float, float]:
    n = len(y_true)
    if n == 0:
        return 0.0, 0.0
    k = max(1, int(np.ceil(n * fraction)))
    true_idx = set(np.argsort(y_true)[-k:].tolist())
    pred_idx = set(np.argsort(y_pred)[-k:].tolist())
    inter = len(true_idx & pred_idx)
    union = len(true_idx | pred_idx)
    return float(inter / k), float(inter / union if union else 1.0)


def _make_estimator(kind: str, seed: int) -> Pipeline:
    if kind == "hist_gradient_boosting":
        model = HistGradientBoostingRegressor(
            loss="squared_error",
            max_iter=350,
            learning_rate=0.04,
            max_leaf_nodes=15,
            min_samples_leaf=20,
            l2_regularization=1.0,
            random_state=seed,
        )
    elif kind == "extra_trees":
        model = ExtraTreesRegressor(
            n_estimators=350,
            max_features=0.80,
            min_samples_leaf=5,
            random_state=seed,
            n_jobs=1,
        )
    else:
        raise ValueError(f"Unsupported estimator kind: {kind}")
    return Pipeline([("impute", SimpleImputer(strategy="median")), ("model", model)])


class UniversalThermalTwin:
    """Portable observational ThermalTwin v2.

    The model predicts *local thermal anomaly* rather than absolute city
    temperature. Absolute temperature is decomposed as:

        T(tile) = observed area/event anchor + learned local anomaly.

    The learned component sees urban morphology and weather context but never
    raw latitude/longitude. This makes the transfer task about urban-form
    structure rather than memorizing city climate or geography.

    This remains an observational model. It is not an intervention causal
    estimator and must not be used to claim causal cooling effects.
    """

    VALID_ESTIMATORS = {"hist_gradient_boosting", "extra_trees", "ensemble"}

    def __init__(
        self,
        *,
        estimator: str = "ensemble",
        base_features: list[str] | None = None,
        random_state: int = 42,
    ) -> None:
        if estimator not in self.VALID_ESTIMATORS:
            raise ValueError(f"estimator must be one of {sorted(self.VALID_ESTIMATORS)}")
        self.estimator = estimator
        self.base_features = base_features or PORTABLE_BASE_FEATURES.copy()
        self.random_state = int(random_state)
        self.features_: list[str] = []
        self.models_: list[Pipeline] = []
        self.development_cities_: list[str] = []
        self.target_definition_ = "temperature_c minus independent observed AOI/event reference temperature"
        self.anchor_definition_ = (
            "FortyGuard environmental area_reference_temperature_c; "
            "legacy target-median fallback is not allowed in the production freeze/blind protocol"
        )

    def _engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        engineered = add_portable_relative_features(df, base_features=self.base_features)
        if self.features_:
            return engineered.reindex(columns=self.features_)
        return engineered

    def fit(self, df: pd.DataFrame, target: str = "temperature_c") -> "UniversalThermalTwin":
        if target not in df.columns:
            raise ValueError(f"Missing target column: {target}")
        y = temperature_anomaly(df, target=target)
        X = add_portable_relative_features(df, base_features=self.base_features)
        mask = y.notna()
        if int(mask.sum()) < 40:
            raise ValueError("Need at least 40 finite target rows for UniversalThermalTwin")
        if X.shape[1] == 0:
            raise ValueError("No usable portable ThermalTwin features")
        self.features_ = list(X.columns)
        X = X.loc[mask, self.features_]
        yy = y.loc[mask].to_numpy(dtype=float)
        fit_df = df.loc[mask]
        weights = city_area_balanced_weights(fit_df)

        if "city" in fit_df.columns:
            self.development_cities_ = sorted(fit_df["city"].dropna().astype(str).unique().tolist())
        else:
            self.development_cities_ = []

        kinds = (
            ["hist_gradient_boosting", "extra_trees"]
            if self.estimator == "ensemble"
            else [self.estimator]
        )
        self.models_ = []
        for i, kind in enumerate(kinds):
            pipe = _make_estimator(kind, self.random_state + i)
            pipe.fit(X, yy, model__sample_weight=weights)
            self.models_.append(pipe)
        return self

    def predict_anomaly(self, df: pd.DataFrame) -> np.ndarray:
        if not self.models_:
            raise RuntimeError("UniversalThermalTwin is not fitted")
        X = self._engineer(df)
        preds = np.vstack([m.predict(X) for m in self.models_])
        return np.mean(preds, axis=0)

    def predict_temperature(self, df: pd.DataFrame, *, target: str = "temperature_c") -> np.ndarray:
        anchor = area_temperature_anchor(df, target=target).to_numpy(dtype=float)
        return anchor + self.predict_anomaly(df)

    # Compatibility alias: for the v2 model, predict() returns local anomaly.
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.predict_anomaly(df)

    def evaluate(self, df: pd.DataFrame, target: str = "temperature_c") -> TransferMetrics:
        y_abs = pd.to_numeric(df[target], errors="coerce")
        anchor = area_temperature_anchor(df, target=target)
        y_anom = y_abs - anchor
        pred_anom = self.predict_anomaly(df)
        mask = y_abs.notna().to_numpy() & np.isfinite(pred_anom) & anchor.notna().to_numpy()
        if int(mask.sum()) < 3:
            raise ValueError("Need at least 3 finite rows for transfer evaluation")

        ya = y_anom.to_numpy(dtype=float)[mask]
        pa = np.asarray(pred_anom, dtype=float)[mask]
        y = y_abs.to_numpy(dtype=float)[mask]
        a = anchor.to_numpy(dtype=float)[mask]
        p_abs = a + pa

        rho = spearmanr(ya, pa).statistic
        recall, jaccard = _hotspot_scores(ya, pa, fraction=0.20)
        areas = int(df.loc[mask, "area"].nunique()) if "area" in df.columns else 1
        return TransferMetrics(
            anomaly_mae_c=float(mean_absolute_error(ya, pa)),
            anomaly_rmse_c=float(mean_squared_error(ya, pa) ** 0.5),
            anomaly_r2=float(r2_score(ya, pa)),
            anomaly_spearman=None if np.isnan(rho) else float(rho),
            top20_hotspot_recall=recall,
            top20_hotspot_jaccard=jaccard,
            anchored_absolute_mae_c=float(mean_absolute_error(y, p_abs)),
            anchored_absolute_rmse_c=float(mean_squared_error(y, p_abs) ** 0.5),
            n=int(mask.sum()),
            areas=areas,
        )

    def canopy_sensitivity(self, df: pd.DataFrame, delta_fraction: float = 0.10) -> np.ndarray:
        """Associational local-anomaly sensitivity, not a causal effect."""
        base = self.predict_anomaly(df)
        cf = df.copy()
        if "canopy_fraction" in cf:
            cf["canopy_fraction"] = (
                pd.to_numeric(cf["canopy_fraction"], errors="coerce").fillna(0.2)
                + delta_fraction
            ).clip(0, 1)
        if "pervious_fraction" in cf:
            cf["pervious_fraction"] = (
                pd.to_numeric(cf["pervious_fraction"], errors="coerce").fillna(0.25)
                + 0.5 * delta_fraction
            ).clip(0, 1)
        if "impervious_fraction" in cf:
            cf["impervious_fraction"] = (
                pd.to_numeric(cf["impervious_fraction"], errors="coerce").fillna(0.55)
                - delta_fraction
            ).clip(0, 1)
        after = self.predict_anomaly(cf)
        return np.clip(base - after, 0.0, None)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> "UniversalThermalTwin":
        obj = joblib.load(path)
        if not isinstance(obj, UniversalThermalTwin):
            raise TypeError(f"Expected UniversalThermalTwin, got {type(obj).__name__}")
        return obj

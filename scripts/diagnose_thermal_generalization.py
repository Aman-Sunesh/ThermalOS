from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, spearmanr, wasserstein_distance
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=RuntimeWarning)

CITIES = ["miami", "houston"]
MORPH = [
    "fg_canopy_fraction",
    "fg_vegetation_fraction",
    "fg_pervious_fraction",
    "fg_impervious_fraction",
    "fg_building_fraction",
    "fg_road_fraction",
]
WEATHER = [
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
EARTH_R_M = 6_371_000.0
OUT = Path("outputs/generalization/comprehensive_diagnostics")
OUT.mkdir(parents=True, exist_ok=True)


def coverage_max(frame: pd.DataFrame, options: list[str]) -> pd.Series:
    cols = [c for c in options if c in frame.columns]
    if not cols:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return frame[cols].apply(pd.to_numeric, errors="coerce").max(axis=1, skipna=True)


def add_satellite_morphology(sat: pd.DataFrame) -> pd.DataFrame:
    s = sat.copy()
    s["fg_building_fraction"] = coverage_max(s, ["building_fraction", "seg_building", "seg_house"])
    s["fg_canopy_fraction"] = coverage_max(s, ["canopy_fraction", "seg_tree"])
    s["fg_road_fraction"] = coverage_max(s, ["road_fraction", "seg_road", "seg_road,_route"])
    s["fg_vegetation_fraction"] = coverage_max(
        s, ["vegetation_fraction", "seg_vegetation", "seg_tree", "seg_grass", "seg_plant"]
    )
    s["fg_impervious_fraction"] = coverage_max(
        s,
        [
            "impervious_fraction", "seg_building", "seg_house", "seg_road", "seg_road,_route",
            "seg_pavement", "seg_sidewalk,_pavement", "seg_path",
        ],
    )
    s["fg_pervious_fraction"] = coverage_max(
        s, ["pervious_fraction", "seg_tree", "seg_grass", "seg_plant", "seg_earth,_ground"]
    )
    return s


def haversine_matrix(lat1, lon1, lat2, lon2):
    lat1 = np.radians(np.asarray(lat1, dtype=float))[:, None]
    lon1 = np.radians(np.asarray(lon1, dtype=float))[:, None]
    lat2 = np.radians(np.asarray(lat2, dtype=float))[None, :]
    lon2 = np.radians(np.asarray(lon2, dtype=float))[None, :]
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_R_M * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def group_hotspot(y: np.ndarray, frac: float = 0.20) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    k = max(1, int(math.ceil(len(y) * frac)))
    idx = np.argsort(y)[-k:]
    out = np.zeros(len(y), dtype=int)
    out[idx] = 1
    return out


def rank_pct(x: pd.Series) -> pd.Series:
    return x.rank(method="average", pct=True)


def load_city(city: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    sat_path = Path("data/interim") / f"{city}_satellite_samples.csv"
    tile_path = Path("data/processed") / f"{city}_tiles.csv"
    if not sat_path.exists() or not tile_path.exists():
        raise FileNotFoundError(f"Missing {sat_path} or {tile_path}")
    sat = add_satellite_morphology(pd.read_csv(sat_path))
    tiles = pd.read_csv(tile_path)
    if "city" not in sat:
        sat["city"] = city
    sat["city_key"] = city

    # Area-level context is independent of sample-level target labels.
    ctx_cols = ["area_reference_temperature_c", *WEATHER]
    ctx_cols = [c for c in ctx_cols if c in tiles.columns]
    ctx = tiles.groupby("area", as_index=False)[ctx_cols].median(numeric_only=True)
    sat = sat.merge(ctx, on="area", how="left")
    sat["source_temperature_c"] = pd.to_numeric(sat["source_temperature_c"], errors="coerce")
    sat["point_temperature_c"] = sat["source_temperature_c"]

    # Create spatially support-matched targets from the full observed FortyGuard thermal field.
    sat["radius200_temperature_c"] = np.nan
    sat["voronoi_temperature_c"] = np.nan
    sat["nearest_tile_temperature_c"] = np.nan
    sat["nearest_tile_distance_m"] = np.nan
    sat["radius200_tile_count"] = 0
    sat["voronoi_tile_count"] = 0

    for area, sidx in sat.groupby("area").groups.items():
        sidx = list(sidx)
        ss = sat.loc[sidx]
        tt = tiles[tiles["area"].astype(str) == str(area)].copy()
        if tt.empty:
            continue
        dist = haversine_matrix(
            ss["lat"].to_numpy(), ss["lon"].to_numpy(), tt["lat"].to_numpy(), tt["lon"].to_numpy()
        )
        temp = pd.to_numeric(tt["temperature_c"], errors="coerce").to_numpy(dtype=float)
        # per-sample point / radius targets
        for i, ridx in enumerate(sidx):
            di = dist[i]
            nn = int(np.nanargmin(di))
            sat.loc[ridx, "nearest_tile_temperature_c"] = temp[nn]
            sat.loc[ridx, "nearest_tile_distance_m"] = di[nn]
            mask = (di <= 200.0) & np.isfinite(temp)
            sat.loc[ridx, "radius200_tile_count"] = int(mask.sum())
            if mask.any():
                sat.loc[ridx, "radius200_temperature_c"] = float(np.nanmedian(temp[mask]))
            else:
                sat.loc[ridx, "radius200_temperature_c"] = float(temp[nn])
        # Voronoi aggregate: each thermal tile belongs to nearest genuine morphology sample.
        nearest_sample = np.argmin(dist, axis=0)
        for i, ridx in enumerate(sidx):
            mask = (nearest_sample == i) & np.isfinite(temp)
            sat.loc[ridx, "voronoi_tile_count"] = int(mask.sum())
            if mask.any():
                sat.loc[ridx, "voronoi_temperature_c"] = float(np.nanmedian(temp[mask]))

    # Portable X-only relative features can be computed at inference from the city's sampled morphology.
    for col in MORPH:
        sat[col] = pd.to_numeric(sat[col], errors="coerce")
        med = sat.groupby("area")[col].transform("median")
        sat[f"{col}__delta_area"] = sat[col] - med
        sat[f"{col}__rank_area"] = sat.groupby("area")[col].rank(pct=True, method="average")

    # Physically directed versions: larger means nominally hotter.
    sat["heat_lack_canopy"] = 1.0 - sat["fg_canopy_fraction"]
    sat["heat_lack_vegetation"] = 1.0 - sat["fg_vegetation_fraction"]
    sat["heat_lack_pervious"] = 1.0 - sat["fg_pervious_fraction"]
    sat["heat_impervious"] = sat["fg_impervious_fraction"]
    sat["heat_building"] = sat["fg_building_fraction"]
    sat["heat_road"] = sat["fg_road_fraction"]

    for support in ["point", "radius200", "voronoi"]:
        tcol = f"{support}_temperature_c"
        sat[f"{support}_anomaly_c"] = sat[tcol] - sat["area_reference_temperature_c"]
        sat[f"{support}_rank"] = sat.groupby("area")[tcol].rank(pct=True, method="average")
        sat[f"{support}_hot20"] = 0
        for area, idx in sat.groupby("area").groups.items():
            vals = sat.loc[list(idx), tcol].to_numpy(dtype=float)
            sat.loc[list(idx), f"{support}_hot20"] = group_hotspot(vals)
    return sat, tiles


def balanced_weights(df: pd.DataFrame) -> np.ndarray:
    # Equal total weight per city, then per AOI within city.
    w = pd.Series(0.0, index=df.index)
    cities = list(df["city_key"].unique())
    for c in cities:
        cidx = df.index[df["city_key"] == c]
        areas = list(df.loc[cidx, "area"].unique())
        for a in areas:
            idx = df.index[(df["city_key"] == c) & (df["area"] == a)]
            w.loc[idx] = 1.0 / (len(cities) * len(areas) * len(idx))
    arr = w.to_numpy(float)
    return arr / arr.mean()


def data_quality(samples: pd.DataFrame, tiles: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for (city, area), g in samples.groupby(["city_key", "area"]):
        t = tiles[city]
        tt = t[t["area"].astype(str) == str(area)]
        row = {
            "city": city,
            "area": area,
            "satellite_samples": len(g),
            "unique_coords": len(g.drop_duplicates(["lat", "lon"])),
            "point_temp_mean": g["point_temperature_c"].mean(),
            "point_temp_std": g["point_temperature_c"].std(),
            "point_temp_min": g["point_temperature_c"].min(),
            "point_temp_max": g["point_temperature_c"].max(),
            "anchor_c": g["area_reference_temperature_c"].median(),
            "point_anomaly_mean": g["point_anomaly_c"].mean(),
            "point_anomaly_std": g["point_anomaly_c"].std(),
            "nearest_tile_distance_m_p50": g["nearest_tile_distance_m"].median(),
            "point_vs_nearest_mae_c": np.mean(np.abs(g["point_temperature_c"] - g["nearest_tile_temperature_c"])),
            "radius200_count_p50": g["radius200_tile_count"].median(),
            "voronoi_count_p50": g["voronoi_tile_count"].median(),
            "all_tiles": len(tt),
            "all_tile_temp_std": pd.to_numeric(tt["temperature_c"], errors="coerce").std(),
        }
        for col in MORPH:
            row[f"{col}_unique"] = int(g[col].nunique(dropna=True))
            row[f"{col}_std"] = float(g[col].std())
        rows.append(row)
    return pd.DataFrame(rows)


def shift_table(samples: pd.DataFrame) -> pd.DataFrame:
    rows = []
    a = samples[samples.city_key == "miami"]
    b = samples[samples.city_key == "houston"]
    cols = MORPH + ["point_anomaly_c", "radius200_anomaly_c", "voronoi_anomaly_c"]
    for col in cols:
        x = pd.to_numeric(a[col], errors="coerce").dropna().to_numpy()
        y = pd.to_numeric(b[col], errors="coerce").dropna().to_numpy()
        if len(x) < 2 or len(y) < 2:
            continue
        pooled = math.sqrt((np.var(x, ddof=1) + np.var(y, ddof=1)) / 2)
        ks = ks_2samp(x, y)
        w = wasserstein_distance(x, y)
        rows.append({
            "feature": col,
            "miami_mean": np.mean(x), "houston_mean": np.mean(y),
            "miami_std": np.std(x, ddof=1), "houston_std": np.std(y, ddof=1),
            "standardized_mean_difference_abs": abs(np.mean(x)-np.mean(y))/(pooled+1e-12),
            "ks_stat": ks.statistic, "ks_p": ks.pvalue,
            "wasserstein": w, "wasserstein_over_pooled_sd": w/(pooled+1e-12),
            "houston_inside_miami_range": np.mean((y >= np.min(x)) & (y <= np.max(x))),
            "miami_inside_houston_range": np.mean((x >= np.min(y)) & (x <= np.max(y))),
        })
    return pd.DataFrame(rows).sort_values("standardized_mean_difference_abs", ascending=False)


def correlation_table(samples: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for city, g in samples.groupby("city_key"):
        for support in ["point", "radius200", "voronoi"]:
            for feat in MORPH:
                rho = spearmanr(g[feat], g[f"{support}_anomaly_c"], nan_policy="omit").statistic
                rr = spearmanr(g[feat], g[f"{support}_rank"], nan_policy="omit").statistic
                rows.append({"city":city,"support":support,"feature":feat,"spearman_anomaly":rho,"spearman_rank":rr})
    return pd.DataFrame(rows)


def domain_auc(samples: pd.DataFrame, features: list[str]) -> float:
    # Leave-one-AOI-out domain-classification AUC: how easily morphology reveals city identity.
    scores=[]; truth=[]
    groups = samples[["city_key","area"]].astype(str).agg("::".join, axis=1)
    for held in groups.unique():
        tr = groups != held; te = groups == held
        if samples.loc[tr,"city_key"].nunique() < 2:
            continue
        pipe=Pipeline([("imp",SimpleImputer(strategy="median")),("scale",StandardScaler()),("m",LogisticRegression(max_iter=2000,C=0.5))])
        y=(samples.loc[tr,"city_key"]=="houston").astype(int)
        pipe.fit(samples.loc[tr,features], y)
        scores.extend(pipe.predict_proba(samples.loc[te,features])[:,1].tolist())
        truth.extend((samples.loc[te,"city_key"]=="houston").astype(int).tolist())
    if len(set(truth)) != 2:
        return float("nan")
    auc = float(roc_auc_score(truth, scores))
    return max(auc, 1.0 - auc)  # 0.5=random, 1.0=perfectly city-identifiable regardless of label orientation


def ood_support(samples: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows=[]
    for src,tgt in [("miami","houston"),("houston","miami")]:
        a=samples[samples.city_key==src][features].apply(pd.to_numeric,errors="coerce").copy()
        b=samples[samples.city_key==tgt][features].apply(pd.to_numeric,errors="coerce").copy()
        med=a.median(); a=a.fillna(med); b=b.fillna(med)
        mu=a.mean(); sd=a.std().replace(0,1)
        A=((a-mu)/sd).to_numpy(); B=((b-mu)/sd).to_numpy()
        # Euclidean NN in standardized morphology space.
        d_cross=np.sqrt(((B[:,None,:]-A[None,:,:])**2).sum(axis=2)).min(axis=1)
        d_self=np.sqrt(((A[:,None,:]-A[None,:,:])**2).sum(axis=2)); np.fill_diagonal(d_self,np.inf)
        d_self=d_self.min(axis=1)
        rows.append({"source_city":src,"target_city":tgt,"source_self_nn_median":np.median(d_self),"target_to_source_nn_median":np.median(d_cross),"ood_distance_ratio":np.median(d_cross)/(np.median(d_self)+1e-12),"target_within_source_feature_box":np.mean(np.all((B>=np.nanmin(A,axis=0))&(B<=np.nanmax(A,axis=0)),axis=1))})
    return pd.DataFrame(rows)


def feature_sets(samples: pd.DataFrame) -> dict[str,list[str]]:
    raw=MORPH.copy()
    ranks=[f"{c}__rank_area" for c in MORPH]
    deltas=[f"{c}__delta_area" for c in MORPH]
    rel=raw+ranks+deltas
    weather=[c for c in WEATHER if c in samples and samples[c].notna().any()]
    physics=["heat_lack_canopy","heat_lack_vegetation","heat_lack_pervious","heat_impervious","heat_building","heat_road"]
    return {"raw":raw,"rank_only":ranks,"relative":rel,"relative_weather":rel+weather,"physics":physics}


def reg_models():
    return {
        "ridge": Pipeline([("imp",SimpleImputer(strategy="median")),("scale",StandardScaler()),("m",Ridge(alpha=3.0))]),
        "positive_linear": Pipeline([("imp",SimpleImputer(strategy="median")),("scale",StandardScaler()),("m",LinearRegression(positive=True))]),
        "extra_trees": Pipeline([("imp",SimpleImputer(strategy="median")),("m",ExtraTreesRegressor(n_estimators=60,max_features=0.8,min_samples_leaf=2,random_state=42,n_jobs=1))]),
        "hist_gb": Pipeline([("imp",SimpleImputer(strategy="median")),("m",HistGradientBoostingRegressor(max_iter=100,learning_rate=0.04,max_leaf_nodes=7,min_samples_leaf=5,l2_regularization=2.0,random_state=42))]),
    }


def clf_models():
    return {
        "logistic": Pipeline([("imp",SimpleImputer(strategy="median")),("scale",StandardScaler()),("m",LogisticRegression(max_iter=2000,C=0.5,class_weight="balanced"))]),
        "extra_trees_clf": Pipeline([("imp",SimpleImputer(strategy="median")),("m",ExtraTreesClassifier(n_estimators=60,max_features=0.8,min_samples_leaf=2,class_weight="balanced",random_state=42,n_jobs=1))]),
    }


def prediction_metrics(test: pd.DataFrame, scores: np.ndarray, support: str) -> dict:
    d=test.copy(); d["__score"]=scores
    spears=[]; recalls=[]; jaccs=[]; pairs=[]; rank_maes=[]
    for _,g in d.groupby(["city_key","area"]):
        if len(g)<3: continue
        y=g[f"{support}_temperature_c"].to_numpy(float); p=g["__score"].to_numpy(float)
        rho=spearmanr(y,p).statistic
        if np.isfinite(rho): spears.append(float(rho))
        true=group_hotspot(y); pred=group_hotspot(p); inter=int(((true==1)&(pred==1)).sum()); union=int(((true==1)|(pred==1)).sum())
        recalls.append(inter/max(1,int(true.sum()))); jaccs.append(inter/max(1,union))
        yr=pd.Series(y).rank(pct=True,method="average").to_numpy(); pr=pd.Series(p).rank(pct=True,method="average").to_numpy(); rank_maes.append(float(np.mean(np.abs(yr-pr))))
        good=tot=0
        for i in range(len(y)):
            for j in range(i+1,len(y)):
                if y[i]==y[j] or p[i]==p[j]: continue
                tot+=1; good+=int((y[i]-y[j])*(p[i]-p[j])>0)
        if tot: pairs.append(good/tot)
    return {
        "macro_spearman": float(np.mean(spears)) if spears else np.nan,
        "macro_top20_recall": float(np.mean(recalls)) if recalls else np.nan,
        "macro_top20_jaccard": float(np.mean(jaccs)) if jaccs else np.nan,
        "macro_pairwise_accuracy": float(np.mean(pairs)) if pairs else np.nan,
        "macro_rank_mae": float(np.mean(rank_maes)) if rank_maes else np.nan,
    }


def make_folds(samples: pd.DataFrame, protocol: str):
    if protocol=="cross_city":
        for held in CITIES:
            yield f"held_city::{held}", samples.city_key!=held, samples.city_key==held
    elif protocol=="within_city_loao":
        for city in CITIES:
            sub=samples.city_key==city
            for area in samples.loc[sub,"area"].unique():
                te=sub & (samples.area==area); tr=sub & ~te
                yield f"{city}::{area}", tr, te
    elif protocol=="all_loao":
        for city in CITIES:
            for area in samples.loc[samples.city_key==city,"area"].unique():
                te=(samples.city_key==city)&(samples.area==area); tr=~te
                yield f"{city}::{area}", tr, te
    else: raise ValueError(protocol)


def fit_with_optional_weight(model, X, y, weights):
    try:
        model.fit(X,y,m__sample_weight=weights)
    except Exception:
        try: model.fit(X,y,sample_weight=weights)
        except Exception: model.fit(X,y)
    return model


def benchmark(samples: pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame]:
    fsets=feature_sets(samples); regs=reg_models(); clfs=clf_models(); rows=[]
    supports=["point","radius200","voronoi"]

    # Stage 1: strict cross-city benchmark for every candidate. This is the
    # primary method-selection diagnostic and is intentionally run first.
    configs=[]
    for support in supports:
        for fs_name, feats in fsets.items():
            for model_name, template in regs.items():
                if model_name=="positive_linear" and fs_name!="physics": continue
                for target_kind in ["anomaly","rank"]:
                    configs.append((support,fs_name,feats,model_name,template,target_kind,"reg"))
            for model_name,template in clfs.items():
                configs.append((support,fs_name,feats,model_name,template,"hot20_classifier","clf"))

    def run_config(cfg, protocol):
        support,fs_name,feats,model_name,template,target_kind,kind=cfg
        local=[]
        for fold,tr,te in make_folds(samples,protocol):
            train=samples.loc[tr].copy(); test=samples.loc[te].copy()
            if kind=="reg":
                target_col=f"{support}_anomaly_c" if target_kind=="anomaly" else f"{support}_rank"
                y=pd.to_numeric(train[target_col],errors="coerce"); ok=y.notna(); train=train.loc[ok]; y=y.loc[ok]
            else:
                y=train[f"{support}_hot20"].astype(int)
                if y.nunique()<2: continue
            if len(train)<10 or len(test)<3: continue
            model=clone(template); w=balanced_weights(train)
            try:
                fit_with_optional_weight(model,train[feats],y,w)
                if kind=="clf": pred=np.asarray(model.predict_proba(test[feats])[:,1],dtype=float)
                else: pred=np.asarray(model.predict(test[feats]),dtype=float)
            except Exception as exc:
                local.append({"support":support,"feature_set":fs_name,"method":model_name,"target_kind":target_kind,"protocol":protocol,"fold":fold,"error":str(exc)})
                continue
            met=prediction_metrics(test,pred,support)
            met["anomaly_mae_c"] = float(np.mean(np.abs(test[f"{support}_anomaly_c"].to_numpy(float)-pred))) if target_kind=="anomaly" else np.nan
            if kind=="clf":
                yy=test[f"{support}_hot20"].astype(int).to_numpy()
                met["average_precision"] = float(average_precision_score(yy,pred)) if len(set(yy))>1 else np.nan
            local.append({"support":support,"feature_set":fs_name,"method":model_name,"target_kind":target_kind,"protocol":protocol,"fold":fold,"n_train":len(train),"n_test":len(test),**met})
        return local

    for cfg in configs:
        rows.extend(run_config(cfg,"cross_city"))

    detail=pd.DataFrame(rows)
    def summarize(frame):
        good=frame[frame.get("macro_spearman",pd.Series(index=frame.index,dtype=float)).notna()].copy()
        keys=["support","feature_set","method","target_kind","protocol"]
        agg=good.groupby(keys,dropna=False).agg(
            folds=("fold","count"), mean_spearman=("macro_spearman","mean"),
            mean_top20_recall=("macro_top20_recall","mean"), mean_top20_jaccard=("macro_top20_jaccard","mean"),
            mean_pairwise_accuracy=("macro_pairwise_accuracy","mean"), mean_rank_mae=("macro_rank_mae","mean"),
            mean_anomaly_mae_c=("anomaly_mae_c","mean"),
        ).reset_index()
        agg["ranking_composite"] = 0.45*agg.mean_spearman.fillna(0)+0.35*agg.mean_top20_recall.fillna(0)+0.20*agg.mean_pairwise_accuracy.fillna(0)
        return agg

    first=summarize(detail)
    strict=first[first.protocol=="cross_city"].sort_values("ranking_composite",ascending=False)
    # Stage 2: deeper AOI diagnostics only for the top strict-transfer methods.
    top_keys=strict.head(12)[["support","feature_set","method","target_kind"]].to_dict("records")
    lookup={(c[0],c[1],c[3],c[5]):c for c in configs}
    for k in top_keys:
        cfg=lookup.get((k["support"],k["feature_set"],k["method"],k["target_kind"]))
        if cfg is None: continue
        rows.extend(run_config(cfg,"within_city_loao"))
        rows.extend(run_config(cfg,"all_loao"))
    detail=pd.DataFrame(rows)
    agg=summarize(detail).sort_values(["protocol","ranking_composite"],ascending=[True,False]).reset_index(drop=True)
    return detail,agg

def write_report(samples, quality, shift, corr, ood, summary):
    cross=summary[summary.protocol=="cross_city"].sort_values("ranking_composite",ascending=False)
    within=summary[summary.protocol=="within_city_loao"].sort_values("ranking_composite",ascending=False)
    best_cross=cross.iloc[0].to_dict() if len(cross) else {}
    best_within=within.iloc[0].to_dict() if len(within) else {}
    raw_auc=domain_auc(samples,MORPH)
    rank_feats=[f"{c}__rank_area" for c in MORPH]
    rank_auc=domain_auc(samples,rank_feats)
    lines=[]
    lines.append("THERMALOS GENERALIZATION DIAGNOSTIC REPORT")
    lines.append("="*72)
    lines.append(f"Samples: {len(samples)} genuine satellite observations; per city={samples.city_key.value_counts().to_dict()}")
    lines.append(f"Per AOI={samples.groupby(['city_key','area']).size().to_dict()}")
    lines.append(f"City-identifiability separability AUC from raw morphology (LOAO): {raw_auc:.3f}")
    lines.append(f"City-identifiability separability AUC from within-AOI morphology ranks (LOAO): {rank_auc:.3f}")
    lines.append("")
    lines.append("BEST STRICT CROSS-CITY METHOD")
    lines.append(json.dumps(best_cross,indent=2,default=str))
    lines.append("")
    lines.append("BEST WITHIN-CITY LEAVE-ONE-AOI METHOD")
    lines.append(json.dumps(best_within,indent=2,default=str))
    lines.append("")
    lines.append("INTERPRETATION RULES")
    lines.append("- Strong within-city but weak cross-city => domain shift / semantics problem.")
    lines.append("- Weak both within- and cross-city => morphology signal is too weak/noisy at current support.")
    lines.append("- Rank/relative features beating raw => city normalization is useful.")
    lines.append("- Radius/Voronoi targets beating point => spatial support mismatch/noise is important.")
    lines.append("- Linear/physics methods beating trees => prefer simpler invariant structure.")
    lines.append("- Direct hotspot classifier beating regression => separate decision-ranking head is justified.")
    lines.append("")
    lines.append("TOP 15 STRICT CROSS-CITY")
    if len(cross): lines.append(cross.head(15).to_string(index=False))
    lines.append("")
    lines.append("TOP FEATURE SHIFTS")
    lines.append(shift.head(12).to_string(index=False))
    lines.append("")
    lines.append("OOD SUPPORT")
    lines.append(ood.to_string(index=False))
    (OUT/"diagnostic_report.txt").write_text("\n".join(lines),encoding="utf-8")
    rec={
        "best_strict_cross_city":best_cross,
        "best_within_city_loao":best_within,
        "domain_separability_auc_raw":raw_auc,
        "domain_separability_auc_rank":rank_auc,
        "do_not_patch_yet":True,
        "note":"Use diagnostics to choose representation/target/model before modifying production code."
    }
    (OUT/"recommendation.json").write_text(json.dumps(rec,indent=2,default=str),encoding="utf-8")
    return raw_auc,rank_auc,cross


def main():
    frames=[]; tiles={}
    for city in CITIES:
        s,t=load_city(city); frames.append(s); tiles[city]=t
    samples=pd.concat(frames,ignore_index=True)
    # Hard integrity checks for current registered development morphology protocol.
    counts=samples.groupby(["city_key","area"]).size()
    print("=== SAMPLE COUNTS ===")
    print(counts.to_string())
    if len(samples)!=90 or not (counts==15).all():
        print("WARNING: expected current protocol is 15 genuine satellite samples per AOI (90 total). Diagnostics will still run.")

    quality=data_quality(samples,tiles); shift=shift_table(samples); corr=correlation_table(samples); ood=ood_support(samples,MORPH)
    quality.to_csv(OUT/"data_quality_by_aoi.csv",index=False)
    shift.to_csv(OUT/"city_distribution_shift.csv",index=False)
    corr.to_csv(OUT/"feature_target_correlations.csv",index=False)
    ood.to_csv(OUT/"cross_city_feature_support.csv",index=False)
    samples.to_csv(OUT/"genuine_satellite_training_table.csv",index=False)

    print("\n=== DATA QUALITY BY AOI ===")
    print(quality[["city","area","satellite_samples","point_temp_std","point_anomaly_std","point_vs_nearest_mae_c","nearest_tile_distance_m_p50","radius200_count_p50","voronoi_count_p50"]].to_string(index=False))
    print("\n=== LARGEST CITY DISTRIBUTION SHIFTS ===")
    print(shift.head(12).to_string(index=False))
    print("\n=== FEATURE SUPPORT ===")
    print(ood.to_string(index=False))

    print("\n=== BENCHMARKING CANDIDATE METHODS ===")
    detail,summary=benchmark(samples)
    detail.to_csv(OUT/"method_fold_results.csv",index=False)
    summary.to_csv(OUT/"method_summary.csv",index=False)
    raw_auc,rank_auc,cross=write_report(samples,quality,shift,corr,ood,summary)
    print(f"Domain separability AUC raw morphology: {raw_auc:.3f}")
    print(f"Domain separability AUC AOI-rank morphology: {rank_auc:.3f}")
    print("\n=== TOP 15 STRICT CROSS-CITY METHODS ===")
    print(cross.head(15).to_string(index=False))
    print("\nSaved diagnostics ->",OUT)
    print("Files: data_quality_by_aoi.csv, city_distribution_shift.csv, feature_target_correlations.csv,")
    print("       cross_city_feature_support.csv, genuine_satellite_training_table.csv,")
    print("       method_fold_results.csv, method_summary.csv, diagnostic_report.txt, recommendation.json")
    print("\nGENERALIZATION DIAGNOSTICS: COMPLETE (NO PRODUCTION CODE MODIFIED)")

if __name__=="__main__":
    main()

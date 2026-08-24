from __future__ import annotations

import html
import json
from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="ThermalOS",
    page_icon="🌡️",
    layout="wide",
)

import pandas as pd
import pydeck as pdk

from thermalos.analytics.policy_lab import run_policy_stress_lab
from thermalos.analytics.robustness import run_robustness
from thermalos.config import available_cities, city_config, generalization_config, interventions_config
from thermalos.copilot import execute_copilot
from thermalos.evidence import build_evidence_ledger, evidence_summary
from thermalos.explain import explain_selected
from thermalos.generalization import run_cross_city_comparison
from thermalos.learning import adaptive_calibration_status, update_intervention_priors
from thermalos.models.interventions import build_candidates
from thermalos.operations.heatops import plan_heatops
from thermalos.optimization.portfolio import apply_portfolio_to_tiles, marginal_value_curve, optimize_portfolio
from thermalos.reporting.dossier import build_dossier_pdf
from thermalos.verification import build_verification_registry, evaluate_post_deployment

ROOT = Path(__file__).resolve().parent

PROJECT_COLORS: dict[str, list[int]] = {
    "tree_canopy": [52, 168, 83, 245],
    "shade_structure": [66, 133, 244, 245],
    "cool_pavement": [154, 160, 166, 245],
    "cool_roof": [244, 160, 65, 245],
    "cooling_node": [171, 71, 188, 245],
}

MECHANISM_LABELS = {
    "shade_evapotranspiration": "Shade + evapotranspiration",
    "radiant_exposure": "Direct radiant-exposure protection",
    "surface_energy_balance": "Surface-energy / albedo intervention",
    "building_surface_albedo": "Building-surface heat rejection",
    "protected_exposure_capacity": "Protected cooling / hydration capacity",
}


@st.cache_data(show_spinner=False)
def load_city(city: str) -> tuple[pd.DataFrame, dict]:
    processed = ROOT / "data" / "processed" / f"{city}_tiles.csv"
    provenance_path = ROOT / "data" / "processed" / f"{city}_provenance.json"
    sample = ROOT / "data" / "sample" / f"{city}_demo_tiles.csv"

    if processed.exists():
        meta: dict = {"processed": True, "synthetic_demo": False, "enrichment": {}}
        if provenance_path.exists():
            try:
                loaded = json.loads(provenance_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    meta.update(loaded)
            except Exception:
                meta["provenance_warning"] = "Could not parse processed provenance file."
        return pd.read_csv(processed), meta

    if sample.exists():
        return pd.read_csv(sample), {
            "processed": False,
            "synthetic_demo": True,
            "enrichment": {},
            "source_file": str(sample),
        }

    raise FileNotFoundError("Run `python scripts/build_demo_dataset.py` first.")


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)
        return out if pd.notna(out) else default
    except (TypeError, ValueError):
        return default


def color_from_score(s: pd.Series) -> list[list[int]]:
    x = pd.to_numeric(s, errors="coerce").fillna(0.5).clip(0, 1)
    # Warm HeatLens ramp: yellow/olive -> orange/red as stress increases.
    return [[int(88 + 167 * v), int(174 - 112 * v), 52, 150] for v in x]


def enrich_selected_for_map(selected: pd.DataFrame, tiles: pd.DataFrame) -> pd.DataFrame:
    if selected.empty:
        return selected.copy()

    optional_context = [
        "temperature_c",
        "exceedance_h",
        "canopy_fraction",
        "transit_stop_count",
        "exposure_multiplier",
        "population",
        "impervious_fraction",
        "building_fraction",
        "road_fraction",
        "thermal_stress_index",
        "poverty_fraction",
        "no_vehicle_fraction",
    ]
    context_cols = ["tile_id"] + (["source_tile_id"] if "source_tile_id" in tiles.columns else []) + [c for c in optional_context if c in tiles.columns]
    context = tiles[context_cols].drop_duplicates("tile_id")
    out = selected.merge(context, on="tile_id", how="left", validate="many_to_one")

    out["marker_color"] = out["intervention"].map(PROJECT_COLORS).apply(
        lambda x: x if isinstance(x, list) else [235, 235, 235, 245]
    )
    out["tooltip_title"] = out["label"].astype(str)
    out["tooltip_line1"] = out.apply(
        lambda r: f"{display_area(r.get('area', ''))} • Cost ${safe_float(r.get('cost_usd')):,.0f}", axis=1
    )
    out["tooltip_line2"] = out.apply(
        lambda r: f"Expected avoided burden: {safe_float(r.get('benefit_expected_person_hours')):,.0f} person-hours",
        axis=1,
    )
    out["tooltip_line3"] = out["reason"].fillna("Click to inspect why this project was selected.").astype(str)
    return out


def build_map_deck(
    tiles: pd.DataFrame,
    project_map: pd.DataFrame,
    active_candidate_id: str | None,
) -> pdk.Deck:
    base = tiles.copy()
    base["color"] = color_from_score(base["thermal_stress_index"])
    base["tooltip_title"] = base["area"].astype(str) + " • HeatLens tile"
    base["tooltip_line1"] = base.apply(
        lambda r: f"Thermal stress: {safe_float(r.get('thermal_stress_index')):.2f} • Temperature: {safe_float(r.get('temperature_c')):.1f} °C",
        axis=1,
    )
    base["tooltip_line2"] = base.apply(
        lambda r: f"Hot duration: {safe_float(r.get('exceedance_h')):.1f} h • Vulnerability score: {safe_float(r.get('vulnerability')):.2f} / 1.00",
        axis=1,
    )
    base["tooltip_line3"] = base.apply(
        lambda r: f"Canopy: {100 * safe_float(r.get('canopy_fraction')):.0f}% • Baseline burden: {safe_float(r.get('baseline_person_hours')):,.0f} person-hours",
        axis=1,
    )

    layers: list[pdk.Layer] = [
        pdk.Layer(
            "GridCellLayer",
            id="thermal-tiles",
            data=base,
            get_position="[lon, lat]",
            get_fill_color="color",
            cell_size=220,
            extruded=False,
            pickable=True,
            auto_highlight=True,
            opacity=0.58,
        )
    ]

    if not project_map.empty:
        projects = project_map.copy()
        projects["marker_radius"] = projects["candidate_id"].apply(
            lambda cid: 92 if cid == active_candidate_id else 68
        )
        projects["marker_line_color"] = projects["candidate_id"].apply(
            lambda cid: [255, 218, 87, 255] if cid == active_candidate_id else [255, 255, 255, 245]
        )
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                id="selected-projects",
                data=projects,
                get_position="[lon, lat]",
                get_radius="marker_radius",
                radius_min_pixels=6,
                radius_max_pixels=15,
                get_fill_color="marker_color",
                get_line_color="marker_line_color",
                line_width_min_pixels=2,
                stroked=True,
                filled=True,
                pickable=True,
                auto_highlight=True,
            )
        )

    return pdk.Deck(
        map_style=None,
        layers=layers,
        initial_view_state=pdk.ViewState(
            latitude=float(base["lat"].mean()),
            longitude=float(base["lon"].mean()),
            zoom=10.2,
            pitch=0,
            controller=True,
        ),
        tooltip={
            "html": "<b>{tooltip_title}</b><br/>{tooltip_line1}<br/>{tooltip_line2}<br/><span style='opacity:.82'>{tooltip_line3}</span>",
            "style": {"backgroundColor": "#17191d", "color": "white", "fontSize": "12px"},
        },
    )


def clicked_project_id(event: object) -> str | None:
    """Return candidate_id from a PyDeck selection event, if a project marker was clicked."""
    try:
        selection = getattr(event, "selection", None)
        objects = getattr(selection, "objects", None)
        if objects is None and isinstance(selection, dict):
            objects = selection.get("objects", {})
        if objects is None:
            return None
        if not isinstance(objects, dict):
            try:
                objects = dict(objects)
            except Exception:
                return None
        picked = objects.get("selected-projects") or []
        if not picked:
            return None
        candidate_id = picked[0].get("candidate_id")
        return str(candidate_id) if candidate_id else None
    except Exception:
        return None


def map_legend(selected: pd.DataFrame) -> None:
    pieces = [
        "<span style='display:inline-flex;align-items:center;gap:6px;margin-right:14px;'>"
        "<span style='width:14px;height:14px;background:linear-gradient(90deg,#9d9f34,#e04b34);display:inline-block;border-radius:2px;'></span>"
        "HeatLens burden (lower → higher)</span>"
    ]
    if not selected.empty:
        seen = selected[["intervention", "label"]].drop_duplicates()
        for _, row in seen.iterrows():
            rgba = PROJECT_COLORS.get(str(row["intervention"]), [235, 235, 235, 245])
            rgb = f"rgb({rgba[0]},{rgba[1]},{rgba[2]})"
            label = html.escape(INTERVENTION_SHORT_LABELS.get(str(row["intervention"]), str(row["label"])))
            pieces.append(
                "<span style='display:inline-flex;align-items:center;gap:6px;margin-right:14px;'>"
                f"<span style='width:12px;height:12px;background:{rgb};display:inline-block;border-radius:50%;border:1px solid #fff;'></span>"
                f"{label}</span>"
            )
    st.markdown(
        "<div style='font-size:0.84rem;opacity:.88;margin:-4px 0 8px 0;'>" + "".join(pieces) + "</div>",
        unsafe_allow_html=True,
    )


def display_area(value: object) -> str:
    return str(value).replace("_", " ")


def planning_benefit(row: pd.Series, impact_basis: str) -> float:
    column = "benefit_low_person_hours" if impact_basis == "conservative" else "benefit_expected_person_hours"
    return safe_float(row.get(column))


def project_option_label(row: pd.Series, rank: int, impact_basis: str) -> str:
    basis_value = planning_benefit(row, impact_basis)
    return (
        f"#{rank} · {row['label']} · {display_area(row['area'])} · "
        f"${safe_float(row['cost_usd']) / 1000:.0f}k · "
        f"{basis_value:,.0f} person-h"
    )


def vulnerability_band(value: float) -> str:
    if value >= 0.70:
        return "High"
    if value >= 0.50:
        return "Elevated"
    if value >= 0.30:
        return "Moderate"
    return "Lower"



INTERVENTION_SHORT_LABELS = {
    "tree_canopy": "Tree canopy",
    "shade_structure": "Shade structures",
    "cool_pavement": "Cool pavement",
    "cool_roof": "Cool roofs",
    "cooling_node": "Cooling / hydration",
}


def section_header(title: str, subtitle: str | None = None, eyebrow: str | None = None) -> None:
    parts = ["<div class='tos-section-head'>"]
    if eyebrow:
        parts.append(f"<div class='tos-eyebrow'>{html.escape(eyebrow)}</div>")
    parts.append(f"<h2>{html.escape(title)}</h2>")
    if subtitle:
        parts.append(f"<p>{html.escape(subtitle)}</p>")
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def pill(text: str, tone: str = "neutral") -> str:
    return f"<span class='tos-pill tos-pill-{tone}'>{html.escape(text)}</span>"


# -----------------------------------------------------------------------------
# PROFESSIONAL APPLICATION SHELL
# -----------------------------------------------------------------------------
st.markdown(
    """
<style>
:root {
  --tos-bg: #09111f;
  --tos-panel: rgba(15, 23, 42, 0.88);
  --tos-panel-2: rgba(17, 29, 49, 0.78);
  --tos-border: rgba(148, 163, 184, 0.16);
  --tos-border-strong: rgba(148, 163, 184, 0.26);
  --tos-text: #f8fafc;
  --tos-muted: #94a3b8;
  --tos-accent: #38bdf8;
  --tos-green: #34d399;
  --tos-amber: #fbbf24;
}

[data-testid="stAppViewContainer"] {
  background:
    radial-gradient(circle at 77% 3%, rgba(14, 165, 233, 0.075), transparent 31rem),
    radial-gradient(circle at 22% 18%, rgba(52, 211, 153, 0.045), transparent 27rem),
    var(--tos-bg);
}
[data-testid="stHeader"] { background: rgba(9, 17, 31, 0.70); }
.block-container {
  max-width: 1540px;
  padding-top: 1.65rem;
  padding-bottom: 4rem;
  padding-left: 2.15rem;
  padding-right: 2.15rem;
}
[data-testid="stSidebar"] {
  background: #0d1627;
  border-right: 1px solid var(--tos-border);
  min-width: 308px;
  max-width: 308px;
}
[data-testid="stSidebar"] > div:first-child { padding-top: 1.35rem; }
[data-testid="stSidebar"] h2 {
  font-size: 1.02rem !important;
  letter-spacing: -0.01em;
  margin-bottom: .75rem;
}
[data-testid="stSidebar"] label, [data-testid="stSidebar"] p { line-height: 1.35; }
[data-testid="stSidebar"] hr { border-color: var(--tos-border); }

/* Cleaner controls */
[data-baseweb="select"] > div,
[data-baseweb="input"] > div,
[data-testid="stNumberInput"] input {
  border-radius: 10px !important;
}
[data-testid="stExpander"] {
  border: 1px solid var(--tos-border) !important;
  border-radius: 12px !important;
  background: rgba(15, 23, 42, 0.42);
}

/* KPI cards */
[data-testid="stMetric"] {
  background: linear-gradient(180deg, rgba(20, 32, 53, .94), rgba(13, 24, 42, .88));
  border: 1px solid var(--tos-border);
  border-radius: 15px;
  padding: 1rem 1.05rem .9rem;
  min-height: 108px;
  box-shadow: 0 10px 28px rgba(0,0,0,.12);
}
[data-testid="stMetricLabel"] p {
  color: var(--tos-muted) !important;
  font-size: .72rem !important;
  font-weight: 700 !important;
  text-transform: uppercase;
  letter-spacing: .055em;
}
[data-testid="stMetricValue"] {
  font-weight: 750 !important;
  letter-spacing: -0.035em;
}
[data-testid="stMetricDelta"] { font-size: .76rem; }

/* Hero */
.tos-hero {
  padding: .3rem 0 .85rem;
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  gap: 2rem;
}
.tos-hero .tos-brand-kicker {
  color: #7dd3fc;
  text-transform: uppercase;
  font-size: .72rem;
  font-weight: 800;
  letter-spacing: .14em;
  margin-bottom: .35rem;
}
.tos-hero h1 {
  margin: 0;
  font-size: clamp(2.25rem, 4vw, 3.55rem);
  line-height: .98;
  letter-spacing: -.055em;
  font-weight: 790;
}
.tos-hero p {
  color: #a8b4c7;
  max-width: 780px;
  margin: .65rem 0 0;
  font-size: .96rem;
}
.tos-hero-badges { display:flex; gap:.5rem; flex-wrap:wrap; justify-content:flex-end; }

/* Provenance card */
.tos-provenance {
  border-radius: 13px;
  padding: .8rem 1rem;
  margin: .2rem 0 1rem;
  border: 1px solid rgba(251, 191, 36, .30);
  background: rgba(113, 63, 18, .20);
  color: #fde68a;
  display:flex;
  gap:.65rem;
  align-items:flex-start;
  font-size:.84rem;
  line-height:1.45;
}
.tos-provenance.real {
  border-color: rgba(52, 211, 153, .28);
  background: rgba(6, 78, 59, .20);
  color:#a7f3d0;
}
.tos-provenance strong { color: inherit; }

/* Section typography */
.tos-section-head { margin: 1.35rem 0 .65rem; }
.tos-section-head .tos-eyebrow {
  color: #7dd3fc;
  text-transform: uppercase;
  letter-spacing: .11em;
  font-size: .68rem;
  font-weight: 800;
  margin-bottom: .18rem;
}
.tos-section-head h2 {
  font-size: 1.45rem;
  letter-spacing: -.025em;
  margin: 0;
}
.tos-section-head p {
  color: var(--tos-muted);
  font-size: .84rem;
  margin: .25rem 0 0;
}

/* Chips */
.tos-chip-row { display:flex; flex-wrap:wrap; gap:.45rem; margin:.7rem 0 .4rem; }
.tos-pill {
  display:inline-flex;
  align-items:center;
  border-radius:999px;
  padding:.34rem .62rem;
  border:1px solid var(--tos-border);
  background:rgba(30,41,59,.62);
  color:#cbd5e1;
  font-size:.74rem;
  font-weight:650;
}
.tos-pill-blue { color:#bae6fd; border-color:rgba(56,189,248,.28); background:rgba(12,74,110,.28); }
.tos-pill-green { color:#bbf7d0; border-color:rgba(52,211,153,.27); background:rgba(6,78,59,.26); }
.tos-pill-amber { color:#fde68a; border-color:rgba(251,191,36,.28); background:rgba(113,63,18,.24); }

/* Card-like bordered containers */
[data-testid="stVerticalBlockBorderWrapper"] {
  border-color: var(--tos-border) !important;
  border-radius: 15px !important;
  background: linear-gradient(180deg, rgba(15,23,42,.72), rgba(11,20,35,.62));
}

/* Dataframes + charts */
[data-testid="stDataFrame"] {
  border: 1px solid var(--tos-border);
  border-radius: 12px;
  overflow: hidden;
}
[data-testid="stPlotlyChart"], [data-testid="stArrowVegaLiteChart"] {
  border-radius: 12px;
}

/* Tabs */
button[data-baseweb="tab"] {
  font-weight: 700 !important;
  padding-left: 1rem !important;
  padding-right: 1rem !important;
}

/* Small utility text */
.tos-mini { color:var(--tos-muted); font-size:.78rem; line-height:1.45; }
.tos-divider { height:1px; background:var(--tos-border); margin:.5rem 0 1rem; }

@media (max-width: 1050px) {
  .block-container { padding-left: 1rem; padding-right: 1rem; }
  .tos-hero { align-items:flex-start; flex-direction:column; }
  .tos-hero-badges { justify-content:flex-start; }
}
</style>
""",
    unsafe_allow_html=True,
)

cfg = interventions_config()
intervention_specs = cfg["interventions"]

# City selection first because policy limits depend on the loaded planning areas.
# Only show cities that already have processed or explicit demo data; blind-city
# configs can exist without exposing their outcomes before the frozen evaluation.
configured_cities = available_cities()
selectable_cities = [
    c for c in configured_cities
    if (ROOT / "data" / "processed" / f"{c}_tiles.csv").exists()
    or (ROOT / "data" / "sample" / f"{c}_demo_tiles.csv").exists()
]
city_labels = {}
evaluation_labels = {
    "miami": "development",
    "houston": "development",
    "phoenix": "clean prospective blind",
    "atlanta": "post-blind v3.1 replay",
    "los_angeles": "post-blind v3.1 replay",
    "las_vegas": "post-blind follow-up",
}

for c in selectable_cities:
    ccfg = city_config(c)
    role = str(ccfg.get("transfer_role", "unspecified"))
    suffix = evaluation_labels.get(c, role.replace("_", " "))
    city_labels[c] = f"{ccfg.get('name', c)} ({suffix})"

with st.sidebar:
    st.markdown("### Planning scenario")
    city = st.selectbox(
        "City",
        selectable_cities,
        format_func=lambda c: city_labels.get(c, c),
        index=selectable_cities.index("miami") if "miami" in selectable_cities else 0,
    )

try:
    tiles, provenance = load_city(city)
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

if "baseline_person_hours" not in tiles:
    st.error(
        "Processed HeatLens features are missing. Run `python scripts/build_city_features.py --city %s`." % city
    )
    st.stop()

available_areas = sorted(tiles["area"].dropna().astype(str).unique().tolist()) if "area" in tiles else []
max_neighborhoods = max(1, len(available_areas))

with st.sidebar:
    budget_m = st.slider("Capital budget", 0.5, 5.0, 2.0, 0.1, format="$%.1fM")
    budget = int(round(budget_m * 1_000_000))

    objective_label = st.selectbox(
        "Planning objective",
        ["Balanced", "Maximum Impact", "Maximum Equity", "Cost Efficiency"],
    )
    objective = objective_label.lower().replace(" ", "_")

    impact_basis_label = st.radio(
        "Optimization basis",
        ["Expected impact", "Conservative impact"],
        index=0,
        help=(
            "Expected impact optimizes the modeled mean effect. Conservative impact optimizes the lower-bound effect estimate."
        ),
    )
    impact_basis = "conservative" if impact_basis_label.startswith("Conservative") else "expected"

    equity_pct = st.slider(
        "Minimum equity-aligned spend",
        0,
        70,
        35,
        5,
        format="%d%%",
        help="Minimum share of deployed capital required in high-vulnerability tiles.",
    )
    equity_min = equity_pct / 100.0

    with st.expander("Policy guardrails", expanded=False):
        max_neighborhood_pct = st.slider(
            "Max budget / neighborhood",
            20,
            100,
            100,
            5,
            format="%d%%",
            help="Cap any one neighborhood at this share of the total available budget.",
        )
        max_intervention_pct = st.slider(
            "Max budget / intervention",
            20,
            100,
            100,
            5,
            format="%d%%",
            help="Cap any one intervention family at this share of the total available budget.",
        )
        min_neighborhoods = st.slider(
            "Minimum neighborhoods served",
            1,
            max_neighborhoods,
            1,
            1,
        )
    max_neighborhood_fraction = max_neighborhood_pct / 100.0
    max_intervention_fraction = max_intervention_pct / 100.0

    with st.expander("Intervention palette", expanded=True):
        all_int = list(intervention_specs.keys())
        enabled = [
            name
            for name in all_int
            if st.checkbox(
                INTERVENTION_SHORT_LABELS.get(name, intervention_specs.get(name, {}).get("label", name)),
                value=True,
                key=f"enabled_{city}_{name}",
            )
        ]
        st.caption(f"{len(enabled)} of {len(all_int)} intervention families enabled")

    st.divider()
    st.caption(
        "Offline-first decision support. Live FortyGuard refresh is a separate data-harvesting step so the demo never waits on API latency."
    )

if not enabled:
    st.error("Enable at least one intervention family in the sidebar.")
    st.stop()

synthetic_demo = bool(provenance.get("synthetic_demo", False))
enrichment = provenance.get("enrichment", {}) if isinstance(provenance.get("enrichment", {}), dict) else {}
real_layers = [
    label
    for key, label in [
        ("tree_canopy", "official/local canopy GIS"),
        ("census_blockgroup_geography_2024", "2024 Census geography"),
        ("acs_blockgroups", "ACS population"),
        ("acs_tracts", "ACS vulnerability"),
        ("gtfs", "GTFS transit"),
        ("environmental_samples", "FortyGuard environment"),
        ("satellite_samples", "FortyGuard satellite"),
    ]
    if enrichment.get(key)
]

mode_badge = pill("Hybrid demo", "amber") if synthetic_demo else pill("Real planning data", "green")
city_cfg = city_config(city)
city_role = str(city_cfg.get("transfer_role", provenance.get("transfer_role", "unspecified")))

evaluation_labels = {
    "miami": "Development city",
    "houston": "Development city",
    "phoenix": "Clean prospective blind",
    "atlanta": "Post-blind v3.1 replay",
    "los_angeles": "Post-blind v3.1 replay",
    "las_vegas": "Post-blind follow-up",
}

role_label = evaluation_labels.get(
    city,
    city_role.replace("_", " ").title(),
)

city_badge = pill(role_label, "blue")
basis_badge = pill("Conservative basis" if impact_basis == "conservative" else "Expected basis")

st.markdown(
    f"""
<div class="tos-hero">
  <div>
    <div class="tos-brand-kicker">Urban heat decision intelligence</div>
    <h1>ThermalOS</h1>
    <p>Turn hyperlocal heat intelligence into an optimized, explainable capital plan under real budget, equity, feasibility, and policy constraints.</p>
  </div>
  <div class="tos-hero-badges">{mode_badge}{city_badge}{basis_badge}</div>
</div>
""",
    unsafe_allow_html=True,
)

if synthetic_demo:
    layer_text = ", ".join(real_layers) if real_layers else "no external enrichment detected"
    st.markdown(
        f"""
<div class="tos-provenance">
  <div>◈</div>
  <div><strong>Hybrid demo mode.</strong> The thermal field is synthetic while the planning layers are real ({html.escape(layer_text)}). Temperature and burden reductions are scenario outputs, not measured Miami impacts.</div>
</div>
""",
        unsafe_allow_html=True,
    )
else:
    layer_text = " + ".join(["FortyGuard thermal intelligence"] + real_layers)
    st.markdown(
        f"<div class='tos-provenance real'><div>●</div><div><strong>Real planning data mode.</strong> {html.escape(layer_text)}. Intervention outcomes are modeled planning scenarios, not measured causal impacts.</div></div>",
        unsafe_allow_html=True,
    )

# ThermalTwin is an observed-temperature-conditioned urban-form/context layer.
# It is intentionally NOT used to calibrate causal intervention effects, and
# ThermalOS does not claim sparse morphology can replace FortyGuard's thermal field.
candidate_result = build_candidates(tiles, cfg)
candidates = candidate_result.candidates

with st.spinner("Solving constrained capital portfolio…"):
    portfolio = optimize_portfolio(
        candidates,
        budget_usd=budget,
        objective=objective,
        impact_basis=impact_basis,
        equity_min_fraction=equity_min,
        enabled_interventions=enabled,
        max_neighborhood_spend_fraction=max_neighborhood_fraction,
        max_intervention_spend_fraction=max_intervention_fraction,
        min_neighborhoods_served=min_neighborhoods,
    )
    selected = explain_selected(portfolio.selected, tiles)
    counter = apply_portfolio_to_tiles(
        tiles,
        selected,
        max_relief_fraction=float(cfg["model"].get("max_tile_relief_fraction", 0.85)),
        impact_basis=impact_basis,
    )

baseline = float(counter["baseline_person_hours"].sum())
after = float(counter["counterfactual_person_hours"].sum())
avoided = baseline - after
reduction_pct = 100 * avoided / baseline if baseline else 0.0
spent = float(selected["cost_usd"].sum()) if len(selected) else 0.0
vuln_spend = float(selected.loc[selected["high_vulnerability"], "cost_usd"].sum()) if len(selected) else 0.0
vuln_share = 100 * vuln_spend / spent if spent else 0.0
protected_mask = counter["modeled_relief_fraction"] >= 0.10
people_proxy = float(pd.to_numeric(counter.loc[protected_mask, "population"], errors="coerce").fillna(0).sum())

# Executive KPI strip
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Capital budget", f"${budget / 1e6:.1f}M")
m2.metric("Funded projects", f"{len(selected):,}")
m3.metric("Modeled relief", f"{reduction_pct:.1f}%")
m4.metric("Exposure-weighted population*", f"{people_proxy:,.0f}")
m5.metric("Equity-aligned spend", f"{vuln_share:.0f}%")

basis_text = "lower-bound effects" if impact_basis == "conservative" else "expected effects"
policy_tone = "green" if (
    max_neighborhood_pct < 100 or max_intervention_pct < 100 or min_neighborhoods > 1 or equity_pct > 0
) else "neutral"
chips = [
    pill(f"{len(candidates):,} feasible candidates", "blue"),
    pill(f"{basis_text}"),
    pill(f"Equity floor {equity_pct}%", policy_tone),
    pill(f"≤ {max_neighborhood_pct}% / neighborhood", policy_tone),
    pill(f"≤ {max_intervention_pct}% / intervention", policy_tone),
    pill(f"≥ {min_neighborhoods} neighborhood{'s' if min_neighborhoods != 1 else ''}", policy_tone),
]
st.markdown("<div class='tos-chip-row'>" + "".join(chips) + "</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='tos-mini'>*Exposure-weighted population is a planning exposure proxy, not a count of clinically protected people. All burden/effect outputs are scenario estimates rather than causal impact evaluations.</div>",
    unsafe_allow_html=True,
)

if selected.empty and portfolio.summary.get("status") == "infeasible_or_no_solution":
    st.warning(
        "No feasible portfolio satisfies this combination of budget, equity floor, intervention availability, and policy constraints. Relax one or more policy controls."
    )

project_map = enrich_selected_for_map(selected, tiles)
project_ranked = project_map.sort_values("benefit_expected_person_hours", ascending=False).reset_index(drop=True)
project_ranked["portfolio_rank"] = range(1, len(project_ranked) + 1)

active_state_key = f"active_project_{city}"
selector_state_key = f"project_selector_{city}"
pending_state_key = f"pending_map_project_{city}"
valid_project_ids = project_ranked["candidate_id"].astype(str).tolist() if not project_ranked.empty else []

if valid_project_ids:
    pending = st.session_state.pop(pending_state_key, None)
    if pending in valid_project_ids:
        st.session_state[active_state_key] = pending
        st.session_state[selector_state_key] = pending
    if st.session_state.get(active_state_key) not in valid_project_ids:
        st.session_state[active_state_key] = valid_project_ids[0]
    if st.session_state.get(selector_state_key) not in valid_project_ids:
        st.session_state[selector_state_key] = st.session_state[active_state_key]

section_header(
    "Decision map",
    "Explore the thermal burden field, funded interventions, and the logic behind each selected project.",
    "HeatLens + CoolCapital",
)

left, right = st.columns([1.62, 1], gap="large")
with left:
    with st.container(border=True):
        if valid_project_ids:
            label_lookup = {
                str(row["candidate_id"]): project_option_label(row, int(row["portfolio_rank"]), impact_basis)
                for _, row in project_ranked.iterrows()
            }
            chosen_project_id = st.selectbox(
                "Selected project",
                valid_project_ids,
                key=selector_state_key,
                format_func=lambda cid: label_lookup.get(str(cid), str(cid)),
                help="Choose a funded project here or click a colored marker on the map.",
            )
            st.session_state[active_state_key] = str(chosen_project_id)
        else:
            chosen_project_id = None

        map_legend(selected)
        map_event = st.pydeck_chart(
            build_map_deck(tiles, project_map, str(chosen_project_id) if chosen_project_id else None),
            width="stretch",
            selection_mode="single-object",
            on_select="rerun",
            key=f"thermal_map_{city}",
        )
        clicked_id = clicked_project_id(map_event)
        if clicked_id in valid_project_ids and clicked_id != chosen_project_id:
            st.session_state[pending_state_key] = clicked_id
            st.rerun()
        st.caption("HeatLens cells show relative thermal burden. Colored markers are funded projects; the gold outline marks the project currently being inspected.")

with right:
    with st.container(border=True):
        st.markdown("#### Digital twin outcome")
        k1, k2 = st.columns(2)
        k1.metric("System-level modeled relief", f"{avoided:,.0f}", f"{reduction_pct:.1f}% lower")
        k1.caption("planning person-hours")
        k2.metric("Budget deployed", f"${spent / 1e6:.2f}M", f"{(100 * spent / budget if budget else 0):.1f}% utilized")
        k2.caption("of available capital")

        compare = pd.DataFrame(
            {"state": ["Current", "ThermalOS plan"], "person-hours": [baseline, after]}
        ).set_index("state")
        st.bar_chart(compare, height=260)
        st.caption(f"{baseline:,.0f} → {after:,.0f} planning person-hours • solver: {portfolio.solver}")

        if not selected.empty:
            mix = selected.groupby("label").size().sort_values(ascending=False)
            area_mix = selected.groupby("area").size().sort_values(ascending=False)
            st.markdown("**Portfolio composition**")
            st.write(" · ".join(f"{label} × {int(count)}" for label, count in mix.items()))
            st.markdown("**Geographic allocation**")
            st.write(" · ".join(f"{display_area(area)} × {int(count)}" for area, count in area_mix.items()))

            served = int(portfolio.summary.get("neighborhoods_served", selected["area"].nunique()))
            actual_area_share = 100 * safe_float(portfolio.summary.get("max_neighborhood_spend_fraction_achieved"))
            actual_int_share = 100 * safe_float(portfolio.summary.get("max_intervention_spend_fraction_achieved"))
            deployed_int_share = 100 * safe_float(portfolio.summary.get("max_intervention_spend_fraction_of_deployed"))
            st.markdown(
                "<div class='tos-chip-row'>"
                + pill(f"Equity {vuln_share:.0f}%", "green")
                + pill(f"{served} areas served", "blue")
                + pill(f"Largest area {actual_area_share:.0f}%")
                + pill(f"Largest type {actual_int_share:.0f}%")
                + "</div>",
                unsafe_allow_html=True,
            )
            st.caption(f"Leading intervention represents {deployed_int_share:.0f}% of deployed spend.")

if valid_project_ids and chosen_project_id:
    active = project_ranked.loc[project_ranked["candidate_id"].astype(str) == str(chosen_project_id)].iloc[0]
    section_header(
        "Project rationale",
        "A concise, auditable explanation of why this intervention-location pair entered the funded portfolio.",
        "Why here?",
    )
    with st.container(border=True):
        explain_left, explain_right = st.columns([1.05, 1.35], gap="large")
        with explain_left:
            color = PROJECT_COLORS.get(str(active.get("intervention")), [235, 235, 235, 245])
            swatch = f"rgb({color[0]},{color[1]},{color[2]})"
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:10px;margin-bottom:4px;'>"
                f"<span style='width:12px;height:12px;background:{swatch};border-radius:50%;border:1px solid white;display:inline-block;'></span>"
                f"<span style='font-size:1.2rem;font-weight:750'>{html.escape(str(active['label']))}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
            display_tile_id = active.get("source_tile_id", active.get("tile_id", ""))
            st.caption(f"Portfolio rank #{int(active['portfolio_rank'])} • {display_area(active['area'])} • tile {display_tile_id}")
            reasons = [r.strip() for r in str(active.get("reason", "")).split(",") if r.strip()]
            st.markdown("**Decision signals**")
            if reasons:
                st.markdown("\n".join(f"- {r[0].upper() + r[1:] if len(r) > 1 else r.upper()}" for r in reasons))
            else:
                st.write("Positive modeled benefit per budget under the current planning constraints.")
            mechanism = MECHANISM_LABELS.get(str(active.get("mechanism")), str(active.get("mechanism", "")))
            if mechanism:
                st.caption(f"Mechanism: {mechanism}")

        with explain_right:
            a1, a2, a3, a4 = st.columns(4)
            a1.metric("Project cost", f"${safe_float(active.get('cost_usd')) / 1000:.0f}k")
            planning_relief = planning_benefit(active, impact_basis)
            relief_label = "Conservative relief" if impact_basis == "conservative" else "Expected relief"
            a2.metric(relief_label, f"{planning_relief:,.0f}")
            a3.metric("Feasibility", f"{100 * safe_float(active.get('feasibility')):.0f}%")
            a4.metric("Canopy", f"{100 * safe_float(active.get('canopy_fraction')):.0f}%")

            b1, b2, b3, b4 = st.columns(4)
            vuln = safe_float(active.get("vulnerability"))
            b1.metric("Vulnerability score", f"{vuln:.2f} / 1.00")
            b1.caption(vulnerability_band(vuln))
            b2.metric("Hot duration", f"{safe_float(active.get('exceedance_h')):.1f} h")
            b3.metric("Tile burden", f"{safe_float(active.get('baseline_person_hours')):,.0f}")
            b4.metric("Transit proxy", f"{safe_float(active.get('transit_stop_count')):.0f}")

            low = safe_float(active.get("benefit_low_person_hours"))
            expected = safe_float(active.get("benefit_expected_person_hours"))
            high = safe_float(active.get("benefit_high_person_hours"))
            value_per_100k = planning_benefit(active, impact_basis) / max(safe_float(active.get("cost_usd")), 1.0) * 100_000
            st.markdown(
                "<div class='tos-chip-row'>"
                + pill(f"Value {value_per_100k:,.0f} person-h / $100k", "blue")
                + pill(f"Range {low:,.0f}–{high:,.0f} person-h")
                + pill(f"{safe_float(active.get('temperature_c')):.1f} °C scenario tile")
                + "</div>",
                unsafe_allow_html=True,
            )
            st.caption("Decision-support scenario estimate only — not a causal impact estimate or procurement recommendation.")

section_header(
    "Capital plan",
    "Review the selected projects, the budget-response frontier, and the scientific guardrails behind the recommendation.",
    "Portfolio",
)
portfolio_tab, frontier_tab, guardrail_tab = st.tabs(["Selected portfolio", "Budget frontier", "Scientific guardrails"])

with portfolio_tab:
    if selected.empty:
        st.info("No feasible projects under the current constraints.")
    else:
        mix = selected.groupby("label").size().sort_values(ascending=False)
        area_mix = selected.groupby("area").size().sort_values(ascending=False)
        st.markdown(
            "<div class='tos-chip-row'>"
            + "".join(pill(f"{label} × {int(count)}", "blue") for label, count in mix.items())
            + "".join(pill(f"{display_area(area)} × {int(count)}") for area, count in area_mix.items())
            + "</div>",
            unsafe_allow_html=True,
        )
        if synthetic_demo and len(area_mix) and int(area_mix.iloc[0]) == len(selected):
            st.info(
                f"This synthetic thermal scenario concentrates the portfolio in {display_area(area_mix.index[0])}. This is a scenario result, not a claim about where real Miami investment should be concentrated."
            )

        show = selected.copy()
        planning_col = "benefit_low_person_hours" if impact_basis == "conservative" else "benefit_expected_person_hours"
        show["planning_relief"] = show[planning_col].round(1)
        show["cost_usd"] = show["cost_usd"].round(0)
        show["value_per_100k"] = (show[planning_col] / show["cost_usd"].clip(lower=1) * 100_000).round(1)
        show["area"] = show["area"].map(display_area)
        compact = show[["label", "area", "cost_usd", "feasibility", "vulnerability", "planning_relief", "value_per_100k"]].sort_values("planning_relief", ascending=False)
        st.dataframe(compact.head(12), width="stretch", hide_index=True)
        if len(compact) > 12:
            with st.expander(f"View all {len(compact)} funded projects"):
                st.dataframe(compact, width="stretch", hide_index=True)

with frontier_tab:
    budgets = [500_000, 1_000_000, 2_000_000, 3_000_000, 4_000_000, 5_000_000]
    frontier_key = f"budget_frontier_{city}"
    frontier_signature = (
        city,
        objective,
        impact_basis,
        round(float(equity_min), 4),
        tuple(sorted(enabled)),
        round(float(max_neighborhood_fraction), 4),
        round(float(max_intervention_fraction), 4),
        int(min_neighborhoods),
    )

    if st.button(
        "Compute budget frontier",
        key=f"compute_budget_frontier_{city}",
        type="primary",
    ):
        with st.spinner("Solving portfolio across six capital budgets…"):
            curve = marginal_value_curve(
                candidates,
                budgets,
                objective=objective,
                impact_basis=impact_basis,
                equity_min_fraction=equity_min,
                enabled_interventions=enabled,
                max_neighborhood_spend_fraction=max_neighborhood_fraction,
                max_intervention_spend_fraction=max_intervention_fraction,
                min_neighborhoods_served=min_neighborhoods,
            )
            st.session_state[frontier_key] = {
                "signature": frontier_signature,
                "curve": curve,
            }

    frontier_payload = st.session_state.get(frontier_key)

    if (
        frontier_payload
        and frontier_payload.get("signature") == frontier_signature
    ):
        curve = frontier_payload["curve"]
        st.markdown(
            "**How much direct project benefit becomes available as the capital envelope grows?**"
        )
        st.line_chart(
            curve.set_index("budget_usd")["first_order_person_hours_avoided"],
            height=340,
        )
        with st.expander("View frontier data"):
            st.dataframe(curve, width="stretch", hide_index=True)
    else:
        st.info(
            "Compute the budget frontier on demand to compare capital envelopes from $0.5M to $5M."
        )

with guardrail_tab:
    st.markdown(
        """
**What ThermalOS is — and is not — claiming**

- **FortyGuard heat and duration layers are inputs, not causal labels.**
- **ThermalTwin is observational/associational context.** FortyGuard supplies the observed thermal field; morphology does not replace it as a universal temperature forecaster.
- Intervention effects are sampled from configurable, literature-bounded uncertainty distributions.
- Shade can reduce radiant human exposure without materially lowering neighborhood air temperature.
- Reflective pavement may reduce surface temperature while worsening radiant comfort in some settings; ThermalOS includes a configurable radiant penalty.
- Costs are planning assumptions until replaced by current local procurement data.
- Population, GTFS, and POIs are exposure proxies, not individual-level movement data or clinical risk predictions.
        """
    )

# -----------------------------------------------------------------------------
# CLOSED-LOOP THERMALOS LAYERS
# Sense -> Understand -> Decide -> Stress-test -> Operate -> Verify -> Learn
# -----------------------------------------------------------------------------
section_header(
    "Closed-loop resilience stack",
    "Stress-test decisions, operate heat events, verify implementation, transfer across cities, and audit every assumption.",
    "ThermalOS layers",
)
layer_tabs = st.tabs([
    "Robustness",
    "Policy Stress Lab",
    "HeatOps",
    "ThermalVerify + Learn",
    "Thermal Copilot",
    "Trust + Dossier",
    "Generalization",
])

scenario_signature = (
    city,
    int(budget),
    objective,
    impact_basis,
    round(float(equity_min), 4),
    tuple(sorted(enabled)),
    round(float(max_neighborhood_fraction), 4),
    round(float(max_intervention_fraction), 4),
    int(min_neighborhoods),
)

with layer_tabs[0]:
    st.markdown("**Does the decision survive uncertainty?**")
    st.caption(
        "Re-optimizes plausible effect/cost worlds under the exact same policy constraints. Selection stability is a stress-test metric, not a guarantee."
    )
    robustness_key = f"robustness_result_{city}"
    if st.button("Run 32-scenario robustness stress test", key=f"run_robustness_{city}", type="primary"):
        with st.spinner("Re-solving plausible effect and cost worlds…"):
            robustness_result = run_robustness(
                candidates,
                budget_usd=budget,
                objective=objective,
                impact_basis=impact_basis,
                equity_min_fraction=equity_min,
                enabled_interventions=enabled,
                max_neighborhood_spend_fraction=max_neighborhood_fraction,
                max_intervention_spend_fraction=max_intervention_fraction,
                min_neighborhoods_served=min_neighborhoods,
                scenarios=32,
            )
            st.session_state[robustness_key] = {"signature": scenario_signature, "result": robustness_result}
    robust_payload = st.session_state.get(robustness_key)
    if robust_payload and robust_payload.get("signature") == scenario_signature:
        rob = robust_payload["result"]
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Portfolio stability", f"{100 * rob.portfolio_stability:.0f}%")
        r2.metric("Median set overlap", f"{100 * rob.median_jaccard:.0f}%")
        r3.metric("Direct relief P10", f"{rob.direct_benefit_p10:,.0f}")
        r4.metric("Direct relief P90", f"{rob.direct_benefit_p90:,.0f}")
        baseline_stability = rob.project_stability[rob.project_stability["baseline_selected"]].copy()
        if len(baseline_stability):
            baseline_stability["selection_stability_pct"] = (100 * baseline_stability["selection_frequency"]).round(0)
            stable_show = baseline_stability[["label", "area", "cost_usd", "benefit_expected_person_hours", "selection_stability_pct"]].copy()
            stable_show["area"] = stable_show["area"].map(display_area)
            st.dataframe(stable_show.head(20), width="stretch", hide_index=True)
        st.caption(rob.method_note)
    elif robust_payload:
        st.info("The planning scenario changed. Re-run the robustness stress test to avoid showing stale stability results.")
    else:
        st.info("Run the stress test to quantify which funded projects remain selected when effect and cost assumptions move within plausible ranges.")

with layer_tabs[1]:
    st.markdown("**What changes when policy priorities change?**")
    st.caption("Separates uncertainty in the model from legitimate differences in public policy priorities.")
    policy_key = f"policy_lab_{city}"
    if st.button("Run policy stress lab", key=f"run_policy_lab_{city}", type="primary"):
        with st.spinner("Solving impact, equity, distribution, and conservative portfolios…"):
            pol = run_policy_stress_lab(
                candidates,
                budget_usd=budget,
                enabled_interventions=enabled,
                area_count=max_neighborhoods,
            )
            st.session_state[policy_key] = {"signature": (city, int(budget), tuple(sorted(enabled))), "result": pol}
    policy_payload = st.session_state.get(policy_key)
    policy_signature = (city, int(budget), tuple(sorted(enabled)))
    if policy_payload and policy_payload.get("signature") == policy_signature:
        pol = policy_payload["result"]
        scenario_show = pol.scenarios.copy()
        scenario_show["equity_spend_pct"] = (100 * scenario_show["equity_spend_fraction"]).round(1)
        scenario_show["largest_area_pct"] = (100 * scenario_show["largest_neighborhood_budget_share"]).round(1)
        st.dataframe(
            scenario_show[["scenario", "projects", "direct_person_hours_avoided", "equity_spend_pct", "neighborhoods_served", "largest_area_pct", "status"]],
            width="stretch",
            hide_index=True,
        )
        frontier = pol.equity_frontier.copy()
        frontier["equity_floor_pct"] = (100 * frontier["equity_floor"]).round(0)
        frontier["direct_relief"] = frontier["direct_person_hours_avoided"]
        st.markdown("**Impact-equity frontier**")
        st.line_chart(frontier.set_index("equity_floor_pct")["direct_relief"], height=280)
        st.caption("A downward slope quantifies the modeled direct-benefit trade-off of increasing the required equity-aligned capital share.")
    else:
        st.info("Run the lab to compare Impact-first, Balanced, Equity-first, Distributed, and Conservative planning philosophies at the same budget.")

with layer_tabs[2]:
    st.markdown("**HeatOps — operational response for the current heat field**")
    st.caption(
        "CoolCapital asks what to build. HeatOps asks where limited temporary cooling resources should be deployed during a heat event. This is operational decision support, not emergency or medical advice."
    )
    ops_budget = st.slider(
        "HeatOps operating budget",
        min_value=20_000,
        max_value=150_000,
        value=60_000,
        step=5_000,
        format="$%d",
        key=f"ops_budget_{city}",
    )
    ops_key = f"heatops_{city}"
    if st.button("Generate HeatOps deployment", key=f"run_heatops_{city}", type="primary"):
        with st.spinner("Prioritizing heat-access gaps and temporary resources…"):
            ops = plan_heatops(
                tiles,
                operating_budget_usd=ops_budget,
                equity_min_fraction=equity_min,
                max_neighborhood_spend_fraction=min(max_neighborhood_fraction, 0.60),
                min_neighborhoods_served=min(min_neighborhoods, max_neighborhoods),
            )
            st.session_state[ops_key] = {"budget": ops_budget, "result": ops}
    ops_payload = st.session_state.get(ops_key)
    if ops_payload and ops_payload.get("budget") == ops_budget:
        ops = ops_payload["result"]
        o1, o2, o3, o4 = st.columns(4)
        o1.metric("Operating actions", f"{len(ops.selected)}")
        o2.metric("Operating spend", f"${float(ops.summary.get('spent_usd', 0)):,.0f}")
        o3.metric("Critical access-gap cells", f"{int(ops.summary.get('critical_access_gap_tiles', 0))}")
        o4.metric("Operating window", str(ops.summary.get("operating_window", "13:30-18:30")))
        if len(ops.selected):
            ops_show = ops.selected[["label", "area", "cost_usd", "benefit_expected_person_hours", "cooling_access_gap", "access_gap_band", "transit_access_score"]].copy()
            ops_show["area"] = ops_show["area"].map(display_area)
            st.dataframe(ops_show, width="stretch", hide_index=True)
        gaps = ops.access.nlargest(12, "cooling_access_gap")[["area", "tile_id", "cooling_access_gap", "access_gap_band", "vulnerability", "transit_access_score", "mobility_friction"]].copy()
        gaps["area"] = gaps["area"].map(display_area)
        with st.expander("Highest cooling-access gaps"):
            st.dataframe(gaps, width="stretch", hide_index=True)
        st.caption(f"Access basis: {ops.summary.get('access_basis', 'not available')}")
    else:
        st.info("Generate a deployment to turn the same thermal, vulnerability, no-vehicle, and GTFS context into short-horizon operational actions.")

with layer_tabs[3]:
    st.markdown("**ThermalVerify — plan the measurement before the project is built**")
    st.caption(
        "Every funded intervention can carry a frozen baseline, matched controls, and 30/90/365-day verification schedule. No post-deployment benefit is invented."
    )
    verify_key = f"verify_registry_{city}"
    if st.button("Prepare verification registry", key=f"prepare_verify_{city}", type="primary"):
        with st.spinner("Matching control cells and freezing project baselines…"):
            registry = build_verification_registry(tiles, selected, controls_per_project=5)
            st.session_state[verify_key] = {"signature": scenario_signature, "result": registry}
    verify_payload = st.session_state.get(verify_key)
    registry = None
    if verify_payload and verify_payload.get("signature") == scenario_signature:
        registry = verify_payload["result"]
        v1, v2, v3 = st.columns(3)
        v1.metric("Baselines captured", f"{len(registry.projects)}")
        v2.metric("Matched controls", f"{len(registry.controls)}")
        learn_state = adaptive_calibration_status(registry.projects)
        v3.metric("Verified local projects", f"{learn_state['verified_projects']}")
        st.dataframe(
            registry.projects[["label", "area", "verification_status", "baseline_temperature_c", "expected_person_hours_avoided", "control_count", "verification_schedule"]].head(20),
            width="stretch",
            hide_index=True,
        )
        st.caption(registry.protocol["claim_boundary"])
        st.markdown("**Adaptive learning hook**")
        st.info(
            f"{learn_state['status']}. ThermalOS will only update city-specific intervention priors after reviewed post-deployment evidence reaches the configured evidence threshold."
        )
        post_file = st.file_uploader(
            "Optional: upload a post-deployment tile snapshot to run the matched-control diagnostic",
            type=["csv"],
            key=f"post_verify_upload_{city}",
        )
        if post_file is not None:
            post_tiles = pd.read_csv(post_file)
            observed = evaluate_post_deployment(registry, post_tiles)
            st.dataframe(observed, width="stretch", hide_index=True)
            if len(observed):
                joined = observed.merge(registry.projects[["candidate_id", "intervention"]], on="candidate_id", how="left")
                reviewed = joined.rename(columns={"observed_cooling_c": "observed_cooling_c"})
                preview = update_intervention_priors(cfg, reviewed, minimum_verified=3)
                with st.expander("Preview local-prior update logic (not automatically applied)"):
                    st.dataframe(preview.audit, width="stretch", hide_index=True)
                    st.caption(preview.status)
    elif verify_payload:
        st.info("The portfolio changed. Prepare a new verification registry so baselines correspond to the current funded projects.")
    else:
        st.info("Prepare the registry to turn each recommendation into an auditable implementation-and-evaluation record.")

with layer_tabs[4]:
    st.markdown("**Thermal Copilot — natural-language control over the real decision engine**")
    st.caption("This is a tool-executing command layer over the same MILP, not a generic heat chatbot. It is deterministic, auditable, and offline-safe.")
    copilot_prompt = st.text_area(
        "Ask ThermalOS",
        value="Give me a $3M balanced plan with at least 50% equity and no neighborhood over 40%.",
        height=92,
        key=f"copilot_prompt_{city}",
    )
    if st.button("Run Copilot command", key=f"run_copilot_{city}", type="primary"):
        state = {
            "budget_usd": budget,
            "objective": objective,
            "impact_basis": impact_basis,
            "equity_min_fraction": equity_min,
            "enabled_interventions": enabled,
            "max_neighborhood_spend_fraction": max_neighborhood_fraction,
            "max_intervention_spend_fraction": max_intervention_fraction,
            "min_neighborhoods_served": min_neighborhoods,
        }
        with st.spinner("Parsing the command and executing the constrained optimizer…"):
            cop = execute_copilot(copilot_prompt, candidates, state, robustness_scenarios=20)
            st.session_state[f"copilot_result_{city}"] = cop
    cop = st.session_state.get(f"copilot_result_{city}")
    if cop is not None:
        st.success(cop.narrative)
        updates = cop.command.updates
        if updates:
            st.markdown("**Parsed planning controls**")
            st.json(updates)
        if len(cop.selected):
            cop_show = cop.selected[["label", "area", "cost_usd", "benefit_expected_person_hours", "vulnerability"]].copy().head(12)
            cop_show["area"] = cop_show["area"].map(display_area)
            st.dataframe(cop_show, width="stretch", hide_index=True)
        if cop.supplemental is not None and len(cop.supplemental):
            with st.expander("Copilot evidence"):
                st.dataframe(cop.supplemental.head(30), width="stretch", hide_index=True)
    st.caption("Examples: “Why wasn't cool pavement selected?”, “Stress-test this plan”, “Use conservative effects and a $4M budget”, “Require at least 3 neighborhoods.”")

with layer_tabs[5]:
    st.markdown("**Evidence Ledger / Trust Center**")
    ledger = build_evidence_ledger(tiles, provenance, cfg)
    ledger_summary = evidence_summary(ledger)
    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Audited layers", ledger_summary["layers"])
    e2.metric("Observed / derived", ledger_summary["real_or_derived_layers"])
    e3.metric("Modeled", ledger_summary["modeled_layers"])
    e4.metric("Policy assumptions", ledger_summary["policy_assumptions"])
    st.dataframe(ledger, width="stretch", hide_index=True)
    st.caption("ThermalOS exposes whether each visible number is observed/model-derived upstream, derived locally, modeled, synthetic, or a policy assumption — and states the claim boundary beside it.")

    current_summary = {
        **portfolio.summary,
        "baseline_person_hours": baseline,
        "counterfactual_person_hours": after,
        "modeled_person_hours_avoided_with_spillover": avoided,
        "modeled_reduction_fraction": avoided / baseline if baseline else 0.0,
        "vulnerable_spend_fraction": vuln_share / 100.0,
    }
    robust_for_report = None
    if robust_payload and robust_payload.get("signature") == scenario_signature:
        rr = robust_payload["result"]
        robust_for_report = {
            "portfolio_stability": rr.portfolio_stability,
            "median_jaccard": rr.median_jaccard,
            "direct_benefit_p10": rr.direct_benefit_p10,
            "direct_benefit_p90": rr.direct_benefit_p90,
        }
    registry_for_report = registry.projects if registry is not None else None
    city_name_for_report = city_config(city).get("name", city)
    dossier_bytes = build_dossier_pdf(
        city_name=city_name_for_report,
        summary=current_summary,
        selected=selected,
        provenance=provenance,
        evidence_ledger=ledger,
        robustness_summary=robust_for_report,
        verification_projects=registry_for_report,
    )
    st.download_button(
        "Download decision dossier (PDF)",
        data=dossier_bytes,
        file_name=f"thermalos_{city}_decision_dossier.pdf",
        mime="application/pdf",
        key=f"download_dossier_{city}",
        type="primary",
    )
    st.download_button(
        "Download evidence ledger (CSV)",
        data=ledger.to_csv(index=False).encode("utf-8"),
        file_name=f"thermalos_{city}_evidence_ledger.csv",
        mime="text/csv",
        key=f"download_ledger_{city}",
    )

with layer_tabs[6]:
    st.markdown("**ThermalTwin context + prospective multi-state system transfer**")
    st.caption(
        "Miami and Houston are development cities because both outcomes have already been inspected. "
        "Development diagnostics showed that sparse morphology alone does not reliably transport hotspot ordering, "
        "so v3 keeps FortyGuard's observed thermal field as the temperature-intelligence input and freezes the actual "
        "ThermalOS decision architecture before Phoenix, Atlanta, and Los Angeles are opened."
    )

    protocol = generalization_config()
    g1, g2, g3 = st.columns(3)
    g1.metric("Development cities", len(protocol.get("development_cities", [])))
    g2.metric("Prospective blind cities", len(protocol.get("blind_cities", [])))
    g3.metric("Satellite contract", f"{protocol.get('satellite_sampling', {}).get('samples_per_area', 15)} / AOI")
    st.code(
        "Development: " + ", ".join(protocol.get("development_cities", []))
        + "\nBlind: " + ", ".join(protocol.get("blind_cities", []))
        + "\nTemperature prediction claim: NO — FortyGuard is the observed thermal-intelligence layer",
        language="text",
    )

    st.markdown("**Development-only system replay**")
    st.caption(
        "The same candidate generator, intervention priors, MILP, policy constraints, and scenario accounting are replayed in Miami and Houston. "
        "This is not a blind result and does not fit a city-specific temperature model."
    )
    cross_key = "cross_city_system_replay_v3"
    if st.button("Run development system replay", key="run_cross_city_transfer", type="primary"):
        with st.spinner("Running the unchanged ThermalOS decision pipeline in both development cities…"):
            try:
                miami_tiles, miami_prov = load_city("miami")
                houston_tiles, houston_prov = load_city("houston")
                cross = run_cross_city_comparison(
                    miami_tiles,
                    houston_tiles,
                    miami_provenance=miami_prov,
                    houston_provenance=houston_prov,
                    intervention_config=cfg,
                    budget_usd=budget,
                )
                st.session_state[cross_key] = {"budget": int(budget), "result": cross}
            except Exception as exc:
                st.error(f"Development system replay could not run: {exc}")

    cross_payload = st.session_state.get(cross_key)
    if cross_payload and cross_payload.get("budget") == int(budget):
        cross = cross_payload["result"]
        comparison = cross.comparison.copy()
        comparison["equity_spend_pct"] = (100 * comparison["equity_spend_fraction"]).round(1)
        comparison["modeled_reduction_pct"] = (100 * comparison["modeled_reduction_fraction"]).round(1)
        st.dataframe(
            comparison[["city", "data_mode", "tiles", "projects", "spent_usd", "equity_spend_pct", "direct_person_hours_avoided", "system_person_hours_avoided", "modeled_reduction_pct", "dominant_intervention", "top_area"]],
            width="stretch",
            hide_index=True,
        )
        t = cross.transfer_metrics
        t1, t2, t3 = st.columns(3)
        t1.metric("Same pipeline", "Yes")
        t2.metric("City temperature refit", "None")
        t3.metric("Morphology-only forecast claim", "No")
        st.caption(t.get("claim_boundary", ""))

    st.markdown("**Why the forecast claim was retired**")
    st.caption(
        "On the Miami/Houston development set, morphology-only anomaly magnitude error was modest but hotspot ranking did not transfer reliably. "
        "ThermalOS therefore uses those diagnostics as a scientific guardrail rather than optimizing until a favorable cross-city score appears."
    )

    freeze_manifest = ROOT / "outputs" / "generalization" / "freeze" / "thermalos_system_transfer_manifest.json"
    release_manifest = ROOT / "RELEASE_MANIFEST.json"

    if freeze_manifest.exists():
        try:
            manifest = json.loads(freeze_manifest.read_text(encoding="utf-8"))
            snapshot = str(manifest.get("pipeline_snapshot_sha256", ""))
            st.success(f"Frozen v3 decision system • pipeline SHA256 {snapshot[:12]}…")
            st.caption(
                "Blind-city evaluation verifies the frozen code/config snapshot and writes a one-time open-log entry. "
                "The success gates are data-contract, decision-feasibility, robustness, policy, and evidence/provenance gates."
            )
        except Exception:
            pass
    elif release_manifest.exists():
        try:
            manifest = json.loads(release_manifest.read_text(encoding="utf-8"))
            snapshot = str(manifest.get("final_pipeline_sha256", ""))
            st.success(f"Frozen ThermalOS release • pipeline SHA256 {snapshot[:12]}…")
            st.caption(
                "The published release uses the frozen multi-state decision architecture, "
                "data contract, robustness, policy, and evidence/provenance gates."
            )
        except Exception:
            st.warning("Release manifest could not be read.")
    else:
        st.warning("Release manifest unavailable.")

st.markdown("<div class='tos-divider'></div>", unsafe_allow_html=True)
st.caption("ThermalOS • Sense → Understand → Decide → Stress-test → Operate → Verify → Learn • observed thermal intelligence + frozen multi-state system transfer")

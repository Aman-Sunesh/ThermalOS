# ThermalOS Architecture

> **Protocol update (2026-08-23):** Houston outcomes have already been inspected, so Houston is now a **development city**, not a blind test. The current prospective blind set is Phoenix (AZ), Atlanta (GA), and Los Angeles (CA). `GENERALIZATION_PROTOCOL.md` is authoritative if older sections below use historical Houston-held-out language.


## Design goal

ThermalOS is deliberately not another heat-map front end. Its architecture closes the urban-heat decision lifecycle:

```text
SENSE → UNDERSTAND → DECIDE → STRESS-TEST → FUND → OPERATE → VERIFY → LEARN
```

FortyGuard provides the hyperlocal temperature-intelligence substrate. ThermalOS turns that substrate into constrained municipal decisions and an auditable feedback loop.

## Closed-loop architecture

```text
                             ┌───────────────────────────────┐
                             │          FortyGuard           │
                             │ heatmap • env • satellite     │
                             └──────────────┬────────────────┘
                                            │ cached validated data
          ┌───────────────────────┬─────────┴────────┬───────────────────────┐
          │                       │                  │                       │
     Census / ACS            Public GIS           GTFS              optional ops data
 population/vulnerability     canopy/context    transit proxy       cooling resources
          │                       │                  │                       │
          └───────────────────────┴──────────┬───────┴───────────────────────┘
                                            ▼
                             ┌───────────────────────────────┐
                             │     Canonical city tiles      │
                             │ thermal • morphology • equity │
                             │ exposure • provenance         │
                             └──────────────┬────────────────┘
                                            ▼
                    ┌──────────────────────────────────────────┐
                    │ HeatLens + EquityLens                    │
                    │ relative thermal burden / exposure       │
                    └─────────────────┬────────────────────────┘
                                      ▼
                    ┌──────────────────────────────────────────┐
                    │ ThermalTwin + intervention evidence      │
                    │ observational transfer diagnostic + separate priors       │
                    └─────────────────┬────────────────────────┘
                                      ▼
                    ┌──────────────────────────────────────────┐
                    │ CoolCapital candidate generator + MILP   │
                    │ budget/equity/geography/type constraints │
                    └───────────┬─────────────────┬────────────┘
                                │                 │
                 ┌──────────────▼───────┐   ┌────▼─────────────────┐
                 │ Robustness Engine     │   │ Policy Stress Lab     │
                 │ uncertainty worlds    │   │ policy Pareto tradeoff │
                 └──────────────┬───────┘   └────┬─────────────────┘
                                └──────────┬──────┘
                                           ▼
                       ┌─────────────────────────────────┐
                       │ Funded capital portfolio         │
                       │ + spillover-aware digital twin   │
                       └───────────┬─────────────┬────────┘
                                   │             │
                       ┌───────────▼─────┐ ┌────▼───────────────┐
                       │ HeatOps          │ │ ThermalVerify       │
                       │ temporary action │ │ baseline + controls │
                       └─────────────────┘ └────────┬────────────┘
                                                   ▼
                                         ┌──────────────────────┐
                                         │ Adaptive learning     │
                                         │ reviewed local priors │
                                         └──────────┬───────────┘
                                                    └────→ next plan

        Thermal Copilot orchestrates the same underlying tools; Trust Center audits them.
```

## Canonical tile table

Every downstream feature consumes the same canonical tile representation. Important families include:

- location: `tile_id`, `area`, `lat`, `lon`;
- thermal: `temperature_c`, exceedance/persistence and environmental context;
- morphology: canopy/building/road/impervious/pervious fractions;
- exposure: allocated population, GTFS stop count and other optional activity proxies;
- equity: poverty / no-vehicle / vulnerability features;
- provenance: city-level record of real, derived and missing layers.

This makes the optimizer city-agnostic: Miami and Houston differ in data values, not decision-engine code.

## HeatLens

HeatLens produces a transparent planning burden, not a medical-risk score.

Conceptually:

```text
thermal_stress = robust combination(temperature, duration, apparent/wet-bulb context)
exposure       = population/activity proxy
burden         = exposure × hot-duration × thermal-stress × vulnerability weighting
```

The exact implementation is in `src/thermalos/features/heatlens.py`.

## ThermalTwin and intervention effects

ThermalOS separates two evidence streams:

1. **Observational thermal response** — a model can learn associations between morphology/context and temperature.
2. **Intervention evidence priors** — each intervention has bounded effect distributions and direct-exposure mechanisms in `configs/interventions.yaml`.

The observational model is never silently treated as causal. Shade, roofs, pavement, tree canopy and cooling/hydration mechanisms remain explicit.

## CoolCapital MILP

Each feasible tile/intervention pair becomes a binary project candidate `x_i`.

Budget:

```math
\sum_i c_i x_i \le B
```

At most one capital project per tile:

```math
\sum_{i \in tile(t)} x_i \le 1
```

Minimum equity-aligned capital share `α`:

```math
\sum_i c_i (v_i-\alpha)x_i \ge 0
```

Optional constraints also cap:

- share of total available budget assigned to one neighborhood;
- share of total available budget assigned to one intervention family;
- minimum number of neighborhoods served.

Expected and conservative impact bases use different benefit columns but the same feasible set and policy language.

## Direct benefit vs system-level relief

These are intentionally different quantities.

- **Direct / first-order relief**: additive project-level benefit used inside the MILP and budget frontier.
- **System-level modeled relief**: post-selection counterfactual accounting with distance-decay spillover and multiplicative overlap correction.

The UI labels both explicitly so a 14k direct frontier value is not confused with a larger spillover-inclusive city-system estimate.

## Robustness Engine

`src/thermalos/analytics/robustness.py`:

1. solves the baseline portfolio on the full candidate universe;
2. builds a competitive repeated-solve pool while preserving every baseline-selected project;
3. samples project benefits from bounded triangular distributions using low/expected/high evidence estimates;
4. perturbs costs with clipped mean-one log-normal factors;
5. re-solves the exact same policy-constrained MILP;
6. reports selection frequency, portfolio stability, Jaccard overlap and P10/P50/P90 direct benefit.

This is a **decision stress test**, not a calibrated probability that the policy succeeds.

## Policy Stress Lab

`src/thermalos/analytics/policy_lab.py` separates uncertain science from legitimate policy choices. It solves named planning philosophies at the same budget and an equity-floor sweep to expose the modeled marginal cost of higher equity requirements.

## HeatOps

`src/thermalos/operations/heatops.py` creates a separate short-horizon action layer. It computes a cooling-access gap from:

- current thermal stress;
- vulnerability;
- no-vehicle mobility friction;
- GTFS transit access;
- optional actual cooling-center access when supplied.

Temporary hydration, shade and mobile-cooling actions are optimized using the same mathematical decision engine but separate operational assumptions in `configs/operations.yaml`.

## ThermalVerify

`src/thermalos/verification.py` closes the recommendation-to-evaluation gap:

- freezes project baseline temperature/burden;
- selects nearby matched controls on thermal/morphology/equity context;
- schedules 30/90/365-day checks;
- never writes a fake post-deployment effect;
- can compute a simple matched-control pre/post diagnostic once a real post-deployment tile snapshot exists.

That diagnostic is explicitly not a complete causal estimator. A production deployment should add weather normalization, repeated observations and quasi-experimental design appropriate to the intervention.

## Adaptive learning

`src/thermalos/learning.py` provides an auditable empirical-Bayes-style update hook. Reviewed local intervention observations are shrunk toward the literature prior according to sample size. A minimum evidence threshold is required and no YAML config is modified automatically.

## Thermal Copilot

`src/thermalos/copilot.py` is deliberately deterministic in the current implementation. It parses planning language into the same formal constraints the sidebar uses and invokes the real MILP / robustness / exclusion-audit tools. This preserves auditability and prevents the demo from depending on an external LLM.

A production interface could place an LLM in front of the exact same typed tool contract without letting free-form model text bypass constraints.

## Evidence Ledger / Trust Center

`src/thermalos/evidence.py` labels each important layer as:

- observed/model-derived upstream;
- observed GIS;
- derived proxy;
- modeled;
- policy assumption;
- synthetic demo.

The evidence class, coverage and claim boundary are surfaced in the UI and exported to the dossier.

## Development transfer + prospective blind multi-state evaluation

`src/thermalos/generalization.py` performs two separate tests:

1. same candidate generator / same optimizer / same policy constraints in Miami and Houston;
2. Miami-fitted observational ThermalTwin predicts Houston temperature **without Houston refitting**.

This tests representation and workflow transfer. It does not prove causal intervention effects transport unchanged between cities.

## Offline-first execution architecture

Live external APIs are data-harvesting dependencies, not presentation dependencies:

```text
live APIs → validated cache → processed city table → deterministic decision tools → UI
```

Every expensive advanced feature runs on demand behind a button and caches its result in Streamlit session state. The core plan remains responsive even if the network is unavailable.

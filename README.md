# ThermalOS

### Live application: https://thermalos.streamlit.app/

**AI operating system for urban heat resilience.**

> Cities already know where it is hot. **ThermalOS tells them what to do about it.**

ThermalOS turns FortyGuard hyperlocal temperature intelligence into an explainable, budget-constrained, equity-aware intervention plan, stress-tests that plan under uncertainty and policy constraints, supports same-day heat operations, prepares post-deployment verification, and learns only from reviewed local evidence.

## The problem

Urban heat teams can increasingly measure **where heat is occurring**, but the operational decision is harder:

> **Given a fixed public budget, what should a city deploy, exactly where and when, to maximize modeled reduction in human heat exposure while satisfying equity, feasibility, and policy constraints?**

A temperature map alone does not answer that question. Cities still need to choose among shade, trees, cool surfaces, cooling access, and operating resources; respect budgets and equity requirements; understand uncertainty; explain decisions; and verify whether deployed interventions actually worked.

## What ThermalOS does

```text
FortyGuard observed thermal intelligence
                 │
                 ▼
              HeatLens
        heat burden + exposure
                 │
                 ▼
             EquityLens
 vulnerability + transit + access
                 │
                 ▼
           CoolCapital MILP
 intervention candidates + $ budget
                 │
        ┌────────┼─────────┐
        ▼        ▼         ▼
  Robustness   Policy    Copilot
    Engine     Stress     control
        │        │         │
        └────────┼─────────┘
                 ▼
          Action / No Action
                 │
        ┌────────┼───────────┐
        ▼        ▼           ▼
     HeatOps   Dossier   ThermalVerify
                 │
                 ▼
         reviewed local evidence
                 │
                 ▼
          adaptive learning
```

### Product layers

- **HeatLens** — observed FortyGuard thermal burden and exposure.
- **EquityLens** — ACS vulnerability, no-vehicle, transit, and access context.
- **CoolCapital** — mixed-integer optimizer for capital intervention portfolios.
- **Decision Robustness Engine** — re-optimizes across uncertain effect/cost worlds.
- **Policy Stress Lab** — tests impact, equity, distribution, concentration, and conservative policy choices.
- **HeatOps + Cooling Access** — allocates short-horizon operational resources such as hydration, temporary shade, and mobile cooling.
- **ThermalVerify** — freezes baselines and matched controls for 30/90/365-day post-deployment verification.
- **Thermal Copilot** — natural-language control of the real optimizer and analysis stack.
- **Trust Center + Decision Dossier** — evidence provenance, claim boundaries, and an auditable PDF brief.
- **Adaptive Learning** — reviewed local verification evidence can update intervention priors through conservative shrinkage.

## 60-second demo

The core system can be understood through four moments:

1. **Phoenix — act.** ThermalOS converts observed heat into a ~$2M, 30-project portfolio that satisfies the frozen 35% equity floor.
2. **Stress the decision.** The same portfolio is re-optimized across 32 uncertain worlds and multiple policy regimes; feasibility and project stability are reported instead of hidden.
3. **Los Angeles — abstain.** The observed event does not cross the frozen intervention threshold, so ThermalOS returns **no action**, spends **$0**, and labels portfolio-only metrics as not applicable.
4. **Verify and explain.** Export the Decision Dossier, inspect Evidence Ledger provenance, ask the Thermal Copilot why an intervention was or was not selected, and prepare ThermalVerify matched controls for post-deployment measurement.

That **act → stress → abstain → verify** loop is the core product.

## Evidence

ThermalOS was exercised across **six U.S. states** using the same decision architecture.

| City | Evaluation role | Final state | Projects | Spend | Equity-aligned spend |
|---|---|---:|---:|---:|---:|
| Miami-Dade, FL | Development | Action plan | 30 | $1,999,381 | 100.0% |
| Houston, TX | Development | Action plan | 27 | $1,999,499 | 72.9% |
| Phoenix, AZ | Prospective blind | Action plan | 30 | $1,999,457 | 36.2% |
| Atlanta, GA | Blind city; final v3.1 replay | Action plan | 29 | $1,998,958 | 82.4% |
| Los Angeles, CA | Blind city; final v3.1 replay | **No action triggered** | 0 | $0 | N/A |
| Las Vegas, NV | Post-blind follow-up | Action plan | 30 | $1,999,573 | 96.8% |

Across all **five action-triggered cities**:

- **32/32** robustness worlds were feasible.
- **100%** of tested policy scenarios were feasible.
- **100%** of the tested equity frontier was feasible.
- Evidence Ledger minimum coverage was **100%**.
- Every action portfolio met the frozen **35% equity-spend floor**.

Los Angeles is intentionally different: the observed event produced no positive baseline heat burden under the frozen threshold, so the correct response was **0 projects and $0 spend**, not a forced portfolio.

### Evaluation provenance

The evaluation record is intentionally transparent:

- **Miami and Houston** are development cities.
- **Phoenix** is the clean prospective blind pass.
- **Atlanta's original blind run** failed the evidence gate because GTFS was missing. Official MARTA GTFS was then imported; the final v3.1 replay passed.
- **Los Angeles' original blind run** exposed an empty-candidate crash on a legitimate no-trigger event. v3.1 added explicit no-action handling; the frozen heat threshold was not lowered or retuned.
- **Las Vegas** is a post-blind follow-up city. Its 40°C threshold was matched to Phoenix before the Las Vegas result was observed.

## Final validation

The frozen release passed:

```text
33 / 33 automated tests
26 / 26 adversarial Feature Gauntlet checks
0 hard failures in the final release stress audit
6 / 6 city/system states valid
```

Negative controls also verify that:

- an impossible 101% equity requirement is rejected;
- Atlanta cannot silently pass without required GTFS evidence;
- zero-burden Los Angeles produces a clean no-action state;
- ThermalVerify does not invent effects when matched controls are unavailable;
- HeatOps does not dispatch resources when heat burden is zero;
- extreme adaptive-learning observations cannot move priors outside configured bounds.

**Frozen core decision-pipeline code/config SHA-256**

```text
80bbb90eb440e6fcce5ae8cb25fe264707b5157d90412d4a28fbfe6087e87eff
```

See [`RELEASE_MANIFEST.md`](RELEASE_MANIFEST.md) for the complete release record.

## Why FortyGuard matters

ThermalOS deliberately does **not** try to replace FortyGuard.

FortyGuard is the observed **temperature-intelligence layer**. ThermalOS is the **decision and action layer** built on top of it:

```text
FortyGuard: Where is the heat?
ThermalOS:  What actions should the city take under real constraints?
```

Sparse morphology is used for intervention suitability, explanation, and scenario context—not as a claim that land-cover fractions alone can reliably forecast unseen-city temperature.

## Scientific guardrail

ThermalOS is a decision-support research prototype.

Intervention effects are configurable **literature-bounded scenario priors**, not learned causal effects. Observational temperature/morphology relationships do not establish intervention causality. ThermalVerify is the mechanism for collecting reviewed post-deployment local evidence before local priors are updated.

Multi-state evaluation tests portability of the **data contract, decision architecture, optimization, robustness, policy, evidence, and verification workflow**. It does not prove identical causal effects across climates.

## Run locally

### 1. Create the environment

Windows CMD:

```cmd
py -3.10 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -e .[dev]
```

### 2. Run the tests

```cmd
.venv\Scripts\python.exe -m pytest -q
```

Expected release result:

```text
33 passed
```

### 3. Launch ThermalOS

```cmd
.venv\Scripts\python.exe -m streamlit run app.py
```

The interactive demo is offline-first and can run from bundled processed data without waiting on live FortyGuard API latency.

## Useful CLI workflows

```cmd
.venv\Scripts\python.exe scripts\run_robustness.py --city phoenix --budget 2000000 --scenarios 32 --pool-size 1200
.venv\Scripts\python.exe scripts\run_policy_stress_lab.py --city phoenix --budget 2000000
.venv\Scripts\python.exe scripts\prepare_verification.py --city phoenix --budget 2000000
.venv\Scripts\python.exe scripts\run_heatops.py --city phoenix --budget 60000
.venv\Scripts\python.exe scripts\run_copilot.py --city phoenix --prompt "Compare a $2M plan with a $4M plan."
.venv\Scripts\python.exe scripts\generate_dossier.py --city phoenix --budget 2000000
.venv\Scripts\python.exe scripts\audit_evidence.py --city phoenix
```

## Optional live-data setup

Copy `.env.example` to `.env` locally and add your own credentials:

```env
FORTYGUARD_API_KEY=...
FORTYGUARD_BASE_URL=https://api.fortyguard.com
CENSUS_API_KEY=...
```

**Never commit or distribute `.env`.**

GTFS feeds can be operator-supplied; the Miami helper defaults to the official Miami-Dade Transit GTFS feed when no URL is provided.

## Repository map

```text
ThermalOS/
├── app.py
├── README.md
├── RELEASE_MANIFEST.md
├── configs/
├── data/
├── docs/
├── outputs/      # generated at runtime
├── scripts/
├── src/thermalos/
└── tests/
```

The release package should exclude `.env`, `.venv`, caches, archived pre-v3.1 source backups, temporary diagnostics, and any secrets.

## Claim boundary

ThermalOS is not a medical tool, engineering design, procurement recommendation, or causal impact evaluation. Population and access fields are planning proxies. The MILP is optimal/feasible relative to its supplied candidate set, assumptions, objective, and constraints.

**ThermalOS converts observed heat intelligence into auditable decisions—and knows when not to act.**

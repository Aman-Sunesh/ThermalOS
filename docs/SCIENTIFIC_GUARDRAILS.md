# Scientific Guardrails

> **Protocol update (2026-08-23):** Houston outcomes have already been inspected, so Houston is now a **development city**, not a blind test. The current prospective blind set is Phoenix (AZ), Atlanta (GA), and Los Angeles (CA). `GENERALIZATION_PROTOCOL.md` is authoritative if older sections below use historical Houston-held-out language.


ThermalOS is ambitious by design, but the value of the product depends on being explicit about uncertainty and evidentiary limits.

## 1. Observational morphology is not causal intervention evidence

A greener tile can be cooler for many reasons besides canopy. The learned thermal-response model is therefore an **observational diagnostic model**. It does not calibrate intervention effects in the production/demo pipeline; those remain bounded by explicit priors until reviewed ThermalVerify evidence exists.

Do not say:

> Adding 10% canopy will causally cool this tile by exactly 1.4°C.

Prefer:

> Under the configured planning evidence model, the intervention is estimated to reduce modeled exposure burden within the stated uncertainty range.

## 2. Upstream model-derived data must still be labeled

FortyGuard outputs are the primary thermal-intelligence source. ThermalOS describes them as real/model-derived upstream planning data rather than pretending every value is a direct thermometer observation.

## 3. Sparse satellite segmentation is a planning proxy

The current Miami morphology enrichment uses 15 FortyGuard satellite samples propagated to nearby thermal tiles. Composite morphology uses non-additive class handling to avoid obvious overlap double counting.

It is not parcel-exact land-cover mapping.

## 4. Shade is not equivalent to ambient air cooling

Shade can strongly reduce radiant exposure while only weakly changing neighborhood air temperature. ThermalOS therefore separates ambient-temperature and direct-exposure mechanisms.

## 5. Cool pavement has radiant trade-offs

Reflective pavement may reduce surface temperature while increasing short-wave radiant load in some pedestrian settings. The configured effect is conservative and can include a radiant penalty.

## 6. Tree cooling is context dependent

Cooling depends on species/traits, canopy density, irrigation/soil moisture, background climate and urban geometry. Local maximum cooling values should never be applied as universal neighborhood means.

## 7. Cost assumptions are policy/scenario inputs

Capital and HeatOps costs are planning bundles unless a dated local procurement source is attached. The optimizer is mathematically exact with respect to its inputs; that does not make an uncertain cost input procurement-grade.

## 8. Population and access are proxies

Allocated residential population does not equal pedestrian presence. GTFS stop density does not equal ridership. No-vehicle households are not a complete mobility model.

The UI uses terms such as **exposure-weighted population** and **transit proxy** rather than claiming exact people protected.

## 9. Cooling-center access requires a verified inventory

CDC evidence supports transportation/access as an important barrier to cooling-center use. HeatOps therefore models mobility friction. If an actual cooling-center inventory is unavailable, the system explicitly labels the result as a transit/mobility access proxy and does not invent facilities.

## 10. Robustness frequency is not a posterior probability

A project selected in 90% of ThermalOS stress-test scenarios is stable under the configured perturbation experiment. It does **not** mean there is a 90% probability that the project will succeed.

## 11. Policy Stress Lab is normative, not scientific truth

Impact-first, equity-first and distributed portfolios encode legitimate value choices. ThermalOS exposes their trade-offs rather than declaring one moral/political objective universally correct.

## 12. Direct and system-level effects are different metrics

The MILP uses direct first-order candidate benefits. Post-selection counterfactual reporting adds distance-decay spillover and overlap correction. These quantities must never share an ambiguous label.

Use:

- **Direct modeled relief / first-order person-hours**;
- **System-level modeled relief / spillover-inclusive scenario**.

## 13. ThermalVerify does not make the system causal by itself

Matched-control pre/post comparison is an evaluation scaffold, not a complete causal study. Stronger production evaluation may require repeated time-matched observations, weather normalization, difference-in-differences, synthetic controls, randomized phased rollout or other quasi-experimental designs.

No post-deployment value is generated until a real post-deployment observation is supplied.

## 14. Adaptive learning requires reviewed evidence

The adaptive hook only previews local-prior updates after a minimum number of reviewed outcomes. It never silently rewrites evidence priors or learns from its own predictions.

This prevents model self-confirmation loops.

## 15. HeatOps is operational decision support, not emergency/medical advice

Temporary hydration/shade/mobile-cooling plans are planning scenarios. Official emergency management agencies, weather services and public-health authorities remain the source for actual emergency instructions.

## 16. Houston transfer is not causal transportability

Miami → Houston can show that the representation and decision workflow transfers to a distinct urban setting. It does not show that every intervention has identical causal effectiveness in both climates.

If Houston is running the bundled demo data, the UI must say **synthetic transfer dataset**.

## 17. Copilot cannot bypass formal constraints

The Copilot converts natural language into typed planning controls and invokes the same optimizer. Free-form generated text is not allowed to create a portfolio outside the mathematical decision engine.

## 18. Evidence Ledger is part of the model, not decoration

Every important decision output should be traceable to an evidence class:

- observed/model-derived upstream;
- observed GIS;
- derived proxy;
- modeled scenario;
- policy assumption;
- synthetic demo.

## 19. Offline-first demo reliability

Live external endpoints are not single points of failure during interactive use. Refresh/harvest is a separate workflow; prepared data are used for the application.

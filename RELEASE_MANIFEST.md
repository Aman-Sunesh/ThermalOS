# ThermalOS Release Manifest

**Release:** `v1.0.0-hackathon`  
**Frozen:** 2026-08-24 01:14 GST (UTC+4)  
**Final pipeline SHA-256:** `80bbb90eb440e6fcce5ae8cb25fe264707b5157d90412d4a28fbfe6087e87eff`  
**Pipeline files hashed:** 45  
**Original pre-blind SHA-256:** `0fa780d55c29a4c152d5933bd072995c780389aad64e98e22d1f39b805e57204`

## Validation

- Current-tree re-validation: **33/33 automated tests passed**.
- Current-tree re-validation: **`compileall` passed**.
- Prior frozen adversarial audit: **26/26 Feature Gauntlet checks passed**.
- Prior frozen hard-stress audit: **0 hard failures; 6/6 city/system states valid**.
- The frozen scientific decision-pipeline code/config remains unchanged at the SHA above.
- Subsequent repository changes were limited to UI, CLI, packaging, and documentation fixes outside the frozen pipeline set; decision logic, thresholds, processed evaluation data, optimization behavior, and frozen scientific settings were unchanged.
- Invalid equity fractions are rejected.
- Missing Atlanta GTFS fails the data contract.
- Los Angeles zero-burden behavior returns an explicit no-action state.
- Failure-injection files were restored byte-for-byte in the prior hardening audit.

## Frozen decision settings

- Capital budget: **$2,000,000**
- Equity-spend floor: **35%**
- Robustness worlds: **32**
- Morphology-only unseen-city temperature prediction claim: **false**

## Final evaluation

| City | Role | Decision | Projects | Spend | Equity spend | Robustness | Evidence |
|---|---|---:|---:|---:|---:|---:|---:|
| Miami-Dade, FL | Development | Action plan | 30 | $1,999,381.45 | 100.0% | 32/32 | 100% |
| Houston, TX | Development | Action plan | 27 | $1,999,499.16 | 72.9% | 32/32 | 100% |
| Phoenix, AZ | Prospective blind | Action plan | 30 | $1,999,457.06 | 36.2% | 32/32 | 100% |
| Atlanta, GA | Blind; final v3.1 replay | Action plan | 29 | $1,998,957.91 | 82.4% | 32/32 | 100% |
| Los Angeles, CA | Blind; final v3.1 replay | **No action triggered** | 0 | $0 | N/A | N/A | 100% |
| Las Vegas, NV | Post-blind follow-up | Action plan | 30 | $1,999,572.90 | 96.8% | 32/32 | 100% |

### Evaluation provenance

- **Miami and Houston** are development cities.
- **Phoenix** is the clean prospective blind pass.
- **Atlanta:** the original blind run failed the evidence gate because GTFS was missing. Official MARTA GTFS was then imported; the final v3.1 replay passed.
- **Los Angeles:** the original blind run exposed an empty-candidate crash on a legitimate no-trigger event. v3.1 added explicit no-action handling; the threshold was not lowered or retuned.
- **Las Vegas:** post-blind follow-up. Its 40°C threshold was matched to Phoenix before Las Vegas results were observed.

## Scientific claim boundary

FortyGuard supplies the observed upstream thermal field. ThermalOS does **not** claim that sparse morphology replaces FortyGuard or reliably predicts unseen-city temperature. Intervention effects remain configurable literature-bounded planning priors, not learned causal effects. Multi-state evaluation tests portability of the decision architecture, data contract, optimization, robustness, policy, provenance, and verification workflow.

## Packaging exclusions

Do not distribute `.env`, `.venv*/`, caches, archived pre-v3.1 backups, temporary diagnostics, or API keys.

Any code change after this freeze requires a new pipeline hash and re-validation.

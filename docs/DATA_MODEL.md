# Canonical ThermalOS Data Model

ThermalOS converts all sources into one row per approximately 100 m tile.

## Required core columns

| Column | Meaning |
|---|---|
| `tile_id` | Stable local tile identifier |
| `city` | City key (`miami`, `houston`) |
| `area` | Study neighborhood/area |
| `lat`, `lon` | Tile centroid |
| `temperature_c` | Representative afternoon temperature |
| `daily_max_temperature_c` | Daily peak/maximum field |
| `exceedance_h` | Hours above configured threshold |
| `persistence_h` | Longest continuous run above threshold |
| `population` | Residential population assigned to tile or exposure proxy |
| `vulnerability` | 0–1 planning vulnerability index |

## Environmental columns

Optional, neutral-filled if unavailable:

- `apparent_temperature_c`
- `heat_index_c`
- `wet_bulb_c`
- `relative_humidity_pct`
- `aqi_us`
- `solar_ghi`

The FortyGuard environmental endpoint is point-based and can be sampled more sparsely than heatmaps. ThermalOS supports nearest-sample or area-level imputation.

## Morphology columns

All expected as fractions in `[0,1]`:

- `canopy_fraction`
- `pervious_fraction`
- `impervious_fraction`
- `building_fraction`
- `road_fraction`
- `vegetation_fraction`

Sources can include Miami-Dade's official canopy service and FortyGuard Premium segmentation. The feature builder records source provenance.

## Human exposure / POI proxies

Optional:

- `transit_stop_count`
- `school_count`
- `park_count`
- `cooling_center_count`
- `exposure_multiplier`

These are proxies. They do not claim exact pedestrian counts.

## Derived HeatLens columns

- `thermal_stress_index` — robust 0–1 relative planning index
- `exposed_population` — population × transparent POI/exposure multiplier
- `baseline_person_hours` — exposure × duration × stress multiplier
- `equity_weighted_person_hours`
- `high_vulnerability` — Boolean used by optimizer equity constraints

## Candidate-project columns

Candidate generation produces a long table with one row per tile × viable intervention:

- `candidate_id`
- `tile_id`
- `intervention`
- `label`
- `feasibility`
- `cost_usd`
- `temp_relief_c_expected`
- `direct_exposure_relief_fraction`
- `benefit_expected_person_hours`
- `benefit_low_person_hours`
- `benefit_high_person_hours`
- `vulnerability`
- `reason`

## Provenance

Each processed run emits `provenance.json` with:

- source files;
- study date/time;
- thresholds;
- fields that were missing/imputed;
- whether a row came from live, cached, or synthetic demo data;
- intervention configuration hash.

This is important because the shipped sample dataset is explicitly synthetic; real-data claims should use harvested data.

# ThermalOS Evidence Matrix

This matrix is the bridge between the literature review and the intervention configuration. It intentionally records **mechanism, confidence, and caveat** rather than treating every intervention as a fixed °C reduction.

| Intervention | Primary mechanism in ThermalOS | Evidence pattern | MVP representation | Major caveat |
|---|---|---|---|---|
| Tree canopy | Shade + evapotranspiration | Consistent cooling, but magnitude varies strongly with climate, urban morphology, canopy traits, and water availability | Uncertain ambient cooling + modest direct exposure relief; suitability favors canopy gaps + pervious space | Observational canopy-temperature association is not causal; water/maintenance constraints matter |
| Shade structures | Solar/radiant shielding | Strong pedestrian thermal-comfort benefit can occur with little ambient-air change | Small ambient effect + large direct exposure relief | Benefit is highly geometry/time/orientation dependent |
| Cool pavement | Surface energy balance / albedo | Can reduce surface temperature; air/comfort effects are smaller/mixed | Conservative ambient effect; suitability requires impervious/road surface; configurable radiant penalty | Reflected short-wave radiation can worsen pedestrian radiant comfort |
| Cool roofs | Roof albedo / building energy balance | Strong roof/building effect; neighborhood air cooling generally more modest | Modest outdoor-tile ambient effect; suitability dominated by building fraction | Core MVP does not monetize building-energy savings, so it may understate total benefit |
| Cooling/hydration node | Protected exposure / access | Directly reduces duration/intensity of outdoor exposure for people who can access it | No ambient cooling; direct exposure-relief fraction; suitability emphasizes vulnerability/activity/access gaps | Capacity, opening hours, accessibility, awareness, and behavior are not fully modeled |

## Configuration philosophy

`configs/interventions.yaml` contains configurable planning priors. Means, bounds, costs, and spillover scales are inputs that can be replaced without changing the optimizer. The UI should report **scenario estimates** and uncertainty, never imply that these defaults are municipal engineering specifications.

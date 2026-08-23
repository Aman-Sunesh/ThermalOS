from __future__ import annotations

import os

import pandas as pd
import requests


DEFAULT_VARIABLES = {
    "population": "B01003_001E",
    "poverty_universe": "B17001_001E",
    "poverty_count": "B17001_002E",
    "households_total": "B08201_001E",
    "no_vehicle_households": "B08201_002E",
}


class CensusError(RuntimeError):
    pass


def fetch_acs(
    state_fips: str,
    county_fips: str,
    *,
    geography: str = "block group",
    api_key: str | None = None,
    year: int = 2024,
    variables: dict[str, str] | None = None,
    timeout_s: float = 60.0,
) -> pd.DataFrame:
    """Fetch ACS 5-year attributes for tracts or block groups.

    For ThermalOS, block groups are preferred in Miami-Dade because the official
    tree-canopy layer is also keyed by 12-digit block-group GEOID.
    """
    if geography not in {"tract", "block group"}:
        raise ValueError("geography must be 'tract' or 'block group'")

    api_key = api_key or os.getenv("CENSUS_API_KEY")
    if not api_key:
        raise CensusError("Set CENSUS_API_KEY for Census Data API access")

    variables = variables or DEFAULT_VARIABLES
    get_vars = ["NAME", *variables.values()]
    params = {
        "get": ",".join(get_vars),
        "for": f"{geography}:*",
        "in": f"state:{state_fips} county:{county_fips}",
        "key": api_key,
    }
    url = f"https://api.census.gov/data/{year}/acs/acs5"
    r = requests.get(url, params=params, timeout=timeout_s)
    if not r.ok:
        raise CensusError(f"ACS request -> {r.status_code}: {r.text[:1000]}")

    rows = r.json()
    if len(rows) < 2:
        return pd.DataFrame()
    df = pd.DataFrame(rows[1:], columns=rows[0])
    rename = {v: k for k, v in variables.items()}
    df = df.rename(columns=rename)
    for col in rename.values():
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if geography == "block group":
        df["geoid_blockgroup"] = (
            df["state"].astype(str)
            + df["county"].astype(str)
            + df["tract"].astype(str)
            + df["block group"].astype(str)
        )
        df["geoid_tract"] = df["geoid_blockgroup"].str[:11]
    else:
        df["geoid_tract"] = (
            df["state"].astype(str)
            + df["county"].astype(str)
            + df["tract"].astype(str)
        )

    if "poverty_count" in df and "poverty_universe" in df:
        denom = df["poverty_universe"].replace(0, pd.NA)
        df["poverty_fraction"] = (df["poverty_count"] / denom).clip(0, 1)

    if "no_vehicle_households" in df and "households_total" in df:
        denom = df["households_total"].replace(0, pd.NA)
        df["no_vehicle_fraction"] = (
            df["no_vehicle_households"] / denom
        ).clip(0, 1)

    return df


def fetch_acs_tracts(
    state_fips: str,
    county_fips: str,
    **kwargs,
) -> pd.DataFrame:
    """Backward-compatible tract wrapper."""
    return fetch_acs(
        state_fips,
        county_fips,
        geography="tract",
        **kwargs,
    )


def fetch_acs_blockgroups(
    state_fips: str,
    county_fips: str,
    **kwargs,
) -> pd.DataFrame:
    return fetch_acs(
        state_fips,
        county_fips,
        geography="block group",
        **kwargs,
    )

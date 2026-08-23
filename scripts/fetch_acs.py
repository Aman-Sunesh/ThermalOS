from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from thermalos.config import available_cities, city_config
from thermalos.data.census import fetch_acs_blockgroups, fetch_acs_tracts


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--city", choices=available_cities(), default="miami")
    p.add_argument(
        "--geography",
        choices=["block-group", "tract"],
        default="block-group",
        help="Block groups are preferred for ThermalOS exposure/equity joins.",
    )
    args = p.parse_args()

    load_dotenv()
    cfg = city_config(args.city)
    if args.geography == "block-group":
        df = fetch_acs_blockgroups(cfg["fips_state"], cfg["fips_county"])
        filename = "acs_blockgroups_2024.csv"
    else:
        df = fetch_acs_tracts(cfg["fips_state"], cfg["fips_county"])
        filename = "acs_tracts_2024.csv"

    out = Path("data/raw") / cfg["city_key"]
    out.mkdir(parents=True, exist_ok=True)
    path = out / filename
    df.to_csv(path, index=False)
    print(f"Saved {len(df):,} ACS rows -> {path}")
    print("Columns:", ", ".join(df.columns))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path

from thermalos.config import available_cities, city_config
from thermalos.data.arcgis import query_feature_layer_geojson


ACS_2024_BLOCKGROUP_LAYER = (
    "https://tigerweb.geo.census.gov/arcgis/rest/services/"
    "TIGERweb/Tracts_Blocks/MapServer/8"
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--city", choices=available_cities(), default="miami")
    args = p.parse_args()

    cfg = city_config(args.city)
    where = f"STATE='{cfg['fips_state']}' AND COUNTY='{cfg['fips_county']}'"
    print(f"Fetching Census ACS-2024 block-group geography for {cfg['name']}...")
    geo = query_feature_layer_geojson(
        ACS_2024_BLOCKGROUP_LAYER,
        where=where,
        out_fields="GEOID,STATE,COUNTY,TRACT,BLKGRP,AREALAND",
        page_size=2000,
        timeout_s=90.0,
    )
    out = Path("data/raw") / cfg["city_key"] / "census_blockgroups_2024.geojson"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(geo), encoding="utf-8")
    print(f"Saved {len(geo.get('features', [])):,} Census block-group polygons -> {out}")


if __name__ == "__main__":
    main()

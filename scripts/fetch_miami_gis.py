from __future__ import annotations

import json
from pathlib import Path

from thermalos.config import city_config
from thermalos.data.arcgis import ArcGISError, layer_metadata, query_feature_layer_geojson


def main() -> None:
    cfg = city_config("miami")
    out = Path("data/raw/miami")
    out.mkdir(parents=True, exist_ok=True)

    # Tree canopy is the required official county layer. Cooling centers and
    # schools are useful exposure/operations context but must not block the
    # pipeline if their third-party service is unavailable.
    sources = [
        ("tree_canopy_blockgroups", cfg["external"]["tree_canopy_layer"], True),
        ("cooling_centers", cfg["external"]["cooling_centers_layer"], False),
        ("public_schools", cfg["external"]["public_schools_layer"], False),
    ]

    required_failures = []
    for name, url, required in sources:
        print("Fetching", name, "(required)" if required else "(optional)")
        try:
            metadata = layer_metadata(url)
            data = query_feature_layer_geojson(url)
            (out / f"{name}.metadata.json").write_text(
                json.dumps(metadata, indent=2), encoding="utf-8"
            )
            (out / f"{name}.geojson").write_text(
                json.dumps(data), encoding="utf-8"
            )
            fields = [f.get("name") for f in metadata.get("fields", [])]
            print(" ", len(data.get("features", [])), "features; fields:", fields[:20])
        except Exception as exc:
            print(f"  {name} FAILED: {exc}")
            if required:
                required_failures.append(f"{name}: {exc}")

    if required_failures:
        raise ArcGISError("Required Miami-Dade GIS source failed: " + " | ".join(required_failures))


if __name__ == "__main__":
    main()

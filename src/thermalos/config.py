from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs"


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _city_config_paths() -> dict[str, Path]:
    out: dict[str, Path] = {}
    reserved = {"data_sources", "interventions", "operations", "generalization"}
    for path in CONFIG_DIR.glob("*.yaml"):
        if path.stem in reserved:
            continue
        try:
            cfg = load_yaml(path)
        except Exception:
            continue
        if "city_key" not in cfg:
            continue
        key = str(cfg["city_key"]).lower().replace("-", "_")
        out[key] = path
        # Filename and common display-name aliases are also accepted.
        out.setdefault(path.stem.lower().replace("-", "_"), path)
    return out


def available_cities() -> list[str]:
    """Canonical city keys with a config file."""
    keys = []
    seen = set()
    for path in sorted(set(_city_config_paths().values())):
        cfg = load_yaml(path)
        key = str(cfg["city_key"]).lower().replace("-", "_")
        if key not in seen:
            seen.add(key)
            keys.append(key)
    return keys


def city_config(city: str) -> dict[str, Any]:
    key = city.lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "miami_dade": "miami",
        "la": "los_angeles",
        "losangeles": "los_angeles",
        "phoenix_maricopa": "phoenix",
        "atlanta_fulton": "atlanta",
    }
    key = aliases.get(key, key)
    mapping = _city_config_paths()
    if key not in mapping:
        raise KeyError(f"Unknown city {city!r}; choose one of {available_cities()}")
    return load_yaml(mapping[key])


def generalization_config() -> dict[str, Any]:
    return load_yaml(CONFIG_DIR / "generalization.yaml")


def intervention_config() -> dict[str, Any]:
    return load_yaml(CONFIG_DIR / "interventions.yaml")


# Backward/ergonomic alias used by CLI scripts.
def interventions_config() -> dict[str, Any]:
    return intervention_config()

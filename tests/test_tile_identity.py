from pathlib import Path

import pandas as pd

from thermalos.config import interventions_config
from thermalos.models.interventions import build_candidates


def test_real_miami_tile_and_candidate_ids_are_citywide_unique():
    path = Path("data/processed/miami_tiles.csv")
    if not path.exists():
        return
    tiles = pd.read_csv(path)
    assert "source_tile_id" in tiles.columns
    assert tiles["tile_id"].is_unique
    candidates = build_candidates(tiles, interventions_config(), seed=42).candidates
    assert candidates["candidate_id"].is_unique

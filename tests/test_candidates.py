from thermalos.config import interventions_config
from thermalos.demo import generate_demo_city
from thermalos.models.interventions import build_candidates


def test_candidates_are_bounded():
    tiles = generate_demo_city("miami", n_side=3)
    c = build_candidates(tiles, interventions_config(), seed=1).candidates
    assert len(c) > 0
    assert (c["cost_usd"] > 0).all()
    assert (c["benefit_low_person_hours"] <= c["benefit_expected_person_hours"] + 1e-9).all()
    assert (c["benefit_expected_person_hours"] <= c["benefit_high_person_hours"] + 1e-9).all()
    assert (c["benefit_high_person_hours"] <= c["baseline_person_hours"] + 1e-9).all()

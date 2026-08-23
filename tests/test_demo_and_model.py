import numpy as np
from sklearn.model_selection import train_test_split

from thermalos.demo import generate_demo_city
from thermalos.models.thermal_twin import ObservationalThermalTwin


def test_demo_is_deterministic():
    a = generate_demo_city("miami", n_side=3, seed=9)
    b = generate_demo_city("miami", n_side=3, seed=9)
    assert np.allclose(a["temperature_c"], b["temperature_c"])


def test_observational_model_runs():
    df = generate_demo_city("miami", n_side=4)
    tr, te = train_test_split(df, test_size=0.25, random_state=1)
    model = ObservationalThermalTwin().fit(tr)
    m = model.evaluate(te)
    assert m.n > 0
    sens = model.canopy_sensitivity(te)
    assert len(sens) == len(te)
    assert (sens >= 0).all()

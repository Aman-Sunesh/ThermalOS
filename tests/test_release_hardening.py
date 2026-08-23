import pytest

from thermalos.system_transfer import apply_system_gates


@pytest.mark.parametrize("bad_floor", [-0.01, 1.01, -1.0, 2.0])
def test_system_gate_rejects_invalid_equity_fraction(bad_floor):
    with pytest.raises(ValueError, match="equity_min_fraction"):
        apply_system_gates({}, {}, equity_min_fraction=bad_floor)

"""Public KS quadrature selection, without running a scientific solve."""

import pytest
from atomiverse._binding import _cpp, extension_available

pytestmark = pytest.mark.skipif(
    not extension_available() or not hasattr(_cpp, "RKS"),
    reason="requires the HF/KS extension",
)


@pytest.mark.parametrize("method,kwargs,default", [
    ("RKS", {"basis": "STO-3G", "xc": "PBE"}, "MatchPySCF"),
    ("UKS", {"basis": "STO-3G", "n_alpha": 1, "n_beta": 0}, "SG0"),
])
def test_ks_grid_policy(method, kwargs, default):
    solver = getattr(_cpp, method)(**kwargs)
    assert solver.grid == default
    for policy in ["MatchPySCF", "SG0", "SG1"]:
        solver.grid = policy
        assert solver.grid == policy
    with pytest.raises(ValueError, match="SG0, SG1 or MatchPySCF"):
        solver.grid = "unknown"
    assert solver.grid == "SG1"

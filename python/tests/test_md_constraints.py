"""SHAKE/RATTLE/LINCS configuration is shared by the drivers and task replay."""
import pytest
from _domains import skip_module_unless
skip_module_unless("ATV_BUILD_MD")
from atomiverse._binding import _cpp
from atomiverse.md import MD, FixLincs, FixShake, FixRattle
from atomiverse.md.integrator import VelocityVerlet
from atomiverse.potentials.pair import LJCut
from atomiverse.neighbor import NeighborList

@pytest.mark.parametrize("constraint", [FixShake, FixRattle, FixLincs])
def test_constraint_rejects_invalid_parameters(constraint):
    for dt in [0, -1, float("inf"), float("nan")]:
        with pytest.raises(ValueError):
            constraint(dt)
    rule = constraint(0.5)
    with pytest.raises(ValueError):
        rule.constrain_bond(-1, 1.09)
    with pytest.raises(ValueError):
        rule.constrain_angle(0, float("nan"))
    rule.constrain_bond(1, 1.09)
    with pytest.raises(ValueError):
        rule.constrain_bond(1, 1.1)

@pytest.mark.parametrize("constraint", [FixShake, FixRattle])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_constraint_registers_on_native_driver(constraint, device):
    if device == "cuda" and not hasattr(_cpp, "_CudaMD"):
        pytest.skip("CUDA driver not built")
    fix = constraint(0.5, tolerance=1e-4, max_iter=900)
    fix.constrain_bond(0, 1.09)
    fix.constrain_angle(1, 1.72)
    sim = MD(device=device)
    sim.set_integrator(VelocityVerlet(0.5))
    sim.set_evaluator(LJCut(1.0, 1.0, 2.0))
    sim.set_neighbor(NeighborList(2.0, 0.5))
    sim.set_observer(fix)
    sim.build()


def test_lincs_registers_on_cpu_and_is_refused_on_cuda():
    with pytest.raises(ValueError):
        FixLincs(0.5, order=0)
    fix = FixLincs(0.5, tolerance=1e-4, max_iter=900, order=6)
    fix.constrain_bond(0, 1.09)
    sim = MD(device="cpu")
    sim.set_integrator(VelocityVerlet(0.5))
    sim.set_evaluator(LJCut(1.0, 1.0, 2.0))
    sim.set_neighbor(NeighborList(2.0, 0.5))
    sim.set_observer(fix)
    sim.build()
    if hasattr(_cpp, "_CudaMD"):
        with pytest.raises(ValueError, match="cpu"):
            MD(device="cuda").set_observer(fix)

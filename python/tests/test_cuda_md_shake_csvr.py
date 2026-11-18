"""SHAKE on X–H bonds together with species CSVR(types, bias="com") on
``MD(device="cuda")``: each species' temperature, counted over the degrees of
freedom the constraints leave, N_f = 3 N_g − 3 − (constraints owned by the
group's atoms), settles at the setpoint.

64 X–H dimers (types 1, 2; one constrained bond each) and 64 free atoms
(type 3), both groups at 50 K (one setpoint, so no heat flows between the
groups and each must sit at it). Counting the dimer group over 3 N_g − 3
would read ~17 % low, outside the 6 % band. The CPU run is identical (same
random stream) and must agree.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import molpy

from atomiverse._binding import _cpp, extension_available
from _domains import skip_module_unless

skip_module_unless("ATV_BUILD_MD")

from atomiverse.md import MD, FixShake
from atomiverse.md.integrator import VelocityVerlet
from atomiverse.md.thermostat import CSVR
from atomiverse.neighbor import NeighborList
from atomiverse.potentials.pair import LJCut

_HAS_CUDA_MD: bool = extension_available() and hasattr(_cpp, "_CudaMD")
needs_cuda = pytest.mark.skipif(not _HAS_CUDA_MD, reason="needs ATV_BUILD_CUDA")

_KB = 8.617343e-5  # [eV/K] (metal)
_MVV2E = 1.0364269e-4  # amu Å²/ps² → eV
_SIDE = 4
_SPACING = 5.0  # [Å]
_BOND = 1.09  # [Å]
_MASS = {1: 12.011, 2: 1.008, 3: 6.94}


def _mixture(seed: int = 5) -> "molpy.Frame":
    rng = np.random.default_rng(seed)
    pos, types, bonds = [], [], []
    for ix in range(_SIDE):
        for iy in range(_SIDE):
            for iz in range(_SIDE):
                c = (np.array([ix, iy, iz]) + 0.25) * _SPACING
                u = rng.normal(size=3)
                u /= np.linalg.norm(u)
                bonds.append((len(pos), len(pos) + 1))
                pos += [c, c + _BOND * u]
                types += [1, 2]
                pos.append(c + 0.5 * _SPACING)
                types.append(3)
    pos = np.asarray(pos)
    types = np.asarray(types, dtype=np.uint32)
    mass = np.array([_MASS[int(t)] for t in types])
    vel = rng.normal(size=pos.shape) * np.sqrt(_KB * 60.0 / (mass * _MVV2E))[:, None]
    vel -= (mass[:, None] * vel).sum(axis=0) / mass.sum()
    atoms = molpy.Block()
    for c, k in enumerate("xyz"):
        atoms.insert(k, np.ascontiguousarray(pos[:, c]))
        atoms.insert("v" + k, np.ascontiguousarray(vel[:, c]))
    atoms.insert("mass", mass)
    atoms.insert("type_id", types)
    atoms.insert("atomic_number", np.where(types == 2, 1, np.where(types == 1, 6, 3)).astype(np.uint32))
    frame = molpy.Frame()
    frame["atoms"] = atoms
    b = np.asarray(bonds, dtype=np.uint32)
    block = molpy.Block()
    block.insert("atomi", np.ascontiguousarray(b[:, 0]))
    block.insert("atomj", np.ascontiguousarray(b[:, 1]))
    block.insert("type_id", np.full(len(b), 1, dtype=np.uint32))  # molpy, 1-based
    frame["bonds"] = block
    frame.box = molpy.Box.cube(_SIDE * _SPACING)
    return frame


def _group_temperature(frame: "molpy.Frame", types, n_constraints: int) -> float:
    a = frame["atoms"]
    sel = np.isin(np.asarray(a["type_id"]), types)
    m = np.asarray(a["mass"], dtype=np.float64)[sel]
    v = np.stack([np.asarray(a[k], dtype=np.float64)[sel] for k in ("vx", "vy", "vz")], axis=1)
    v = v - (m[:, None] * v).sum(axis=0) / m.sum()
    dof = 3 * m.size - 3 - n_constraints
    return float(np.sum(m[:, None] * v * v) * _MVV2E / (dof * _KB))


def _bond_lengths(frame: "molpy.Frame") -> np.ndarray:
    a = frame["atoms"]
    xyz = np.stack([np.asarray(a[k], dtype=np.float64) for k in "xyz"], axis=1)
    i = np.asarray(frame["bonds"]["atomi"], dtype=int)
    j = np.asarray(frame["bonds"]["atomj"], dtype=int)
    d = xyz[j] - xyz[i]
    d -= _SIDE * _SPACING * np.round(d / (_SIDE * _SPACING))
    return np.linalg.norm(d, axis=1)


def _run(device: str):
    frame = _mixture()
    sim = MD(device=device, units="metal")
    sim.set_neighbor(NeighborList(8.5, 0.5))
    sim.set_evaluator(LJCut(0.0104, 3.4, 8.5, s12=0.0))
    sim.set_integrator(VelocityVerlet(0.001))
    shake = FixShake(0.001, tolerance=1e-8, max_iter=500)
    shake.constrain_bond(1, _BOND)  # molpy type_id 1, as stored
    sim.set_observer(shake)
    sim.set_thermostat(CSVR(50.0, 0.02, 0.001, 31, types=[1, 2], bias="com"))
    sim.set_thermostat(CSVR(50.0, 0.02, 0.001, 32, types=[3], bias="com"))
    temps = []
    for _ in range(80):
        sim.run(frame, 25)
        temps.append(
            (
                _group_temperature(frame, [1, 2], 64),
                _group_temperature(frame, [1, 2], 0),
                _group_temperature(frame, [3], 0),
            )
        )
    return frame, np.asarray(temps[30:])


@needs_cuda
def test_shake_species_csvr_group_temperatures() -> None:
    frame, late = _run("cuda")
    np.testing.assert_allclose(_bond_lengths(frame), _BOND, rtol=0, atol=1e-6)
    dimers, dimers_unconstrained, atoms = late.mean(axis=0)
    assert abs(dimers - 50.0) < 0.06 * 50.0, dimers
    assert abs(atoms - 50.0) < 0.06 * 50.0, atoms
    # The constraint DOF matter: the same KE over 3 N_g − 3 reads low.
    assert dimers_unconstrained < 0.9 * 50.0, dimers_unconstrained
    # Same random stream on the CPU: the same trajectory.
    _, cpu_late = _run("cpu")
    np.testing.assert_allclose(late, cpu_late, rtol=1e-6)

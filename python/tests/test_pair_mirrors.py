"""``PairLJCut`` / ``PairCoulCut`` mirror fields: energy shift, per-type rows,
special weights and Coulomb constant, each checked on a two- or three-atom
closed form."""

from __future__ import annotations

import math

import numpy as np
import pytest

from _domains import skip_module_unless

skip_module_unless("ATV_BUILD_MD", "ATV_BUILD_POTENTIALS")

import molpy  # noqa: E402

from atomiverse.md import MD  # noqa: E402
from atomiverse.md.integrator import VelocityVerlet  # noqa: E402
from atomiverse.neighbor import NeighborList  # noqa: E402
from atomiverse.potentials.pair import CoulCut, LJCut, LJTypeCoeff  # noqa: E402

CUTOFF = 8.0
EPS, SIG = 0.01, 3.0


def _frame(xyz, types, charges, bonds=()) -> molpy.Frame:
    frame = molpy.Frame()
    atoms = molpy.Block()
    xyz = np.asarray(xyz, dtype=np.float64) + 10.0
    for col, key in enumerate(("x", "y", "z")):
        atoms.insert(key, xyz[:, col].copy())
    atoms.insert("atomic_number", np.full(len(types), 6, dtype=np.uint64))
    atoms.insert("mass", np.full(len(types), 12.0))
    atoms.insert("type_id", np.asarray(types, dtype=np.uint64))
    atoms.insert("charge", np.asarray(charges, dtype=np.float64))
    frame["atoms"] = atoms
    if bonds:
        b = molpy.Block()
        rows = np.asarray(bonds, dtype=np.uint64)
        b.insert("atomi", rows[:, 0].copy())
        b.insert("atomj", rows[:, 1].copy())
        b.insert("type_id", np.zeros(len(rows), dtype=np.uint64))
        frame["bonds"] = b
    frame.box = molpy.Box.cube(40.0)
    return frame


def _energy(frame, *terms) -> float:
    sim = MD()
    sim.set_integrator(VelocityVerlet(0.5))
    sim.set_neighbor(NeighborList(CUTOFF, 1.0))
    for term in terms:
        sim.set_evaluator(term)
    sim.run(frame, 0)
    return float(frame.meta["energy"])


def _lj(r: float, eps: float = EPS, sig: float = SIG) -> float:
    return 4 * eps * ((sig / r) ** 12 - (sig / r) ** 6)


@pytest.mark.parametrize("shift", [True, False])
def test_single_type_energy_shift_switch(shift: bool) -> None:
    r = 3.5
    energy = _energy(_frame([(0, 0, 0), (r, 0, 0)], [0, 0], [0, 0]),
                     LJCut(EPS, SIG, CUTOFF, energy_shift=shift))
    expected = _lj(r) - (_lj(CUTOFF) if shift else 0.0)
    assert energy == pytest.approx(expected, rel=1e-12)


def test_table_explicit_cross_row_overrides_mixing_and_shift_is_per_pair() -> None:
    r = 3.5
    table = LJCut([LJTypeCoeff(0, 0, EPS, SIG), LJTypeCoeff(1, 1, 0.002, 2.5),
                   LJTypeCoeff(0, 1, 0.005, 3.3)], CUTOFF)
    energy = _energy(_frame([(0, 0, 0), (r, 0, 0)], [0, 1], [0, 0]), table)
    assert energy == pytest.approx(_lj(r, 0.005, 3.3) - _lj(CUTOFF, 0.005, 3.3), rel=1e-12)


@pytest.mark.parametrize("shift", [True, False])
def test_table_energy_shift_switch(shift: bool) -> None:
    r = 3.5
    table = LJCut([LJTypeCoeff(0, 0, EPS, SIG)], CUTOFF, energy_shift=shift)
    energy = _energy(_frame([(0, 0, 0), (r, 0, 0)], [0, 0], [0, 0]), table)
    expected = _lj(r) - (_lj(CUTOFF) if shift else 0.0)
    assert energy == pytest.approx(expected, rel=1e-12)


def test_coulomb_constant_and_special_weights() -> None:
    # Chain 0-1-2-3 on x: 1-2 and 1-3 pairs dropped, the 1-4 pair (0, 3) at
    # weight 0.25, with an explicit constant k.
    k = 14.39914
    q = [0.3, -0.1, 0.2, -0.4]
    frame = _frame([(1.5 * i, 0, 0) for i in range(4)], [0] * 4, q,
                   bonds=[(0, 1), (1, 2), (2, 3)])
    energy = _energy(frame, CoulCut(CUTOFF, 0.0, 0.0, 0.25, coulomb=k))
    assert energy == pytest.approx(0.25 * k * q[0] * q[3] / 4.5, rel=1e-12)
    assert math.isfinite(energy)

"""Gas-phase (vacuum) GAFF by composition: ``lj/cut`` + ``coul/cut`` long cutoffs.

There is no vacuum mode in Atomiverse. molpy's GAFF force field states its
nonbonded part as two styles, ``pair lj/cut`` and ``pair coul/cut``; setting
both styles' ``cutoff`` beyond the molecule, in a box where no periodic image
comes within it, and translating them style by style must reproduce the Amber
``ntb=0, cut=999`` model exactly: every pair, Lorentz-Berthelot LJ, bare
Coulomb with the force field's own constant, its 1-2/1-3/1-4 scales and no
energy shift. The reference is an independent all-pairs NumPy sum built
straight from the molpy force field (not from the translated Atomiverse
tables), with special-bond tiers from a breadth-first walk of the bond graph.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np
import pytest

from _domains import skip_module_unless

skip_module_unless("ATV_BUILD_MD", "ATV_BUILD_MINIMIZER", "ATV_BUILD_POTENTIALS")

import molpy  # noqa: E402

from atomiverse import forcefield as avff  # noqa: E402
from atomiverse.forcefield import KCAL_PER_MOL  # noqa: E402
from atomiverse.md import MD  # noqa: E402
from atomiverse.md.integrator import VelocityVerlet  # noqa: E402
from atomiverse.minimizer import LBFGS, ForceConverged, Minimize  # noqa: E402
from atomiverse.neighbor import NeighborList  # noqa: E402
from atomiverse.potentials.pair import CoulCut, LJCut  # noqa: E402

#: Longer than the molecule (benzyl alcohol spans < 7 Å).
CUTOFF = 30.0
SKIN = 1.0
#: Every periodic image is >= BOX - extent > CUTOFF + SKIN away.
BOX = 80.0


@pytest.fixture(scope="module")
def typed():
    """Benzyl alcohol typed by molpy's GAFF2 typifiers, Gasteiger charges.

    The ring makes some pairs reachable at two bond distances, which the
    nearest-tier rule of the special-bond list has to resolve.
    """
    mol = molpy.io.smiles.read_smiles("OCc1ccccc1").to_atomistic()
    mol, _ = molpy.conformer.Conformer(seed=1).generate(mol)
    labelled = molpy.ff.typifier.AtdTypifier(parameter_set="gaff2").typify(mol)
    typifier = molpy.ff.typifier.GaffTypifier(parameter_set="gaff2")
    typed = typifier.typify(labelled)
    frame = typed.to_frame()
    charges = np.asarray(molpy.ff.charge.GasteigerModel().assign(typed), dtype=np.float64)
    frame["atoms"].insert("charge", charges)
    xyz = _xyz(frame)
    shift = np.full(3, BOX / 2.0) - xyz.mean(axis=0)
    for col, key in enumerate(("x", "y", "z")):
        frame["atoms"][key] = xyz[:, col] + shift[col]
    frame.box = molpy.Box.cube(BOX)
    forcefield = typifier.forcefield()
    for name in ("lj/cut", "coul/cut"):
        forcefield.get_style("pair", name).set(cutoff=CUTOFF)
    return frame, forcefield


def _xyz(frame) -> np.ndarray:
    return np.stack([np.asarray(frame["atoms"][k], dtype=np.float64) for k in ("x", "y", "z")], axis=1)


def _tiers(n: int, bonds: np.ndarray) -> np.ndarray:
    """Bond-graph distance per pair, capped at 4 (0 on the diagonal)."""
    adj: list[list[int]] = [[] for _ in range(n)]
    for i, j in bonds:
        adj[int(i)].append(int(j))
        adj[int(j)].append(int(i))
    tier = np.full((n, n), 4, dtype=int)
    for s in range(n):
        tier[s, s] = 0
        queue = deque([s])
        while queue:
            a = queue.popleft()
            if tier[s, a] >= 3:
                continue
            for b in adj[a]:
                if tier[s, b] > tier[s, a] + 1:
                    tier[s, b] = tier[s, a] + 1
                    queue.append(b)
    return tier


def _reference(frame, forcefield, xyz: np.ndarray) -> tuple[float, np.ndarray]:
    """All-pairs Amber vacuum energy [eV] and forces [eV/Å]."""
    atoms = frame["atoms"]
    n = atoms.n_rows
    lj = {t.itom.name: t.params for t in forcefield.get_style("pair", "lj/cut").get_types()}
    types = [str(t) for t in np.asarray(atoms["type"])]
    eps = np.array([lj[t]["epsilon"] for t in types]) * KCAL_PER_MOL
    sig = np.array([lj[t]["sigma"] for t in types])
    q = np.asarray(atoms["charge"], dtype=np.float64)
    coul = forcefield.get_style("pair", "coul/cut").params
    coulomb = float(coul["coulomb"]) * KCAL_PER_MOL / float(coul.get("dielectric", 1.0))
    s_lj, s_coul = forcefield.special_bonds
    w_lj = np.array([1.0, *s_lj, 1.0])  # by tier 0..4; tier 0 is unused
    w_coul = np.array([1.0, *s_coul, 1.0])
    bonds = np.stack([np.asarray(frame["bonds"][k]) for k in ("atomi", "atomj")], axis=1)
    tier = _tiers(n, bonds)

    energy = 0.0
    forces = np.zeros((n, 3))
    for i in range(n):
        for j in range(i + 1, n):
            d = xyz[i] - xyz[j]
            r = math.sqrt(d @ d)
            e_ij = math.sqrt(eps[i] * eps[j])
            s_ij = 0.5 * (sig[i] + sig[j])
            sr6 = (s_ij / r) ** 6
            a, b = w_lj[tier[i, j]], w_coul[tier[i, j]]
            energy += a * 4.0 * e_ij * (sr6 * sr6 - sr6) + b * coulomb * q[i] * q[j] / r
            dedr = (
                a * -24.0 * e_ij * (2.0 * sr6 * sr6 - sr6) / r
                - b * coulomb * q[i] * q[j] / (r * r)
            )
            f = -dedr * d / r
            forces[i] += f
            forces[j] -= f
    return energy, forces


def _evaluate(system: avff.Compiled, xyz: np.ndarray | None = None, *, pair_only: bool = True):
    """Energy [eV] and forces [eV/Å] of the vacuum model at ``xyz``."""
    frame = system.frame.copy()
    if xyz is not None:
        for col, key in enumerate(("x", "y", "z")):
            frame["atoms"][key] = xyz[:, col].copy()
    sim = MD()
    sim.set_integrator(VelocityVerlet(0.5))
    sim.set_neighbor(NeighborList(system.cutoff, SKIN))
    for term in system.evaluators:
        if not pair_only or isinstance(term, (LJCut, CoulCut)):
            sim.set_evaluator(term)
    sim.run(frame, 0)
    forces = np.stack([np.asarray(frame["atoms"][k], dtype=np.float64) for k in ("fx", "fy", "fz")], axis=1)
    return float(frame.meta["energy"]), forces


def test_gaff_nonbonded_is_two_cutoff_styles_at_the_style_cutoff(typed):
    frame, forcefield = typed
    system = avff.compile(forcefield, frame)
    assert [type(t) for t in system.evaluators[:2]] == [LJCut, CoulCut]
    assert system.cutoff == CUTOFF


def test_vacuum_energy_and_forces_match_all_pairs_reference(typed):
    frame, forcefield = typed
    system = avff.compile(forcefield, frame)
    xyz = _xyz(system.frame)
    energy, forces = _evaluate(system)
    e_ref, f_ref = _reference(frame, forcefield, xyz)
    assert energy == pytest.approx(e_ref, rel=1e-10, abs=1e-12)
    scale = np.abs(f_ref).max()
    np.testing.assert_allclose(forces, f_ref, rtol=0.0, atol=1e-10 * scale)


def test_vacuum_forces_are_the_negative_energy_gradient(typed):
    frame, forcefield = typed
    system = avff.compile(forcefield, frame)
    xyz0 = _xyz(system.frame)
    _, forces = _evaluate(system, xyz0, pair_only=False)
    h = 1e-5
    rng = np.random.default_rng(7)
    for atom in rng.choice(len(xyz0), size=4, replace=False):
        for axis in range(3):
            plus, minus = xyz0.copy(), xyz0.copy()
            plus[atom, axis] += h
            minus[atom, axis] -= h
            e_plus, _ = _evaluate(system, plus, pair_only=False)
            e_minus, _ = _evaluate(system, minus, pair_only=False)
            fd = -(e_plus - e_minus) / (2.0 * h)
            assert forces[atom, axis] == pytest.approx(fd, rel=1e-6, abs=1e-7)


def test_minimize_and_md_run_without_a_kspace(typed):
    frame, forcefield = typed
    system = avff.compile(forcefield, frame)

    def install(driver):
        driver.set_neighbor(NeighborList(system.cutoff, SKIN))
        for term in system.evaluators:
            driver.set_evaluator(term)

    e0, _ = _evaluate(system, pair_only=False)
    opt = Minimize()
    opt.set_optimizer(LBFGS())
    opt.add_criterion(ForceConverged(ftol=1e-4))
    install(opt)
    relaxed = system.frame.copy()
    opt.run(relaxed, 500)
    e1 = float(relaxed.meta["energy"])
    assert math.isfinite(e1) and e1 < e0

    sim = MD()
    sim.set_integrator(VelocityVerlet(0.5))
    install(sim)
    sim.run(relaxed, 20)
    assert math.isfinite(float(relaxed.meta["energy"]))

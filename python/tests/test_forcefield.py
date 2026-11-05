"""atomiverse.forcefield: a molpy force field, translated style by style.

Hand-built molpy force fields (``def_style``) on 2-4 atom frames, so every
expected number is a closed form: the unit conversion, the per-type LJ table
and its cross term under each mixing rule, the style-stated shift and Coulomb
constant, the bond and torsion energies, the split of a multi-term periodic
dihedral into CHARMM rows and the exact periodic → cvff improper conversion.
One seam test feeds a molecule typed by molpy's GAFF2 typifiers through the
translation.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from _domains import skip_module_unless

skip_module_unless("ATV_BUILD_MD")

import molpy  # noqa: E402

from atomiverse import forcefield as avff  # noqa: E402
from atomiverse.forcefield import KCAL_PER_MOL, ForceFieldError  # noqa: E402
from atomiverse.md import MD  # noqa: E402
from atomiverse.md.integrator import VelocityVerlet  # noqa: E402
from atomiverse.neighbor import NeighborList  # noqa: E402
from atomiverse.potentials.improper import Cvff  # noqa: E402
from atomiverse.potentials.pair import CoulCut, LJCut  # noqa: E402

BOX = 40.0
CUTOFF = 10.0
#: The Amber Coulomb constant [kcal·Å/(mol·e²)] — deliberately not
#: Atomiverse's built-in 332.06371, so the translation must carry it.
AMBER_COULOMB = 332.0522173


def _forcefield(
    *, dihedral_terms=None, improper=None, mixing="arithmetic", shift=None
) -> molpy.ff.forcefield.ForceField:
    """Two atom types: ``a`` (C-like) and ``b`` (H-like), Amber form."""
    ff = molpy.ff.forcefield.ForceField("test", units="real")
    atom = ff.def_style("atom", "full")
    a = atom.def_type("a", mass=12.0)
    b = atom.def_type("b", mass=1.0)
    lj_params = {"cutoff": CUTOFF, "mixing": mixing}
    if shift is not None:
        lj_params["shift"] = shift
    lj = ff.def_style("pair", "lj/cut", lj_params)
    lj.def_type("a", a, a, epsilon=0.1, sigma=3.4)
    lj.def_type("b", b, b, epsilon=0.02, sigma=2.6)
    ff.def_style(
        "pair",
        "coul/cut",
        {"cutoff": CUTOFF, "coulomb": AMBER_COULOMB, "dielectric": 1.0},
    )
    ff.set_special_bonds([0.0, 0.0, 0.5], [0.0, 0.0, 5.0 / 6.0])
    ff.def_style("bond", "harmonic").def_type("a-a", a, a, k=300.0, r0=1.5)
    ff.def_style("angle", "harmonic").def_type("a-a-a", a, a, a, k=50.0, theta0=110.0)
    if dihedral_terms is not None:
        params = {}
        for i, (k, n, phase) in enumerate(dihedral_terms, 1):
            params |= {f"k{i}": k, f"periodicity{i}": n, f"phase{i}": phase}
        ff.def_style("dihedral", "periodic").def_type("a-a-a-a", a, a, a, a, **params)
    if improper is not None:
        k, n, phase = improper
        ff.def_style("improper", "periodic").def_type(
            "a-a-a-a", a, a, a, a, k=k, periodicity=n, phase=phase
        )
    return ff


def _frame(xyz, types, charges, *, bonds=(), angles=(), dihedrals=(), impropers=()):
    frame = molpy.Frame()
    atoms = molpy.Block()
    xyz = np.asarray(xyz, dtype=np.float64)
    for col, key in enumerate(("x", "y", "z")):
        atoms.insert(key, xyz[:, col].copy())
    atoms.insert("atomic_number", np.array([6 if t == "a" else 1 for t in types], dtype=np.uint64))
    atoms.insert("mass", np.array([12.0 if t == "a" else 1.0 for t in types]))
    atoms.insert("type", np.array(types))
    atoms.insert("charge", np.asarray(charges, dtype=np.float64))
    frame["atoms"] = atoms
    for name, rows, label in (
        ("bonds", bonds, "a-a"),
        ("angles", angles, "a-a-a"),
        ("dihedrals", dihedrals, "a-a-a-a"),
        ("impropers", impropers, "a-a-a-a"),
    ):
        if not rows:
            continue
        rows = np.asarray(rows, dtype=np.uint64)
        block = molpy.Block()
        for col, end in enumerate(("atomi", "atomj", "atomk", "atoml")[: rows.shape[1]]):
            block.insert(end, rows[:, col].copy())
        block.insert("type", np.array([label] * len(rows)))
        frame[name] = block
    frame.box = molpy.Box.cube(BOX)
    return frame


def _energy(system: avff.Compiled) -> float:
    sim = MD()
    sim.set_integrator(VelocityVerlet(0.5))
    sim.set_neighbor(NeighborList(system.cutoff, 1.0))
    for term in system.evaluators:
        sim.set_evaluator(term)
    frame = system.frame.copy()
    sim.run(frame, 0)
    return float(frame.meta["energy"])


def _chain(n: int, *, spacing: float = 1.5):
    """A zigzag-free chain on x, then bent into a 90° dihedral when n = 4."""
    if n == 4:
        return [(0.0, 0.0, 0.0), (1.5, 0.0, 0.0), (1.5, 1.5, 0.0), (1.5, 1.5, 1.5)]
    return [(spacing * i, 0.0, 0.0) for i in range(n)]


def test_kcal_per_mol_is_the_codata_value():
    assert KCAL_PER_MOL == pytest.approx(0.04336410424, rel=1e-10)


def test_like_kinds_split_by_type_not_element():
    frame = _frame([(0, 0, 0), (5, 0, 0), (10, 0, 0)], ["a", "b", "a"], [0.0] * 3)
    system = avff.compile(_forcefield(), frame)
    kinds = np.asarray(system.frame["atoms"]["type_id"])
    assert kinds[0] == kinds[2] != kinds[1]
    # One molpy style, one evaluator: lj/cut then coul/cut, in style order.
    assert [type(t) for t in system.evaluators] == [LJCut, CoulCut]
    assert system.cutoff == CUTOFF


def test_lj_cross_term_mixes_lorentz_berthelot_in_ev():
    r = 4.0
    frame = _frame([(0, 0, 0), (r, 0, 0)], ["a", "b"], [0.0, 0.0])
    energy = _energy(avff.compile(_forcefield(), frame))
    eps = math.sqrt(0.1 * 0.02) * KCAL_PER_MOL
    sigma = (3.4 + 2.6) / 2
    expected = 4 * eps * ((sigma / r) ** 12 - (sigma / r) ** 6)
    assert energy == pytest.approx(expected, rel=1e-9, abs=1e-12)


def test_bonded_pair_is_excluded_and_bond_energy_is_k_dr_squared():
    r = 1.7
    frame = _frame([(0, 0, 0), (r, 0, 0)], ["a", "a"], [0.0, 0.0], bonds=[(0, 1)])
    energy = _energy(avff.compile(_forcefield(), frame))
    assert energy == pytest.approx(300.0 * KCAL_PER_MOL * (r - 1.5) ** 2, rel=1e-9)


def test_multi_term_dihedral_becomes_one_charmm_row_per_term():
    terms = [(1.0, 1, 0.0), (0.5, 3, 180.0), (0.0, 2, 0.0)]
    frame = _frame(
        _chain(4),
        ["a"] * 4,
        [0.0] * 4,
        bonds=[(0, 1), (1, 2), (2, 3)],
        angles=[(0, 1, 2), (1, 2, 3)],
        dihedrals=[(0, 1, 2, 3)],
    )
    system = avff.compile(_forcefield(dihedral_terms=terms), frame)
    # The zero-amplitude term contributes nothing and is not emitted.
    assert system.frame["dihedrals"].n_rows == 2
    # Right angles everywhere: angle energy is 2 k (90° - 110°)², the
    # torsion φ = 90° gives k1 (1 + cos 90°) + k2 (1 + cos(270° - 180°)).
    angle = 2 * 50.0 * math.radians(20.0) ** 2
    torsion = 1.0 * (1 + math.cos(math.radians(90.0))) + 0.5 * (
        1 + math.cos(math.radians(3 * 90.0 - 180.0))
    )
    # 1-4 pair (atoms 0 and 3, r = 1.5·√3) at half weight; charges are zero.
    r14 = 1.5 * math.sqrt(3.0)
    lj14 = 0.5 * 4 * 0.1 * ((3.4 / r14) ** 12 - (3.4 / r14) ** 6)
    expected = (angle + torsion + lj14) * KCAL_PER_MOL
    assert _energy(system) == pytest.approx(expected, rel=1e-8)


def test_periodic_improper_becomes_cvff_with_the_sign_of_its_phase():
    frame = _frame(
        _chain(4),
        ["a"] * 4,
        [0.0] * 4,
        bonds=[(0, 1), (1, 2), (2, 3)],
        impropers=[(0, 1, 2, 3)],
    )
    system = avff.compile(_forcefield(improper=(1.1, 2, 180.0)), frame)
    improper = system.evaluators[-1]
    assert isinstance(improper, Cvff)
    # φ = 90°: 1.1 (1 - cos 180°) = 2.2 kcal/mol, plus the 1-3/1-4 pair
    # terms the bonds leave: 1-4 at half weight; 1-3 excluded.
    r14 = 1.5 * math.sqrt(3.0)
    lj14 = 0.5 * 4 * 0.1 * ((3.4 / r14) ** 12 - (3.4 / r14) ** 6)
    assert _energy(system) == pytest.approx((2.2 + lj14) * KCAL_PER_MOL, rel=1e-8)


def test_a_style_without_a_counterpart_is_refused():
    ff = _forcefield()
    ff.def_style("bond", "morse")
    frame = _frame([(0, 0, 0), (1.5, 0, 0)], ["a", "a"], [0.0, 0.0])
    with pytest.raises(ForceFieldError, match="bond morse"):
        avff.compile(ff, frame)


def test_an_unparameterized_type_is_refused_by_name():
    frame = _frame([(0, 0, 0), (1.5, 0, 0)], ["a", "a"], [0.0, 0.0], bonds=[(0, 1)])
    ff = _forcefield()
    ff.get_style("bond", "harmonic").del_type("a-a")
    with pytest.raises(ForceFieldError, match="a-a"):
        avff.compile(ff, frame)


def test_a_frame_without_charges_is_refused():
    frame = _frame([(0, 0, 0), (1.5, 0, 0)], ["a", "a"], [0.0, 0.0])
    atoms = molpy.Block()
    for key in ("x", "y", "z", "atomic_number", "mass", "type"):
        atoms.insert(key, np.asarray(frame["atoms"][key]))
    frame["atoms"] = atoms
    with pytest.raises(KeyError, match="charge"):
        avff.compile(_forcefield(), frame)


def test_gaff2_typed_molecule_translates_completely():
    """The seam: molpy's GAFF2 typing of an ion pair reaches every style."""
    mol = molpy.io.smiles.read_smiles("C[N+](C)(C)CCBr").to_atomistic()
    mol, _ = molpy.conformer.Conformer(seed=1).generate(mol)
    typed = molpy.ff.typifier.AtdTypifier(parameter_set="gaff2").typify(mol)
    typifier = molpy.ff.typifier.GaffTypifier(parameter_set="gaff2")
    typed = typifier.typify(typed)
    frame = typed.to_frame()
    charges = molpy.ff.charge.GasteigerModel().assign(typed)
    frame["atoms"].insert("charge", np.asarray(charges))
    frame.box = molpy.Box.cube(BOX)

    system = avff.compile(typifier.forcefield(), frame)
    out = system.frame
    n_types = len(set(np.asarray(frame["atoms"]["type"])))
    assert len(set(np.asarray(out["atoms"]["type_id"]))) == n_types
    assert out["bonds"].n_rows == frame["bonds"].n_rows
    assert out["angles"].n_rows == frame["angles"].n_rows
    assert out["dihedrals"].n_rows >= 1
    names = [type(t).__name__ for t in system.evaluators]
    assert names[:4] == ["PairLJCut", "PairCoulCut", "BondHarmonic", "AngleHarmonic"]
    assert np.isclose(np.asarray(out["atoms"]["charge"]).sum(), 1.0)


def test_mixing_rule_comes_from_the_style():
    r = 4.0
    frame = _frame([(0, 0, 0), (r, 0, 0)], ["a", "b"], [0.0, 0.0])
    energy = _energy(avff.compile(_forcefield(mixing="geometric"), frame))
    eps = math.sqrt(0.1 * 0.02) * KCAL_PER_MOL
    sigma = math.sqrt(3.4 * 2.6)
    expected = 4 * eps * ((sigma / r) ** 12 - (sigma / r) ** 6)
    assert energy == pytest.approx(expected, rel=1e-9, abs=1e-12)


def test_shift_follows_the_style_parameter():
    r = 4.0
    frame = _frame([(0, 0, 0), (r, 0, 0)], ["a", "a"], [0.0, 0.0])

    def lj(x: float) -> float:
        return 4 * 0.1 * KCAL_PER_MOL * ((3.4 / x) ** 12 - (3.4 / x) ** 6)

    unshifted = _energy(avff.compile(_forcefield(), frame))
    shifted = _energy(avff.compile(_forcefield(shift=1.0), frame))
    assert unshifted == pytest.approx(lj(r), rel=1e-10)
    assert shifted == pytest.approx(lj(r) - lj(CUTOFF), rel=1e-10)


def test_coulomb_uses_the_constant_the_style_states():
    r = 3.0
    frame = _frame([(0, 0, 0), (r, 0, 0)], ["b", "b"], [0.4, -0.3])
    ff = _forcefield()
    ff.get_style("pair", "lj/cut").get_type_by_name("b").set(epsilon=0.0)
    energy = _energy(avff.compile(ff, frame))
    expected = AMBER_COULOMB * KCAL_PER_MOL * 0.4 * -0.3 / r
    assert energy == pytest.approx(expected, rel=1e-12)


def test_cutoff_is_read_from_the_style_params():
    frame = _frame([(0, 0, 0), (4.0, 0, 0)], ["a", "b"], [0.1, -0.1])
    ff = _forcefield()
    for name in ("lj/cut", "coul/cut"):
        ff.get_style("pair", name).set(cutoff=25.0)
    assert avff.compile(ff, frame).cutoff == 25.0


def test_an_ewald_style_is_refused_by_name():
    ff = _forcefield()
    ff.def_style("pair", "coul/long/pme", {"cutoff": CUTOFF, "alpha": 0.3})
    frame = _frame([(0, 0, 0), (4.0, 0, 0)], ["a", "a"], [0.0, 0.0])
    with pytest.raises(ForceFieldError, match="coul/long/pme"):
        avff.compile(ff, frame)

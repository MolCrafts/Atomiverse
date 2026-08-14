"""AM1 semiempirical Python surface tests (spec: python-api-namespaces-am1).

Exposes the C++-accepted ``cpu::semiempirical::AM1Evaluator`` (AM1: Dewar,
Zoebisch, Healy, Stewart, J. Am. Chem. Soc. 1985, 107, 3902-3909, DOI
10.1021/ja00299a024) through ``atv.single_point``. The domain oracle is the
one already pinned in ``tests/cpu/semiempirical/test_am1_scaffold.cpp``
(AM1 reference 23.2.5, ``AM1 1SCF XYZ``): tetrahedral CH4 with H along the
body diagonals at h=0.629118 A per axis component (C-H ~ 1.0897 A) gives
Mulliken charges C = -0.262125 e and H = +0.065531 e. No new reference
values are introduced here.

Tolerances: the molrs.Frame ``charge`` column is checked to 2.5e-3 e
(covers single-precision storage on an ``ATV_REAL_STORAGE=float`` wheel);
``last_result().mulliken_charges`` is always double and is checked to
5.0e-5 e, mirroring the C++ unit test's own tolerance.

Units: typed frame metadata energy [eV]; ``electronic_energy_hartree`` [Hartree];
``heat_of_formation_kcal_mol`` [kcal/mol]; charges [e].
"""

from __future__ import annotations

import numpy as np
import pytest
import molrs

import atomiverse as av

# ── Pinned oracle (cited in tests/cpu/semiempirical/test_am1_scaffold.cpp) ──

_H = 0.629118  # [A] per-axis H displacement -> C-H = 1.0897 A
_ORACLE_C = -0.262125  # [e]
_ORACLE_H = 0.065531  # [e]
_TOL_FRAME_CHARGE = 2.5e-3  # [e] float-storage wheel tolerance
_TOL_RESULT_CHARGE = 5.0e-5  # [e] mirrors the C++ EXPECT_NEAR tolerance


def _make_ch4_frame() -> "molrs.Frame":
    """Tetrahedral CH4 molrs.Frame (type = atomic number Z; C at origin,
    H along the body diagonals) — same geometry as the C++ oracle test."""
    frame = molrs.Frame()
    atoms = molrs.Block()
    atoms.insert("x", np.array([0.0, _H, -_H, -_H, _H], dtype=np.float64))
    atoms.insert("y", np.array([0.0, _H, -_H, _H, -_H], dtype=np.float64))
    atoms.insert("z", np.array([0.0, _H, _H, -_H, -_H], dtype=np.float64))
    atoms.insert(
        "mass",
        np.array([12.011, 1.008, 1.008, 1.008, 1.008], dtype=np.float64),
    )
    atoms.insert("atomic_number", np.array([6, 1, 1, 1, 1], dtype=np.uint32))
    frame["atoms"] = atoms
    frame.box = molrs.Box.cube(20.0)
    return frame


@pytest.fixture
def ch4_frame() -> "molrs.Frame":
    return _make_ch4_frame()


@pytest.fixture(scope="module")
def am1_ch4() -> tuple["molrs.Frame", "av.AM1Evaluator"]:
    """One converged AM1 CH4 single point, shared read-only by the
    oracle assertions below."""
    frame = _make_ch4_frame()
    evaluator = av.AM1Evaluator()
    av.single_point(frame, evaluator)
    return frame, evaluator


# ── AM1 construction (parameters assigned in the constructor) ───────


class TestAM1:
    def test_default_construction(self) -> None:
        """AM1() carries the documented defaults."""
        m = av.AM1()
        assert m.charge == 0
        assert m.multiplicity == 1
        assert m.max_iter == 200
        assert m.density_tol == pytest.approx(1e-8)
        assert m.energy_tol == pytest.approx(1e-10)

    def test_kwargs_roundtrip(self) -> None:
        """Constructor kwargs land on the matching members."""
        m = av.AM1(
            charge=-1,
            multiplicity=2,
            max_iter=50,
            density_tol=1e-6,
            energy_tol=1e-8,
        )
        assert m.charge == -1
        assert m.multiplicity == 2
        assert m.max_iter == 50
        assert m.density_tol == pytest.approx(1e-6)
        assert m.energy_tol == pytest.approx(1e-8)

    def test_fields_writable(self) -> None:
        """Members remain read-write after construction."""
        m = av.AM1()
        m.charge = 1
        m.max_iter = 300
        assert m.charge == 1
        assert m.max_iter == 300


class TestAM1EvaluatorConstruction:
    def test_default_constructible(self) -> None:
        assert av.AM1Evaluator() is not None

    def test_constructible_from_kwargs(self) -> None:
        assert av.AM1Evaluator(charge=0, multiplicity=1) is not None

    def test_constructible_from_method(self) -> None:
        assert av.AM1Evaluator(av.AM1(max_iter=77)) is not None

    def test_method_accessor_roundtrip(self) -> None:
        """method() exposes the owned AM1 with construction parameters."""
        evaluator = av.AM1Evaluator(max_iter=77)
        assert evaluator.method().max_iter == 77


# ── CH4 single point: energy channel + charge write-back ────────────


class TestAM1SinglePointCH4:
    def test_energy_written_back_finite_negative(self, am1_ch4) -> None:
        """Typed frame metadata energy [eV] is finite and negative."""
        frame, _ = am1_ch4
        assert "scalars" not in frame.keys()
        assert "stress" not in frame.keys()
        energy = frame.meta["energy"]
        assert isinstance(energy, float)
        assert np.isfinite(energy)
        assert energy < 0.0

    def test_frame_charge_column_matches_oracle(self, am1_ch4) -> None:
        """The molrs.Frame charge column carries the 5 Mulliken charges
        within the float-storage tolerance (write_back contract)."""
        frame, _ = am1_ch4
        charges = np.asarray(frame["atoms"]["charge"], dtype=np.float64)
        assert charges.shape == (5,)
        assert charges[0] == pytest.approx(_ORACLE_C, abs=_TOL_FRAME_CHARGE)
        for i in range(1, 5):
            assert charges[i] == pytest.approx(_ORACLE_H, abs=_TOL_FRAME_CHARGE), (
                f"atom {i}"
            )

    def test_last_result_mulliken_charges_match_oracle(self, am1_ch4) -> None:
        """last_result().mulliken_charges (always double) match the pinned
        oracle at the C++ unit test's own tolerance."""
        _, evaluator = am1_ch4
        charges = evaluator.last_result().mulliken_charges
        assert len(charges) == 5
        assert charges[0] == pytest.approx(_ORACLE_C, abs=_TOL_RESULT_CHARGE)
        for i in range(1, 5):
            assert charges[i] == pytest.approx(_ORACLE_H, abs=_TOL_RESULT_CHARGE), (
                f"atom {i}"
            )

    def test_last_result_converged(self, am1_ch4) -> None:
        _, evaluator = am1_ch4
        assert evaluator.last_result().converged is True

    def test_last_result_iterations_positive(self, am1_ch4) -> None:
        _, evaluator = am1_ch4
        assert evaluator.last_result().iterations > 0

    def test_last_result_electronic_energy_finite(self, am1_ch4) -> None:
        """electronic_energy_hartree [Hartree] is readable and finite."""
        _, evaluator = am1_ch4
        assert np.isfinite(evaluator.last_result().electronic_energy_hartree)

    def test_last_result_heat_of_formation_finite(self, am1_ch4) -> None:
        """heat_of_formation_kcal_mol [kcal/mol] is readable and finite."""
        _, evaluator = am1_ch4
        assert np.isfinite(evaluator.last_result().heat_of_formation_kcal_mol)


# ── Fail-loud boundary: unsupported element ─────────────────────────


class TestUnsupportedElement:
    def test_z99_raises_runtime_error(self) -> None:
        """Z=99 has no AM1 parameters — single_point raises RuntimeError
        across the binding boundary (no fallback, no crash)."""
        frame = molrs.Frame()
        atoms = molrs.Block()
        atoms.insert("x", np.array([0.0], dtype=np.float64))
        atoms.insert("y", np.array([0.0], dtype=np.float64))
        atoms.insert("z", np.array([0.0], dtype=np.float64))
        atoms.insert("mass", np.array([252.0], dtype=np.float64))
        atoms.insert("atomic_number", np.array([99], dtype=np.uint32))
        frame["atoms"] = atoms
        frame.box = molrs.Box.cube(20.0)

        with pytest.raises(RuntimeError):
            av.single_point(frame, av.AM1Evaluator())

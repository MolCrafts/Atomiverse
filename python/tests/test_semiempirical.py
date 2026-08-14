"""AM1 semiempirical Python surface tests (specs: python-api-namespaces-am1,
py-am1-forces).

Exposes the C++-accepted ``cpu::semiempirical::AM1Evaluator`` (AM1: Dewar,
Zoebisch, Healy, Stewart, J. Am. Chem. Soc. 1985, 107, 3902-3909, DOI
10.1021/ja00299a024) through ``atv.single_point`` (energy [eV] + Mulliken
charges [e]) and, per spec py-am1-forces, through ``AM1Evaluator.forces``
(nuclear forces [eV/Å] returned as an (N, 3) array and written back to
fx/fy/fz, same convention as the SCF binding's ``forces``).

No new reference values are introduced here; every golden below is
hard-coded and cited:

===================================  =========================================
Pinned quantity                      Source (hard-coded golden)
===================================  =========================================
CH4 Mulliken q(C) = -0.262125 e,     MOPAC AM1 reference 23.2.5,
q(H) = +0.065531 e at h = 0.629118   ``AM1 1SCF XYZ``, transcribed. NOTE: the
Å per axis component (C-H ~ 1.0897   C++ unit test that used to carry this pin
Å)                                   (``tests/cpu/semiempirical/
                                     test_am1_scaffold.cpp``) has been deleted
                                     and no in-tree file re-derives it; the
                                     surviving C++ AM1 anchors are the two
                                     regressions below, which pin water, not
                                     CH4. The geometry alone also lives in
                                     ``regressions/launcher/lj_parity/
                                     atv_am1.py:17``.
Water AM1 optimum r(O-H) = 0.961267  ``regressions/semiempirical/
Å, angle(HOH) = 103.53103 deg, and   am1-water-mopac-energy/
its stationarity gate max|F| <=      reg_am1-water-mopac-energy.cpp:40-41,58``
5e-3 eV/Å                            (OpenMOPAC / MOPAC 7.00 ``AM1 PRECISE EF``
                                     transcript)
Off-equilibrium water fixture        ``regressions/semiempirical/
O(0,0,0) H(1.05,0,0) H(-0.26,1,0),   am1_water_force/
net-force gate |sum F_d| <= 1e-5     reg_am1_water_force.cpp:36-46,94,101``
eV/Å, non-triviality gate
sum|F|^2 >= 1e-10
===================================  =========================================

The water coordinates are DERIVED here from the pinned internal
coordinates with the same trigonometry as the C++ case (C2v frame: O at the
origin, H1 on +x, H2 in the xy plane at the full HOH angle), never
transcribed from a rounded Cartesian table.

Tolerances: the molrs.Frame ``charge`` column is checked to 2.5e-3 e
(covers single-precision storage on an ``ATV_REAL_STORAGE=float`` wheel);
``last_result().mulliken_charges`` is always double and is checked to
5.0e-5 e, mirroring the C++ unit test's own tolerance. Force columns
carry the same storage caveat and use 1e-4 eV/Å (the numerical-force
tolerance test_scf.py already uses for fx/fy/fz write-back); values that
never touch a storage column (``last_result()`` fields, ``frame.meta``)
are double on both sides and are compared at 1e-10.

Units: typed frame metadata energy [eV]; ``electronic_energy_hartree`` [Hartree];
``heat_of_formation_kcal_mol`` [kcal/mol]; charges [e]; forces [eV/Å].
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import molrs

import atomiverse as av

# ── Pinned CH4 oracle (MOPAC AM1 reference 23.2.5, `AM1 1SCF XYZ`; see the
# module docstring — the C++ unit test that carried it no longer exists) ──

_H = 0.629118  # [A] per-axis H displacement -> C-H = 1.0897 A
_ORACLE_C = -0.262125  # [e]
_ORACLE_H = 0.065531  # [e]
_TOL_FRAME_CHARGE = 2.5e-3  # [e] float-storage wheel tolerance
_TOL_RESULT_CHARGE = 5.0e-5  # [e] mirrors the C++ EXPECT_NEAR tolerance


def _make_ch4_frame() -> "molrs.Frame":
    """Tetrahedral CH4 molrs.Frame (type = atomic number Z; C at origin,
    H along the body diagonals) — the geometry the MOPAC pin was taken at,
    also spelled out in regressions/launcher/lj_parity/atv_am1.py:17."""
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


# ── py-am1-forces: water fixtures + gates borrowed from the C++ cases ──

#: MOPAC 7.00 `AM1 PRECISE EF` optimized internal coordinates of water,
#: reg_am1-water-mopac-energy.cpp:40-41.
_R_OH_MOPAC = 0.961267  # [Å]
_HOH_MOPAC_DEG = 103.53103  # [degrees]
#: The same case's cubic cell (isolation only; AM1 here is non-periodic).
_BOX_MOPAC = 30.0  # [Å]
#: reg_am1_water_force.cpp:47 uses a 20 Å cell for its distorted fixture.
_BOX_FORCE = 20.0  # [Å]

#: kStationaryTolEvA — reg_am1-water-mopac-energy.cpp:58. MOPAC stopped at
#: RMS gradient 0.00141 kcal/mol/Å, so the published optimum must also be a
#: stationary point of ATV's own AM1 surface.
_TOL_STATIONARY_EV_A = 5.0e-3  # [eV/Å]
#: Net-force (translation-invariance) gate — reg_am1_water_force.cpp:101.
_TOL_NET_FORCE_EV_A = 1.0e-5  # [eV/Å]
#: Non-triviality gate: sum of squared force components on the distorted
#: fixture — reg_am1_water_force.cpp:94.
_MIN_FORCE_SQ_EV_A = 1.0e-10  # [(eV/Å)²]
#: fx/fy/fz are StorageReal columns; test_scf.py grades the same write-back
#: at this window, which covers an ATV_REAL_STORAGE=float wheel.
_TOL_FRAME_FORCE_EV_A = 1.0e-4  # [eV/Å]
#: frame.meta['energy'] and AM1Result fields are double on both sides of the
#: molrs boundary — no narrowing, so the exact-energy tier applies.
_TOL_ENERGY_EV = 1.0e-10  # [eV]
#: O-H stretch used for the displacement-response case (well above the SCF
#: energy tolerance, well below anharmonicity).
_STRETCH_A = 0.01  # [Å]


def _c2v_water_coords(oh_angstrom: float, hoh_degrees: float) -> "np.ndarray":
    """Water Cartesians in MOPAC's C2v frame [Å].

    Same trigonometry as reg_am1-water-mopac-energy.cpp:66-85: O at the
    origin, H1 on +x at *oh_angstrom*, H2 at the full HOH angle in the xy
    plane. Deriving rather than transcribing keeps this file free of a
    second, rounded copy of the pinned geometry.

    @param oh_angstrom: O-H bond length [Å] (both bonds; C2v).
    @param hoh_degrees: H-O-H bond angle [degrees].
    @return (3, 3) positions [Å]; row 0 = O, rows 1-2 = H.
    """
    theta = math.radians(hoh_degrees)
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [oh_angstrom, 0.0, 0.0],
            [oh_angstrom * math.cos(theta), oh_angstrom * math.sin(theta), 0.0],
        ],
        dtype=np.float64,
    )


def _make_water_frame(coords: "np.ndarray", box_angstrom: float) -> "molrs.Frame":
    """Water molrs.Frame (O, H, H) at *coords* in a cubic cell.

    @param coords: (3, 3) positions [Å]; row 0 = O, rows 1-2 = H.
    @param box_angstrom: Cubic cell edge [Å]; isolation only.
    @return molrs.Frame with x/y/z/mass/atomic_number columns.
    """
    frame = molrs.Frame()
    atoms = molrs.Block()
    for axis, key in enumerate(("x", "y", "z")):
        atoms.insert(key, np.ascontiguousarray(coords[:, axis], dtype=np.float64))
    atoms.insert("mass", np.array([15.999, 1.008, 1.008], dtype=np.float64))
    atoms.insert("atomic_number", np.array([8, 1, 1], dtype=np.uint32))
    frame["atoms"] = atoms
    frame.box = molrs.Box.cube(box_angstrom)
    return frame


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


# ── py-am1-forces: forces(frame) — array out AND fx/fy/fz written back ──


class TestAM1Forces:
    """``AM1Evaluator.forces(frame)``: nuclear forces [eV/Å].

    Each solve is shared by a class-scoped fixture (an AM1 SCF plus the
    fixed-density displacement sweep is the expensive part). The fixtures
    are ``@staticmethod`` because a class-scoped fixture written as an
    instance method binds an instance pytest throws away before the tests
    run; nothing here needs ``self``.
    """

    @pytest.fixture(scope="class")
    @staticmethod
    def ch4_forces() -> tuple["molrs.Frame", "np.ndarray", "av.AM1Evaluator"]:
        """One AM1 force evaluation on the pinned CH4 geometry."""
        frame = _make_ch4_frame()
        evaluator = av.AM1Evaluator()
        forces = np.asarray(evaluator.forces(frame), dtype=np.float64)
        return frame, forces, evaluator

    @pytest.fixture(scope="class")
    @staticmethod
    def water_optimum() -> tuple["molrs.Frame", "np.ndarray"]:
        """AM1 forces at MOPAC's published AM1 optimum for water."""
        frame = _make_water_frame(
            _c2v_water_coords(_R_OH_MOPAC, _HOH_MOPAC_DEG), _BOX_MOPAC
        )
        forces = np.asarray(av.AM1Evaluator().forces(frame), dtype=np.float64)
        return frame, forces

    @pytest.fixture(scope="class")
    @staticmethod
    def water_stretched() -> tuple["molrs.Frame", "np.ndarray"]:
        """The same optimum with H1 pushed +0.01 Å along its own O-H axis.

        H1 sits on +x with O at the origin, so the O->H1 unit vector IS +x
        and the stretch is a pure bond elongation — no rotation, no angle
        change, which is what makes the sign of F_x(H1) readable.
        """
        coords = _c2v_water_coords(_R_OH_MOPAC, _HOH_MOPAC_DEG)
        coords[1, 0] += _STRETCH_A
        frame = _make_water_frame(coords, _BOX_MOPAC)
        forces = np.asarray(av.AM1Evaluator().forces(frame), dtype=np.float64)
        return frame, forces

    @pytest.fixture(scope="class")
    @staticmethod
    def water_distorted() -> tuple["molrs.Frame", "np.ndarray"]:
        """The distorted fixture of reg_am1_water_force.cpp:36-46."""
        coords = np.array(
            [[0.0, 0.0, 0.0], [1.05, 0.0, 0.0], [-0.26, 1.0, 0.0]],
            dtype=np.float64,
        )
        frame = _make_water_frame(coords, _BOX_FORCE)
        forces = np.asarray(av.AM1Evaluator().forces(frame), dtype=np.float64)
        return frame, forces

    # ── shape / finiteness / write-back ──

    def test_returns_n_by_three_array(
        self, ch4_forces: tuple["molrs.Frame", "np.ndarray", "av.AM1Evaluator"]
    ) -> None:
        _, forces, _ = ch4_forces
        assert forces.shape == (5, 3)

    def test_forces_finite(
        self, ch4_forces: tuple["molrs.Frame", "np.ndarray", "av.AM1Evaluator"]
    ) -> None:
        _, forces, _ = ch4_forces
        assert np.all(np.isfinite(forces))

    def test_force_columns_written_back(
        self, ch4_forces: tuple["molrs.Frame", "np.ndarray", "av.AM1Evaluator"]
    ) -> None:
        """forces() publishes fx/fy/fz on the SAME molrs.Frame, matching the
        returned array (the write-back contract MD and SCF already keep).

        CH4 is pinned at C-H = 1.0897 Å, which is not AM1's own methane
        minimum, so the H forces are non-zero along the body diagonals and
        every Cartesian component of them is non-zero.
        """
        frame, forces, _ = ch4_forces
        fx = np.asarray(frame["atoms"]["fx"], dtype=np.float64)
        fy = np.asarray(frame["atoms"]["fy"], dtype=np.float64)
        fz = np.asarray(frame["atoms"]["fz"], dtype=np.float64)
        assert fx.shape == (5,)
        assert np.any(fx != 0.0), "fx is all zeros — forces were not written back"
        assert fx == pytest.approx(forces[:, 0], abs=_TOL_FRAME_FORCE_EV_A)
        assert fy == pytest.approx(forces[:, 1], abs=_TOL_FRAME_FORCE_EV_A)
        assert fz == pytest.approx(forces[:, 2], abs=_TOL_FRAME_FORCE_EV_A)

    def test_energy_written_back_finite_negative(
        self, ch4_forces: tuple["molrs.Frame", "np.ndarray", "av.AM1Evaluator"]
    ) -> None:
        """forces() also lands the total energy [eV]: bound, so negative
        (the same gate as reg_am1_water_force.cpp:77-81)."""
        frame, _, _ = ch4_forces
        energy = frame.meta["energy"]
        assert isinstance(energy, float)
        assert np.isfinite(energy)
        assert energy < 0.0

    def test_forces_translation_invariance(
        self, ch4_forces: tuple["molrs.Frame", "np.ndarray", "av.AM1Evaluator"]
    ) -> None:
        """Σ_A F_A = 0 per component, at the C++ case's own net-force gate
        (reg_am1_water_force.cpp:101, 1e-5 eV/Å): a rigid translation of an
        isolated molecule cannot change its energy."""
        _, forces, _ = ch4_forces
        assert forces.sum(axis=0) == pytest.approx(np.zeros(3), abs=_TOL_NET_FORCE_EV_A)

    # ── AM1Result force / energy fields ──

    def test_result_forces_field(
        self, ch4_forces: tuple["molrs.Frame", "np.ndarray", "av.AM1Evaluator"]
    ) -> None:
        """last_result().forces_ev_a is the flat (fx0, fy0, fz0, fx1, …)
        layout of NDDOResult, holding exactly the doubles that were
        returned — same buffer, no conversion, so exact equality."""
        _, forces, evaluator = ch4_forces
        flat = evaluator.last_result().forces_ev_a
        assert len(flat) == 15
        assert np.asarray(flat, dtype=np.float64).tolist() == forces.ravel().tolist()

    def test_result_total_energy_ev_matches_frame_meta(
        self, ch4_forces: tuple["molrs.Frame", "np.ndarray", "av.AM1Evaluator"]
    ) -> None:
        """AM1Result.total_energy_ev is the number AM1 writes to
        frame.meta['energy'] (src/cpu/semiempirical/am1_solver.cpp:88-92);
        both are double, so the exact-energy tier applies."""
        frame, _, evaluator = ch4_forces
        total = evaluator.last_result().total_energy_ev
        assert np.isfinite(total)
        assert total == pytest.approx(frame.meta["energy"], abs=_TOL_ENERGY_EV)

    # ── Physics: stationarity, non-triviality, displacement response ──

    def test_forces_stationary_at_mopac_optimum(
        self, water_optimum: tuple["molrs.Frame", "np.ndarray"]
    ) -> None:
        """MOPAC's AM1 optimum is stationary on ATV's AM1 surface too.

        Borrowed gate: kStationaryTolEvA = 5e-3 eV/Å
        (reg_am1-water-mopac-energy.cpp:58,167-181). If ATV's AM1 is the
        same functional as MOPAC's, the geometry MOPAC converged to must
        have a vanishing gradient here as well.
        """
        _, forces = water_optimum
        max_force = float(np.max(np.abs(forces)))
        assert max_force <= _TOL_STATIONARY_EV_A, (
            f"max|F| = {max_force!r} eV/Å at MOPAC's AM1 optimum exceeds the "
            f"C++ case's stationarity gate {_TOL_STATIONARY_EV_A!r}"
        )

    def test_forces_do_not_vanish_off_optimum(
        self, water_distorted: tuple["molrs.Frame", "np.ndarray"]
    ) -> None:
        """The distorted fixture keeps the stationarity gate honest.

        Same non-triviality gate as reg_am1_water_force.cpp:94: sum of
        squared components >= 1e-10 (eV/Å)². This deliberately does NOT
        borrow kOffOptimumFloorEvA = 0.5 eV/Å — that floor is pinned for
        the MOPAC *input* geometry (1.0 Å, 105°), not for this fixture.
        """
        _, forces = water_distorted
        force_sq = float(np.sum(forces**2))
        assert force_sq >= _MIN_FORCE_SQ_EV_A, (
            f"forces vanished on distorted water (sum|F|² = {force_sq!r})"
        )

    def test_distorted_forces_translation_invariance(
        self, water_distorted: tuple["molrs.Frame", "np.ndarray"]
    ) -> None:
        """Σ_A F_A = 0 on the distorted fixture — this is exactly the check
        reg_am1_water_force.cpp:100-106 runs, at its own 1e-5 eV/Å gate."""
        _, forces = water_distorted
        assert forces.sum(axis=0) == pytest.approx(np.zeros(3), abs=_TOL_NET_FORCE_EV_A)

    def test_stretched_bond_raises_energy(
        self,
        water_optimum: tuple["molrs.Frame", "np.ndarray"],
        water_stretched: tuple["molrs.Frame", "np.ndarray"],
    ) -> None:
        """E(optimum + 0.01 Å O-H stretch) > E(optimum).

        The optimum is a minimum, so any displacement raises the energy;
        with an O-H force constant of order 50 eV/Å² the rise is ~2.6e-3 eV,
        far above the SCF energy tolerance (1e-10 eV by default). This is the
        cheapest statement that the force channel and the energy channel
        describe the same surface.
        """
        optimum_frame, _ = water_optimum
        stretched_frame, _ = water_stretched
        assert stretched_frame.meta["energy"] > optimum_frame.meta["energy"]

    def test_stretched_bond_force_points_back_at_oxygen(
        self, water_stretched: tuple["molrs.Frame", "np.ndarray"]
    ) -> None:
        """The stretched H is pulled back toward O: F(H1)·d̂ < 0.

        d̂ = +x is the O->H1 direction (O is at the origin, H1 on +x), so a
        restoring force has a negative x component. The magnitude must also
        clear the stationarity gate — otherwise the sign would just be
        reading the residual gradient noise of the optimum rather than a
        real response to the displacement.
        """
        _, forces = water_stretched
        projection = float(forces[1, 0])
        assert projection < -_TOL_STATIONARY_EV_A, (
            f"F_x(H1) = {projection!r} eV/Å does not restore the stretched "
            f"O-H bond (expected clearly negative)"
        )


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

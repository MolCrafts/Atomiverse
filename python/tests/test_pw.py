"""Python PW entry points, Frame publication and existing workflow checks.

The Si2 fixture keeps a=5.43 Å, a primitive FCC cell, a +.08 Å displacement,
GTH-PBE-q4, Gamma, 8/32 Ry, eight electrons and four fixed-occupation bands.
LSDA means Slater exchange plus VWN5 correlation. Energy and all six force
components are compared with an actual QE7.5 execution archived under
``tools/diagnostics/nc-extension/python-fcc``. The reference uses QE's native
GTH reader and records input, stdout, executable/source hashes and settings.
The force reference sums the printed contributions before QE's subtraction
of the total force; that subtraction cannot validate translation invariance.

The original bounds remain 1e-3 eV for energy, 5e-3 eV/Å for each force
component, and 1e-4 eV/Å for the native force sum. The independent net-force
check also exercises the XC quadrature and finite SCF accuracy. This fixture
is not a claim of cutoff-converged Si material properties.

``PWResult.energy`` is Ry, ``energy_ha`` is Hartree; Frame energy is eV,
forces are eV/Å, and returned stress is eV/Å³. Frame stress stores the
extensive virial in eV. Stress symmetry/sign and small dynamics checks retain
their existing settings and bounds; they do not use a new magnitude pin.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import molpy

from atomiverse._binding import _cpp, extension_available

_HAS_PW: bool = extension_available() and hasattr(_cpp, "PW")

if _HAS_PW:
    from atomiverse.pw import PW, PWResult, GthPseudopotential


# ── Fixture geometry / settings (regressions/pw/rks) ──────

#: Si lattice constant [Å] — reg_cuda-pw-forces.cu:32.
_A: float = 5.43
#: Displacement of atom 1 along x [Å] so the forces are non-zero and the KB
#: projectors are exercised — reg_cuda-pw-forces.cu:33.
_STRETCH: float = 0.08
#: In-tree analytic GTH parameterization for Si (no external file).
_GTH_SI: str = "GTH-PBE-q4"
_ECUT_RY: float = 8.0
_K_MESH: int = 1  # Γ only
_TOL: float = 1.0e-6
_MAX_ITER: int = 40

#: Hartree → eV, include/atomiverse/core/units.h:40.
_HARTREE_EV: float = 27.211386245988

# ── Independently executed QE reference ──────────────────────────────

_QE_REFERENCE = json.loads(
    (Path(__file__).resolve().parents[2]
     / "tools/diagnostics/nc-extension/python-fcc/reference.json").read_text()
)
_QE_LSDA = next(run for run in _QE_REFERENCE["runs"] if run["name"] == "lsda14")
#: Preserve each component, including its sign; do not use a native freeze.
_PIN_FORCE_EV_A = np.asarray(_QE_LSDA["preprojection_forces_ry_bohr"]) * (
    .5 * _HARTREE_EV / .529177210903
)
_TOL_FORCE_PIN_EV_A: float = 5.0e-3
_MIN_MAX_FORCE_EV_A: float = 0.05
_PIN_FORCE_PATH_ENERGY_EV: float = _QE_LSDA["energy_ry"] * .5 * _HARTREE_EV
_TOL_PIN_ENERGY_EV: float = 1.0e-3

#: Σ_i F_i = 0 gate [eV/Å]: numerical force tier (see the module docstring).
_TOL_NET_FORCE_EV_A: float = 1.0e-4
#: fx/fy/fz are StorageReal columns — same window test_scf.py grades
#: force write-back at, which also covers an ATV_REAL_STORAGE=float wheel.
_TOL_FRAME_FORCE_EV_A: float = 1.0e-4

#: kSymTol of reg_pw-analytic-stress-si.cpp:28 [eV/Å³]: the off-diagonal
#: symmetry residual an incomplete ecut/k-mesh still leaves behind.
_TOL_STRESS_SYM_EV_A3: float = 0.025
#: Non-vacuity floor for the stress [eV/Å³]: an all-zero return would pass
#: every symmetry gate above, so require a real hydrostatic component.
_MIN_STRESS_EV_A3: float = 1.0e-3

# ── Single-H cell (regressions/dynamics/bomd) ─────────────────────────

_H_BOX_A: float = 6.0
_GTH_H: str = "GTH-PBE-q1"
_H_ECUT_RY: float = 5.0
_H_MIX_BETA: float = 0.5
_H_MAX_ITER: int = 60
#: One unpaired electron ⇒ N↑ − N↓ = multiplicity − 1 = 1 with fixed
#: occupations (pw_solver.h:57-60).
_H_MULTIPLICITY: int = 2
_PIN_H_MAGNETIZATION: float = 1.0
_TOL_MAGNETIZATION: float = 1.0e-6

# ── PW BOMD budget (pw-bomd-h speed settings, Si2 geometry) ──────────

_BOMD_DT_FS: float = 0.5
_BOMD_STEPS: int = 3
_BOMD_TOL: float = 1.0e-4
_BOMD_MAX_ITER: int = 12

#: The four ATV::electronic::Smearing spellings of pw_solver.h:79-84.
_SMEARING_NAMES: tuple[str, ...] = (
    "none",
    "fermi_dirac",
    "methfessel_paxton",
    "gaussian",
)


# ── Frame / method builders ──────────────────────────────────────────


def _si2_frame(stretch: float, with_box: bool = True) -> "molpy.Frame":
    """Two-atom Si fcc primitive cell as a ``molpy.Frame``.

    Same cell and basis as reg_cuda-pw-forces.cu:60-83: rows
    [[0,a/2,a/2],[a/2,0,a/2],[a/2,a/2,0]] (symmetric, so the row/column
    convention of the cell matrix cannot flip the fixture), Si at the origin
    and at (a/4 + *stretch*, a/4, a/4).

    @param stretch: Displacement of atom 1 along x [Å]; 0 is the ideal
        diamond geometry.
    @param with_box: When false the cell is left unset — the reachable
        degenerate case for the fail-loud gate (molrs itself refuses a
        singular cell matrix, raising ValueError("SingularCell")).
    @return molpy.Frame with x/y/z [Å], mass [amu] and atomic_number columns.
    """
    frame = molpy.Frame()
    atoms = molpy.Block()
    atoms.insert("x", np.array([0.0, _A / 4.0 + stretch], dtype=np.float64))
    atoms.insert("y", np.array([0.0, _A / 4.0], dtype=np.float64))
    atoms.insert("z", np.array([0.0, _A / 4.0], dtype=np.float64))
    atoms.insert("mass", np.full(2, 28.0855, dtype=np.float64))
    atoms.insert("atomic_number", np.full(2, 14, dtype=np.uint32))
    frame["atoms"] = atoms
    if with_box:
        half = _A / 2.0
        frame.box = molpy.Box(
            molpy.core.Cell.matrix(
                [[0.0, half, half], [half, 0.0, half], [half, half, 0.0]]
            )
        )
    return frame


def _si2_method(xc: str = "PBE") -> "PW":
    """``PW`` carrying the pinned Si2 settings: identity as constructor
    arguments, knobs as members (no config bag).

    @param xc: Exchange-correlation functional name (identity, fixed at
        construction).
    """
    pw = PW(potential=GthPseudopotential(14, _GTH_SI), ecut_ry=_ECUT_RY, k_mesh=_K_MESH, xc=xc)
    pw.tol = _TOL
    pw.max_iter = _MAX_ITER
    return pw


def _si2_bomd_method() -> "PW":
    """The same method at the BOMD speed settings of reg_pw-bomd-h.cpp:44-45
    (tol 1e-4, max_iter 12) — a step must stay ~0.3 s."""
    pw = _si2_method()
    pw.tol = _BOMD_TOL
    pw.max_iter = _BOMD_MAX_ITER
    return pw


def _h_atom_frame() -> "molpy.Frame":
    """One H atom at the origin of a 6 Å cube (reg_pw-bomd-h.cpp:28-38)."""
    frame = molpy.Frame()
    atoms = molpy.Block()
    atoms.insert("x", np.zeros(1, dtype=np.float64))
    atoms.insert("y", np.zeros(1, dtype=np.float64))
    atoms.insert("z", np.zeros(1, dtype=np.float64))
    atoms.insert("mass", np.full(1, 1.008, dtype=np.float64))
    atoms.insert("atomic_number", np.full(1, 1, dtype=np.uint32))
    frame["atoms"] = atoms
    frame.box = molpy.Box.cube(_H_BOX_A)
    return frame


# ── Knobs: the members `cpu::pw::PW` publishes ───────────────────────


@pytest.mark.skipif(not _HAS_PW, reason="extension built without ATV_BUILD_PW")
class TestPwKnobDefaults:
    """The new members carry the C++ defaults of pw_solver.h:117-139."""

    def test_xc_defaults_to_pbe(self) -> None:
        """``xc`` is a functional *name* on the Python surface; the C++
        default is ``XCType::PBE`` (pw_solver.h:122)."""
        assert _si2_method().xc == "PBE"

    def test_spin_defaults_to_restricted(self) -> None:
        """Default is restricted, multiplicity 1 (pw_solver.h:126,130)."""
        pw = _si2_method()
        assert pw.spin_polarized is False
        assert pw.multiplicity == 1

    def test_smearing_defaults_to_none(self) -> None:
        """Fixed occupations by default (pw_solver.h:133)."""
        assert _si2_method().smearing == "none"

    def test_smearing_parameter_defaults(self) -> None:
        """degauss [Ry], MP order and extra bands (pw_solver.h:134-139)."""
        pw = _si2_method()
        assert pw.degauss == pytest.approx(0.02)
        assert pw.mp_order == 1
        assert pw.extra_bands == 4


@pytest.mark.skipif(not _HAS_PW, reason="extension built without ATV_BUILD_PW")
class TestPwKnobRoundTrip:
    """Assignment → read-back for each new member (pass-through only; the
    physics of these switches is pinned by regressions/pw/)."""

    @pytest.mark.parametrize("name", ("LSDA", "PBE"))
    def test_xc_name_roundtrip(self, name: str) -> None:
        """A supported functional name survives the enum mapping."""
        assert _si2_method(xc=name).xc == name

    def test_xc_unknown_raises(self) -> None:
        """An unsupported functional fails loud at construction and lists the
        supported names — same no-fallback contract as RKS(xc=...)
        (xc_names.h)."""
        with pytest.raises(ValueError, match="LSDA|PBE"):
            _si2_method(xc="NOTAFUNCTIONAL")

    @pytest.mark.parametrize("name", _SMEARING_NAMES)
    def test_smearing_name_roundtrip(self, name: str) -> None:
        """Every ``ATV::electronic::Smearing`` spelling of pw_solver.h:79-84 round-trips."""
        pw = _si2_method()
        pw.smearing = name
        assert pw.smearing == name

    def test_smearing_parameters_roundtrip(self) -> None:
        """Width [Ry], Methfessel-Paxton order and extra bands are plain
        members, set beside the smearing kind."""
        pw = _si2_method()
        pw.smearing = "gaussian"
        pw.degauss = 0.01
        pw.mp_order = 2
        pw.extra_bands = 8
        assert pw.smearing == "gaussian"
        assert pw.degauss == pytest.approx(0.01)
        assert pw.mp_order == 2
        assert pw.extra_bands == 8

    def test_spin_knobs_roundtrip(self) -> None:
        """Collinear switch + multiplicity (pw_solver.h:126-130)."""
        pw = _si2_method()
        pw.spin_polarized = True
        pw.multiplicity = 3
        assert pw.spin_polarized is True
        assert pw.multiplicity == 3


@pytest.mark.skipif(not _HAS_PW, reason="extension built without ATV_BUILD_PW")
class TestPwIdentity:
    """Pseudopotential, cutoff, k-mesh and functional are constructor
    arguments, fixed afterwards."""

    def test_identity_reads_back(self) -> None:
        pw = _si2_method(xc="LSDA")
        assert pw.potential.atomic_number == 14
        assert pw.potential.valence == 4
        assert pw.ecut_ry == pytest.approx(_ECUT_RY)
        assert pw.k_mesh == _K_MESH
        assert pw.xc == "LSDA"

    @pytest.mark.parametrize("name", ("potential", "ecut_ry", "k_mesh", "xc"))
    def test_identity_is_read_only(self, name: str) -> None:
        pw = _si2_method()
        with pytest.raises(AttributeError):
            setattr(pw, name, getattr(pw, name))

    def test_no_pseudopotential_raises(self) -> None:
        with pytest.raises(TypeError):
            PW(ecut_ry=_ECUT_RY, k_mesh=_K_MESH)

    def test_null_pseudopotential_raises(self) -> None:
        with pytest.raises((TypeError, ValueError)):
            PW(potential=None, ecut_ry=_ECUT_RY, k_mesh=_K_MESH)


# ── forces(frame): (N,3) [eV/Å] + write-back ─────────────────────────


@pytest.mark.skipif(not _HAS_PW, reason="extension built without ATV_BUILD_PW")
class TestPwForces:
    """``PW.forces(frame)`` — analytic local + Ewald + KB ionic forces.

    Each fixture is class-scoped: one SCF plus the nonlocal band re-solve is
    the expensive part (~0.5 s on this cell). They are ``@staticmethod``
    because a class-scoped fixture written as an instance method binds an
    instance pytest throws away before the tests run.
    """

    @pytest.fixture(scope="class")
    @staticmethod
    def default_forces() -> tuple["molpy.Frame", "np.ndarray"]:
        """Forces on the stretched Si2 cell with the default functional."""
        frame = _si2_frame(_STRETCH)
        pw = _si2_method()
        pw.build(frame)
        forces = np.asarray(pw.forces(frame), dtype=np.float64)
        return frame, forces

    @pytest.fixture(scope="class")
    @staticmethod
    def lsda_forces() -> tuple["molpy.Frame", "np.ndarray"]:
        """The pinned run: same cell, ``xc = LSDA`` (REFERENCE.md settings)."""
        frame = _si2_frame(_STRETCH)
        pw = _si2_method(xc="LSDA")
        pw.build(frame)
        forces = np.asarray(pw.forces(frame), dtype=np.float64)
        return frame, forces

    def test_returns_n_by_three_array(
        self, default_forces: tuple["molpy.Frame", "np.ndarray"]
    ) -> None:
        """One row per atom, three Cartesian components [eV/Å]."""
        _, forces = default_forces
        assert forces.shape == (2, 3)

    def test_forces_finite(
        self, default_forces: tuple["molpy.Frame", "np.ndarray"]
    ) -> None:
        _, forces = default_forces
        assert np.all(np.isfinite(forces))

    def test_forces_translation_invariance(
        self, default_forces: tuple["molpy.Frame", "np.ndarray"]
    ) -> None:
        """Σ_i F_i = 0 per component: a rigid translation of the whole basis
        through a periodic cell cannot change the energy. Graded at the
        numerical force tier (1e-4 eV/Å) — the discrete FFT grid and the
        finite SCF tolerance leave ~3e-5 eV/Å behind, not zero."""
        _, forces = default_forces
        assert forces.sum(axis=0) == pytest.approx(np.zeros(3), abs=_TOL_NET_FORCE_EV_A)

    def test_forces_non_vacuous(
        self, default_forces: tuple["molpy.Frame", "np.ndarray"]
    ) -> None:
        """The +0.08 Å stretch must produce a real force — the same
        non-vacuity floor the CUDA parity regression uses
        (REFERENCE.md:42)."""
        _, forces = default_forces
        max_force = float(np.max(np.abs(forces)))
        assert max_force > _MIN_MAX_FORCE_EV_A

    def test_force_columns_written_back(
        self, default_forces: tuple["molpy.Frame", "np.ndarray"]
    ) -> None:
        """fx/fy/fz land on the SAME molpy.Frame and match the returned
        array — the write-back contract the SCF and AM1 bindings keep."""
        frame, forces = default_forces
        atoms = frame["atoms"]
        fx = np.asarray(atoms["fx"], dtype=np.float64)
        fy = np.asarray(atoms["fy"], dtype=np.float64)
        fz = np.asarray(atoms["fz"], dtype=np.float64)
        assert fx.shape == (2,)
        assert np.any(fx != 0.0), "fx is all zeros — forces were not written back"
        assert fx == pytest.approx(forces[:, 0], abs=_TOL_FRAME_FORCE_EV_A)
        assert fy == pytest.approx(forces[:, 1], abs=_TOL_FRAME_FORCE_EV_A)
        assert fz == pytest.approx(forces[:, 2], abs=_TOL_FRAME_FORCE_EV_A)

    def test_energy_written_back_finite(
        self, default_forces: tuple["molpy.Frame", "np.ndarray"]
    ) -> None:
        """forces() runs its own SCF, so it also lands the total energy
        [eV] as typed frame metadata: finite and bound (negative)."""
        frame, _ = default_forces
        energy = frame.meta["energy"]
        assert isinstance(energy, float)
        assert np.isfinite(energy)
        assert energy < 0.0

    def test_lsda_all_force_components_match_qe(
        self, lsda_forces: tuple["molpy.Frame", "np.ndarray"]
    ) -> None:
        """Actual QE7.5 preprojection forces at the original component bound."""
        _, forces = lsda_forces
        np.testing.assert_allclose(forces, _PIN_FORCE_EV_A,
                                   atol=_TOL_FORCE_PIN_EV_A, rtol=0)

    def test_lsda_force_path_energy_matches_qe(
        self, lsda_forces: tuple["molpy.Frame", "np.ndarray"]
    ) -> None:
        """Actual QE7.5 total energy, published to the same Frame in eV."""
        frame, _ = lsda_forces
        assert frame.meta["energy"] == pytest.approx(
            _PIN_FORCE_PATH_ENERGY_EV, abs=_TOL_PIN_ENERGY_EV
        )


# ── stress(frame): Voigt 6-vector [eV/Å³] ────────────────────────────


@pytest.mark.skipif(not _HAS_PW, reason="extension built without ATV_BUILD_PW")
class TestPwStress:
    """``PW.stress(frame)`` — Nielsen-Martin analytic cell stress on the
    *ideal* (unstretched) primitive cell, where crystal symmetry makes the
    answer predictable without a magnitude pin."""

    @pytest.fixture(scope="class")
    @staticmethod
    def ideal_stress() -> tuple["molpy.Frame", "np.ndarray", "np.ndarray"]:
        """Stress on ideal diamond Si; also returns the x column recorded
        before the call, for the immutability check."""
        frame = _si2_frame(0.0)
        x_before = np.asarray(frame["atoms"]["x"], dtype=np.float64).copy()
        pw = _si2_method()
        pw.build(frame)
        stress = np.asarray(pw.stress(frame), dtype=np.float64)
        # Frame stores extensive virial [eV]; the returned stress is [eV/Å³].
        np.testing.assert_allclose(
            np.asarray(frame.meta["stress"]), stress * frame.box.volume()
        )
        assert np.isfinite(frame.meta["energy"])
        return frame, stress, x_before

    def test_returns_voigt_six_vector(
        self, ideal_stress: tuple["molpy.Frame", "np.ndarray", "np.ndarray"]
    ) -> None:
        """Voigt order xx, yy, zz, xy, xz, yz."""
        _, stress, _ = ideal_stress
        assert stress.shape == (6,)

    def test_stress_finite(
        self, ideal_stress: tuple["molpy.Frame", "np.ndarray", "np.ndarray"]
    ) -> None:
        _, stress, _ = ideal_stress
        assert np.all(np.isfinite(stress))

    def test_offdiagonal_vanishes_by_symmetry(
        self, ideal_stress: tuple["molpy.Frame", "np.ndarray", "np.ndarray"]
    ) -> None:
        """σ_xy = σ_xz = σ_yz = 0 for cubic diamond Si; graded at the
        regression's soft window kSymTol = 0.025 eV/Å³
        (reg_pw-analytic-stress-si.cpp:28), which is what an incomplete
        cutoff and a Γ-only mesh still leave behind."""
        _, stress, _ = ideal_stress
        assert stress[3:] == pytest.approx(np.zeros(3), abs=_TOL_STRESS_SYM_EV_A3)

    def test_diagonal_is_isotropic(
        self, ideal_stress: tuple["molpy.Frame", "np.ndarray", "np.ndarray"]
    ) -> None:
        """σ_xx = σ_yy = σ_zz: the three cubic axes are equivalent, so the
        hydrostatic stress cannot single one out. Same soft window."""
        _, stress, _ = ideal_stress
        diagonal = stress[:3]
        assert float(np.max(diagonal) - np.min(diagonal)) <= _TOL_STRESS_SYM_EV_A3

    def test_diagonal_is_positive_under_compression(
        self, ideal_stress: tuple["molpy.Frame", "np.ndarray", "np.ndarray"]
    ) -> None:
        """Mechanical / CP2K sign σ = −(1/Ω)∂E/∂ε: a = 5.43 Å sits below the
        PseudoDojo EOS a0 ≈ 5.468 Å (pw-analytic-stress-si/REFERENCE.md), so
        the cell is compressed and the diagonal is positive. This also keeps
        the two symmetry gates above honest — an all-zero return would pass
        both."""
        _, stress, _ = ideal_stress
        assert float(np.min(stress[:3])) > _MIN_STRESS_EV_A3

    def test_stress_does_not_move_the_atoms(
        self, ideal_stress: tuple["molpy.Frame", "np.ndarray", "np.ndarray"]
    ) -> None:
        """stress() reports a cell derivative; it writes no geometry back."""
        frame, _, x_before = ideal_stress
        x_after = np.asarray(frame["atoms"]["x"], dtype=np.float64)
        assert x_after == pytest.approx(x_before, abs=1e-12)


# ── PWResult channels beyond energy ──────────────────────────────────


@pytest.mark.skipif(not _HAS_PW, reason="extension built without ATV_BUILD_PW")
class TestPwResultFields:
    """The ``PWResult`` members of pw_solver.h:52-70 that the binding never
    exposed: magnetization, e_fermi, n_elec_occ, e_entropy, n_kpts."""

    @pytest.fixture(scope="class")
    @staticmethod
    def si2_result() -> "PWResult":
        """One restricted, fixed-occupation SCF on the stretched cell."""
        frame = _si2_frame(_STRETCH)
        pw = _si2_method()
        pw.build(frame)
        result = pw.run(frame)
        assert frame.meta["energy"] == pytest.approx(
            result.energy * 13.605693, rel=1e-12
        )
        return result

    def test_n_kpts_is_one_at_gamma(self, si2_result: "PWResult") -> None:
        """k_mesh = 1 is the Γ point only, so the Monkhorst-Pack set has a
        single member (pw_solver.h:65-66)."""
        assert si2_result.n_kpts == 1

    def test_e_fermi_finite(self, si2_result: "PWResult") -> None:
        """Documented as 0 with fixed occupations (pw_solver.h:62); the gate
        here is only that the channel exists and carries a real number."""
        assert np.isfinite(si2_result.e_fermi)

    def test_n_elec_occ_finite(self, si2_result: "PWResult") -> None:
        """Σ w f (pw_solver.h:63) — the fixed-occupation path leaves it at
        0, so this asserts finiteness rather than an electron count."""
        assert np.isfinite(si2_result.n_elec_occ)

    def test_magnetization_zero_when_restricted(self, si2_result: "PWResult") -> None:
        """Exactly zero when ``spin_polarized == false`` (pw_solver.h:57-59)."""
        assert si2_result.magnetization == pytest.approx(0.0, abs=_TOL_MAGNETIZATION)

    def test_entropy_zero_without_smearing(self, si2_result: "PWResult") -> None:
        """−TS enters the energy only with metallic smearing
        (pw_solver.h:64); this run has ``smearing = none``."""
        assert si2_result.e_entropy == pytest.approx(0.0, abs=1e-12)


# ── Collinear spin-polarized path driven from Python ─────────────────


@pytest.mark.skipif(not _HAS_PW, reason="extension built without ATV_BUILD_PW")
class TestPwSpinPolarized:
    """One unpaired electron: the open-shell guard must be reachable from
    Python, not just from C++."""

    @pytest.fixture(scope="class")
    @staticmethod
    def h_atom_result() -> "PWResult":
        """Spin-polarized doublet H in the pw-bomd-h cell."""
        frame = _h_atom_frame()
        pw = PW(potential=GthPseudopotential(1, _GTH_H), ecut_ry=_H_ECUT_RY, k_mesh=_K_MESH)
        pw.tol = _TOL
        pw.max_iter = _H_MAX_ITER
        pw.mix_beta = _H_MIX_BETA
        pw.spin_polarized = True
        pw.multiplicity = _H_MULTIPLICITY
        pw.build(frame)
        return pw.run(frame)

    def test_converges(self, h_atom_result: "PWResult") -> None:
        assert h_atom_result.converged is True

    def test_energy_finite_and_bound(self, h_atom_result: "PWResult") -> None:
        """Total energy [Ry] of a bound H atom is finite and negative."""
        assert np.isfinite(h_atom_result.energy)
        assert h_atom_result.energy < 0.0

    def test_magnetization_is_one_unpaired_electron(
        self, h_atom_result: "PWResult"
    ) -> None:
        """Fixed occupations set N↑ − N↓ = multiplicity − 1 exactly
        (pw_solver.h:57-60), i.e. 1 electron for a doublet."""
        assert h_atom_result.magnetization == pytest.approx(
            _PIN_H_MAGNETIZATION, abs=_TOL_MAGNETIZATION
        )


# ── Fail-loud boundary: no cell ──────────────────────────────────────


@pytest.mark.skipif(not _HAS_PW, reason="extension built without ATV_BUILD_PW")
class TestPwFailsLoud:
    """A plane-wave basis is defined by the cell; without one the run today
    silently returns a NaN energy. The binding must refuse instead, naming
    the box."""

    @staticmethod
    def _boxless_case() -> tuple["molpy.Frame", "PW"]:
        """The Si2 basis with ``frame.box`` left unset. ``max_iter`` is cut
        to 2 so that a binding which does *not* guard fails fast (today it
        runs the SCF anyway and reports a NaN energy)."""
        frame = _si2_frame(_STRETCH, with_box=False)
        pw = _si2_method()
        pw.max_iter = 2
        return frame, pw

    def test_fixture_really_has_no_box(self) -> None:
        """Guard on the guard: the degenerate case must be degenerate."""
        frame, _ = self._boxless_case()
        assert frame.box is None

    def test_run_without_box_raises(self) -> None:
        """``build`` is lazy inside ``run`` (src/cpu/planewave/pw_solver.cpp:
        219-221), so the guard may fire in either call — both are inside the
        raises block."""
        frame, pw = self._boxless_case()
        with pytest.raises(RuntimeError, match="(?i)box"):
            pw.build(frame)
            pw.run(frame)

    def test_forces_without_box_raises(self) -> None:
        frame, pw = self._boxless_case()
        with pytest.raises(RuntimeError, match="(?i)box"):
            pw.build(frame)
            pw.forces(frame)


# ── PW as an MD force source (plane-wave BOMD) ───────────────────────


@pytest.mark.skipif(not _HAS_PW, reason="extension built without ATV_BUILD_PW")
class TestPwBomd:
    """``MD.set_evaluator(pw)`` — the C++ twin is
    regressions/dynamics/bomd (reg_pw-bomd-h.cpp:39-60), whose speed settings
    (tol 1e-4, max_iter 12) are reused here on the stretched Si2 cell so one
    step stays ~0.3 s. Three steps is a smoke, not an energy-drift test:
    the drift pin belongs to that regression.

    The MD names are imported inside the bodies for the
    same build-capability reason as test_md.py's AM1 class — they are absent
    from a wheel built without the matching domain.
    """

    @pytest.fixture(scope="class")
    @staticmethod
    def bomd_si2() -> tuple["molpy.Frame", "np.ndarray"]:
        """One 3-step plane-wave BOMD run; returns the advanced frame and
        the x column it started from."""
        from atomiverse.md import MD
        from atomiverse.md.integrator import VelocityVerlet

        frame = _si2_frame(_STRETCH)
        x_before = np.asarray(frame["atoms"]["x"], dtype=np.float64).copy()
        sim = MD()
        sim.set_integrator(VelocityVerlet(_BOMD_DT_FS))
        sim.set_evaluator(_si2_bomd_method())
        sim.run(frame, _BOMD_STEPS)
        return frame, x_before

    def test_set_evaluator_accepts_pw(self) -> None:
        """The evaluator dispatch must accept the plane-wave force source.
        Checked without running so the failure is unambiguous — this is the
        whole point of the spec row."""
        from atomiverse.md import MD
        from atomiverse.md.integrator import VelocityVerlet

        sim = MD()
        sim.set_integrator(VelocityVerlet(_BOMD_DT_FS))
        sim.set_evaluator(_si2_bomd_method())

    def test_potential_energy_finite_and_bound(
        self, bomd_si2: tuple["molpy.Frame", "np.ndarray"]
    ) -> None:
        """PE [eV] after the run is finite and negative (bound crystal, not
        a blow-up) — same gate as reg_pw-bomd-h.cpp:66-72."""
        frame, _ = bomd_si2
        energy = frame.meta["energy"]
        assert isinstance(energy, float)
        assert np.isfinite(energy)
        assert energy < 0.0

    def test_positions_advanced(
        self, bomd_si2: tuple["molpy.Frame", "np.ndarray"]
    ) -> None:
        """The atoms moved and the new positions came back into the SAME
        molpy.Frame. Velocities start at zero here, so motion is possible
        only if plane-wave forces actually reached the integrator."""
        frame, x_before = bomd_si2
        x_after = np.asarray(frame["atoms"]["x"], dtype=np.float64)
        assert not np.allclose(x_before, x_after)

    def test_velocities_finite(
        self, bomd_si2: tuple["molpy.Frame", "np.ndarray"]
    ) -> None:
        """No NaN/inf leaked into the velocity columns."""
        atoms = bomd_si2[0]["atoms"]
        for component in ("vx", "vy", "vz"):
            values = np.asarray(atoms[component], dtype=np.float64)
            assert np.all(np.isfinite(values)), f"{component} is not finite"

    def test_forces_finite_and_non_zero(
        self, bomd_si2: tuple["molpy.Frame", "np.ndarray"]
    ) -> None:
        """The forces written back on the last step are finite and, on the
        stretched cell, not identically zero."""
        fx = np.asarray(bomd_si2[0]["atoms"]["fx"], dtype=np.float64)
        assert np.all(np.isfinite(fx))
        assert np.any(fx != 0.0)

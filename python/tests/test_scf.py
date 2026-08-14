"""Gaussian SCF single-point Python surface tests (spec: py-scf-single-point).

Binds the concrete mean-field methods ``RHF`` / ``UHF`` / ``RKS`` / ``UKS``
as single-point evaluators over a ``molrs.Frame``: ``run(frame)`` writes the
total energy [eV] into ``frame.meta["energy"]``, ``forces(frame)`` returns an
(N, 3) array [eV/Å] and writes ``fx``/``fy``/``fz`` back to the frame, and
``energy()`` / ``converged()`` / ``iterations()`` report the last run
(Hartree / flag / count). Parameters are **constructor arguments** — basis and
functional are strings, there is no Config bag and no factory.

No new oracles: every number below is the value an existing C++ regression
already pins, cited per test. All goldens are hard-coded — nothing here runs
PySCF, reads a file, or touches the network.

===================================  =========================================
Pinned quantity                      Source (hard-coded golden)
===================================  =========================================
E_RHF/STO-3G(H₂O) = −74.961754 Ha    regressions/scf/h2o_rhf_sto3g/REFERENCE.md
                                     (LibreTexts §28.2; window 1e-3 Ha)
E_UHF/6-31G*(N₂⁺) =                  regressions/scf/n2_ionization/REFERENCE.md
  −108.365971102534 Ha               (PySCF 2.14.0, conv_tol 1e-12)
E_B3LYP/def2-SVP(H₂O) =              regressions/scf/b3lyp-fused-jk-h2o/
  −76.35814183099154 Ha              REFERENCE.md (PySCF 2.14.0; window 1e-6 Ha)
∂E/∂z(O) = −6.142775911808e-02       regressions/scf/h2o_rhf_sto3g_grad/
  Ha/Bohr                            REFERENCE.md:34 (PySCF 2.14.0 analytic)
===================================  =========================================

Sign convention (verified, not assumed): Atomiverse publishes **forces**
F = −∂E/∂R in eV/Å — see
``regressions/scf/h2o_rhf_sto3g_grad/reg_h2o_rhf_sto3g_grad.cpp:69-73``, which
negates the reported force to recover the PySCF gradient. The pinned oxygen
gradient of −6.142775911808e-02 Ha/Bohr is therefore a **positive** z force of
+3.158742374 eV/Å.

Unit factors are the ``core/units.h`` constants verbatim
(``kEvInHartree = 1/27.211386245988``, ``kAngstromInBohr =
1.8897261245650618``) so the conversions here are bit-comparable with the C++
regressions; the rounded 0.529177210544 Bohr radius is deliberately not used
(it differs from the pair above in the 8th significant digit).

Each SCF solve costs seconds, so a class shares **one** run through a
``scope="class"`` fixture. Those fixtures are ``@staticmethod`` on purpose: a
class-scoped fixture written as an instance method is deprecated in pytest
(``PytestRemovedIn10Warning``) because the instance it binds to is thrown away
before the tests run. Nothing here needs ``self`` — the run is handed to the
tests as a return value, never as an instance attribute.

Capability gate: the SCF methods exist only on an extension built with
``ATV_BUILD_HF=ON`` (same contract as PW), so the compute classes are guarded
by ``_HAS_HF``. ``atomiverse.scf`` itself imports on every build.
"""

from __future__ import annotations

import importlib
import math

import numpy as np
import pytest
import molrs

from atomiverse._binding import _cpp, extension_available

_HAS_HF: bool = extension_available() and hasattr(_cpp, "RHF")

if _HAS_HF:
    from atomiverse.scf import RHF, RKS, UHF, UKS
else:  # capability-gated wheel: names resolve to nothing, classes skip
    RHF = RKS = UHF = UKS = None  # type: ignore[misc, assignment]


# ── Unit factors (core/units.h) ─────────────────────────────────────

_HARTREE_IN_EV: float = 27.211386245988  # units::kHartreeInEv
_ANGSTROM_IN_BOHR: float = 1.8897261245650618  # units::kAngstromInBohr
# units::kHartreePerBohrInEvPerAngstrom = 51.422067474673923
_HARTREE_PER_BOHR_IN_EV_PER_ANGSTROM: float = _HARTREE_IN_EV * _ANGSTROM_IN_BOHR


# ── Hard-coded goldens + their stated windows ───────────────────────

# regressions/scf/h2o_rhf_sto3g/REFERENCE.md (LibreTexts §28.2, PySCF-checked).
_E_RHF_H2O_STO3G_HA: float = -74.961754
_TOL_RHF_H2O_HA: float = 1.0e-3  # the regression's own milli-Hartree window

# regressions/scf/n2_ionization/REFERENCE.md (PySCF 2.14.0, conv_tol 1e-12).
_E_UHF_N2_CATION_HA: float = -108.365971102534
# The C++ case grades the DERIVED ionization energies at ±1e-3 eV, i.e.
# 1e-3 / 27.211386245988 = 3.7e-5 Ha on the underlying total energies. 1e-4 Ha
# is a deliberately looser band on the raw total energy of the cation alone.
_TOL_UHF_N2_HA: float = 1.0e-4

# regressions/scf/b3lyp-fused-jk-h2o/REFERENCE.md (PySCF 2.14.0, ac-006).
_E_RKS_B3LYP_H2O_DEF2SVP_HA: float = -76.35814183099154
_TOL_RKS_B3LYP_HA: float = 1.0e-6  # the regression's own window

# regressions/scf/h2o_rhf_sto3g_grad/REFERENCE.md:34 — PySCF analytic dE/dz on
# oxygen [Ha/Bohr]; F = −∂E/∂R, so the force is positive along +z.
_GRAD_O_Z_HA_PER_BOHR: float = -6.142775911808e-02
_FORCE_O_Z_EV_PER_ANGSTROM: float = (
    -_GRAD_O_Z_HA_PER_BOHR * _HARTREE_PER_BOHR_IN_EV_PER_ANGSTROM
)
_TOL_FORCE_EV_PER_ANGSTROM: float = 1.0e-4  # numerical-force tolerance


# ── Geometries ──────────────────────────────────────────────────────

# regressions/scf/h2o_rhf_sto3g/reg_h2o_rhf_sto3g.cpp:22-26 — derived from the
# citation geometry with the same formulas (R = 0.95 Å, ∠HOH = 104.5°), NOT
# from the rounded table in that case's REFERENCE.md.
_R_OH_ANGSTROM: float = 0.95
_HALF_HOH_RAD: float = 0.5 * 104.5 * math.pi / 180.0
_H_X: float = _R_OH_ANGSTROM * math.sin(_HALF_HOH_RAD)  # 0.7511550950566511
_H_Z: float = _R_OH_ANGSTROM * math.cos(_HALF_HOH_RAD)  # 0.5816064160327268

_BOX_ANGSTROM: float = 20.0  # gas-phase cell; does not enter the isolated SCF


def _make_frame(
    atomic_numbers: list[int],
    masses: list[float],
    xyz: list[tuple[float, float, float]],
) -> "molrs.Frame":
    """Build a gas-phase molecular ``molrs.Frame`` in a 20 Å cubic cell.

    @param atomic_numbers: Nuclear charge Z per atom (the SCF species column).
    @param masses: Atomic masses [amu] per atom.
    @param xyz: Cartesian positions [Å] per atom.
    @return A ``molrs.Frame`` carrying the x/y/z/mass/atomic_number columns.
    """
    assert len(atomic_numbers) == len(masses) == len(xyz)
    frame = molrs.Frame()
    atoms = molrs.Block()
    atoms.insert("x", np.array([p[0] for p in xyz], dtype=np.float64))
    atoms.insert("y", np.array([p[1] for p in xyz], dtype=np.float64))
    atoms.insert("z", np.array([p[2] for p in xyz], dtype=np.float64))
    atoms.insert("mass", np.array(masses, dtype=np.float64))
    atoms.insert("atomic_number", np.array(atomic_numbers, dtype=np.uint32))
    frame["atoms"] = atoms
    frame.box = molrs.Box.cube(_BOX_ANGSTROM)
    return frame


def _h2o_libretexts_frame() -> "molrs.Frame":
    """Water at R(O–H) = 0.95 Å / ∠HOH = 104.5° (LibreTexts pin geometry)."""
    return _make_frame(
        [8, 1, 1],
        [15.999, 1.008, 1.008],
        [(0.0, 0.0, 0.0), (_H_X, 0.0, _H_Z), (-_H_X, 0.0, _H_Z)],
    )


def _h2o_pyscf_frame() -> "molrs.Frame":
    """Water at the PySCF oracle preset geometry (`pin_scf.py --geom water`)."""
    return _make_frame(
        [8, 1, 1],
        [15.999, 1.008, 1.008],
        [
            (0.0, 0.0, 0.1173),
            (0.0, 0.7572, -0.4692),
            (0.0, -0.7572, -0.4692),
        ],
    )


def _n2_frame() -> "molrs.Frame":
    """N₂ at the experimental bond length 1.0977 Å (n2_ionization geometry)."""
    return _make_frame([7, 7], [14.007, 14.007], [(0.0, 0.0, 0.0), (0.0, 0.0, 1.0977)])


def _o2_frame() -> "molrs.Frame":
    """O₂ at 1.2075 Å — the triplet ground state, an open-shell UKS smoke."""
    return _make_frame([8, 8], [15.999, 15.999], [(0.0, 0.0, 0.0), (0.0, 0.0, 1.2075)])


@pytest.fixture
def n2_frame() -> "molrs.Frame":
    return _n2_frame()


# ── The module exists on every build (capability in __all__) ────────


def test_scf_module_importable() -> None:
    """``atomiverse.scf`` imports on every build (PW-style capability
    module): the wheel reflects what was compiled instead of vanishing."""
    importlib.import_module("atomiverse.scf")


def test_scf_module_exports_track_the_build() -> None:
    """The four method names are exported iff the extension carries HF."""
    scf = importlib.import_module("atomiverse.scf")
    for name in ("RHF", "UHF", "RKS", "UKS"):
        assert (name in scf.__all__) is _HAS_HF, name


# ── RHF: the flagship STO-3G water pin ──────────────────────────────


@pytest.mark.skipif(not _HAS_HF, reason="extension built without ATV_BUILD_HF")
class TestRHF:
    @pytest.fixture(scope="class")
    @staticmethod
    def rhf_h2o() -> tuple["molrs.Frame", "RHF"]:
        """One RHF/STO-3G water single point, shared read-only below.

        Constructor arguments only (iron law 3): ``RHF()`` carries the
        documented defaults — STO-3G, charge 0, tol 1e-9 Ha, 100 iterations.
        """
        frame = _h2o_libretexts_frame()
        method = RHF()
        method.run(frame)
        return frame, method

    def test_converged(self, rhf_h2o: tuple["molrs.Frame", "RHF"]) -> None:
        """The SCF loop reports convergence (no silent non-stationary read)."""
        _, method = rhf_h2o
        assert method.converged() is True

    def test_iterations_positive(self, rhf_h2o: tuple["molrs.Frame", "RHF"]) -> None:
        """``iterations()`` reports the loop count of the last run."""
        _, method = rhf_h2o
        assert method.iterations() > 0

    def test_frame_meta_energy_matches_libretexts_pin(
        self, rhf_h2o: tuple["molrs.Frame", "RHF"]
    ) -> None:
        """``run()`` writes the total energy [eV] to ``frame.meta["energy"]``;
        converted to Hartree it reproduces the −74.961754 Ha pin of
        regressions/scf/h2o_rhf_sto3g/REFERENCE.md within its 1e-3 Ha window."""
        frame, _ = rhf_h2o
        energy_ev = frame.meta["energy"]
        assert isinstance(energy_ev, float)
        assert np.isfinite(energy_ev)
        energy_ha = energy_ev / _HARTREE_IN_EV
        assert energy_ha == pytest.approx(_E_RHF_H2O_STO3G_HA, abs=_TOL_RHF_H2O_HA)

    def test_energy_accessor_is_hartree(
        self, rhf_h2o: tuple["molrs.Frame", "RHF"]
    ) -> None:
        """``energy()`` is the same total energy in **Hartree** (the C++
        ``MeanFieldMethod::energy()`` currency), not eV."""
        _, method = rhf_h2o
        assert method.energy() == pytest.approx(
            _E_RHF_H2O_STO3G_HA, abs=_TOL_RHF_H2O_HA
        )

    def test_no_result_blocks_on_frame(
        self, rhf_h2o: tuple["molrs.Frame", "RHF"]
    ) -> None:
        """Global results ride typed meta — no invented result block."""
        frame, _ = rhf_h2o
        assert "scalars" not in frame.keys()
        assert "stress" not in frame.keys()


# ── UHF: open-shell doublet cation ──────────────────────────────────


@pytest.mark.skipif(not _HAS_HF, reason="extension built without ATV_BUILD_HF")
class TestUHF:
    @pytest.fixture(scope="class")
    @staticmethod
    def uhf_n2_cation() -> "UHF":
        """N₂⁺ doublet: multiplicity resolves α/β from N_e = ΣZ − charge."""
        frame = _n2_frame()
        method = UHF(basis="6-31G*", charge=1, multiplicity=2, tol=1e-12, max_iter=400)
        method.run(frame)
        return method

    def test_converged(self, uhf_n2_cation: "UHF") -> None:
        assert uhf_n2_cation.converged() is True

    def test_energy_matches_pyscf_pin(self, uhf_n2_cation: "UHF") -> None:
        """E_UHF(N₂⁺) reproduces the −108.365971102534 Ha PySCF 2.14.0 pin of
        regressions/scf/n2_ionization/REFERENCE.md within 1e-4 Ha (that case
        grades the derived IPs at ±1e-3 eV ≈ 3.7e-5 Ha)."""
        assert uhf_n2_cation.energy() == pytest.approx(
            _E_UHF_N2_CATION_HA, abs=_TOL_UHF_N2_HA
        )


# ── RKS: hybrid functional on a polarized basis ─────────────────────


@pytest.mark.skipif(not _HAS_HF, reason="extension built without ATV_BUILD_HF")
class TestRKS:
    @pytest.fixture(scope="class")
    @staticmethod
    def rks_b3lyp_h2o() -> "RKS":
        """B3LYP/def2-SVP water — the spherical-d hybrid path."""
        frame = _h2o_pyscf_frame()
        method = RKS(functional="B3LYP", basis="def2-SVP", tol=1e-10, max_iter=200)
        method.run(frame)
        return method

    def test_converged(self, rks_b3lyp_h2o: "RKS") -> None:
        assert rks_b3lyp_h2o.converged() is True

    def test_energy_matches_pyscf_pin(self, rks_b3lyp_h2o: "RKS") -> None:
        """E_B3LYP/def2-SVP(H₂O) reproduces the −76.35814183099154 Ha PySCF
        2.14.0 pin of regressions/scf/b3lyp-fused-jk-h2o/REFERENCE.md within
        that case's own 1e-6 Ha window."""
        assert rks_b3lyp_h2o.energy() == pytest.approx(
            _E_RKS_B3LYP_H2O_DEF2SVP_HA, abs=_TOL_RKS_B3LYP_HA
        )


# ── UKS: open-shell DFT smoke (ac-001's fourth method) ──────────────


@pytest.mark.skipif(not _HAS_HF, reason="extension built without ATV_BUILD_HF")
class TestUKS:
    @pytest.fixture(scope="class")
    @staticmethod
    def uks_o2_triplet() -> tuple["molrs.Frame", "UKS"]:
        """Triplet O₂ / LSDA / STO-3G. Smoke only — no pinned oracle for
        this combination exists in-tree, so nothing numeric is asserted."""
        frame = _o2_frame()
        method = UKS(functional="LSDA", charge=0, multiplicity=3)
        method.run(frame)
        return frame, method

    def test_converged(self, uks_o2_triplet: tuple["molrs.Frame", "UKS"]) -> None:
        _, method = uks_o2_triplet
        assert method.converged() is True

    def test_energy_finite_and_negative(
        self, uks_o2_triplet: tuple["molrs.Frame", "UKS"]
    ) -> None:
        _, method = uks_o2_triplet
        energy = method.energy()
        assert np.isfinite(energy)
        assert energy < 0.0

    def test_frame_meta_energy_written(
        self, uks_o2_triplet: tuple["molrs.Frame", "UKS"]
    ) -> None:
        """Open-shell KS writes the same typed meta channel [eV] as RHF."""
        frame, method = uks_o2_triplet
        energy_ev = frame.meta["energy"]
        assert isinstance(energy_ev, float)
        assert energy_ev / _HARTREE_IN_EV == pytest.approx(method.energy(), abs=1e-6)


# ── forces(): array out AND fx/fy/fz written back ───────────────────


@pytest.mark.skipif(not _HAS_HF, reason="extension built without ATV_BUILD_HF")
class TestScfForces:
    @pytest.fixture(scope="class")
    @staticmethod
    def rhf_forces_h2o() -> tuple["molrs.Frame", np.ndarray]:
        """RHF/STO-3G analytic forces at the pinned-gradient geometry.

        tol/max_iter mirror regressions/scf/h2o_rhf_sto3g_grad
        (``RHF{STO3G, 0, 1e-12, 200}``) so the comparison is against the pin
        at the settings the pin was taken at.
        """
        frame = _h2o_pyscf_frame()
        method = RHF(tol=1e-12, max_iter=200)
        forces = method.forces(frame)
        return frame, np.asarray(forces, dtype=np.float64)

    def test_returns_n_by_three_array(
        self, rhf_forces_h2o: tuple["molrs.Frame", np.ndarray]
    ) -> None:
        _, forces = rhf_forces_h2o
        assert forces.shape == (3, 3)

    def test_forces_finite(
        self, rhf_forces_h2o: tuple["molrs.Frame", np.ndarray]
    ) -> None:
        _, forces = rhf_forces_h2o
        assert np.all(np.isfinite(forces))

    def test_force_columns_written_back(
        self, rhf_forces_h2o: tuple["molrs.Frame", np.ndarray]
    ) -> None:
        """``forces()`` publishes fx/fy/fz on the molrs.Frame, matching the
        returned array (same write-back contract as MD)."""
        frame, forces = rhf_forces_h2o
        fx = np.asarray(frame["atoms"]["fx"], dtype=np.float64)
        fy = np.asarray(frame["atoms"]["fy"], dtype=np.float64)
        fz = np.asarray(frame["atoms"]["fz"], dtype=np.float64)
        assert fx.shape == (3,)
        assert np.any(np.abs(fz) > 0.0)
        assert fx == pytest.approx(forces[:, 0], abs=_TOL_FORCE_EV_PER_ANGSTROM)
        assert fy == pytest.approx(forces[:, 1], abs=_TOL_FORCE_EV_PER_ANGSTROM)
        assert fz == pytest.approx(forces[:, 2], abs=_TOL_FORCE_EV_PER_ANGSTROM)

    def test_oxygen_z_force_matches_pinned_gradient(
        self, rhf_forces_h2o: tuple["molrs.Frame", np.ndarray]
    ) -> None:
        """F_z(O) = −∂E/∂z(O) at the PySCF pin: −(−6.142775911808e-02 Ha/Bohr)
        = +3.158742374 eV/Å (regressions/scf/h2o_rhf_sto3g_grad/REFERENCE.md:34;
        sign convention read off reg_h2o_rhf_sto3g_grad.cpp:69-73)."""
        _, forces = rhf_forces_h2o
        assert forces[0, 2] == pytest.approx(
            _FORCE_O_Z_EV_PER_ANGSTROM, abs=_TOL_FORCE_EV_PER_ANGSTROM
        )

    def test_forces_sum_to_zero(
        self, rhf_forces_h2o: tuple["molrs.Frame", np.ndarray]
    ) -> None:
        """Translational invariance: Σ_A F_A = 0 (the C++ case grades the
        equivalent gradient sum at 1e-8 Ha/Bohr ≈ 5e-7 eV/Å)."""
        _, forces = rhf_forces_h2o
        assert forces.sum(axis=0) == pytest.approx(
            np.zeros(3), abs=_TOL_FORCE_EV_PER_ANGSTROM
        )


# ── Fail-loud boundaries (no fallback basis / functional / occupation) ──


@pytest.mark.skipif(not _HAS_HF, reason="extension built without ATV_BUILD_HF")
class TestScfFailsLoud:
    def test_unknown_basis_raises(self) -> None:
        """An unsupported basis name fails at construction and lists the
        supported names (no silent fallback to STO-3G)."""
        with pytest.raises(ValueError, match="STO-3G"):
            RHF(basis="6-311++G(3df,3pd)")

    def test_unknown_functional_raises(self) -> None:
        """An unsupported functional name fails at construction and lists the
        supported functionals."""
        with pytest.raises(ValueError, match="B3LYP"):
            RKS(functional="M06-2X")

    def test_bad_multiplicity_raises_at_run(self, n2_frame: "molrs.Frame") -> None:
        """A multiplicity of the wrong parity for the electron count is only
        detectable once N_e = ΣZ − charge is known, so it fails at ``run``:
        neutral N₂ has 14 electrons and cannot be a doublet."""
        method = UHF(multiplicity=2)
        with pytest.raises(ValueError, match="(?i)multiplicit"):
            method.run(n2_frame)

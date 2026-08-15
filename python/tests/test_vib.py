"""Harmonic frequencies + RRHO thermochemistry from Python (spec: py-vib).

``atomiverse.vibration.Hessian`` is the CPU central-finite-difference
Hessian driver (``cpu::vib::Hessian``, ``include/atomiverse/cpu/vibration/
hessian.h``): it displaces every Cartesian coordinate by ±h (h is fixed at
1.0e-3 Å in ``displace_step.h:17``), differences the forces produced by
whatever evaluator sits in the force slot, symmetrizes, mass-weights,
projects out translations/rotations PySCF-style and diagonalizes. The
result object carries the wavenumbers, the normal modes, the Cartesian
Hessian and the rigid-rotor/harmonic-oscillator thermochemistry.

Mode count is the *projected* count, never "3N with five near-zero
entries": ``hessian_diagonalizer.cpp:236-284`` builds the
translation/rotation projector, QR-orthogonalizes it, and diagonalizes only
the complement, so ``frequencies`` has exactly 3N−6 entries (3N−5 for a
linear molecule; empty for a lone atom, which ``hessian_diagonalizer.cpp:
190-197`` short-circuits before the projector). A diatomic therefore reports
exactly one frequency — the same invariant the C++ regression asserts
(``regressions/semiempirical/se-analytic-hessian/
reg_se-analytic-hessian.cpp:92-97``).

The physics gate
----------------
Two Ar atoms at the Lennard-Jones minimum have a closed-form stretch
frequency, so this file needs no external oracle:

    V(r)  = 4 eps [ (sigma/r)^12 - (sigma/r)^6 ]
    r_min = 2^(1/6) sigma
    k     = V''(r_min) = 72 eps / (2^(1/3) sigma^2)
    nu~   = sqrt(k / mu) / (2 pi c),      mu = m_Ar / 2

The two LJ parameters are the verified conftest values (eps = 0.0104 eV,
sigma = 3.4 Å) and m_Ar = 39.948 amu is the IUPAC-2013 weight the domain
itself looks up (``atomic_masses.h:68``) — no new reference value is
introduced. ``k`` is DERIVED below from the closed form and converted with
the CODATA-2014 constants the engine uses
(``include/atomiverse/cpu/vibration/nist_constants.h``), by two independent
unit paths (SI and atomic-unit/PySCF) that agree to ~1e-16 relative, and
the pinned literal 26.456347759377 cm⁻¹ is checked against both.

Note the frame contract: this domain reads the atom's **atomic number**
(``kAtom_type``, from the molrs ``atomic_number`` column) and looks the mass
up per Z — ``mass_weighting.cpp:26-28``. The per-atom ``mass`` column is
present in the fixtures because MD-style frames carry it, but it is IGNORED
here; changing it cannot move a frequency.

Tolerances
----------
=================================  =========================================
frequency vs. the closed form      1e-3 relative. The central-difference
                                   truncation at h = 1e-3 Å is
                                   (h^2/6) V''''/V'' = 4.25e-6 on k, i.e.
                                   2.12e-6 on nu~ — so 1e-3 is ~470x
                                   headroom, sized to also cover an
                                   ATV_REAL_STORAGE=float wheel (single-
                                   precision positions perturb the ±h
                                   displacement by ~5e-4 relative).
two analytic derivations agree     1e-12 relative (measured 1.3e-16)
max |H - H^T|                      1e-8 eV/Å^2, the gate
                                   reg_se-analytic-hessian.cpp:85 uses
ZPE vs 0.5 h c nu~                 1e-9 Hartree (the two constant sets
                                   differ by 7e-14 Hartree here)
semiempirical smoke frequency      > 100 cm^-1, the soft floor of
                                   reg_se-analytic-hessian.cpp:104
=================================  =========================================

Units: frequencies [cm^-1]; Hessian [eV/Å^2]; energies [Hartree];
entropies [Hartree/K]; positions [Å]; masses [amu].
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import molrs

import atomiverse as av
from atomiverse._binding import _cpp, extension_available
from atomiverse.neighbor import NeighborList
from atomiverse.potentials.pair import LJCut
from atomiverse.vibration import Hessian, HessianResult

# Short-circuited on ``extension_available()``: the missing-extension
# sentinel raises ImportError (not AttributeError), so a bare ``hasattr``
# would break collection on a pure-Python wheel instead of letting conftest
# skip the file (idiom from test_md.py / test_surface.py).
_HAS_AM1: bool = extension_available() and hasattr(_cpp, "AM1Evaluator")


# ── CODATA-2014 constants, verbatim from the engine's own header ──────
# include/atomiverse/cpu/vibration/nist_constants.h (which mirrors
# pyscf/data/nist.py). Only the RAW constants are transcribed; every
# derived quantity is composed here exactly as the header composes it, so
# no rounded magic literal enters the derivation.

_PI: float = 3.141592653589793  # nist_constants.h:23
_PLANCK_JS: float = 6.626070040e-34  # nist_constants.h:26  [J*s]
_E_MASS_KG: float = 9.10938356e-31  # nist_constants.h:32  [kg]
_AVOGADRO: float = 6.022140857e23  # nist_constants.h:35  [1/mol]
_BOHR_A: float = 0.52917721092  # nist_constants.h:41  [Å]
_LIGHT_SPEED_SI: float = 299792458.0  # nist_constants.h:53  [m/s]
_E_CHARGE_C: float = 1.6021766208e-19  # nist_constants.h:56  [C]

_HBAR_JS: float = _PLANCK_JS / (2.0 * _PI)  # nist_constants.h:29
_ATOMIC_MASS_KG: float = 1.0e-3 / _AVOGADRO  # nist_constants.h:38
_BOHR_SI_M: float = _BOHR_A * 1.0e-10  # nist_constants.h:44
#: nist_constants.h:50 — hbar^2 / (m_e a0^2), NOT the tabulated literal.
_HARTREE2J: float = _HBAR_JS * _HBAR_JS / (_E_MASS_KG * _BOHR_SI_M * _BOHR_SI_M)
_HARTREE2EV: float = _HARTREE2J / _E_CHARGE_C  # nist_constants.h:62

#: NIST CODATA hartree-inverse-metre relationship, 1 Eh = 219474.6313702
#: cm^-1 (https://physics.nist.gov/cgi-bin/cuu/Value?hrminv). Composing the
#: same factor from the constants above (h c / E_h) gives 219474.63111558,
#: a 1.2e-9 relative difference — 7e-14 Hartree on the ZPE checked below,
#: far inside its 1e-9 Hartree window.
_CM1_TO_HARTREE: float = 1.0 / 219474.6313702


# ── Ar dimer fixture parameters ───────────────────────────────────────
# The LJ parameters and the cutoff are the verified conftest values
# (conftest.py:84,101 — Ar, eps = 0.0104 eV, sigma = 3.4 Å, cutoff 8.5 Å).

_EPS_EV: float = 0.0104  # LJ well depth [eV]
_SIGMA_A: float = 3.4  # LJ sigma [Å]
_CUTOFF_A: float = 8.5  # pair cutoff [Å]
_SKIN_A: float = 1.0  # neighbor-list skin [Å]
_BOX_A: float = 40.0  # cubic cell edge [Å] (isolation: > 2 * cutoff)
_Z_AR: int = 18  # argon atomic number
#: IUPAC-2013 standard atomic weight of Ar — atomic_masses.h:68. This is
#: what the domain looks up from Z; the frame's `mass` column is ignored.
_MASS_AR_AMU: float = 39.948

#: LJ minimum r_min = 2^(1/6) sigma = 3.816370964251868 Å.
_R_MIN_A: float = 2.0 ** (1.0 / 6.0) * _SIGMA_A

#: Analytic well curvature k = V''(r_min) = 72 eps / (2^(1/3) sigma^2)
#: [eV/Å^2]. Derivation: V'(r) = 4 eps (-12 sigma^12 / r^13 + 6 sigma^6 /
#: r^7); V''(r) = 4 eps (156 sigma^12 / r^14 - 42 sigma^6 / r^8); at
#: r = 2^(1/6) sigma this collapses to 4 eps 2^(-4/3) (78 - 42) / sigma^2.
_K_EV_PER_A2: float = 72.0 * _EPS_EV / (2.0 ** (1.0 / 3.0) * _SIGMA_A**2)

#: Reduced mass of the homonuclear dimer, mu = m/2 [amu].
_MU_AMU: float = _MASS_AR_AMU / 2.0


def _stretch_wavenumber_si() -> float:
    """Analytic Ar2 stretch wavenumber [cm^-1] via the SI route.

    k [eV/Å^2] -> [J/m^2] by (e / 1e-20); mu [amu] -> [kg] by ATOMIC_MASS;
    omega = sqrt(k/mu) [rad/s]; nu~ = omega / (2 pi c) [1/m] -> [1/cm].

    @return The harmonic stretch wavenumber [cm^-1].
    """
    k_si = _K_EV_PER_A2 * _E_CHARGE_C / 1.0e-20
    mu_si = _MU_AMU * _ATOMIC_MASS_KG
    omega = math.sqrt(k_si / mu_si)
    return omega / (2.0 * _PI * _LIGHT_SPEED_SI) * 1.0e-2


def _stretch_wavenumber_atomic_units() -> float:
    """Analytic Ar2 stretch wavenumber [cm^-1] via the engine's own route.

    Mirrors ``mass_weighting.cpp:40-47`` + ``hessian_diagonalizer.cpp:296-
    306``: convert eV/Å^2 -> Hartree/Bohr^2, divide by the amu reduced mass
    to get the atomic-unit force constant, take the square root, and scale
    by ``nist::au2wavenumber()``.

    @return The harmonic stretch wavenumber [cm^-1].
    """
    # nist_constants.h:69-70
    ev_per_a2_to_hartree_per_bohr2 = (1.0 / _HARTREE2EV) * (_BOHR_A * _BOHR_A)
    k_au = _K_EV_PER_A2 * ev_per_a2_to_hartree_per_bohr2
    freq_au = math.sqrt(k_au / _MU_AMU)
    # nist_constants.h:82-94
    au2hz = math.sqrt(_HARTREE2J / (_ATOMIC_MASS_KG * _BOHR_SI_M * _BOHR_SI_M)) / (
        2.0 * _PI
    )
    return freq_au * au2hz / _LIGHT_SPEED_SI * 1.0e-2


#: The pinned analytic stretch wavenumber [cm^-1], 14 significant digits.
#: Both derivations above reproduce it; it is written out so a reader can
#: see the number the gate is about without running the file.
_NU_ANALYTIC_CM1: float = 26.456347759377

# ── Tolerances ────────────────────────────────────────────────────────

#: Frequency window vs. the closed form. Budget: 2.12e-6 relative from the
#: h = 1e-3 Å central-difference truncation (see the module docstring).
_TOL_FREQ_REL: float = 1.0e-3
#: The two analytic unit paths agree to 1.3e-16; graded 4 decades looser.
_TOL_DERIVATION_REL: float = 1.0e-12
#: Hessian asymmetry gate — reg_se-analytic-hessian.cpp:85.
_TOL_HESSIAN_ASYM_EV_A2: float = 1.0e-8
#: ZPE closed-form window [Hartree] (constant-set drift is 7e-14 here).
_TOL_ZPE_HARTREE: float = 1.0e-9
#: Soft physical floor for a semiempirical stretch — reg_se-analytic-
#: hessian.cpp:104 uses exactly this bound for the DFTB2 H2 case.
_MIN_SE_FREQ_CM1: float = 100.0


# ── Fixtures ──────────────────────────────────────────────────────────


def _ar_dimer_frame(atomic_number: int = _Z_AR) -> "molrs.Frame":
    """Two Ar atoms on the z axis at exactly the LJ minimum separation.

    @param atomic_number Z written to the ``atomic_number`` column (the
           engine's atom type, and the key of the mass lookup).
    @return A molrs.Frame with x/y/z [Å], mass [amu] and atomic_number.
    """
    frame = molrs.Frame()
    atoms = molrs.Block()
    atoms.insert("x", np.zeros(2, dtype=np.float64))
    atoms.insert("y", np.zeros(2, dtype=np.float64))
    atoms.insert("z", np.array([0.0, _R_MIN_A], dtype=np.float64))
    # Present because MD-style frames carry it; the vibration domain reads
    # masses from Z (mass_weighting.cpp:26-28) and never touches this.
    atoms.insert("mass", np.full(2, _MASS_AR_AMU, dtype=np.float64))
    atoms.insert("atomic_number", np.full(2, atomic_number, dtype=np.uint32))
    frame["atoms"] = atoms
    frame.box = molrs.Box.cube(_BOX_A)
    return frame


@pytest.fixture
def ar_dimer_frame() -> "molrs.Frame":
    """Ar2 at r_min — the analytic-frequency fixture."""
    return _ar_dimer_frame()


def _lj_hessian(**kwargs: str) -> "Hessian":
    """Hessian driver wired to the conftest LJ evaluator + neighbor list.

    @param kwargs Forwarded to ``Hessian`` (``frequency_tsv`` /
           ``mode_xyz_prefix`` in the file-output cases).
    @return A configured, not-yet-run driver.
    """
    driver = Hessian(**kwargs)
    driver.set_evaluator(LJCut(_EPS_EV, _SIGMA_A, _CUTOFF_A))
    driver.set_neighbor(NeighborList(_CUTOFF_A, _SKIN_A))
    return driver


@pytest.fixture
def ar_dimer_result(ar_dimer_frame: "molrs.Frame") -> "HessianResult":
    """The finite-difference result for Ar2 at the LJ minimum."""
    return _lj_hessian().run(ar_dimer_frame)


# ── The analytic golden is derived, not transcribed ───────────────────


class TestAnalyticReference:
    def test_derivation_paths_agree_with_the_pinned_value(self) -> None:
        """The SI and atomic-unit routes both reproduce 26.456347759377
        cm^-1, so the gate below rests on a derivation rather than on a
        transcribed number."""
        nu_si = _stretch_wavenumber_si()
        nu_au = _stretch_wavenumber_atomic_units()
        assert nu_si == pytest.approx(nu_au, rel=_TOL_DERIVATION_REL)
        assert nu_si == pytest.approx(_NU_ANALYTIC_CM1, rel=_TOL_DERIVATION_REL)
        assert nu_au == pytest.approx(_NU_ANALYTIC_CM1, rel=_TOL_DERIVATION_REL)


# ── ac-001: FD Hessian + frequencies from Python ──────────────────────


class TestArDimerFrequencies:
    def test_dimer_stretch_frequency_matches_analytic(
        self, ar_dimer_result: "HessianResult"
    ) -> None:
        """The one projected mode of Ar2 is the analytic well-curvature
        stretch, inside the finite-difference truncation budget."""
        assert ar_dimer_result.frequencies.shape == (1,)
        freq = float(ar_dimer_result.frequencies[0])
        assert freq == pytest.approx(_NU_ANALYTIC_CM1, rel=_TOL_FREQ_REL), (
            f"Ar2 stretch {freq!r} cm^-1 differs from the closed-form "
            f"{_NU_ANALYTIC_CM1!r} cm^-1 by more than {_TOL_FREQ_REL} "
            "relative (finite-difference truncation alone is 2.1e-6)"
        )

    def test_mode_count_is_the_projected_count(
        self, ar_dimer_result: "HessianResult"
    ) -> None:
        """3N-5 = 1 for a diatomic: translations and rotations are
        PROJECTED OUT, not returned as near-zero entries."""
        assert ar_dimer_result.n_atoms == 2
        assert ar_dimer_result.frequencies.ndim == 1
        assert ar_dimer_result.frequencies.size == 1

    def test_dimer_is_linear_no_imaginary(
        self, ar_dimer_result: "HessianResult"
    ) -> None:
        """A diatomic is a linear rotor, and the LJ minimum is a genuine
        minimum — no imaginary (negative-wavenumber) mode."""
        assert ar_dimer_result.is_linear is True
        assert ar_dimer_result.n_imaginary == 0

    def test_last_result_matches_run(self, ar_dimer_frame: "molrs.Frame") -> None:
        """``run()`` returns the result AND caches it on the driver."""
        driver = _lj_hessian()
        returned = driver.run(ar_dimer_frame)
        cached = driver.last_result()
        assert isinstance(returned, HessianResult)
        assert isinstance(cached, HessianResult)
        np.testing.assert_allclose(
            np.asarray(cached.frequencies),
            np.asarray(returned.frequencies),
            rtol=0.0,
            atol=0.0,
        )


# ── ac-002: RRHO thermochemistry readable ─────────────────────────────


class TestRrhoThermochemistry:
    def test_rrho_fields_finite(self, ar_dimer_result: "HessianResult") -> None:
        """Every published RRHO quantity is finite and physically signed:
        a positive zero-point energy and a positive total entropy at
        298.15 K / 1 atm."""
        res = ar_dimer_result
        assert res.zpe > 0.0
        assert math.isfinite(res.zpe)
        for value in (res.u_tot, res.h_tot, res.g_tot, res.g_thermal):
            assert math.isfinite(value)
        assert res.s_tot > 0.0
        assert res.s_vib >= 0.0
        for value in (res.s_trans, res.s_rot, res.s_elec):
            assert math.isfinite(value)

    def test_zpe_is_the_one_mode_closed_form(
        self, ar_dimer_result: "HessianResult"
    ) -> None:
        """With exactly one real mode the harmonic ZPE reduces to
        0.5 h c nu~; graded against the driver's OWN frequency so the
        finite-difference error cancels and only the constant set is
        under test."""
        freq_cm1 = float(ar_dimer_result.frequencies[0])
        expected_hartree = 0.5 * freq_cm1 * _CM1_TO_HARTREE
        assert ar_dimer_result.zpe == pytest.approx(
            expected_hartree, abs=_TOL_ZPE_HARTREE
        )


class TestHessianMatrix:
    def test_hessian_matrix_shape_symmetric(
        self, ar_dimer_result: "HessianResult"
    ) -> None:
        """The Cartesian Hessian is a finite, symmetric [3N,3N] array in
        eV/Å^2 — the same gate reg_se-analytic-hessian.cpp:71-87 applies."""
        hessian = np.asarray(ar_dimer_result.hessian)
        assert hessian.shape == (6, 6)
        assert np.all(np.isfinite(hessian))
        max_asym = float(np.max(np.abs(hessian - hessian.T)))
        assert max_asym <= _TOL_HESSIAN_ASYM_EV_A2, (
            f"max |H - H^T| = {max_asym!r} eV/Å^2 exceeds {_TOL_HESSIAN_ASYM_EV_A2}"
        )

    def test_normal_modes_shape(self, ar_dimer_result: "HessianResult") -> None:
        """Normal modes are lifted back to the full 3N Cartesian basis:
        [3N, n_modes], one column per published frequency."""
        modes = np.asarray(ar_dimer_result.normal_modes)
        assert modes.shape == (6, 1)
        assert np.all(np.isfinite(modes))


# ── File outputs: OFF unless a path is given ──────────────────────────


class TestFileOutputs:
    def test_no_files_written_by_default(
        self,
        ar_dimer_frame: "molrs.Frame",
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A default ``Hessian()`` writes NOTHING to the working directory.

        The C++ preset defaults to ``"frequencies.tsv"`` / prefix ``"mode"``
        (``src/cpu/vibration/hessian.cpp:75-79``), which litters the CWD of
        whoever imports the binding; the Python default is the empty path =
        no recorder output at all.
        """
        monkeypatch.chdir(tmp_path)
        _lj_hessian().run(ar_dimer_frame)

        assert not (tmp_path / "frequencies.tsv").exists()
        assert list(tmp_path.rglob("*.xyz")) == []
        stray = sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*"))
        assert stray == [], f"default Hessian() littered the CWD: {stray}"

    def test_frequency_tsv_written_when_path_given(
        self, ar_dimer_frame: "molrs.Frame", tmp_path: Path
    ) -> None:
        """A non-empty ``frequency_tsv`` writes the TSV, and it parses with
        the header + one row per mode + ``#`` trailer that
        ``src/cpu/vibration/frequency_report.cpp:29-49`` emits."""
        tsv = tmp_path / "f.tsv"
        result = _lj_hessian(frequency_tsv=str(tsv)).run(ar_dimer_frame)

        assert tsv.exists()
        lines = tsv.read_text(encoding="utf-8").splitlines()
        assert lines[0].split("\t") == ["index", "frequency_cm-1", "type"]

        rows = [ln.split("\t") for ln in lines[1:] if not ln.startswith("#")]
        assert len(rows) == int(result.frequencies.size)
        assert rows[0][0] == "0"
        assert float(rows[0][1]) == pytest.approx(
            float(result.frequencies[0]), rel=1.0e-12
        )
        assert rows[0][2] == "real"

        trailer = {
            ln[1:].split("\t")[0].strip(): ln[1:].split("\t")[1]
            for ln in lines
            if ln.startswith("#")
        }
        assert int(trailer["n_modes"]) == int(result.frequencies.size)
        assert int(trailer["n_imaginary"]) == result.n_imaginary
        assert float(trailer["zpe_hartree"]) == pytest.approx(result.zpe, rel=1.0e-12)

    def test_one_mode_xyz_per_real_mode(
        self, ar_dimer_frame: "molrs.Frame", tmp_path: Path
    ) -> None:
        """A non-empty ``mode_xyz_prefix`` writes one animation per REAL
        mode, named ``<prefix>_mode<k>.xyz``
        (``normal_mode_writer.h:75-77``; imaginary modes are skipped)."""
        prefix = tmp_path / "m"
        result = _lj_hessian(mode_xyz_prefix=str(prefix)).run(ar_dimer_frame)

        n_real = int(np.sum(np.asarray(result.frequencies) > 0.0))
        written = sorted(p.name for p in tmp_path.rglob("*.xyz"))
        assert written == ["m_mode0.xyz"]
        assert len(written) == n_real

        # Valid multi-frame XYZ: every frame starts with the atom count.
        first_line = (
            (tmp_path / "m_mode0.xyz").read_text(encoding="utf-8").splitlines()[0]
        )
        assert int(first_line.strip()) == 2


# ── Frame contract: fail loud on an element the mass table lacks ──────


class TestUnsupportedElement:
    def test_unsupported_z_raises(self) -> None:
        """The mass lookup covers Z = 1..18 (``atomic_masses.h:69-73``) and
        has no fallback. Z = 99 must surface as a ``ValueError`` that names
        the atomic number, not as the bare ``IndexError`` nanobind produces
        from an untranslated ``std::out_of_range``.
        """
        frame = _ar_dimer_frame(atomic_number=99)
        with pytest.raises(ValueError, match="(?i)atomic number"):
            _lj_hessian().run(frame)


# ── The force slot takes any bound evaluator ──────────────────────────


class TestSemiempiricalForceSource:
    @pytest.mark.skipif(
        not _HAS_AM1, reason="extension built without ATV_BUILD_SEMIEMPIRICAL"
    )
    def test_am1_force_source_smoke(self) -> None:
        """AM1 in the force slot gives H2 one real stretch — the same
        structural gate reg_se-analytic-hessian.cpp:92-108 applies to its
        DFTB2 twin (soft floor 100 cm^-1; the absolute wavenumber is not
        pinned here, the SE oracle loop owns that)."""
        frame = molrs.Frame()
        atoms = molrs.Block()
        atoms.insert("x", np.array([0.0, 0.74], dtype=np.float64))
        atoms.insert("y", np.zeros(2, dtype=np.float64))
        atoms.insert("z", np.zeros(2, dtype=np.float64))
        atoms.insert("mass", np.full(2, 1.008, dtype=np.float64))
        atoms.insert("atomic_number", np.ones(2, dtype=np.uint32))
        frame["atoms"] = atoms
        frame.box = molrs.Box.cube(20.0)

        driver = Hessian()
        driver.set_evaluator(av.AM1Evaluator())
        result = driver.run(frame)

        assert result.frequencies.shape == (1,)
        freq = float(result.frequencies[0])
        assert math.isfinite(freq)
        assert freq > _MIN_SE_FREQ_CM1

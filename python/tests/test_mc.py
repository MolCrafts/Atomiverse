"""Monte Carlo Python surface tests (spec: py-mc).

Binds the claimed atomic-fluid MC domain — canonical NVT (``MC``),
isothermal-isobaric (``NptMC``) and grand-canonical μVT (``GCMC``) — through
``atomiverse.montecarlo``, with the same categorical-setter handle shape as
MD (``set_evaluator`` / ``set_neighbor`` / ``set_mover`` / ``run``).

What is (and is not) checked here
--------------------------------
Behavioural invariants only. The quantitative ensemble physics lives in
``regressions/mc/`` and stays there:

* ``regressions/mc/npt_lj`` — ⟨V⟩ = (N+1)k_BT/P + N·B₂ and the exact NPT
  width σ_V = √(N+1)·k_BT/P.
* ``regressions/mc/gcmc_lj`` — ⟨N⟩(z) against the Mayer activity expansion
  ρ = z − 2B₂z² at two activities.

This file asserts only what a *binding* can break: the handle constructs,
the chain advances (acceptance strictly inside (0,1) — a frozen chain and a
"accept everything" chain both fail), the statistics are per-verb and
cumulative across ``run()`` calls, the results are written back into the
molrs Frame (positions, box, and — for μVT — a rebuilt ``atoms`` block at
the new N), and the build gates fail loud.

Hard-coded goldens (μVT)
------------------------
The one number with physics in it is the μVT chemical potential. It is a
**hard-coded golden**, derived once from published relations and pinned
below; ``TestGcmc.test_mu_matches_the_pinned_activity`` re-derives it from
the unit constants so the pin can never silently rot.

============================  ===============================================
Pinned quantity               Source
============================  ===============================================
z ≡ e^{βμ}/Λ³, so             Frenkel & Smit, *Understanding Molecular
μ = k_BT·ln(z·Λ³)             Simulation* (2nd ed. 2002) §5.6.1 eq. (5.6.9);
                              transcribed in ``regressions/mc/gcmc_lj/
                              REFERENCE.md`` and used at
                              ``reg_gcmc_lj.cpp:84``
Λ = h/√(2π m k_BT)            same REFERENCE.md; engine implementation
                              ``core/units.h:171`` (``lambda_thermal``)
k_B = 8.617333262e-5 eV/K,    ``core/units.h:138``, ``:161``, ``:149``
h = 4.135667696 eV·fs,
amu·Å²/fs² → eV = 103.6427
ρ(z) = z − 2·B₂(T)·z² + O(z³) Hill, *An Introduction to Statistical
                              Thermodynamics* (Dover 1986) §15-3 eq. (15-24);
                              ``regressions/mc/gcmc_lj/REFERENCE.md``
B₂(T*=1.5, LJ cut+shifted at  quadrature of McQuarrie, *Statistical
2.5σ) = −1.7801 σ³            Mechanics* (2000) eq. (12-30) — the same
                              integral ``reg_gcmc_lj.cpp:45-72`` evaluates
============================  ===============================================

The μVT band asserted here is deliberately **loose** — [0.2×, 5×] of the
ideal-gas ⟨N⟩ = z·V = 10.80 atoms. Three effects justify the width and none
is worth tightening in a pytest: the second-order Mayer term lifts the
prediction to 12.72 atoms (+18%), the first *neglected* term is worth a
further ≈0.6 atoms, and an 8k-trial chain on ~11 atoms is far too short for
a percent-level mean. Tightening it is ``regressions/mc/gcmc_lj``'s job,
which runs 450k trials in a 12σ box at two activities.

Units: temperature [K]; pressure [eV/Å³] (LAMMPS sign, positive =
compressive); μ [eV]; mass [amu]; lengths [Å]; volumes [Å³]; energies [eV].
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import molrs

from atomiverse.montecarlo import (
    GCMC,
    MC,
    InsertDeleteMove,
    MoveSelector,
    NptMC,
    TranslateMove,
    VolumeMove,
)
from atomiverse.neighbor import NeighborList
from atomiverse.potentials.pair import LJCut

# ── Unit constants, transcribed from core/units.h ───────────────────

_KB_EV_PER_K: float = 8.617333262e-5  # core/units.h:138 [eV/K]
_H_PLANCK_EV_FS: float = 4.135667696  # core/units.h:161 [eV·fs]
_MASS_V2_TO_ENERGY: float = 103.6427  # core/units.h:149 [amu·Å²/fs² → eV]

# ── Argon-flavoured LJ, the same parameters regressions/mc/ samples ──

_EPS_EV: float = 120.0 * _KB_EV_PER_K  # ε/k_B = 120 K → 0.01034 eV
_SIGMA_A: float = 3.4  # [Å]
_MASS_AMU: float = 39.948  # argon [amu]
_Z_ARGON: int = 18  # atomic number stamped on μVT insertions

#: Supercritical vapour, T* = T k_B/ε = 1.5 — the state point every
#: regressions/mc/ case uses; no phase boundary anywhere near it.
_T_MC_K: float = 1.5 * 120.0  # 180 K

# ── NVT case: the conftest Ar lattice at sigma spacing ──────────────

#: The conftest lattice spacing IS σ, so the cutoff is taken at σ too: the
#: sampled potential is then the purely repulsive core (u(σ) = 0, and
#: PairLJCut is cut-AND-shifted so the shift is zero here). A perfect
#: lattice therefore starts at E = 0 with no overlap, and every trial that
#: pushes a pair inside σ costs energy — which is what makes the
#: acceptance ratio land strictly inside (0, 1).
_NVT_CUTOFF_A: float = 3.4
_NVT_SKIN_A: float = 1.0
#: 27 atoms → side 3 → L = 10.2 Å, so cutoff + skin = 4.4 < L/2 = 5.1.
_NVT_N_ATOMS: int = 27
_NVT_DELTA_A: float = 0.3  # displacement half-width [Å]
_NVT_TRIALS: int = 300
_NVT_SEED: int = 42
_NVT_T_K: float = 300.0

#: frame.meta['energy'] and the handle's own bookkeeping are both AccReal
#: (double) on either side of the molrs boundary — no narrowing, so the
#: exact-energy tier applies (same reasoning as test_semiempirical.py:127).
_TOL_ENERGY_EV: float = 1.0e-10

# ── NPT case ────────────────────────────────────────────────────────

_NPT_N_ATOMS: int = 125  # 5³ — a clean simple-cubic lattice
_NPT_RHO_RED: float = 0.6  # ρσ³ — a compressed fluid, so displacements
#                            are rejected often enough that the cumulative
#                            acceptance can never reach 1.0
_NPT_CUTOFF_A: float = 2.0 * _SIGMA_A  # 6.8 Å
_NPT_SKIN_A: float = 1.0
#: P* = Pσ³/ε. Well below the fluid's own pressure at ρ* = 0.6, T* = 1.5,
#: so the box EXPANDS monotonically: the minimum-image margin
#: (cutoff + skin = 7.8 Å vs L/2 = 10.08 Å at the start) only grows.
_NPT_P_RED: float = 0.05
_NPT_DELTA_A: float = 1.0  # displacement half-width [Å]
_NPT_DV_MAX_A3: float = 400.0  # volume half-width [Å³], ≈ 5% of V₀
_NPT_TRIALS: int = 500
_NPT_SEED: int = 1337

# ── μVT (GCMC) case ─────────────────────────────────────────────────

_GCMC_CUTOFF_A: float = 2.5 * _SIGMA_A  # 8.5 Å — the regression's cutoff
_GCMC_SKIN_A: float = 1.0
#: L = 6σ = 20.4 Å (the regression's 12σ box shrunk to keep the chain
#: cheap). cutoff + skin = 9.5 < L/2 = 10.2, so minimum image holds.
_GCMC_BOX_A: float = 6.0 * _SIGMA_A
_GCMC_VOLUME_A3: float = _GCMC_BOX_A**3  # 8489.664 Å³ = 216 σ³
#: Reduced activity zσ³ on the dilute vapour branch. The regression samples
#: 0.010 and 0.030 in a 12σ box (⟨N⟩ ≈ 18 and 55); this file's box is 6σ, so
#: the activity is raised to keep ⟨N⟩ ≈ 11 rather than ≈ 4 — at ⟨N⟩ = 4 the
#: instantaneous P(N = 0) is ~1.4% and every "the cell is not empty" assert
#: would be a coin flip. At ⟨N⟩ = 10.8 it is 2e-5.
_GCMC_Z_RED: float = 0.05
_GCMC_Z_PER_A3: float = _GCMC_Z_RED / _SIGMA_A**3  # [Å⁻³]

#: HARD-CODED GOLDEN — μ = k_BT·ln(z·Λ³) at T = 180 K, m = 39.948 amu,
#: zσ³ = 0.05, σ = 3.4 Å. Derivation (all inputs pinned in the module
#: docstring's table):
#:     k_BT   = 8.617333262e-5 × 180        = 0.0155111998716    eV
#:     Λ      = h/√(2π m k_BT · 103.6427)   = 0.20588093156041   Å
#:     Λ³                                   = 8.7266663949720e-3 Å³
#:     z      = 0.05 / 3.4³                 = 1.2721351516385e-3 Å⁻³
#:     μ      = k_BT · ln(z Λ³)             = −0.17695844425991  eV
_GCMC_MU_EV: float = -0.17695844425991422
_GCMC_LAMBDA_A: float = 0.20588093156041126

#: Ideal-gas ⟨N⟩ = z·V = 10.80 atoms. The second-order Mayer term lifts this
#: to 12.72 (ρ = z − 2B₂z², B₂ = −69.966 Å³ = −1.7801 σ³) and the first
#: NEGLECTED term is worth ≈ 0.64 atoms — both far inside the asserted band.
_GCMC_N_IDEAL: float = _GCMC_Z_PER_A3 * _GCMC_VOLUME_A3
_GCMC_N_BAND_LO: float = 0.2 * _GCMC_N_IDEAL  # 2.16
_GCMC_N_BAND_HI: float = 5.0 * _GCMC_N_IDEAL  # 54.0

#: 8 atoms on a 2³ sub-lattice: every pair is 10.2 Å apart, beyond the 8.5 Å
#: cutoff, so E₀ = 0 exactly — no overlap can freeze the chain with a NaN
#: ΔE. 8 is BELOW ⟨N⟩ ≈ 11, so the chain must climb to the answer.
_GCMC_N_START: int = 8
#: Trial budget. Sized for wall time, not for precision: ~8k trials on ~11
#: atoms is a fraction of a second, and the assertions above are decided by
#: the band width, not by the standard error. If this file ever creeps past
#: a few seconds, these three numbers are the ones to cut.
_GCMC_EQUILIBRATE: int = 3000
_GCMC_BLOCKS: int = 50
_GCMC_TRIALS_PER_BLOCK: int = 100
_GCMC_SEED: int = 97
#: The regression's own frozen-chain gate (reg_gcmc_lj.cpp:147).
_MIN_ACCEPTANCE: float = 0.01


# ── Helpers ─────────────────────────────────────────────────────────


def _cubic_lattice_frame(n_atoms: int, box_len: float) -> "molrs.Frame":
    """Simple-cubic Ar ``molrs.Frame``: *n_atoms* rows in a cubic cell.

    A lattice start (never a random one) is deliberate and is the same
    choice ``reg_gcmc_lj.cpp:92`` documents: random placement can overlap
    two atoms, and an infinite initial energy freezes the Markov chain with
    a NaN ΔE forever.

    @param n_atoms Number of Ar atoms to place.
    @param box_len Cubic cell edge [Å].
    @return A molrs.Frame carrying x/y/z [Å], mass [amu] and atomic_number.
    """
    # Smallest side with side³ >= n_atoms, by integer search: floating cube
    # roots land on either side of an exact cube (math.cbrt(27) is
    # 3.0000000000000004, whose ceil is 4) and would silently thin the
    # lattice out.
    side = 1
    while side**3 < n_atoms:
        side += 1
    spacing = box_len / side
    coords = np.empty((n_atoms, 3), dtype=np.float64)
    for i in range(n_atoms):
        coords[i, 0] = (i % side + 0.5) * spacing
        coords[i, 1] = ((i // side) % side + 0.5) * spacing
        coords[i, 2] = (i // (side * side) + 0.5) * spacing
    frame = molrs.Frame()
    atoms = molrs.Block()
    atoms.insert("x", np.ascontiguousarray(coords[:, 0]))
    atoms.insert("y", np.ascontiguousarray(coords[:, 1]))
    atoms.insert("z", np.ascontiguousarray(coords[:, 2]))
    atoms.insert("mass", np.full(n_atoms, _MASS_AMU, dtype=np.float64))
    atoms.insert("atomic_number", np.full(n_atoms, _Z_ARGON, dtype=np.uint32))
    frame["atoms"] = atoms
    frame.box = molrs.Box.cube(box_len)
    return frame


def _lj(cutoff: float) -> "LJCut":
    """The argon LJ pair style at *cutoff* [Å]."""
    return LJCut(_EPS_EV, _SIGMA_A, cutoff)


# ── 1. Construction ─────────────────────────────────────────────────


class TestMcConstruction:
    """The three presets and the four movers construct, and the
    ensemble/mover parameters reject nonsense before any sampling."""

    def test_nvt_constructs(self) -> None:
        assert MC(_NVT_T_K, _NVT_SEED) is not None

    def test_npt_constructs(self) -> None:
        pressure = _NPT_P_RED * _EPS_EV / _SIGMA_A**3
        assert NptMC(_T_MC_K, pressure, _NPT_SEED) is not None

    def test_gcmc_constructs(self) -> None:
        assert GCMC(_T_MC_K, _GCMC_MU_EV, _MASS_AMU, _GCMC_SEED) is not None

    def test_translate_move_constructs(self) -> None:
        assert TranslateMove(_NVT_DELTA_A) is not None

    def test_volume_move_constructs(self) -> None:
        assert VolumeMove(_NPT_DV_MAX_A3) is not None

    def test_insert_delete_move_constructs(self) -> None:
        assert InsertDeleteMove(_Z_ARGON, _MASS_AMU) is not None

    def test_move_selector_chains_and_returns_self(self) -> None:
        """``add`` returns the selector so mixed ensembles read as one
        expression (the C++ ref-qualified chaining contract)."""
        selector = MoveSelector()
        assert selector.add(TranslateMove(_NVT_DELTA_A), 0.5) is selector

    def test_zero_temperature_raises(self) -> None:
        """β = 1/k_BT is undefined at T = 0 — every preset refuses it."""
        with pytest.raises(ValueError):
            MC(0.0, _NVT_SEED)

    def test_negative_temperature_raises(self) -> None:
        with pytest.raises(ValueError):
            MC(-10.0, _NVT_SEED)

    def test_npt_zero_temperature_raises(self) -> None:
        with pytest.raises(ValueError):
            NptMC(0.0, 1.0e-5, _NPT_SEED)

    def test_gcmc_zero_temperature_raises(self) -> None:
        with pytest.raises(ValueError):
            GCMC(0.0, _GCMC_MU_EV, _MASS_AMU, _GCMC_SEED)

    def test_gcmc_zero_mass_raises(self) -> None:
        """Λ = h/√(2πmk_BT) diverges at m = 0."""
        with pytest.raises(ValueError):
            GCMC(_T_MC_K, _GCMC_MU_EV, 0.0, _GCMC_SEED)

    def test_zero_displacement_raises(self) -> None:
        """A zero-width proposal never moves anything."""
        with pytest.raises(ValueError):
            TranslateMove(0.0)

    def test_negative_displacement_raises(self) -> None:
        with pytest.raises(ValueError):
            TranslateMove(-0.3)

    def test_zero_volume_half_width_raises(self) -> None:
        with pytest.raises(ValueError):
            VolumeMove(0.0)

    def test_insert_delete_zero_mass_raises(self) -> None:
        with pytest.raises(ValueError):
            InsertDeleteMove(_Z_ARGON, 0.0)


# ── 2. Canonical (NVT) sampling ─────────────────────────────────────


def _nvt_handle() -> "MC":
    """NVT Metropolis on the conftest Ar lattice, ready to ``run``."""
    sim = MC(_NVT_T_K, _NVT_SEED)
    sim.set_evaluator(_lj(_NVT_CUTOFF_A))
    sim.set_neighbor(NeighborList(_NVT_CUTOFF_A, _NVT_SKIN_A))
    sim.set_mover(TranslateMove(_NVT_DELTA_A))
    return sim


class TestNvtMc:
    """Canonical Metropolis on the conftest Ar lattice, enlarged to 27
    atoms so that cutoff + skin stays below L/2."""

    @pytest.fixture
    def n_atoms(self) -> int:
        """Override the conftest default (8) — 27 gives L = 10.2 Å."""
        return _NVT_N_ATOMS

    def test_acceptance_ratio_strictly_inside_unit_interval(
        self, argon_frame: "molrs.Frame"
    ) -> None:
        """The chain neither freezes nor accepts everything.

        Both failure modes are silent otherwise: a NaN ΔE rejects every
        trial (ratio 0) and a dropped energy term accepts every trial
        (ratio 1). The attempts/accepts guards run FIRST so that a
        divide-by-zero ratio of 0.0 can never be read as "in range".
        """
        sim = _nvt_handle()
        sim.run(argon_frame, _NVT_TRIALS)
        assert sim.attempts() == _NVT_TRIALS
        assert sim.accepts() > 0, "frozen chain: no trial move was accepted"
        assert sim.accepts() < sim.attempts(), (
            "every trial accepted — the Boltzmann factor is not being applied"
        )
        ratio = sim.acceptance_ratio()
        assert 0.0 < ratio < 1.0

    def test_acceptance_ratio_matches_the_tallies(
        self, argon_frame: "molrs.Frame"
    ) -> None:
        """``acceptance_ratio()`` is exactly accepts/attempts."""
        sim = _nvt_handle()
        sim.run(argon_frame, _NVT_TRIALS)
        assert sim.acceptance_ratio() == pytest.approx(
            sim.accepts() / sim.attempts(), abs=1.0e-12
        )

    def test_displace_verb_carries_every_trial(
        self, argon_frame: "molrs.Frame"
    ) -> None:
        """A translate-only ensemble tallies every trial under
        ``"displace"``, and the verb-less call is the aggregate."""
        sim = _nvt_handle()
        sim.run(argon_frame, _NVT_TRIALS)
        assert sim.attempts("displace") == _NVT_TRIALS
        assert sim.attempts("displace") == sim.attempts()
        assert sim.accepts("displace") == sim.accepts()

    def test_current_energy_is_finite(self, argon_frame: "molrs.Frame") -> None:
        sim = _nvt_handle()
        sim.run(argon_frame, _NVT_TRIALS)
        energy = sim.current_energy()
        assert isinstance(energy, float)
        assert np.isfinite(energy)

    def test_frame_meta_energy_is_the_chain_energy(
        self, argon_frame: "molrs.Frame"
    ) -> None:
        """The molrs Frame carries the LAST ACCEPTED energy, bit-identical
        to the handle's own bookkeeping — a rejected trial's energy must
        never survive the write-back."""
        sim = _nvt_handle()
        sim.run(argon_frame, _NVT_TRIALS)
        meta_energy = float(argon_frame.meta["energy"])
        assert np.isfinite(meta_energy)
        assert meta_energy == pytest.approx(sim.current_energy(), abs=_TOL_ENERGY_EV)

    def test_positions_written_back(self, argon_frame: "molrs.Frame") -> None:
        """Accepted displacements land in the SAME molrs.Frame."""
        x_before = np.asarray(argon_frame["atoms"]["x"]).copy()
        sim = _nvt_handle()
        sim.run(argon_frame, _NVT_TRIALS)
        assert sim.accepts() > 0, "frozen chain: nothing could have moved"
        x_after = np.asarray(argon_frame["atoms"]["x"])
        assert not np.allclose(x_before, x_after), (
            "accepted trial moves must change the positions in the frame"
        )

    def test_atom_count_is_conserved(self, argon_frame: "molrs.Frame") -> None:
        """NVT is fixed-N: nothing is inserted or deleted."""
        sim = _nvt_handle()
        sim.run(argon_frame, _NVT_TRIALS)
        assert sim.n_atoms() == _NVT_N_ATOMS
        assert sim.attempts("insert") == 0
        assert sim.attempts("delete") == 0


# ── 3. Chain continuation ───────────────────────────────────────────


class TestChainContinuation:
    """Consecutive ``run()`` calls on ONE handle extend ONE Markov chain
    (spec py-mc DECIDED #1) — the standard
    ``run(equilibrate); run(sample)`` pattern. A per-call reset would show
    up as halved tallies."""

    @pytest.fixture
    def n_atoms(self) -> int:
        return _NVT_N_ATOMS

    def test_attempts_accumulate_across_runs(self, argon_frame: "molrs.Frame") -> None:
        sim = _nvt_handle()
        sim.run(argon_frame, _NVT_TRIALS // 2)
        first = sim.attempts()
        sim.run(argon_frame, _NVT_TRIALS // 2)
        assert first == _NVT_TRIALS // 2
        assert sim.attempts() == _NVT_TRIALS, (
            "run() must continue the chain, not restart it"
        )

    def test_accepts_accumulate_across_runs(self, argon_frame: "molrs.Frame") -> None:
        sim = _nvt_handle()
        sim.run(argon_frame, _NVT_TRIALS // 2)
        first_accepts = sim.accepts()
        sim.run(argon_frame, _NVT_TRIALS // 2)
        assert sim.accepts() >= first_accepts
        assert sim.accepts() == sim.accepts("displace")


# ── 4. Isothermal-isobaric (NPT) sampling ───────────────────────────


def _npt_box_length() -> float:
    """Cubic edge [Å] holding ``_NPT_N_ATOMS`` atoms at ρσ³ = ρ*."""
    volume = _NPT_N_ATOMS / _NPT_RHO_RED * _SIGMA_A**3
    return volume ** (1.0 / 3.0)


def _npt_handle() -> "NptMC":
    """NPT-MC with a 50/50 translate + volume move mix."""
    pressure = _NPT_P_RED * _EPS_EV / _SIGMA_A**3  # [eV/Å³]
    sim = NptMC(_T_MC_K, pressure, _NPT_SEED)
    sim.set_evaluator(_lj(_NPT_CUTOFF_A))
    sim.set_neighbor(NeighborList(_NPT_CUTOFF_A, _NPT_SKIN_A))
    sim.set_mover(
        MoveSelector()
        .add(TranslateMove(_NPT_DELTA_A), 0.5)
        .add(VolumeMove(_NPT_DV_MAX_A3), 0.5)
    )
    return sim


@pytest.fixture
def npt_frame() -> "molrs.Frame":
    """125 Ar atoms on a 5³ lattice at ρσ³ = 0.6."""
    return _cubic_lattice_frame(_NPT_N_ATOMS, _npt_box_length())


class TestNptMc:
    """The volume degree of freedom is sampled and written back."""

    def test_both_verbs_are_attempted(self, npt_frame: "molrs.Frame") -> None:
        """The MoveSelector fires exactly one child per trial, and both
        children are drawn — registering two movers side by side would
        instead fire BOTH every trial."""
        sim = _npt_handle()
        sim.run(npt_frame, _NPT_TRIALS)
        assert sim.attempts("displace") > 0
        assert sim.attempts("box") > 0
        assert sim.attempts("displace") + sim.attempts("box") == _NPT_TRIALS
        assert sim.attempts() == _NPT_TRIALS

    def test_displacement_acceptance_strictly_inside_unit_interval(
        self, npt_frame: "molrs.Frame"
    ) -> None:
        sim = _npt_handle()
        sim.run(npt_frame, _NPT_TRIALS)
        assert sim.accepts("displace") > 0, "frozen displacement chain"
        assert sim.accepts("displace") < sim.attempts("displace"), (
            "every displacement accepted at rho* = 0.6 — no Boltzmann factor"
        )
        assert 0.0 < sim.acceptance_ratio("displace") < 1.0

    def test_volume_moves_are_accepted(self, npt_frame: "molrs.Frame") -> None:
        """A rejected-forever volume move is the classic symptom of a
        broken box rollback or a missing N·k_BT·ln(V'/V) Jacobian."""
        sim = _npt_handle()
        sim.run(npt_frame, _NPT_TRIALS)
        assert sim.acceptance_ratio("box") > 0.0

    def test_current_volume_is_finite_and_positive(
        self, npt_frame: "molrs.Frame"
    ) -> None:
        sim = _npt_handle()
        sim.run(npt_frame, _NPT_TRIALS)
        volume = sim.current_volume()
        assert isinstance(volume, float)
        assert np.isfinite(volume)
        assert volume > 0.0

    def test_box_written_back(self, npt_frame: "molrs.Frame") -> None:
        """The accepted box lands in the molrs Frame, not only in the
        handle's bookkeeping."""
        volume_before = npt_frame.box.volume()
        sim = _npt_handle()
        sim.run(npt_frame, _NPT_TRIALS)
        volume_after = npt_frame.box.volume()
        assert sim.accepts("box") > 0, "no accepted volume move to write back"
        assert volume_after != pytest.approx(volume_before, rel=1.0e-12), (
            "an accepted volume move must change the molrs Frame box"
        )
        assert volume_after == pytest.approx(sim.current_volume(), rel=1.0e-10)

    def test_no_grand_canonical_verbs(self, npt_frame: "molrs.Frame") -> None:
        """Insert/delete belong to μVT — NPT never proposes them."""
        sim = _npt_handle()
        sim.run(npt_frame, _NPT_TRIALS)
        assert sim.attempts("insert") == 0
        assert sim.attempts("delete") == 0
        assert sim.n_atoms() == _NPT_N_ATOMS


# ── 5. Grand-canonical (μVT) sampling ───────────────────────────────


def _gcmc_handle() -> "GCMC":
    """μVT sampler at the pinned μ, 50/50 translate + insert/delete."""
    sim = GCMC(_T_MC_K, _GCMC_MU_EV, _MASS_AMU, _GCMC_SEED)
    sim.set_evaluator(_lj(_GCMC_CUTOFF_A))
    sim.set_neighbor(NeighborList(_GCMC_CUTOFF_A, _GCMC_SKIN_A))
    sim.set_mover(
        MoveSelector()
        .add(TranslateMove(0.4 * _SIGMA_A), 0.5)
        .add(InsertDeleteMove(_Z_ARGON, _MASS_AMU), 0.5)
    )
    return sim


@pytest.fixture
def gcmc_frame() -> "molrs.Frame":
    """8 well-separated Ar atoms in the 6σ cell — BELOW ⟨N⟩ ≈ 10.8, so the
    chain must climb to the answer and a frozen chain cannot fake a pass."""
    return _cubic_lattice_frame(_GCMC_N_START, _GCMC_BOX_A)


class TestGcmc:
    """μVT: variable N, the loose ideal-gas bound, and the atoms-block
    rebuild on write-back."""

    def test_mu_matches_the_pinned_activity(self) -> None:
        """Re-derive the hard-coded μ golden from the pinned relations
        (Frenkel & Smit eq. 5.6.9 + Λ = h/√(2πmk_BT)) so the constant in
        this file can never silently drift from the physics it claims."""
        kt = _KB_EV_PER_K * _T_MC_K
        lam = _H_PLANCK_EV_FS / math.sqrt(
            2.0 * math.pi * _MASS_AMU * kt * _MASS_V2_TO_ENERGY
        )
        assert lam == pytest.approx(_GCMC_LAMBDA_A, rel=1.0e-12)
        mu = kt * math.log(_GCMC_Z_PER_A3 * lam**3)
        assert mu == pytest.approx(_GCMC_MU_EV, rel=1.0e-12)

    def test_insert_and_delete_are_both_accepted(
        self, gcmc_frame: "molrs.Frame"
    ) -> None:
        """The regression's own frozen-chain gate (reg_gcmc_lj.cpp:147):
        both μVT verbs must actually get through, or ⟨N⟩ below is just the
        starting configuration."""
        sim = _gcmc_handle()
        sim.run(gcmc_frame, _GCMC_EQUILIBRATE)
        assert sim.attempts("insert") > 0
        assert sim.attempts("delete") > 0
        assert sim.acceptance_ratio("insert") > _MIN_ACCEPTANCE
        assert sim.acceptance_ratio("delete") > _MIN_ACCEPTANCE

    def test_mean_population_inside_the_ideal_gas_band(
        self, gcmc_frame: "molrs.Frame"
    ) -> None:
        """⟨N⟩ sits within [0.2×, 5×] of the ideal-gas z·V = 10.80.

        LOOSE on purpose — see the module docstring. The quantitative test
        (⟨N⟩ against ρ = z − 2B₂z² at two activities, 5σ statistical + the
        first neglected Mayer term) is regressions/mc/gcmc_lj.
        """
        sim = _gcmc_handle()
        sim.run(gcmc_frame, _GCMC_EQUILIBRATE)
        populations: list[int] = []
        for _ in range(_GCMC_BLOCKS):
            sim.run(gcmc_frame, _GCMC_TRIALS_PER_BLOCK)
            populations.append(sim.n_atoms())
        n_avg = float(np.mean(populations))
        assert max(populations) > _GCMC_N_START, (
            "the chain never climbed above its starting population — it is "
            "coasting on the initial configuration"
        )
        assert _GCMC_N_BAND_LO < n_avg < _GCMC_N_BAND_HI, (
            f"<N> = {n_avg:.2f} is outside the ideal-gas band "
            f"[{_GCMC_N_BAND_LO:.2f}, {_GCMC_N_BAND_HI:.2f}] "
            f"around z*V = {_GCMC_N_IDEAL:.2f}"
        )

    def test_population_actually_varies(self, gcmc_frame: "molrs.Frame") -> None:
        """N is a fluctuating observable in μVT — a constant N over the
        whole chain means insert/delete never committed."""
        sim = _gcmc_handle()
        populations: list[int] = []
        for _ in range(_GCMC_BLOCKS):
            sim.run(gcmc_frame, _GCMC_TRIALS_PER_BLOCK)
            populations.append(sim.n_atoms())
        assert len(set(populations)) > 1, (
            "N never changed across the chain — μVT is not sampling N"
        )

    def test_atoms_block_rebuilt_at_the_new_population(
        self, gcmc_frame: "molrs.Frame"
    ) -> None:
        """Write-back contract (spec py-mc DECIDED #2): ``to_molrs``
        rejects a changed row count, so the binding rebuilds the molrs
        ``atoms`` block. Every column must come back at the new N.

        Checked after EVERY block rather than once at the end: whether N
        differs from the start at one arbitrary instant is a coin flip,
        whereas "at least one block ran at a different N" over the 50
        blocks is not — and the invariant itself must hold at every one of
        them.
        """
        sim = _gcmc_handle()
        seen: set[int] = set()
        for _ in range(_GCMC_BLOCKS):
            sim.run(gcmc_frame, _GCMC_TRIALS_PER_BLOCK)
            n_now = sim.n_atoms()
            seen.add(n_now)
            atoms = gcmc_frame["atoms"]
            for column in ("x", "y", "z", "mass"):
                assert len(np.asarray(atoms[column])) == n_now, (
                    f"column '{column}' was not rebuilt at the new "
                    f"population N = {n_now}"
                )
        assert seen != {_GCMC_N_START}, (
            "fixture drift: this case needs a chain that changed N"
        )

    def test_inserted_atoms_carry_the_configured_species(
        self, gcmc_frame: "molrs.Frame"
    ) -> None:
        """``InsertDeleteMove(z, mass)`` stamps the species onto every
        inserted row — the mass MUST be the one GCMC built Λ from."""
        sim = _gcmc_handle()
        sim.run(gcmc_frame, _GCMC_EQUILIBRATE)
        masses = np.asarray(gcmc_frame["atoms"]["mass"])
        assert masses.size == sim.n_atoms()
        assert masses.size > 0, "the chain emptied the cell — nothing to check"
        assert masses == pytest.approx(np.full(sim.n_atoms(), _MASS_AMU), abs=1.0e-6)

    def test_current_energy_is_finite(self, gcmc_frame: "molrs.Frame") -> None:
        sim = _gcmc_handle()
        sim.run(gcmc_frame, _GCMC_EQUILIBRATE)
        assert np.isfinite(sim.current_energy())
        assert float(gcmc_frame.meta["energy"]) == pytest.approx(
            sim.current_energy(), abs=_TOL_ENERGY_EV
        )


# ── 6. Build gates ──────────────────────────────────────────────────


class TestMcErrors:
    """A half-configured sampler fails loud, before any trial."""

    @pytest.fixture
    def n_atoms(self) -> int:
        return _NVT_N_ATOMS

    def test_no_mover_raises(self, argon_frame: "molrs.Frame") -> None:
        """There is no default trial move — without a Mover the loop has
        nothing to propose."""
        sim = MC(_NVT_T_K, _NVT_SEED)
        sim.set_evaluator(_lj(_NVT_CUTOFF_A))
        sim.set_neighbor(NeighborList(_NVT_CUTOFF_A, _NVT_SKIN_A))
        with pytest.raises(RuntimeError, match="(?i)mover"):
            sim.run(argon_frame, 1)

    def test_no_evaluator_raises(self, argon_frame: "molrs.Frame") -> None:
        """Metropolis scores an energy; with no energy source there is
        nothing to score."""
        sim = MC(_NVT_T_K, _NVT_SEED)
        sim.set_mover(TranslateMove(_NVT_DELTA_A))
        with pytest.raises(RuntimeError, match="(?i)potential"):
            sim.run(argon_frame, 1)

    def test_pair_evaluator_without_neighbor_raises(
        self, argon_frame: "molrs.Frame"
    ) -> None:
        """A pair style is a NeighborConsumer; the error names the setter
        that fixes it."""
        sim = MC(_NVT_T_K, _NVT_SEED)
        sim.set_evaluator(_lj(_NVT_CUTOFF_A))
        sim.set_mover(TranslateMove(_NVT_DELTA_A))
        with pytest.raises(RuntimeError, match="set_neighbor"):
            sim.run(argon_frame, 1)

    def test_npt_no_mover_raises(self, npt_frame: "molrs.Frame") -> None:
        sim = NptMC(_T_MC_K, _NPT_P_RED * _EPS_EV / _SIGMA_A**3, _NPT_SEED)
        sim.set_evaluator(_lj(_NPT_CUTOFF_A))
        sim.set_neighbor(NeighborList(_NPT_CUTOFF_A, _NPT_SKIN_A))
        with pytest.raises(RuntimeError, match="(?i)mover"):
            sim.run(npt_frame, 1)

    def test_gcmc_no_mover_raises(self, gcmc_frame: "molrs.Frame") -> None:
        sim = GCMC(_T_MC_K, _GCMC_MU_EV, _MASS_AMU, _GCMC_SEED)
        sim.set_evaluator(_lj(_GCMC_CUTOFF_A))
        sim.set_neighbor(NeighborList(_GCMC_CUTOFF_A, _GCMC_SKIN_A))
        with pytest.raises(RuntimeError, match="(?i)mover"):
            sim.run(gcmc_frame, 1)

    def test_unknown_verb_raises(self, argon_frame: "molrs.Frame") -> None:
        """The statistics accessors take a verb name, not free text."""
        sim = _nvt_handle()
        sim.run(argon_frame, 1)
        with pytest.raises(ValueError):
            sim.attempts("wiggle")

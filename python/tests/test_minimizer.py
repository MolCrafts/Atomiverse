"""Structural tests for the typed CPU Minimize Python surface.

The first five cases pin the driver skeleton on LBFGS — construction,
phase validation, device refusal, and the shape of a run's result.
Everything below the ``py-minimize-full`` banner covers the rest of the
claimed CPU minimize domain:

* ``CG`` and ``FIRE`` as alternative optimizers, judged by the SAME
  ``ForceConverged`` criterion the LBFGS path uses (the C++ contract: an
  optimizer publishes metrics, a criterion judges them).
* the four variable-cell relaxers plus ``PressureConverged`` /
  ``StressConverged``.
* ac-003: ``Minimize.set_evaluator`` accepts every component type
  ``MD.set_evaluator`` accepts, because both facades read ONE dispatch
  table.
* the late-setter contract already pinned for MD
  (``test_md.py::TestMdLateSetters``): reconfiguring a driver after its
  first ``run()`` fails loud, while re-running the same configuration
  stays legal.

New names are imported INSIDE the test bodies (the discipline
``test_md.py`` documents for ``TestCoulombPme`` / ``TestXyzRecorder``):
until the bindings land the import IS the failure and the five
pre-existing cases still collect and pass.

Physical parameters are the Ar LJ set used throughout this suite
(eps = 0.0104 eV, sigma = 3.4 A, r_c = 2.5 sigma) — no new reference
values are introduced. The invariants come from the matching C++
regressions, cited per case.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
import molpy

import atomiverse as av
from atomiverse._binding import _cpp, extension_available
from atomiverse.longrange import CoulLong
from atomiverse.minimizer import ForceConverged, LBFGS, Minimize, MinimizeResult
from atomiverse.neighbor import NeighborList
from atomiverse.potentials.angle import Harmonic as AngleHarmonic
from atomiverse.potentials.bond import Harmonic as BondHarmonic
from atomiverse.potentials.dihedral import MultiHarmonic as DihedralMultiHarmonic
from atomiverse.potentials.pair import CoulCut, LJCut


def make_minimize() -> Minimize:
    opt = Minimize()
    opt.add_criterion(
        ForceConverged(
            ftol=0.05,
            rms_ftol=0.03,
            max_dtol=0.002,
            rms_dtol=0.001,
            etol=1e-5,
        )
    )
    opt.set_optimizer(LBFGS(memory=5, alpha0=0.01, max_ls=10))
    opt.set_neighbor(NeighborList(8.5, 0.3))
    opt.set_evaluator(LJCut(0.0104, 3.4, 8.5))
    return opt


def test_canonical_identity_exports() -> None:
    assert av.Minimize is Minimize
    assert av.LBFGS is LBFGS
    assert av.ForceConverged is ForceConverged
    assert av.MinimizeResult is MinimizeResult


def test_typed_composition_builds() -> None:
    opt = make_minimize()
    opt.build()
    assert "built" in repr(opt)


def test_wrong_phase_fails_before_build() -> None:
    with pytest.raises(ValueError, match="Unknown minimizer phase"):
        Minimize().on("md.force")


def test_gpu_is_explicitly_deferred() -> None:
    with pytest.raises(ValueError, match="(?i)gpu|cuda"):
        Minimize(device="cuda")


def test_run_returns_control_result_and_mutates_same_frame(argon_frame: object) -> None:
    opt = make_minimize()
    result = opt.run(argon_frame, 1)
    assert isinstance(result, MinimizeResult)
    assert result.iterations == 1
    assert isinstance(result.converged, bool)
    assert isinstance(result.stop_reason, str)


# ═══════════════════════════════════════════════════════════════════
# py-minimize-full: CG / FIRE / cell relax / shared evaluator dispatch
# ═══════════════════════════════════════════════════════════════════

# ── Ar LJ parameters, shared by every case below ────────────────────
#
# Identical to regressions/minimizer/lj_dimer, fire_lj_dimer and
# cell_lj_pressure: eps = 0.0104 eV, sigma = 3.4 A, r_c = 2.5 sigma.

_EPS: float = 0.0104  # [eV]
_SIGMA: float = 3.4  # [A]
_CUTOFF: float = 8.5  # [A] = 2.5 sigma
_SKIN: float = 0.3  # [A]
_MASS_AR: float = 39.948  # [amu]
_Z_AR: int = 18

#: HARD-CODED GOLDEN — analytic minimum of the 12-6 Lennard-Jones pair
#: potential, r* = 2^(1/6) sigma. Not measured here; see
#: regressions/minimizer/fire_lj_dimer/REFERENCE.md ("Analytic LJ minimum").
_R_STAR: float = 2.0 ** (1.0 / 6.0) * _SIGMA  # 3.816371 A

#: Isolation cell for the dimer cases: 40 A, as in both dimer regressions.
_DIMER_BOX: float = 40.0  # [A]

#: Geometry tolerance on r*, from reg_lj_dimer.cpp / reg_fire_lj_dimer.cpp
#: (`std::fabs(r - r_star) < 0.01`).
_R_TOL: float = 0.01  # [A]


def _dimer_frame(separation: float, box_length: float = _DIMER_BOX) -> "molpy.Frame":
    """Two Ar atoms *separation* A apart along x in a cubic cell.

    Column set matches the conftest Ar fixture (x/y/z/mass/atomic_number):
    the driver materializes forces itself and geometry optimization needs
    no velocities.

    @param separation: Initial interatomic distance [A].
    @param box_length: Cubic cell edge [A].
    @return A two-atom molpy.Frame.
    """
    frame = molpy.Frame()
    atoms = molpy.Block()
    atoms.insert("x", np.array([0.0, separation], dtype=np.float64))
    atoms.insert("y", np.zeros(2, dtype=np.float64))
    atoms.insert("z", np.zeros(2, dtype=np.float64))
    atoms.insert("mass", np.full(2, _MASS_AR, dtype=np.float64))
    atoms.insert("atomic_number", np.full(2, _Z_AR, dtype=np.uint32))
    frame["atoms"] = atoms
    frame.box = molpy.Box.cube(box_length)
    return frame


def _separation(frame: "molpy.Frame") -> float:
    """Distance between the two atoms of a dimer frame [A].

    Plain Euclidean, no minimum image: the dimer sits at ~4 A inside a
    40 A cell, so the direct distance IS the nearest image.

    @param frame: Two-atom molpy.Frame.
    @return Interatomic distance [A].
    """
    atoms = frame["atoms"]
    delta = np.array(
        [
            float(np.asarray(atoms[key], dtype=np.float64)[1])
            - float(np.asarray(atoms[key], dtype=np.float64)[0])
            for key in ("x", "y", "z")
        ],
        dtype=np.float64,
    )
    return float(np.linalg.norm(delta))


def _lj_dimer_minimize(optimizer: object, criterion: object) -> Minimize:
    """Ar LJ minimizer carrying *optimizer* and *criterion*.

    @param optimizer: Component for ``set_optimizer``.
    @param criterion: Component for ``add_criterion``.
    @return An unbuilt Minimize ready to ``run``.
    """
    opt = Minimize()
    opt.set_optimizer(optimizer)
    opt.add_criterion(criterion)
    opt.set_neighbor(NeighborList(_CUTOFF, _SKIN))
    opt.set_evaluator(LJCut(_EPS, _SIGMA, _CUTOFF))
    return opt


# ── ac-001: CG and FIRE reach the analytic LJ minimum ───────────────


def test_cg_minimizes_dimer_to_rmin() -> None:
    """Polak-Ribiere+ CG drives a stretched Ar dimer onto r* = 2^(1/6) sigma.

    Recipe and gates are regressions/minimizer/lj_dimer (the LBFGS twin):
    start at 1.2 r*, |r - r*| < 0.01 A, and the run must stop because the
    criterion said so — ``force_converged`` — not because the iteration
    budget ran out. CG's convergence tolerances are NOT its constructor's
    business (cpu/minimize/cg.h:53-56): they live on ForceConverged, which
    is exactly what makes CG and LBFGS interchangeable here.
    """
    from atomiverse.minimizer import CG

    frame = _dimer_frame(1.2 * _R_STAR)
    opt = _lj_dimer_minimize(
        CG(),
        ForceConverged(
            ftol=0.01, rms_ftol=0.005, max_dtol=0.05, rms_dtol=0.02, etol=1e-6
        ),
    )
    result = opt.run(frame, 200)

    assert abs(_separation(frame) - _R_STAR) < _R_TOL, (
        f"CG stopped at r = {_separation(frame)!r} A, not the analytic "
        f"minimum r* = {_R_STAR!r} A"
    )
    assert result.converged
    assert "force_converged" in result.stop_reason


def test_fire_minimizes_dimer_to_rmin() -> None:
    """FIRE reaches the same analytic minimum from further out.

    ``dt0`` / ``dt_max`` are the LJ retune the C++ regression mandates
    (reg_fire_lj_dimer.cpp:30-58 and its REFERENCE.md): ASE's FIRE-1.0
    defaults assume O(1) eV/A forces, while the Ar dimer residual is
    ~1e-2 eV/A, so the pseudo-time scale is grown to keep the steps
    useful. Start at 1.5 r*, budget 2000 iterations, tolerance 1e-3 eV/A —
    all from the same regression.
    """
    from atomiverse.minimizer import FIRE

    frame = _dimer_frame(1.5 * _R_STAR)
    opt = _lj_dimer_minimize(
        FIRE(dt0=1.0, dt_max=20.0),
        ForceConverged(
            ftol=1e-3, rms_ftol=5e-4, max_dtol=0.05, rms_dtol=0.02, etol=1e-6
        ),
    )
    result = opt.run(frame, 2000)

    assert abs(_separation(frame) - _R_STAR) < _R_TOL, (
        f"FIRE stopped at r = {_separation(frame)!r} A, not the analytic "
        f"minimum r* = {_R_STAR!r} A"
    )
    assert result.converged
    assert "force_converged" in result.stop_reason


# ── ac-002: variable-cell relaxation ────────────────────────────────
#
# Recipe verbatim from regressions/minimizer/cell_lj_pressure
# (reg_cell_lj_pressure.cpp + REFERENCE.md): two Ar atoms 3.0 A apart in a
# 6.0 A cubic cell — compressed well inside the LJ minimum, so the virial
# pressure is positive (compressive) and a P_target = 0 relaxer must
# expand the cell.

_CELL_L0: float = 6.0  # [A] compressed cubic edge
_CELL_DX: float = 3.0  # [A] atom separation
_CELL_ALPHA: float = 0.08  # scale gain (regression value)
_CELL_MAX_SCALE: float = 0.08  # cap on |s-1| per step
_CELL_BULK_MOD: float = 0.5  # [eV/A^3] soft modulus
_CELL_PTOL: float = 5e-4  # [eV/A^3] PressureConverged tolerance

#: Pass gates of regressions/minimizer/cell_lj_pressure/REFERENCE.md
#: ("Pass"): L1 > L0 + 0.05 A and |P1| < 5e-2 eV/A^3 — a skeleton
#: expansion test, deliberately NOT an EOS equilibrium (spec
#: py-minimize-full, DECIDED 2026-08-15).
_CELL_MIN_EXPANSION: float = 0.05  # [A]
_CELL_MAX_ABS_P: float = 5e-2  # [eV/A^3]


def _compressed_cell_frame() -> "molpy.Frame":
    """The cell_lj_pressure starting state: 2 Ar atoms in a 6 A cube.

    @return A two-atom molpy.Frame under compressive virial pressure.
    """
    return _dimer_frame(_CELL_DX, box_length=_CELL_L0)


def _pressure(frame: "molpy.Frame") -> float:
    """Isotropic virial pressure P = tr(sigma) / (3V) [eV/A^3].

    ``kStress`` is the EXTENSIVE Voigt virial [eV] for every classical
    producer (core/frame_schema.h), which is why this divides by the cell
    volume — the same formula reg_cell_lj_pressure.cpp:18-22 and
    ``cpu::PressureConverged`` use.

    @param frame: Frame carrying the ``stress`` meta entry and a box.
    @return Pressure [eV/A^3].
    """
    stress = np.asarray(frame.meta["stress"], dtype=np.float64)
    volume = float(frame.box.volume())
    return float((stress[0] + stress[1] + stress[2]) / (3.0 * volume))


def _cell_relax_minimize(optimizer: object, criterion: object) -> Minimize:
    """The cell_lj_pressure driver recipe around *optimizer*.

    @param optimizer: Variable-cell component for ``set_optimizer``.
    @param criterion: Component for ``add_criterion``.
    @return An unbuilt Minimize ready to ``run``.
    """
    opt = Minimize()
    opt.set_optimizer(optimizer)
    opt.add_criterion(criterion)
    opt.set_neighbor(NeighborList(_CUTOFF, _SKIN))
    opt.set_evaluator(LJCut(_EPS, _SIGMA, _CUTOFF))
    return opt


def test_cell_relax_expands_compressed_cell() -> None:
    """IsotropicCellRelax expands the compressed cell and drops |P|.

    Both gates are the regression's own
    (regressions/minimizer/cell_lj_pressure/REFERENCE.md "Pass"), read off
    the molpy.Frame the Python facade hands back: the relaxed cell is
    written into ``frame.box`` and the virial into ``frame.meta["stress"]``
    by the same ``to_molrs`` boundary MD's NPT cases rely on.

    The 200-iteration budget is the regression's; it deliberately stops on
    the budget rather than on ``PressureConverged`` (measured L1 = 6.3212 A,
    P1 = 1.07e-3 eV/A^3 there), so nothing about convergence is asserted
    here — that is the next case's job.
    """
    from atomiverse.minimizer import IsotropicCellRelax, PressureConverged

    frame = _compressed_cell_frame()
    opt = _cell_relax_minimize(
        IsotropicCellRelax(
            P_target=0.0,
            alpha0=_CELL_ALPHA,
            max_scale=_CELL_MAX_SCALE,
            bulk_mod=_CELL_BULK_MOD,
        ),
        PressureConverged(P_target=0.0, ptol=_CELL_PTOL),
    )
    opt.run(frame, 200)

    length = float(np.asarray(frame.box.lengths, dtype=np.float64)[0])
    assert length > _CELL_L0 + _CELL_MIN_EXPANSION, (
        f"the compressed cell must expand: L1 = {length!r} A is not more "
        f"than L0 + {_CELL_MIN_EXPANSION} = "
        f"{_CELL_L0 + _CELL_MIN_EXPANSION} A"
    )
    assert abs(_pressure(frame)) < _CELL_MAX_ABS_P, (
        f"|P1| = {abs(_pressure(frame))!r} eV/A^3 exceeds the regression's "
        f"{_CELL_MAX_ABS_P} eV/A^3 gate"
    )


def test_pressure_converged_stops_the_cell_relax() -> None:
    """PressureConverged is effective: it ENDS the loop, early.

    Same recipe and the same 5e-4 eV/A^3 tolerance as the case above; only
    the iteration budget differs. Feasibility is measured, not guessed —
    the C++ recipe reaches |P| < 5e-4 eV/A^3 after 559 iterations (at 200
    it is still 1.07e-3 and stops with an empty ``stop_reason``), so a
    1000-iteration budget must terminate strictly early and name the
    criterion. A criterion that never fires, or one that fires on
    iteration 0 without expanding the cell, fails this.
    """
    from atomiverse.minimizer import IsotropicCellRelax, PressureConverged

    frame = _compressed_cell_frame()
    budget = 1000
    opt = _cell_relax_minimize(
        IsotropicCellRelax(
            P_target=0.0,
            alpha0=_CELL_ALPHA,
            max_scale=_CELL_MAX_SCALE,
            bulk_mod=_CELL_BULK_MOD,
        ),
        PressureConverged(P_target=0.0, ptol=_CELL_PTOL),
    )
    result = opt.run(frame, budget)

    assert result.converged
    assert "pressure_converged" in result.stop_reason
    assert result.iterations < budget, (
        "the loop must stop because the pressure criterion fired, not "
        "because the iteration budget ran out"
    )
    # ... and it fired on a RELAXED cell, not on the untouched compressed
    # one. The final residual itself is deliberately not asserted against
    # 5e-4: it is the criterion's own threshold, so re-asserting it here
    # would pin a 0.03% margin (measured 4.998e-4) with no extra meaning.
    length = float(np.asarray(frame.box.lengths, dtype=np.float64)[0])
    assert length > _CELL_L0 + _CELL_MIN_EXPANSION


def test_stress_converged_is_accepted_as_a_criterion() -> None:
    """StressConverged constructs with its per-axis mask and installs.

    Accept-level only: the criterion judges the INTENSIVE stress of a
    variable-cell run, and its own effectiveness is the anisotropic /
    triclinic story, not this binding's.
    """
    from atomiverse.minimizer import StressConverged

    criterion = StressConverged(
        sigma_target=0.0, stol=1e-2, check_x=True, check_y=True, check_z=True
    )
    assert Minimize().add_criterion(criterion) is not None


def test_anisotropic_cell_relax_is_accepted_as_an_optimizer() -> None:
    """AnisotropicCellRelax constructs with its per-axis relax mask.

    The mask is the CP2K ``CELL_OPT`` / ``CONSTRAINT`` twin
    (cpu/minimize/anisotropic_cell_relax.h:22-34): a masked axis is never
    scaled and never enters the residual.
    """
    from atomiverse.minimizer import AnisotropicCellRelax

    optimizer = AnisotropicCellRelax(
        P_target=0.0,
        alpha0=0.05,
        max_scale=0.05,
        bulk_mod=1.0,
        relax_x=True,
        relax_y=True,
        relax_z=False,
    )
    assert Minimize().set_optimizer(optimizer) is not None


def test_triclinic_cell_relax_is_accepted_as_an_optimizer() -> None:
    """TriclinicCellRelax constructs with its strain knobs.

    Full Voigt strain, h <- (I + eps) h (cpu/minimize/triclinic_cell_relax.h:
    16-24) — hence ``max_strain`` rather than the cubic relaxers'
    ``max_scale``.
    """
    from atomiverse.minimizer import TriclinicCellRelax

    optimizer = TriclinicCellRelax(alpha0=0.02, max_strain=0.05, bulk_mod=1.0)
    assert Minimize().set_optimizer(optimizer) is not None


def test_joint_isotropic_relax_is_accepted_as_an_optimizer() -> None:
    """JointIsotropicRelax constructs with cell AND atom step knobs.

    ``atom_step`` is what makes it joint: it moves the atoms along the unit
    force while it scales the cell (cpu/minimize/joint_cell_relax.h:22-33).
    """
    from atomiverse.minimizer import JointIsotropicRelax

    optimizer = JointIsotropicRelax(
        P_target=0.0,
        atom_step=0.01,
        alpha0=0.05,
        max_scale=0.05,
        bulk_mod=1.0,
    )
    assert Minimize().set_optimizer(optimizer) is not None


# ── ac-003: ONE evaluator dispatch table ────────────────────────────
#
# Both facades read ONE table (`atomiverse._dispatch`), so every force
# source MD accepts, Minimize accepts. A second copy would be free to
# drift; these cases pin the shared object, not two agreeing dicts.
#
# IDENTITY ASSERTION, chosen deliberately: the contract pinned below is
# `atomiverse.md._EVALUATOR_DISPATCH is atomiverse.minimizer._EVALUATOR_DISPATCH`
# — both facade attributes already exist under those names today, so the
# assertion adds no new naming, and it holds for ANY internal module the
# shared table ends up living in (spec says python/src/atomiverse/
# _dispatch.py). Importing the internal module by name and pinning the
# table's name there would overfit a private detail the spec leaves to
# the implementer.

#: Concrete samples of the evaluator types MD accepts unconditionally.
#: Built lazily so a missing binding fails inside the test, never at
#: collection. Constructor shapes come from bind_md.cpp: the bonded
#: styles take PER-TYPE LISTS, not scalars.
_MD_EVALUATOR_SAMPLES: tuple[tuple[str, Callable[[], object]], ...] = (
    ("PairLJCut", lambda: LJCut(_EPS, _SIGMA, _CUTOFF)),
    ("PairCoulCut", lambda: CoulCut(_CUTOFF)),
    ("CoulLong", lambda: CoulLong(3.5 / _CUTOFF, _CUTOFF)),
    ("BondHarmonic", lambda: BondHarmonic([1.0], [1.0])),
    ("AngleHarmonic", lambda: AngleHarmonic([1.0], [109.5])),
    ("DihedralMultiHarmonic", lambda: DihedralMultiHarmonic([[0.0, 0.0, 0.0]])),
)

#: AM1 is compile-optional (ATV_BUILD_SEMIEMPIRICAL) and is the marquee
#: case for the generalized dispatch: Minimize + AM1Evaluator is a real
#: C++ capability (regressions/semiempirical/am1_water_minimize).
_HAS_SEMIEMPIRICAL: bool = extension_available() and hasattr(_cpp, "AM1Evaluator")


@pytest.mark.parametrize(
    "factory",
    [sample[1] for sample in _MD_EVALUATOR_SAMPLES],
    ids=[sample[0] for sample in _MD_EVALUATOR_SAMPLES],
)
def test_minimize_accepts_md_evaluators(factory: Callable[[], object]) -> None:
    """Every force source MD takes, Minimize takes.

    Accept-level: the assertion is that ``set_evaluator`` does not refuse.
    Running a bonded or electrostatic Hamiltonian would need a topology /
    charge-carrying frame, which is a force-field concern this dispatch
    test has no business owning.
    """
    assert Minimize().set_evaluator(factory()) is not None


@pytest.mark.skipif(
    not _HAS_SEMIEMPIRICAL,
    reason="extension built without ATV_BUILD_SEMIEMPIRICAL",
)
def test_minimize_accepts_am1_evaluator() -> None:
    """A semiempirical method is a legal force source for minimization.

    The C++ twin is regressions/semiempirical/am1_water_minimize:
    ``Minimize`` + ``AM1Evaluator`` + LBFGS. Accept-level here.
    """
    from atomiverse.semiempirical import AM1Evaluator

    assert Minimize().set_evaluator(AM1Evaluator()) is not None


def test_evaluator_dispatch_is_one_shared_table() -> None:
    """MD and Minimize read the SAME dispatch object, not two copies.

    Identity, not equality: two dicts that happen to agree today drift
    apart at the next binding, and the drift shows up as a force source
    one driver accepts and the other refuses.
    """
    import atomiverse.md as md_facade
    import atomiverse.minimizer as minimizer_facade

    assert md_facade._EVALUATOR_DISPATCH is minimizer_facade._EVALUATOR_DISPATCH


def test_minimize_dispatch_covers_every_md_type() -> None:
    """The behavioral half of ac-003, including capability-gated types.

    ``KSpacePME`` / ``AM1Evaluator`` / ``PWEvaluator`` join MD's table only
    on an extension that carries them; asserting key-set equality covers
    them without constructing one.
    """
    import atomiverse.md as md_facade
    import atomiverse.minimizer as minimizer_facade

    assert set(minimizer_facade._EVALUATOR_DISPATCH) == set(
        md_facade._EVALUATOR_DISPATCH
    )


def test_set_evaluator_still_rejects_unknown_type() -> None:
    """Generalizing the dispatch must not make it an accept-all."""
    with pytest.raises(TypeError, match="PairLJCut"):
        Minimize().set_evaluator(object())


# ── A chargeless frame under a Coulomb model is refused ─────────────
#
# ONE dispatch table (ac-003 above) means Minimize accepts every force
# source MD accepts — including the three Coulomb styles. It does NOT yet
# mean Minimize *guards* them: `MdHandle::require_charge_column`
# (python/src/bindings/bind_md.cpp) rejects a molrs Frame with no `charge`
# column before an electrostatic MD run, and `MinimizeHandle::run`
# (python/src/bindings/bind_minimizer.cpp:215) has no equivalent. The
# consequence is the exact failure mode that guard was written for:
# `cpu::from_molrs` allocates the optional `charge` column and zero-fills
# it, so a forgotten charge column and a deliberately neutral system are
# indistinguishable by the time the optimizer runs — the geometry relaxes
# under LJ (or under nothing) while the electrostatics the caller asked
# for is silently absent, and the result looks plausible.
#
# The refusal belongs to the SAME boundary for both drivers: the molrs
# Frame is the last place where a column's absence is visible.


class TestMinimizeFailsLoud:
    """Minimize refuses setups whose silence would look like a result."""

    def test_coulomb_minimize_missing_charge_raises(
        self, argon_frame: "molpy.Frame"
    ) -> None:
        """A Coulomb evaluator on a chargeless frame must not optimize.

        Twin of test_md.py::TestCoulombPme::test_coul_cut_missing_charge_raises,
        same fixture and same 3.0 A cutoff: the Ar fixture's cell edge is
        6.8 A, so L/2 = 3.4 A still satisfies the minimum-image convention
        and the missing `charge` column is the only defect in the setup.

        Accept-level on the message: `match="charge"` pins WHAT was wrong,
        not the wording, exactly as the MD twin does.
        """
        assert "charge" not in argon_frame["atoms"].keys(), (
            "fixture drift: this case needs a frame WITHOUT a charge column"
        )

        opt = Minimize()
        opt.set_optimizer(LBFGS())
        opt.set_neighbor(NeighborList(3.0, 0.5))
        opt.set_evaluator(CoulCut(3.0))

        with pytest.raises(RuntimeError, match="charge"):
            opt.run(argon_frame, 1)


# ── Late reconfiguration is refused (the MD/MC late-setter contract) ─
#
# Mirror of test_md.py::TestMdLateSetters, which documents the underlying
# defect: `Driver::build()` consumes its pending registrations, so a
# setter called after `run()` cannot reach the compiled plan — it is
# silently dropped, and the NEXT run() dies in a re-build with a
# component-validation error naming the wrong culprit.
#
# bind_mc.cpp's `require_configurable_` is the shipped shape: a
# std::runtime_error naming the refused setter. nanobind maps that to
# RuntimeError, so these cases read the same whether the guard lands in
# the facade or in bind_minimizer.cpp.

#: Contract substring pinned for the implementer, identical to
#: test_md.py's ``_LATE_SETTER_MATCH``. The refusal must contain it and,
#: for the named setters, must also name the setter it refused.
_LATE_SETTER_MATCH: str = r"already run"


class TestMinimizeLateSetters:
    """Reconfiguring a Minimize after its first ``run()`` fails loud.

    Re-running an UNCHANGED configuration stays legal — Minimize is
    re-runnable by contract (spec py-minimize-full, DECIDED 2026-08-15),
    and chunked optimization is the normal way to interleave analysis.
    """

    #: Two iterations per leg: the minimizer's phase order runs
    #: `search_direction` BEFORE `force`, so iteration 0 has no forces yet
    #: and moves nothing (measured on the C++ driver). One iteration would
    #: therefore be a no-op leg and could not show that leg two advances.
    _ITERS: int = 2

    @staticmethod
    def _stretched_dimer_minimize() -> Minimize:
        """LBFGS on a dimer far from r*, so every leg has work left."""
        return _lj_dimer_minimize(
            LBFGS(memory=5, alpha0=0.01, max_ls=10),
            ForceConverged(
                ftol=0.01, rms_ftol=0.005, max_dtol=0.05, rms_dtol=0.02, etol=1e-6
            ),
        )

    def test_late_set_evaluator_raises(self) -> None:
        """A force term added after the run cannot reach the plan."""
        opt = self._stretched_dimer_minimize()
        opt.run(_dimer_frame(1.2 * _R_STAR), self._ITERS)
        with pytest.raises(RuntimeError, match=_LATE_SETTER_MATCH) as excinfo:
            opt.set_evaluator(LJCut(_EPS, _SIGMA, _CUTOFF))
        assert "set_evaluator" in str(excinfo.value), (
            "the refusal must name the setter it refused, like bind_mc.cpp's "
            "require_configurable_"
        )

    def test_late_set_optimizer_raises(self) -> None:
        """Swapping the optimizer mid-chain is a new configuration.

        The minimizer's own version of test_md.py's integrator case: a
        re-build that happened to satisfy its component checks would
        optimize *something*, just not with the algorithm the caller
        configured.
        """
        opt = self._stretched_dimer_minimize()
        opt.run(_dimer_frame(1.2 * _R_STAR), self._ITERS)
        with pytest.raises(RuntimeError, match=_LATE_SETTER_MATCH) as excinfo:
            opt.set_optimizer(LBFGS(memory=5, alpha0=0.01, max_ls=10))
        assert "set_optimizer" in str(excinfo.value)

    def test_late_set_neighbor_raises(self) -> None:
        """The neighbor list is part of the compiled plan too."""
        opt = self._stretched_dimer_minimize()
        opt.run(_dimer_frame(1.2 * _R_STAR), self._ITERS)
        with pytest.raises(RuntimeError, match=_LATE_SETTER_MATCH) as excinfo:
            opt.set_neighbor(NeighborList(_CUTOFF, _SKIN))
        assert "set_neighbor" in str(excinfo.value)

    def test_late_add_criterion_raises(self) -> None:
        """Changing the stop policy after the fact is a reconfiguration.

        Loosening a tolerance changes no physics, which is exactly why it
        must refuse: it is the cheapest way to make the next ``run()``
        report a convergence that the compiled plan never judged.
        """
        opt = self._stretched_dimer_minimize()
        opt.run(_dimer_frame(1.2 * _R_STAR), self._ITERS)
        with pytest.raises(RuntimeError, match=_LATE_SETTER_MATCH) as excinfo:
            opt.add_criterion(
                ForceConverged(
                    ftol=0.5, rms_ftol=0.3, max_dtol=0.5, rms_dtol=0.3, etol=1e-3
                )
            )
        assert "add_criterion" in str(excinfo.value)

    def test_late_on_phase_raises(self) -> None:
        """The escape hatch refuses at ``on()``, not at ``add()``.

        Same ruling as MD (test_md.py::TestMdLateSetters): ``on(phase)`` is
        the registration entry point and the earliest clean observable, so
        ``PhaseBuilder`` needs no guard of its own.
        """
        opt = self._stretched_dimer_minimize()
        opt.run(_dimer_frame(1.2 * _R_STAR), self._ITERS)
        with pytest.raises(RuntimeError, match=_LATE_SETTER_MATCH):
            opt.on("force")

    def test_two_runs_same_config_still_advance(self) -> None:
        """Chunked minimization stays legal and keeps making progress.

        The refusal above is about *changing* the configuration, never
        about running again. A guard that turned ``run()`` itself into a
        one-shot would pass every case above and break this one.
        """
        frame = _dimer_frame(1.2 * _R_STAR)
        opt = self._stretched_dimer_minimize()

        opt.run(frame, self._ITERS)
        r_mid = _separation(frame)

        result = opt.run(frame, self._ITERS)
        r_after = _separation(frame)

        assert result.iterations == self._ITERS
        assert r_after != r_mid, (
            "the second run() of an unchanged configuration must keep "
            "optimizing the same compiled plan"
        )
        assert abs(r_after - _R_STAR) < abs(r_mid - _R_STAR), (
            "the second leg must move the dimer TOWARD the minimum"
        )

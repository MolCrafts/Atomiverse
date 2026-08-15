"""CPU geometry minimization — categorical setters matching C++.

Mirrors ``cpu::minimizer::Minimize`` (headers under ``cpu/minimize/``)::

    opt = Minimize()
    opt.set_optimizer(LBFGS(memory=5, alpha0=0.02, max_ls=8))
    opt.add_criterion(ForceConverged(ftol=0.5, …))  # optional
    opt.set_evaluator(PairLJCut(…))
    opt.set_neighbor(NeighborList(…))
    opt.build()
    result = opt.run(frame, max_iter)

Geometry optimization walks the atoms (and optionally the periodic box)
downhill on the potential energy surface until the forces vanish — a
*minimum*, the structure the system would relax into at 0 K. It is not
dynamics: there is no time, no temperature and no velocity, only a sequence of
steps chosen to lower the energy.

Choosing an optimizer
---------------------
The slot is singular — the seven components below are alternatives, never a
stack. Three work at fixed cell:

``LBFGS(memory, alpha0, max_ls)``
    Limited-memory quasi-Newton: reconstructs the curvature of the energy
    surface from the last ``memory`` steps and backtracks along the resulting
    direction. The usual first choice; fastest near a minimum.
``CG(alpha0)``
    Polak-Ribière⁺ conjugate gradient: steepest descent plus a weighted memory
    of the previous direction. Lower memory, and the better choice when
    L-BFGS keeps discarding its curvature history.
``FIRE(dt0, dt_max, …)``
    Damped dynamics with no line search — a fictitious unit-mass trajectory
    whose velocity is steered towards the force. Cheap per step and robust far
    from the minimum. Its defaults are ASE's and assume forces of order
    1 eV/Å; on a soft system (an argon Lennard-Jones dimer sits at ~10⁻²
    eV/Å) grow the pseudo-time scale — ``FIRE(dt0=1.0, dt_max=20.0)``, the
    retune ``regressions/minimizer/fire_lj_dimer`` uses.

Note ``alpha0`` means a step size in Å²/eV for ``LBFGS`` and ``CG``, but a
dimensionless velocity-mixing fraction for ``FIRE``.

The other four move the periodic **box**: ``IsotropicCellRelax`` (one scale
factor, volume only), ``AnisotropicCellRelax`` (three box edges independently,
maskable per axis), ``TriclinicCellRelax`` (full strain, so the cell may
shear) and ``JointIsotropicRelax`` (volume *and* a steepest-descent atom
step).

Criteria: the optimizer never judges itself
-------------------------------------------
Termination is a separate object, and that separation is what makes the
optimizers interchangeable: an optimizer publishes force / displacement /
energy metrics and a *criterion* judges them. ``ForceConverged`` is the
fixed-cell policy (and the build-time default; all five of its tolerances must
hold at once). A variable-cell run judges ``PressureConverged`` (isotropic
mean) or ``StressConverged`` (per axis, maskable) instead, because a cell
relaxer moves the box and can sit at zero force while far from the target
pressure. The slot is plural and verdicts compose by severity, so a joint
atom-plus-cell relaxation can carry both a force and a pressure test.

What a variable-cell run needs (the honest gates)
-------------------------------------------------
A relaxer follows the virial — the configurational stress an evaluator
accumulates while it computes forces, carried on the frame as the ``stress``
entry in Voigt order (xx, yy, zz, xy, xz, yz) and **extensive**, i.e. a
whole-cell energy in eV rather than a stress. Both criteria divide it by the
cell volume, so targets and tolerances are pressures in eV/Å³
(1 bar ≈ 6.2415×10⁻⁷ eV/Å³, positive = compressive).

Three things follow, none of which the API can check for you:

* **The force source must write that virial.** The classical evaluators
  (``PairLJCut``, ``PairCoulCut``, ``CoulLong`` + ``KSpacePME``, the bonded
  styles) do, by default. ``AM1Evaluator`` writes none — it is a molecular
  method with no cell — and ``PWEvaluator`` is installed here as a
  gradient-only copy, so neither drives a cell step from Python. With a zero
  virial the cell simply never moves, and a ``PressureConverged(P_target=0)``
  reports success on the untouched cell.
* **The first iteration is a free one.** A step is computed from the stress
  the *previous* iteration's force evaluation left behind, and iteration 0 has
  no such evaluation — it reads whatever the frame arrived with, which is zero
  on a freshly built one. At the default ``P_target=0`` that makes its scale
  factor exactly 1, so the cell starts moving on iteration 1.
* **Match the criterion to the algorithm.** A relaxer publishes its pressure
  residual into the metric slot ``ForceConverged`` reads as a force, so that
  pairing does not fail loudly — it silently compares eV/Å³ against a
  tolerance meant for eV/Å.

Evaluators, and running under the MPI launcher
----------------------------------------------
Every force source the MD facade accepts is accepted here — both read one
dispatch table (``atomiverse._dispatch``), exposed on each as
``_EVALUATOR_DISPATCH``. One exception applies only when this interpreter is
hosted by the multi-rank MPI launcher: the configuration is re-sent to the
worker ranks as a task payload each run, and that payload carries
``PairLJCut``, the ``NeighborList`` and the seven optimizers only. A run
configured with any other force source raises ``RuntimeError`` naming what it
cannot carry, rather than optimizing with a term silently missing.

Configuration is one-shot: the execution plan is compiled exactly once, so
after the first completed :meth:`Minimize.run` every setter and
:meth:`Minimize.on` raises ``RuntimeError``. Running again is still legal and
re-optimizes from the frame's current state.
"""

from __future__ import annotations

from atomiverse._binding import _cpp
from atomiverse._dispatch import (
    EVALUATOR_DISPATCH as _EVALUATOR_DISPATCH,
    unsupported_evaluator_message as _unsupported_evaluator_message,
)

_PHASE_MAP: dict[str, object] = {
    "search_direction": _cpp.MinimizePhase.search_direction,
    "line_search": _cpp.MinimizePhase.line_search,
    "force": _cpp.MinimizePhase.force,
    "convergence_check": _cpp.MinimizePhase.convergence_check,
    "end_of_step": _cpp.MinimizePhase.end_of_step,
}

# Optimizer types accepted by set_optimizer. Unconditional: they all live in
# `cpu/minimize/`, which `ATV_BUILD_MINIMIZER` (ON by default) compiles as a
# unit — there is no per-optimizer build flag to probe for.
_OPTIMIZER_TYPES: tuple[type, ...] = (
    _cpp.LBFGS,
    _cpp.CG,
    _cpp.FIRE,
    _cpp.IsotropicCellRelax,
    _cpp.AnisotropicCellRelax,
    _cpp.TriclinicCellRelax,
    _cpp.JointIsotropicRelax,
)

# Criterion types accepted by add_criterion, likewise unconditional
# (`cpu/driver/criteria.h` is part of the driver, not of a method domain).
_CRITERION_TYPES: tuple[type, ...] = (
    _cpp.ForceConverged,
    _cpp.PressureConverged,
    _cpp.StressConverged,
)

_PHASE_DISPATCH: dict[type, str] = {
    _cpp.PairLJCut: "add_pair_lj_cut",
    _cpp.NeighborList: "add_neighbor_list",
}


class PhaseBuilder:
    """Fluent builder returned by :meth:`Minimize.on` (escape hatch)."""

    def __init__(self, cpp_builder: object) -> None:
        self._cpp = cpp_builder

    def add(self, component: object, order: int = 0) -> PhaseBuilder:
        method_name = _PHASE_DISPATCH.get(type(component))
        if method_name is None:
            supported = ", ".join(t.__name__ for t in _PHASE_DISPATCH)
            raise TypeError(
                f"Cannot add {type(component).__name__} to a minimizer phase. "
                f"Supported: {supported}"
            )
        getattr(self._cpp, method_name)(component, order)
        return self


class Minimize:
    """Geometry-minimization driver for the CPU backend.

    Parameters
    ----------
    convergence : ForceConverged | PressureConverged | StressConverged, optional
        Stop policy. If omitted, C++ ``install_defaults()`` attaches a
        default ``ForceConverged`` at build time.
    device : str, optional
        Execution backend. v1 accepts only ``"cpu"``.
    """

    def __init__(
        self,
        convergence: object | None = None,
        device: str = "cpu",
    ) -> None:
        if device != "cpu":
            raise ValueError(
                f"device={device!r}: GPU minimization is deferred; "
                "only device='cpu' is supported"
            )
        self.device = device
        self._cpp = _cpp.Minimize()
        self._built = False
        if convergence is not None:
            self.add_criterion(convergence)

    def set_optimizer(self, component: object) -> Minimize:
        """Install the optimizer (``Minimize::set_optimizer``).

        The slot is singular — ``LBFGS``, ``CG``, ``FIRE`` and the four
        variable-cell relaxers are alternatives. A relaxer moves the *box*,
        so pair it with ``PressureConverged`` or ``StressConverged``:
        ``ForceConverged`` would judge a quantity the relaxer is not driving.
        A relaxer also needs a force source that writes the virial, and skips
        its first iteration — see this module's docstring for both gates.
        """
        if type(component) not in _OPTIMIZER_TYPES:
            supported = ", ".join(t.__name__ for t in _OPTIMIZER_TYPES)
            raise TypeError(
                f"set_optimizer does not support {type(component).__name__}. "
                f"Supported: {supported}"
            )
        self._cpp.set_optimizer(component)
        self._built = False
        return self

    def set_evaluator(self, component: object) -> Minimize:
        """Register an energy/force source (``Driver::set_evaluator``, plural).

        The slot is plural: each call appends another term and the driver sums
        them. The accepted types are exactly those ``MD.set_evaluator``
        accepts — one shared dispatch table — so a bonded force field, an
        Ewald split, a semiempirical method or plane-wave DFT all drive a
        geometry optimization.

        Two caveats this dispatch cannot catch for you:

        * ``CoulLong`` and ``KSpacePME`` are the two halves of one Ewald
          split. Register both, with the same ``alpha`` — the driver rejects
          a lone half at :meth:`build`, not here.
        * every Coulomb style reads the per-atom ``charge`` column [e] of the
          frame.
        """
        method_name = _EVALUATOR_DISPATCH.get(type(component))
        if method_name is None:
            raise TypeError(_unsupported_evaluator_message(component))
        getattr(self._cpp, method_name)(component)
        self._built = False
        return self

    def set_neighbor(self, component: object) -> Minimize:
        """Install the neighbor list (``Driver::set_neighbor``)."""
        if type(component) is not _cpp.NeighborList:
            raise TypeError(
                f"set_neighbor expects NeighborList, got {type(component).__name__}"
            )
        self._cpp.set_neighbor(component)
        self._built = False
        return self

    def add_criterion(self, criterion: object) -> Minimize:
        """Attach a termination criterion (``Driver::add_criterion``).

        The slot is plural and the verdicts compose by severity
        (Stop > Converged > Continue), which is how a joint atom-plus-cell
        relaxation gets both a force and a pressure test.
        """
        if type(criterion) not in _CRITERION_TYPES:
            supported = ", ".join(t.__name__ for t in _CRITERION_TYPES)
            raise TypeError(
                f"add_criterion does not support {type(criterion).__name__}. "
                f"Supported: {supported}"
            )
        self._cpp.add_criterion(criterion)
        self._built = False
        return self

    def on(self, phase: str) -> PhaseBuilder:
        """Escape hatch for explicit phase binding.

        Raises ``RuntimeError`` once :meth:`run` has completed: ``on`` is the
        registration entry point, so the refusal lands here rather than at
        ``PhaseBuilder.add``.
        """
        cpp_phase = _PHASE_MAP.get(phase)
        if cpp_phase is None:
            valid = ", ".join(f'"{p}"' for p in _PHASE_MAP)
            raise ValueError(f"Unknown minimizer phase {phase!r}. Valid: {valid}")
        # The handle owns the already-run guard and is asked FIRST: a refused
        # registration must leave this object's build state untouched.
        cpp_builder = self._cpp.on(cpp_phase)
        self._built = False
        return PhaseBuilder(cpp_builder)

    def build(self) -> None:
        """Compile the execution plan (called automatically by ``run``)."""
        self._cpp.build()
        self._built = True

    def run(self, frame: object, max_iter: int) -> object:
        """Optimize ``frame`` for at most ``max_iter`` iterations.

        Returns a :class:`MinimizeResult` with loop-control metadata.
        Geometry / energy / forces — and, for a variable-cell run, the box and
        the ``stress`` metadata entry — are written back into *frame*.

        The plan is compiled on the FIRST call only. A completed run freezes
        the configuration (every setter refuses afterwards), so calling ``run``
        again re-runs that same compiled plan from the frame's current state.

        Under the MPI launcher this raises ``RuntimeError`` when the
        configuration names a force source the task payload cannot carry; the
        message names the components. See the module docstring.
        """
        if not self._built:
            self.build()
        return self._cpp.run(frame, max_iter)

    def __repr__(self) -> str:
        state = "built" if self._built else "not built"
        return f"Minimize({state})"


# Optimizers (`cpu/minimize/`) and the loop-termination criteria
# (`cpu/driver/criteria.h`) share this module: `cpu/minimize/` is the one
# backend-root directory the whole geometry-optimization surface belongs to,
# and ForceConverged has been re-exported from here since the first binding.
LBFGS = _cpp.LBFGS
CG = _cpp.CG
FIRE = _cpp.FIRE
IsotropicCellRelax = _cpp.IsotropicCellRelax
AnisotropicCellRelax = _cpp.AnisotropicCellRelax
TriclinicCellRelax = _cpp.TriclinicCellRelax
JointIsotropicRelax = _cpp.JointIsotropicRelax
ForceConverged = _cpp.ForceConverged
PressureConverged = _cpp.PressureConverged
StressConverged = _cpp.StressConverged
MinimizeResult = _cpp.MinimizeResult

__all__ = [
    "Minimize",
    "LBFGS",
    "CG",
    "FIRE",
    "IsotropicCellRelax",
    "AnisotropicCellRelax",
    "TriclinicCellRelax",
    "JointIsotropicRelax",
    "ForceConverged",
    "PressureConverged",
    "StressConverged",
    "MinimizeResult",
    "PhaseBuilder",
]

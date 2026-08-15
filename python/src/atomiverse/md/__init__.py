"""OOP Molecular Dynamics driver — operates directly on a ``molrs.Frame``.

Mirrors the C++ categorical API on ``cpu::md::MD`` (headers under
``cpu/dynamics/``)::

    sim = MD()
    sim.set_evaluator(PairLJCut(...))
    sim.set_neighbor(NeighborList(...))
    sim.set_integrator(VelocityVerlet(dt))
    sim.set_thermo(200)
    sim.build()
    sim.run(frame, 1000)

The driver borrows the ``molrs.Frame`` for the duration of :meth:`MD.run`:
the C++ side materializes a transient execution frame, advances it, and
writes positions / velocities / forces / energy back into the *same*
``molrs.Frame``. Global results are exact-dtype typed metadata entries in
``frame.meta``, read back as plain Python values.
"""

from __future__ import annotations

from atomiverse._binding import _cpp
from atomiverse._dispatch import (
    EVALUATOR_DISPATCH as _EVALUATOR_DISPATCH,
    unsupported_evaluator_message as _unsupported_evaluator_message,
)

# ---------------------------------------------------------------------------
# Phase name → C++ enum (escape hatch for on(phase).add)
# ---------------------------------------------------------------------------
_PHASE_MAP: dict[str, object] = {
    "initial_integrate": _cpp.Phase.initial_integrate,
    "pre_force": _cpp.Phase.pre_force,
    "force": _cpp.Phase.force,
    "post_force": _cpp.Phase.post_force,
    "final_integrate": _cpp.Phase.final_integrate,
    "end_of_step": _cpp.Phase.end_of_step,
}

# Evaluator types → typed C++ setter (plural set_evaluator). Imported, not
# built here: `atomiverse._dispatch` owns THE table and `atomiverse.minimizer`
# imports the same object, so a force source bound for one driver is bound for
# both. A second copy would be free to drift, accepting a component here and
# refusing it there.

# Recorder types accepted by MD.set_record. Both are unconditional: the
# wheel forces ATV_BUILD_IO=ON through the CMake module DAG, so the
# trajectory writer is never capability-probed (unlike KSpacePME above).
_RECORDER_TYPES: tuple[type, ...] = (_cpp.Thermo, _cpp.XYZ)

# Escape-hatch phase builder dispatch
_PHASE_DISPATCH: dict[type, str] = {
    _cpp.PairLJCut: "add_pair_lj_cut",
    _cpp.NeighborList: "add_neighbor_list",
    _cpp.BondHarmonic: "add_bond_harmonic",
    _cpp.AngleHarmonic: "add_angle_harmonic",
    _cpp.DihedralMultiHarmonic: "add_dihedral_multi_harmonic",
    _cpp.Thermo: "add_thermo",
}


class PhaseBuilder:
    """Fluent builder returned by :meth:`MD.on` (escape hatch).

    Prefer the categorical setters (``set_evaluator``, ``set_neighbor``, …).
    Use ``on(phase).add`` only when you need an explicit phase or order.
    """

    def __init__(self, cpp_builder: object) -> None:
        self._cpp = cpp_builder

    def add(self, component: object, order: int = 0) -> PhaseBuilder:
        """Add *component* at this phase with the given execution *order*."""
        method_name = _PHASE_DISPATCH.get(type(component))
        if method_name is None:
            supported = ", ".join(t.__name__ for t in _PHASE_DISPATCH)
            raise TypeError(
                f"Cannot add {type(component).__name__} to a phase. "
                f"Supported: {supported}"
            )
        getattr(self._cpp, method_name)(component, order)
        return self


class MD:
    """Molecular dynamics driver (CPU backend).

    Parameters
    ----------
    dt : float, optional
        Timestep in fs. If provided, calls ``set_integrator(VelocityVerlet(dt))``.
    device : str, optional
        Execution backend, PyTorch-style. v1 accepts only ``"cpu"``;
        any other value raises ``ValueError`` before any compute.

    Example (mirrors ``examples/07_md_lj.cpp``)::

        import atomiverse as av

        sim = av.MD()
        sim.set_evaluator(av.PairLJCut(0.0104, 3.4, 8.5))
        sim.set_neighbor(av.NeighborList(8.5, 0.5))
        sim.set_integrator(av.VelocityVerlet(1.0))
        sim.set_thermo(200)
        sim.build()
        sim.run(frame, 1000)

    Configuration is one-shot: the execution plan is compiled exactly once,
    so after the first completed :meth:`run` every setter and :meth:`on`
    raises ``RuntimeError``. Chunked runs of the *same* configuration stay
    legal — a different configuration means a new ``MD`` object.
    """

    def __init__(self, dt: float | None = None, device: str = "cpu") -> None:
        if device != "cpu":
            raise ValueError(
                f"device={device!r}: CUDA bindings are not yet available; "
                "only device='cpu' is supported (backend selection is a "
                "constructor parameter, not a namespace — see the package "
                "docstring)"
            )
        self.device = device
        self._cpp = _cpp.MD()
        self._built = False
        if dt is not None:
            self.set_integrator(_cpp.VelocityVerlet(dt))

    # ------------------------------------------------------------------
    # Categorical setters (LAMMPS-shaped; match cpu::md::MD / Driver)
    # ------------------------------------------------------------------

    def set_integrator(self, component: object) -> MD:
        """Install the time integrator (``cpu::md::MD::set_integrator``).

        Exactly one integrator advances positions and velocities.
        ``VelocityVerlet`` covers NVE (fixed particle number, volume and
        energy) and, with a thermostat added, NVT (fixed temperature
        instead of energy). For NPT (fixed pressure instead of volume),
        ``NoseHooverNPT`` (LAMMPS ``fix npt``) and ``ParrinelloRahmanNPT``
        (the same scheme with a full, shearable cell matrix) *replace* the
        plain integrator — they integrate, thermostat and barostat in one
        step, so do not add a separate thermostat or barostat with them.
        """
        if type(component) not in (
            _cpp.VelocityVerlet,
            _cpp.NoseHooverNPT,
            _cpp.ParrinelloRahmanNPT,
        ):
            raise TypeError(
                f"set_integrator expects VelocityVerlet, NoseHooverNPT, or "
                f"ParrinelloRahmanNPT, got {type(component).__name__}"
            )
        self._cpp.set_integrator(component)
        self._built = False
        return self

    def set_thermostat(self, component: object) -> MD:
        """Install a thermostat (``cpu::md::MD::set_thermostat``)."""
        if type(component) not in (_cpp.NoseHoover, _cpp.Langevin, _cpp.CSVR):
            raise TypeError(
                f"set_thermostat expects NoseHoover, Langevin, or CSVR, got "
                f"{type(component).__name__}"
            )
        self._cpp.set_thermostat(component)
        self._built = False
        return self

    def set_barostat(self, component: object) -> MD:
        """Install a barostat (``cpu::md::MD::set_barostat``).

        A barostat holds the system at a target pressure by rescaling the
        periodic cell. ``BerendsenBarostat`` rides *alongside* the
        integrator and a thermostat, so a Berendsen NPT run is
        ``set_integrator(VelocityVerlet(dt))`` +
        ``set_thermostat(...)`` + ``set_barostat(BerendsenBarostat(...))``.
        The NPT integrators (``NoseHooverNPT``, ``ParrinelloRahmanNPT``)
        replace the integrator instead and take no separate barostat.

        Pressure targets are in eV/Å³ (positive means compressive);
        1 bar ≈ 6.2415e-7 eV/Å³.
        """
        if type(component) not in (_cpp.BerendsenBarostat,):
            raise TypeError(
                f"set_barostat expects BerendsenBarostat, got "
                f"{type(component).__name__}"
            )
        self._cpp.set_barostat(component)
        self._built = False
        return self

    def set_observer(self, component: object) -> MD:
        """Install an observer (``Driver::set_observer``).

        v1 supports ``InitVelocity`` (Maxwell–Boltzmann seeding).
        """
        if type(component) is not _cpp.InitVelocity:
            raise TypeError(
                f"set_observer expects InitVelocity, got {type(component).__name__}"
            )
        self._cpp.set_observer(component)
        self._built = False
        return self

    def set_evaluator(self, component: object) -> MD:
        """Register an energy/force source (``Driver::set_evaluator``, plural).

        The slot is plural: each call appends another term and the driver
        sums them, so a full force field is several calls (a pair style, a
        bond style, an electrostatic model, …). The accepted component
        types are whatever this extension binds; an unsupported one raises
        ``TypeError`` listing them.

        ``AM1Evaluator`` belongs here too: registering it makes the run
        Born-Oppenheimer molecular dynamics — one semiempirical SCF solve
        per step, forces from it. It needs no ``NeighborList`` (AM1 is
        molecular, not a pair style) and the driver installs its own
        gradient-enabled copy, so results land on the frame rather than on
        the object you passed.

        ``PWEvaluator`` is the same story on a plane-wave Hamiltonian:
        registering it makes the run plane-wave Born-Oppenheimer dynamics,
        with analytic ionic forces from the converged density. It likewise
        takes no ``NeighborList``, and it additionally needs the frame to
        carry a ``box`` — a plane-wave basis is defined by the cell. Each
        step costs *two* SCF solves (the energy solve, then the force
        path's own), so size the cell, ``ecut_ry`` and ``max_iter`` for the
        number of steps you actually want.

        Three caveats, because they are the ones this dispatch cannot catch
        for you:

        * ``CoulLong`` and ``KSpacePME`` are the two halves of one Ewald
          split. Register both, with the same ``alpha`` — the driver
          rejects a lone half at :meth:`build`, not here.
        * every Coulomb style reads the per-atom ``charge`` column [e] of
          the frame, and :meth:`run` refuses a frame that has none.
        * ``PairDpd`` is a thermostat as well as a potential — its
          dissipative and random terms are what hold the temperature — so
          register it *instead of* ``NoseHoover`` / ``Langevin`` / ``CSVR``,
          and give it the same ``dt`` as the integrator. Nothing here
          refuses a second thermostat beside it; the result is simply
          double-thermostatted.
        """
        method_name = _EVALUATOR_DISPATCH.get(type(component))
        if method_name is None:
            raise TypeError(_unsupported_evaluator_message(component))
        getattr(self._cpp, method_name)(component)
        self._built = False
        return self

    def set_neighbor(self, component: object) -> MD:
        """Install the neighbor list (``Driver::set_neighbor``)."""
        if type(component) is not _cpp.NeighborList:
            raise TypeError(
                f"set_neighbor expects NeighborList, got {type(component).__name__}"
            )
        self._cpp.set_neighbor(component)
        self._built = False
        return self

    def set_thermo(self, every: int) -> MD:
        """Set thermo print cadence (``Driver::set_thermo(N)``)."""
        self._cpp.set_thermo(int(every))
        self._built = False
        return self

    def set_record(self, component: object) -> MD:
        """Install a recorder (``Driver::set_record``, plural).

        Two recorders are bound: ``Thermo(freq)``, which maps to the
        driver's ``set_thermo(freq)`` print cadence, and ``XYZ(filename,
        freq)``, the trajectory writer that binds at ``end_of_step`` and
        owns its own stride gate. An unsupported component raises
        ``TypeError`` naming both.
        """
        if type(component) in _RECORDER_TYPES:
            self._cpp.set_record(component)
            self._built = False
            return self
        supported = ", ".join(t.__name__ for t in _RECORDER_TYPES)
        raise TypeError(
            f"set_record does not support {type(component).__name__}. "
            f"Supported: {supported}"
        )

    def on(self, phase: str) -> PhaseBuilder:
        """Escape hatch: explicit phase binding.

        Prefer categorical setters. *phase* is one of:
        ``initial_integrate``, ``pre_force``, ``force``, ``post_force``,
        ``final_integrate``, ``end_of_step``.

        Raises ``RuntimeError`` once :meth:`run` has completed: ``on`` is the
        registration entry point, so the refusal lands here rather than at
        ``PhaseBuilder.add``.
        """
        cpp_phase = _PHASE_MAP.get(phase)
        if cpp_phase is None:
            valid = ", ".join(f'"{p}"' for p in _PHASE_MAP)
            raise ValueError(f"Unknown phase {phase!r}. Valid phases: {valid}")
        # The handle owns the already-run guard and is asked FIRST: a refused
        # registration must leave this object's build state untouched.
        cpp_builder = self._cpp.on(cpp_phase)
        self._built = False
        return PhaseBuilder(cpp_builder)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def build(self) -> None:
        """Compile the execution plan (called automatically by ``run``)."""
        self._cpp.build()
        self._built = True

    def run(self, frame: object, n_steps: int) -> None:
        """Run *n_steps* of MD on *frame*.

        *frame* is a ``molrs.Frame``. The C++ driver borrows it, materializes
        a transient execution frame, advances it, and writes
        positions / velocities / forces / energy back into *frame* in place.

        Raises ``RuntimeError`` before any compute if an electrostatic
        evaluator is registered and *frame* has no per-atom ``charge``
        column [e]: once the frame reaches C++ a missing column is
        indistinguishable from deliberate zeros, so it is caught here.
        Insert an explicit zero column to mean zero.

        The plan is compiled on the FIRST call only. A completed run freezes
        the configuration (every setter refuses afterwards), so calling
        ``run`` again re-runs that same compiled plan on the frame's current
        state — it never recompiles.
        """
        if not self._built:
            self.build()
        self._cpp.run(frame, n_steps)

    def enable_timing(self) -> None:
        """Enable per-component timing."""
        self._cpp.enable_timing()

    def summary(self) -> str:
        """Return a timing summary string."""
        return self._cpp.summary()

    def __repr__(self) -> str:
        state = "built" if self._built else "not built"
        return f"MD({state})"


# InitVelocity lives next to MD (cpu/dynamics/init.h); integrators,
# thermostats and barostats live in the sibling modules `md.integrator`,
# `md.thermostat` and `md.barostat`.
InitVelocity = _cpp.InitVelocity

__all__ = ["MD", "InitVelocity", "PhaseBuilder"]

"""OOP Molecular Dynamics driver — operates directly on a ``molrs.Frame``.

The driver borrows the ``molrs.Frame`` for the duration of :meth:`MD.run`:
the C++ side materializes a transient execution frame, advances it, and
writes positions / velocities / forces / energy back into the *same*
``molrs.Frame``. Energy, kinetic energy, conserved energy, and stress are
exact-dtype ``molrs.MetaValue`` entries in ``frame.meta``; they are never
synthetic blocks. Atomiverse never owns a separate Frame data type.
"""

from __future__ import annotations

from atomiverse._binding import _cpp

# ---------------------------------------------------------------------------
# Phase name → C++ enum
# ---------------------------------------------------------------------------
_PHASE_MAP: dict[str, object] = {
    "initial_integrate": _cpp.Phase.initial_integrate,
    "pre_force": _cpp.Phase.pre_force,
    "force": _cpp.Phase.force,
    "post_force": _cpp.Phase.post_force,
    "final_integrate": _cpp.Phase.final_integrate,
    "end_of_step": _cpp.Phase.end_of_step,
}

# ---------------------------------------------------------------------------
# Component dispatch tables
# ---------------------------------------------------------------------------
_DRIVER_DISPATCH: dict[type, str] = {
    _cpp.VelocityVerlet: "add_velocity_verlet",
    _cpp.InitVelocity: "add_init_velocity",
    _cpp.NoseHoover: "add_nose_hoover",
}

_PHASE_DISPATCH: dict[type, str] = {
    _cpp.PairLJCut: "add_pair_lj_cut",
    _cpp.NeighborList: "add_neighbor_list",
    _cpp.BondHarmonic: "add_bond_harmonic",
    _cpp.AngleHarmonic: "add_angle_harmonic",
    _cpp.DihedralMultiHarmonic: "add_dihedral_multi_harmonic",
    _cpp.Thermo: "add_thermo",
}


class PhaseBuilder:
    """Fluent builder returned by :meth:`MD.on`.

    Example::

        sim.on("force").add(av.NeighborList(cutoff=8.5), order=-10) \\
                       .add(av.PairLJCut(epsilon=0.01, sigma=3.4, cutoff=8.5))
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
        Timestep in fs.  If provided, a ``VelocityVerlet`` integrator
        is added automatically.
    device : str, optional
        Execution backend, PyTorch-style.  v1 accepts only ``"cpu"``;
        any other value raises ``ValueError`` before any compute.

    Example::

        import numpy as np
        import molrs
        import atomiverse as av

        frame = molrs.Frame()
        atoms = molrs.Block()
        atoms.insert("x", x); atoms.insert("y", y); atoms.insert("z", z)
        atoms.insert("mass", m); atoms.insert("type", t)
        frame["atoms"] = atoms

        sim = av.MD(dt=0.001)
        sim.add(av.InitVelocity(temperature=10.0, seed=42))
        sim.on("force").add(av.NeighborList(cutoff=8.5), order=-10)
        sim.on("force").add(av.PairLJCut(epsilon=0.01, sigma=3.4, cutoff=8.5))
        sim.run(frame, 1000)
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
            self.add(_cpp.VelocityVerlet(dt))

    # ------------------------------------------------------------------
    # Component registration
    # ------------------------------------------------------------------

    def add(self, component: object) -> MD:
        """Add a component that has default phase bindings.

        Supported: ``VelocityVerlet``, ``InitVelocity``, ``NoseHoover``.
        For phase-specific components use :meth:`on`.
        """
        method_name = _DRIVER_DISPATCH.get(type(component))
        if method_name is None:
            supported = ", ".join(t.__name__ for t in _DRIVER_DISPATCH)
            raise TypeError(
                f"Cannot add {type(component).__name__} directly. "
                f"Use sim.on(phase).add() instead.  "
                f"Driver-level components: {supported}"
            )
        getattr(self._cpp, method_name)(component)
        self._built = False
        return self

    def on(self, phase: str) -> PhaseBuilder:
        """Return a :class:`PhaseBuilder` for explicit phase binding.

        *phase* is one of: ``"initial_integrate"``, ``"pre_force"``,
        ``"force"``, ``"post_force"``, ``"final_integrate"``,
        ``"end_of_step"``.
        """
        cpp_phase = _PHASE_MAP.get(phase)
        if cpp_phase is None:
            valid = ", ".join(f'"{p}"' for p in _PHASE_MAP)
            raise ValueError(f"Unknown phase {phase!r}. Valid phases: {valid}")
        self._built = False
        return PhaseBuilder(self._cpp.on(cpp_phase))

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

        Parameters
        ----------
        frame : molrs.Frame
            System state. Must carry an ``atoms`` block with
            ``x, y, z, type`` columns (``mass`` required for MD). Global
            results are typed entries in ``frame.meta``.
        n_steps : int
            Number of MD steps to advance.
        """
        if not self._built:
            self.build()
        self._cpp.run(frame, n_steps)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def enable_timing(self) -> None:
        """Enable per-component timing."""
        self._cpp.enable_timing()

    def summary(self) -> str:
        """Return a timing summary string."""
        return self._cpp.summary()

    def __repr__(self) -> str:
        state = "built" if self._built else "not built"
        return f"MD({state})"


# InitVelocity's canonical home is ``atomiverse.md`` (cpu/md/init.h has no
# finer category); integrators and thermostats live in the sibling modules
# ``atomiverse.md.integrator`` / ``atomiverse.md.thermostat``.
InitVelocity = _cpp.InitVelocity

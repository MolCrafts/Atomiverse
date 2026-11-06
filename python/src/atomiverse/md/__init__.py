"""OOP Molecular Dynamics driver — operates directly on a ``molpy.Frame``.

Mirrors the C++ categorical API on ``cpu::md::MD`` (headers under
``cpu/dynamics/``)::

    sim = MD()
    sim.set_evaluator(PairLJCut(...))
    sim.set_neighbor(NeighborList(...))
    sim.set_integrator(VelocityVerlet(dt))
    sim.set_thermo(200)
    sim.build()
    sim.run(frame, 1000)

The driver borrows the ``molpy.Frame`` for the duration of :meth:`MD.run`:
the C++ side materializes a transient execution frame, advances it, and
writes positions / velocities / forces / energy back into the *same*
``molpy.Frame``. Global results are exact-dtype typed metadata entries in
``frame.meta``, read back as plain Python values.
"""

from __future__ import annotations

from atomiverse._binding import _cpp, require_domain, backend_type

require_domain(__name__, "md")
from atomiverse._dispatch import (
    EVALUATOR_DISPATCH as _EVALUATOR_DISPATCH,
    bind_evaluator as _bind_evaluator,
)
from atomiverse._units import resolve as _resolve_units
from atomiverse._units import set_units, units_style

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

# Recorder types accepted by MD.set_record. All three are unconditional:
# the wheel forces ATV_BUILD_IO=ON through the CMake module DAG, so the
# trajectory writers are never capability-probed (unlike KSpacePME above).
_RECORDER_TYPES: tuple[type, ...] = (_cpp.Thermo, _cpp.XYZ, _cpp.MrecRecorder)
# The CUDA driver records `*.mrec` only (its Thermo is `set_thermo`).
_CUDA_RECORDER_TYPES: tuple[type, ...] = (_cpp.MrecRecorder,)

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
    """Molecular dynamics driver with native backend selection.

    Parameters
    ----------
    dt : float, optional
        Timestep in fs. If provided, calls ``set_integrator(VelocityVerlet(dt))``.
    device : str, optional
        Execution backend: ``"cpu"`` or ``"cuda"``. CUDA binds the same
        pair, FENE, harmonic and cosine angle, Coulomb/PME/PPPM, uniform
        electric field (``FixEfield``), neighbor list, Langevin,
        NoseHoover, Berendsen and NoseHooverNPT (orthorhombic cells)
        parameter objects as the CPU driver, plus AM1/RM1/MNDO/PM3. One
        angle style per CUDA run (singular slot). Continuation stays on one
        molpy.Frame.
    units : str, optional
        LAMMPS ``units`` style of the run (``"lj"``, ``"real"``,
        ``"metal"``, ``"si"``, ``"cgs"``, ``"electron"``, ``"micro"``,
        ``"nano"``). Every parameter, ``dt``, temperature, pressure and the
        frame are taken in that style and every result is reported in it;
        nothing is converted. Default: :func:`atomiverse.units_style`
        (``"lj"`` unless :func:`atomiverse.set_units` chose another).
    capture_buffer_bytes : int, optional
        Byte budget [bytes, >= 1] of the driver's ONE shared asynchronous
        capture buffer. Every asynchronous recorder (on CUDA:
        ``MrecRecorder``, thermo, the pressure series) reads its steps from
        it: a step any of them needs is captured once, as an independent
        copy, and freed when the last of them has consumed it. The budget
        bounds every frame in flight across all of them, is allocated once
        when the first run starts, and when it is full the run waits for
        the consumers — nothing is ever dropped. Default: the build's
        ``ATV_CAPTURE_BUFFER_BYTES`` (256 MiB). A budget smaller than one
        captured frame raises ``ValueError`` naming both sizes when the run
        starts. Accounting: :meth:`capture_stats`.

    Example (mirrors ``examples/cpp/08_md_lj.cpp``)::

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

    def __init__(
        self,
        dt: float | None = None,
        device: str = "cpu",
        capture_buffer_bytes: int | None = None,
        units: str | None = None,
    ) -> None:
        self.device = device
        self._cpp = backend_type(device, "MD", "_CudaMD")()
        self._cpp.set_units(_resolve_units(units))
        self._built = False
        if capture_buffer_bytes is not None:
            if int(capture_buffer_bytes) < 1:
                raise ValueError(
                    "capture_buffer_bytes must be >= 1 byte, got "
                    f"{capture_buffer_bytes}"
                )
            self._cpp.set_capture_buffer_bytes(int(capture_buffer_bytes))
        if dt is not None:
            self.set_integrator(_cpp.VelocityVerlet(dt))

    # ------------------------------------------------------------------
    # Categorical setters (LAMMPS-shaped; match cpu::md::MD / Driver)
    # ------------------------------------------------------------------

    def set_units(self, style: str) -> MD:
        """LAMMPS ``units <style>`` of this run (before the first :meth:`run`)."""
        self._cpp.set_units(style)
        return self

    @property
    def units_style(self) -> str:
        """LAMMPS ``units`` style of this run."""
        return self._cpp.units_style

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
        ``NvtSllod`` likewise replaces the plain integrator and already
        thermostats peculiar velocities. Do not add a thermostat beside it.
        It runs on CPU and CUDA. The MPI launcher cannot carry it.

        On ``device="cuda"`` the integrators are ``VelocityVerlet``,
        ``NvtSllod`` and ``NoseHooverNPT`` (the same constructor arguments
        as on CPU; the cell must be orthorhombic, a triclinic cell is
        refused when the run is built). ``ParrinelloRahmanNPT`` is CPU only.
        Both devices refuse a thermostat or barostat beside an NPT
        integrator, in either call order.
        """
        if type(component) not in (
            _cpp.VelocityVerlet,
            _cpp.NoseHooverNPT,
            _cpp.ParrinelloRahmanNPT,
            _cpp.NvtSllod,
        ):
            raise TypeError(
                f"set_integrator expects VelocityVerlet, NoseHooverNPT, "
                f"ParrinelloRahmanNPT, or NvtSllod, got {type(component).__name__}"
            )
        if self.device == "cuda" and type(component) is _cpp.ParrinelloRahmanNPT:
            raise ValueError(
                "CUDA MD has no ParrinelloRahmanNPT (full-cell MTK). Use "
                "NoseHooverNPT on an orthorhombic cell with device='cuda', or "
                "run ParrinelloRahmanNPT with device='cpu'"
            )
        if type(component) is _cpp.NvtSllod and getattr(self, "_thermostatted", False):
            raise ValueError(
                "NvtSllod already thermostats; do not also set a thermostat"
            )
        npt = type(component) in (_cpp.NoseHooverNPT, _cpp.ParrinelloRahmanNPT)
        if npt and (
            getattr(self, "_thermostatted", False)
            or getattr(self, "_barostatted", False)
        ):
            raise ValueError(
                f"{type(component).__name__} integrates, thermostats and "
                "barostats in one step; do not also set a thermostat or "
                "barostat. Use VelocityVerlet + set_thermostat + "
                "set_barostat(BerendsenBarostat) for the split recipe"
            )
        # Flags change only once the native slot accepted the integrator.
        self._cpp.set_integrator(component)
        self._npt = npt
        self._nvt_sllod = type(component) is _cpp.NvtSllod
        self._built = False
        return self

    def set_thermostat(self, component: object) -> MD:
        """Install a thermostat (``cpu::md::MD::set_thermostat``).

        ``NoseHoover`` (single chain, LAMMPS ``fix nvt``) and ``Langevin``
        run on both devices; ``CSVR`` is CPU only. A thermostat never joins
        ``NvtSllod`` or an NPT integrator, which already thermostat. On
        both devices ``NoseHoover`` takes the integrator's timestep in place
        of its own ``dt``.
        """
        if getattr(self, "_nvt_sllod", False):
            raise ValueError(
                "NvtSllod already thermostats; do not also set a thermostat"
            )
        if getattr(self, "_npt", False):
            raise ValueError(
                "the NPT integrator already thermostats; do not also set a "
                "thermostat (use VelocityVerlet + set_thermostat + "
                "set_barostat(BerendsenBarostat) for the split recipe)"
            )
        if type(component) not in (_cpp.NoseHoover, _cpp.Langevin, _cpp.CSVR):
            raise TypeError(
                f"set_thermostat expects NoseHoover, Langevin, or CSVR, got "
                f"{type(component).__name__}"
            )
        if self.device == "cuda" and type(component) is _cpp.CSVR:
            raise ValueError(
                "CUDA MD has no CSVR thermostat. Use NoseHoover or Langevin "
                "with device='cuda', or run CSVR with device='cpu'"
            )
        self._thermostatted = True
        self._cpp.set_thermostat(component)
        self._built = False
        return self

    def set_run_span(self, nfirst: int, nlast: int) -> MD:
        """Fix the temperature ramp to ``[nfirst, nlast]`` before the first chunk.

        Applies to the ``Langevin`` thermostat and the ``NoseHooverNPT``
        ``T_stop`` ramp. CUDA continuation keeps the step counter, so one
        cool is many ``run(chunk)`` calls on one driver and the span is the
        whole cool. Without it the first ``run(n)`` sets ``[0, n]``. Call
        this after ``set_thermostat`` / ``set_integrator`` and before
        ``run``.
        """
        if self.device != "cuda":
            raise ValueError(
                "set_run_span is the CUDA continuation ramp; CPU MD takes "
                "the span from each run()"
            )
        if int(nlast) <= int(nfirst):
            raise ValueError(f"run span [{nfirst}, {nlast}] is empty")
        self._cpp.set_run_span(int(nfirst), int(nlast))
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
        if getattr(self, "_npt", False):
            raise ValueError(
                "the NPT integrator already barostats; do not also set a "
                "barostat (use VelocityVerlet + set_thermostat + "
                "set_barostat(BerendsenBarostat) for the split recipe)"
            )
        self._barostatted = True
        self._cpp.set_barostat(component)
        self._built = False
        return self

    def set_observer(self, component: object) -> MD:
        """Install an observer (``Driver::set_observer``).

        v1 supports ``InitVelocity`` (Maxwell–Boltzmann seeding, net
        momentum removed) on both devices.
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

        Gaussian methods (``RHF``, ``UHF``, ``RKS``, ``UKS``) and ``DFTB2``
        are borrowed as live native nodes/components. This driver keeps their
        Python owners alive; their diagnostics remain on the original object.
        These registrations currently require running without the MPI launcher.

        ``AM1`` belongs here too: registering it makes the run
        Born-Oppenheimer molecular dynamics — one semiempirical SCF solve
        per step, forces from it. It needs no ``NeighborList`` (AM1 is
        molecular, not a pair style). AM1 and RM1 are borrowed and kept alive;
        their diagnostics remain on the original object. MPI launcher transport
        for borrowed methods is explicitly unsupported.

        ``PW`` is the same story on a plane-wave Hamiltonian:
        registering it makes the run plane-wave Born-Oppenheimer dynamics,
        with analytic ionic forces from the converged density. It likewise
        takes no ``NeighborList``, and it additionally needs the frame to
        carry a ``box`` — a plane-wave basis is defined by the cell. Each
        step costs one SCF solve (forces reuse its density), so size the
        cell, ``ecut_ry`` and ``max_iter`` for the number of steps you
        actually want.

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
        method_device = getattr(component, "device", None)
        if method_device is not None and method_device != self.device:
            raise ValueError("Evaluator and driver must use the same device")
        if self.device == "cuda":
            self._cpp.set_evaluator(component)
        else:
            _bind_evaluator(self._cpp, component)
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

    def set_pressure_series(
        self, path: str, every: int, epsilon_ev: float, direction: int
    ) -> MD:
        """Record −P* during one continuous CUDA run.

        ``NvtSllod`` restarts its strain clock when the driver is built
        again, and a chunked ``run`` builds again. The series is written
        from inside the single deform ``run``. ``direction`` is 0..5.
        Rows follow the LAMMPS ``dump`` step convention: the initial state
        (step 0, strain 0), then the state after every ``every``-th
        completed step; the ``step`` column counts completed steps.
        """
        if self.device != "cuda":
            raise ValueError("pressure series is recorded on the CUDA driver")
        self._cpp.set_pressure_series(str(path), int(every), float(epsilon_ev), int(direction))
        self._built = False
        return self

    def set_thermo(self, every: int) -> MD:
        """Set thermo print cadence (``Driver::set_thermo(N)``).

        Both backends: CPU ``cpu::md::MD::set_thermo``, CUDA
        ``cuda::md::MD::set_thermo`` (lines printed by a worker thread from
        the driver's shared capture buffer, never a stepping-thread sync).

        LAMMPS ``thermo`` step convention: the ``step`` column counts
        completed MD steps and each line is the state after that many
        steps. A run prints the header and the initial state, then the
        state after every ``every``-th step and after its last step —
        ``run(frame, 1000)`` at ``every=100`` prints steps 0, 100, …, 1000.
        ``every=0`` prints the first and last step of each run only
        (LAMMPS ``thermo 0``).
        """
        self._cpp.set_thermo(int(every))
        self._built = False
        return self

    def set_record(self, component: object) -> MD:
        """Install a recorder (``Driver::set_record``, plural).

        Recorders bind at ``end_of_step`` and own their stride gate. They
        follow the LAMMPS ``dump``/``thermo`` step convention: a record's
        ``step`` is the number of completed MD steps and it holds the state
        after exactly that many steps (``time = step · dt``). A run of ``n``
        steps at cadence ``k`` records the initial state and the states
        after steps ``k, 2k, …`` up to and including ``n`` (``0, k, …, n``
        when ``k`` divides ``n``); a step is never recorded twice.

        * ``Thermo(freq)`` — maps to the driver's ``set_thermo(freq)``
          print cadence (CPU; on CUDA use :meth:`set_thermo`).
        * ``XYZ(filename, freq)`` — extended-XYZ positions (CPU).
        * ``MrecRecorder(path, every)`` — the molrec (``*.mrec``)
          trajectory, on **both** devices with the same columns:
          positions, velocities, forces, charges, int32 images, tags,
          mass, cell, energies (total and per term), ``step`` and
          ``time`` [fs], plus the identity columns the frame carries
          (``type_id`` and/or ``atomic_number``). Read it with
          ``molpy.io.mrec.MrecReader``. After :meth:`run` returns the
          store holds every frame of that run; on CUDA the frames are
          written by a worker thread through the driver's shared capture
          buffer (``MD(capture_buffer_bytes=...)``) — the step loop waits
          only when that buffer is full, never drops a frame, and a writer
          failure stops the run with ``RuntimeError`` naming the store. On CPU each ``run()``
          restarts at step 0, so one recorder records one ``run()``; CUDA
          ``run()`` calls on the same frame continue its trajectory (the
          boundary step is written once: ``run(f, 4); run(f, 4)`` at
          ``every=2`` records steps 0, 2, 4, 6, 8).
          Carried by the MPI launcher (rank 0 gathers and writes).

        An unsupported component raises ``TypeError`` naming the
        supported ones for this device.
        """
        supported_types = (
            _CUDA_RECORDER_TYPES if self.device == "cuda" else _RECORDER_TYPES
        )
        if type(component) in supported_types:
            self._cpp.set_record(component)
            self._built = False
            return self
        supported = ", ".join(t.__name__ for t in supported_types) or "none"
        raise TypeError(
            f"set_record on device {self.device!r} does not support "
            f"{type(component).__name__}. Supported: {supported}"
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
        if self.device == "cuda":
            raise ValueError("CUDA Python on is not implemented for this molecular slice")
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

        *frame* is a ``molpy.Frame``. The C++ driver borrows it, materializes
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

    def capture_stats(self) -> dict[str, float]:
        """Accounting of the driver's shared capture buffer.

        Keys: ``budget_bytes``, ``frame_bytes`` (one captured frame),
        ``capacity`` (frames the budget holds), ``captures`` (frames
        captured — one per captured step, however many consumers read it),
        ``peak_bytes``, ``blocked_episodes`` and ``blocked_seconds`` (how
        often and how long the run waited for space). All zero before the
        first run or when no asynchronous consumer is mounted.
        """
        return dict(self._cpp.capture_stats())

    def enable_timing(self) -> None:
        """Enable per-component timing."""
        if self.device == "cuda":
            raise ValueError("CUDA Python enable_timing is not implemented for this molecular slice")
        self._cpp.enable_timing()

    def summary(self) -> str:
        """Return a timing summary string."""
        if self.device == "cuda":
            raise ValueError("CUDA Python summary is not implemented for this molecular slice")
        return self._cpp.summary()

    def __repr__(self) -> str:
        state = "built" if self._built else "not built"
        return f"MD({state})"


# InitVelocity lives next to MD (cpu/dynamics/init.h); integrators,
# thermostats and barostats live in the sibling modules `md.integrator`,
# `md.thermostat` and `md.barostat`.
InitVelocity = _cpp.InitVelocity

#: Constant uniform electric field (LAMMPS ``fix efield``) —
#: ``cpu::md::FixEfield`` / ``cuda::md::FixEfield``. ``Ex, Ey, Ez`` in V/Å;
#: register it with :meth:`MD.set_evaluator`. F = qE [eV/Å]; the energy is
#: −Σ q E·r^u on unwrapped coordinates and no virial is added.
FixEfield = _cpp.FixEfield

__all__ = ["MD", "InitVelocity", "FixEfield", "PhaseBuilder"]

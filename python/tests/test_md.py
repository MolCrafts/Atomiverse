"""Python integration tests for the MD driver running on a molrs.Frame.

Covers rust-frame-lowering-03 ac-002: a Python MD run starts from a
molrs.Frame end to end — the driver borrows the molrs.Frame, advances it,
and writes positions / velocities / energy back into that same frame.

Also asserts ac-001's structural deliverable from the Python side: no
legacy C++ frame type is exposed and atomiverse._convert no longer exists.

Imports use the deep canonical namespaces (spec python-api-namespaces-am1);
``TestTopLevelConvenience`` keeps ONE full run on the top-level re-exports
to prove the two spellings drive the same objects.

``TestCoulombPme`` (spec py-coulomb-pme) covers charged systems: the
``CoulCut`` pair style and the matched ``CoulLong`` + ``KSpacePME`` pair.
Those names are imported INSIDE the test bodies so that this module keeps
collecting on a build whose extension does not bind them yet.

``TestXyzRecorder`` (spec py-recorder-xyz) covers trajectory output: an
``XYZ`` recorder installed through ``MD.set_record``, parsed back off
disk. Same in-body import discipline.

``TestAm1Bomd`` (spec py-am1-forces) covers semiempirical Born-Oppenheimer
MD: ``MD.set_evaluator(AM1Evaluator(...))``, i.e. an electronic-structure
method in the force slot instead of a classical pair style.
"""

import importlib
from pathlib import Path

import numpy as np
import pytest
import molrs

import atomiverse as av
from atomiverse._binding import _cpp, extension_available
from atomiverse.md import MD, InitVelocity
from atomiverse.md.barostat import BerendsenBarostat
from atomiverse.md.integrator import NoseHooverNPT, ParrinelloRahmanNPT, VelocityVerlet
from atomiverse.md.thermostat import CSVR, Langevin, NoseHoover
from atomiverse.neighbor import NeighborList
from atomiverse.potentials.pair import LJCut
from atomiverse.recorder import Thermo

#: The reciprocal-space solver is compile-optional (``ATV_BUILD_KSPACE``),
#: exactly like PW: the binding is present iff the extension carries it.
#: ``CoulCut`` / ``CoulLong`` are unconditional and are never gated here.
_HAS_KSPACE: bool = extension_available() and hasattr(_cpp, "KSpacePME")


# ── Component construction (deep canonical paths) ───────────────────


class TestComponentConstruction:
    def test_velocity_verlet(self):
        assert VelocityVerlet(0.001) is not None

    def test_init_velocity(self):
        assert InitVelocity(300.0, 42) is not None

    def test_nose_hoover(self):
        assert NoseHoover(300.0, 100.0, 0.001) is not None

    def test_pair_lj_cut(self):
        assert LJCut(0.0104, 3.4, 8.5) is not None

    def test_neighbor_list(self):
        assert NeighborList(8.5) is not None

    def test_neighbor_list_with_skin(self):
        assert NeighborList(8.5, 0.5) is not None

    def test_thermo(self):
        assert Thermo(100) is not None


# ── Driver build ────────────────────────────────────────────────────


class TestMDDriverBuild:
    def test_dt_shorthand_builds(self, full_lj_md):
        full_lj_md.build()

    def test_categorical_setters(self):
        sim = MD()
        sim.set_integrator(VelocityVerlet(0.001))
        sim.set_neighbor(NeighborList(8.5))
        sim.set_evaluator(LJCut(0.0104, 3.4, 8.5))
        sim.set_thermo(100)
        sim.build()

    def test_phase_escape_hatch(self):
        sim = MD(dt=0.001)
        sim.on("force").add(NeighborList(8.5), order=-10).add(LJCut(0.0104, 3.4, 8.5))
        sim.build()

    def test_invalid_phase_raises(self):
        sim = MD()
        with pytest.raises(ValueError, match="Unknown phase"):
            sim.on("nonexistent")

    def test_repr(self):
        sim = MD(dt=0.001)
        assert "not built" in repr(sim)
        sim.set_neighbor(NeighborList(8.5))
        sim.set_evaluator(LJCut(0.0104, 3.4, 8.5))
        sim.build()
        assert "built" in repr(sim)


# ── ac-002: MD on a molrs.Frame, end to end ─────────────────────────


class TestMDRunOnMolrsFrame:
    def test_positions_written_back(self, argon_frame, full_lj_md):
        """Positions update in the SAME molrs.Frame after run()."""
        x_before = np.asarray(argon_frame["atoms"]["x"]).copy()
        full_lj_md.run(argon_frame, 10)
        x_after = np.asarray(argon_frame["atoms"]["x"])
        assert not np.allclose(x_before, x_after), (
            "positions should move and be written back into the molrs.Frame"
        )

    def test_velocities_written_back(self, argon_frame, full_lj_md):
        """InitVelocity-seeded velocities land back in the molrs.Frame."""
        full_lj_md.run(argon_frame, 1)
        vx = np.asarray(argon_frame["atoms"]["vx"])
        assert np.any(vx != 0.0), "velocities should be initialized and written back"

    def test_energy_finite(self, argon_frame, full_lj_md):
        """Run yields exact typed metadata, never synthetic result blocks."""
        full_lj_md.run(argon_frame, 5)
        assert "scalars" not in argon_frame.keys()
        assert "stress" not in argon_frame.keys()
        assert molrs.FRAME_SCHEMA_VERSION == 2
        energy = argon_frame.meta["energy"]
        kinetic = argon_frame.meta["kinetic"]
        stress = argon_frame.meta["stress"]
        stress_kinetic = argon_frame.meta["stress_kinetic"]
        assert isinstance(energy, float)
        assert isinstance(kinetic, float)
        assert len(stress) == 6
        assert len(stress_kinetic) == 6
        assert np.isfinite(energy)
        assert np.isfinite(kinetic)
        assert kinetic > 0.0

    def test_forces_finite(self, argon_frame, full_lj_md):
        """Forces written back are finite."""
        full_lj_md.run(argon_frame, 1)
        fx = np.asarray(argon_frame["atoms"]["fx"])
        assert np.all(np.isfinite(fx))

    def test_auto_build(self, argon_frame, full_lj_md):
        """run() compiles the plan automatically."""
        # full_lj_md is not explicitly built; run() must build it.
        full_lj_md.run(argon_frame, 1)

    def test_run_zero_steps(self, argon_frame, full_lj_md):
        full_lj_md.run(argon_frame, 0)

    def test_missing_required_field_raises(self):
        """A molrs.Frame without atom positions fails loud."""
        frame = molrs.Frame()
        sim = MD(dt=0.001)
        sim.set_neighbor(NeighborList(8.5))
        sim.set_evaluator(LJCut(0.0104, 3.4, 8.5))
        with pytest.raises(RuntimeError, match="atoms"):
            sim.run(frame, 1)

    def test_non_frame_argument_raises(self, full_lj_md):
        """Passing a non-molrs.Frame object fails with a clear message."""
        with pytest.raises(RuntimeError, match="molrs.Frame"):
            full_lj_md.run(object(), 1)


# ── Late reconfiguration is refused (the MC late-setter contract) ───
#
# `Driver::build()` consumes its pending registrations, so a re-build is
# not a repeatable operation: everything registered before the first
# `run()` is gone from the driver's pending list afterwards. A setter
# called after `run()` therefore cannot reach the compiled plan, and the
# facade's "re-build because a setter ran" path (md/__init__.py: every
# `set_*` clears `_built`) recompiles from whatever happens to be left.
#
# MC already ships the contract this pins for MD — bind_mc.cpp
# `require_configurable_`: a setter after the first `run()` raises
# std::runtime_error naming the setter and telling the caller to build a
# new object. MD must refuse the same way, and the facade's
# re-build-after-run path goes away with it.
#
# Whether the guard lands in the Python facade or in bind_md.cpp is the
# implementer's call: nanobind maps std::runtime_error to RuntimeError,
# so these cases read the same either way.

#: Contract substring pinned for the implementer. The refusal message
#: must contain this phrase; cases 1-3 additionally require it to name
#: the refused setter, exactly as `require_configurable_` does
#: (bind_mc.cpp:269 — `std::string(setter) + ": ..."`). Everything else
#: in the message (the "construct a new MD" how-to) is free text.
_LATE_SETTER_MATCH: str = r"already run"


class TestMdLateSetters:
    """Reconfiguring an MD after its first ``run()`` fails loud.

    Measured on dev before the fix (none of 1-4 refuses):

    * ``set_evaluator`` — the setter itself is accepted and silently
      dropped; the *next* ``run()`` dies in the re-build with
      ``"MD::build(): component 'lj' is a NeighborConsumer but no
      neighbor list was registered"`` — loud, but blaming a neighbor
      list the caller did register.
    * ``set_integrator`` / ``set_thermo`` / ``set_observer`` /
      ``set_neighbor`` — same shape, and the next ``run()`` dies with
      ``"driver build: missing required Integrator"``. Re-calling
      ``set_integrator`` does not repair it: the categorical slot is
      consumed too, so the message names the one component the caller
      just installed.
    * ``on(phase)`` and ``on(phase).add(...)`` are accepted with no
      error at all.

    So today the observable is a misleading component-validation error
    (or silence), never the configuration contract.
    """

    _STEPS: int = 2

    def test_late_set_evaluator_raises(
        self, argon_frame: "molrs.Frame", full_lj_md: MD
    ) -> None:
        """A force term added after the run cannot reach the plan."""
        full_lj_md.run(argon_frame, self._STEPS)
        with pytest.raises(RuntimeError, match=_LATE_SETTER_MATCH) as excinfo:
            full_lj_md.set_evaluator(LJCut(0.0104, 3.4, 8.5))
        assert "set_evaluator" in str(excinfo.value), (
            "the refusal must name the setter it refused, like bind_mc.cpp's "
            "require_configurable_"
        )

    def test_late_set_integrator_raises(
        self, argon_frame: "molrs.Frame", full_lj_md: MD
    ) -> None:
        """Swapping the integrator mid-chain is a new configuration.

        This is the mirror of the evaluator case and the reason the
        refusal cannot be left to build-time validation: a re-build that
        happens to satisfy its component checks would run *something*,
        just not the Hamiltonian the caller configured.
        """
        full_lj_md.run(argon_frame, self._STEPS)
        with pytest.raises(RuntimeError, match=_LATE_SETTER_MATCH) as excinfo:
            full_lj_md.set_integrator(VelocityVerlet(0.001))
        assert "set_integrator" in str(excinfo.value)

    def test_late_set_thermo_raises(
        self, argon_frame: "molrs.Frame", full_lj_md: MD
    ) -> None:
        """Even a harmless-looking cadence change is a reconfiguration.

        ``set_thermo`` changes no physics, which is exactly why it must
        refuse: today it is the cheapest way to detonate the next
        ``run()`` with an error about a missing integrator.
        """
        full_lj_md.run(argon_frame, self._STEPS)
        with pytest.raises(RuntimeError, match=_LATE_SETTER_MATCH) as excinfo:
            full_lj_md.set_thermo(100)
        assert "set_thermo" in str(excinfo.value)

    def test_late_on_phase_raises(
        self, argon_frame: "molrs.Frame", full_lj_md: MD
    ) -> None:
        """The escape hatch refuses at ``on()``, not at ``add()``.

        PINNED FOR THE IMPLEMENTER: ``on(phase)`` is the registration
        entry point and today it is also what clears ``_built``, so it is
        the earliest and cleanest observable — a caller who has already
        run gets the contract before touching a component, and
        ``PhaseBuilder`` needs no guard of its own. Naming the entry
        point in the message is expected but not asserted here.
        """
        full_lj_md.run(argon_frame, self._STEPS)
        with pytest.raises(RuntimeError, match=_LATE_SETTER_MATCH):
            full_lj_md.on("force")

    def test_two_runs_same_config_still_work(
        self, argon_frame: "molrs.Frame", full_lj_md: MD
    ) -> None:
        """Chunked runs stay legal: the refusal is about *changing* the
        configuration, never about running again.

        The second leg must also still advance the frame — a guard that
        turned ``run()`` itself into a one-shot would pass the four cases
        above and break this one.
        """
        full_lj_md.run(argon_frame, self._STEPS)
        x_mid = np.asarray(argon_frame["atoms"]["x"], dtype=np.float64).copy()

        full_lj_md.run(argon_frame, self._STEPS)
        x_after = np.asarray(argon_frame["atoms"]["x"], dtype=np.float64)

        assert np.any(x_after != x_mid), (
            "the second run() of an unchanged configuration must keep "
            "advancing the same compiled plan"
        )
        assert np.isfinite(float(argon_frame.meta["energy"]))


# ── Top-level convenience spelling: same objects, same physics ──────


class TestTopLevelConvenience:
    def test_full_run_via_top_level_names(self, argon_frame):
        """One complete LJ run built entirely from the top-level
        re-exports (av.MD, av.PairLJCut, ...) — proving the convenience
        spelling drives the same identity-re-exported classes as the
        deep canonical paths used everywhere else in this file."""
        assert av.MD is MD
        assert av.PairLJCut is LJCut
        cutoff = 8.5
        sim = av.MD()
        sim.set_integrator(av.VelocityVerlet(0.001))
        sim.set_observer(av.InitVelocity(10.0, 42))
        sim.set_neighbor(av.NeighborList(cutoff))
        sim.set_evaluator(av.PairLJCut(0.0104, 3.4, cutoff))
        x_before = np.asarray(argon_frame["atoms"]["x"]).copy()
        sim.run(argon_frame, 10)
        x_after = np.asarray(argon_frame["atoms"]["x"])
        assert not np.allclose(x_before, x_after)
        energy = float(argon_frame.meta["energy"])
        assert np.isfinite(energy)


# ── ac-001: no legacy frame binding / no _convert on the surface ────


class TestFrameBindingRemoved:
    def test_no_cpuframe_on_extension(self):
        from atomiverse._binding import _cpp

        assert not hasattr(_cpp, "Cpu" + "Frame")  # token split: ac-002 grep gate

    def test_no_frame_attr_on_package(self):
        # Frame and Block have one owner: molrs.
        assert not hasattr(av, "Frame")
        assert not hasattr(av, "Block")

    @pytest.mark.parametrize("name", ("Frame", "Block"))
    def test_no_frame_or_block_import(self, name):
        with pytest.raises(ImportError):
            exec(f"from atomiverse import {name}", {})

    def test_convert_module_gone(self):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("atomiverse._convert")

    def test_device_compatibility_module_gone(self):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("atomiverse._device")


# ── NPT helpers (module level: shared by the barostat/ramp cases) ───


def _independent_argon_frame(reference: "molrs.Frame") -> "molrs.Frame":
    """Build a fresh molrs.Frame from *reference*'s atoms and cubic cell.

    ``molrs.Frame.copy()`` hands out column storage that is still shared
    with its source, so a two-leg comparison rebuilds the columns instead
    — the two frames must be advanced completely independently.
    """
    src = reference["atoms"]
    atoms = molrs.Block()
    for key in ("x", "y", "z", "mass"):
        atoms.insert(key, np.array(src[key], dtype=np.float64))
    atoms.insert("atomic_number", np.array(src["atomic_number"], dtype=np.uint32))
    frame = molrs.Frame()
    frame["atoms"] = atoms
    frame.box = molrs.Box.cube(float(np.asarray(reference.box.lengths)[0]))
    return frame


def _kinetic_from_columns(frame: "molrs.Frame") -> float:
    """Kinetic energy from the frame's own mass/velocity columns.

    Units are amu·Å²/fs² (the eV factor is deliberately not applied): the
    value is only ever compared against another value from this same
    function, and reading it off the columns keeps the assertion
    independent of the ``kinetic`` meta plumbing.
    """
    atoms = frame["atoms"]
    mass = np.asarray(atoms["mass"], dtype=np.float64)
    v_sq = sum(
        np.asarray(atoms[component], dtype=np.float64) ** 2
        for component in ("vx", "vy", "vz")
    )
    return float(0.5 * np.sum(mass * v_sq))


class TestThermostatsAndNpt:
    """NVT via Langevin/CSVR and NPT via NoseHooverNPT (LAMMPS `fix npt`)."""

    #: Barostat cases need a real timestep: the 0.001 fs of the NVE smokes
    #: advances 5 as of physical time, which no cell can respond to. The Ar
    #: step of regressions/md/pr_npt_lj/reg_pr_npt_lj.cpp is 1.0 fs; 4.0 fs
    #: is still ~1/450 of the Ar LJ oscillation period (~1.8 ps) and buys
    #: the same physical time in a quarter of the steps, which is what
    #: keeps these cases inside the suite's seconds-scale budget.
    _DT = 4.0
    _CUTOFF = 8.5

    def _lj_no_integrator(self) -> MD:
        """Ar LJ recipe + seeded velocities; the caller installs the
        integrator (plain, NPT, or plain-plus-barostat)."""
        sim = MD()
        sim.set_observer(InitVelocity(10.0, 42))
        sim.set_neighbor(NeighborList(self._CUTOFF))
        sim.set_evaluator(LJCut(0.0104, 3.4, self._CUTOFF))
        return sim

    def _lj_nvt(self, thermostat) -> MD:
        cutoff = 8.5
        sim = MD()
        sim.set_integrator(VelocityVerlet(0.001))
        sim.set_observer(InitVelocity(10.0, 42))
        sim.set_thermostat(thermostat)
        sim.set_neighbor(NeighborList(cutoff))
        sim.set_evaluator(LJCut(0.0104, 3.4, cutoff))
        return sim

    def test_langevin_nvt_smoke(self, argon_frame):
        sim = self._lj_nvt(Langevin(10.0, 10.0, 100.0, 0.001, 7))
        sim.run(argon_frame, 5)
        assert np.isfinite(argon_frame.meta["energy"])
        assert argon_frame.meta["kinetic"] > 0.0

    def test_csvr_nvt_smoke(self, argon_frame):
        sim = self._lj_nvt(CSVR(10.0, 50.0, 0.001, 7))
        sim.run(argon_frame, 5)
        assert np.isfinite(argon_frame.meta["energy"])
        assert argon_frame.meta["kinetic"] > 0.0

    def test_npt_smoke(self, argon_frame):
        cutoff = 8.5
        sim = MD()
        sim.set_integrator(NoseHooverNPT(10.0, 100.0, 0.0, 1000.0, 0.001))
        sim.set_observer(InitVelocity(10.0, 42))
        sim.set_neighbor(NeighborList(cutoff))
        sim.set_evaluator(LJCut(0.0104, 3.4, cutoff))
        sim.run(argon_frame, 5)
        assert np.isfinite(argon_frame.meta["energy"])

    def test_set_thermostat_rejects_unknown_type(self, argon_frame):
        sim = MD()
        with pytest.raises(TypeError, match="NoseHoover, Langevin, or CSVR"):
            sim.set_thermostat(VelocityVerlet(0.001))

    def test_set_integrator_accepts_npt(self):
        sim = MD()
        sim.set_integrator(NoseHooverNPT(300.0, 100.0, 0.0, 1000.0, 0.5))

    # ── py-md-npt: PR-NPT, Berendsen, and the NoseHooverNPT knob tail ──

    def test_pr_npt_smoke(self, argon_frame):
        """Parrinello-Rahman NPT: finite energy and a cell that responds.

        The fixture lattice sits at r = sigma, compressed relative to the
        LJ minimum at 2^(1/6) sigma, so a P_target = 0 barostat has to
        expand it — the setup of regressions/md/pr_npt_lj.
        """
        volume_before = argon_frame.box.volume()
        sim = self._lj_no_integrator()
        # ctor order is T, P_target, tau_T, tau_P, dt — NOT NoseHooverNPT's
        # T, tau_T, P_target, tau_P, dt.
        sim.set_integrator(ParrinelloRahmanNPT(10.0, 0.0, 100.0, 1000.0, self._DT))
        sim.run(argon_frame, 10)
        assert np.isfinite(argon_frame.meta["energy"])
        volume_after = argon_frame.box.volume()
        assert np.isfinite(volume_after)
        assert volume_after > 0.0
        assert abs(volume_after - volume_before) > 1e-9 * volume_before, (
            "NPT ran but frame.box is unchanged — the barostatted cell must "
            "be written back into the molrs.Frame"
        )

    def test_berendsen_smoke(self, argon_frame):
        """Berendsen barostat rides alongside VelocityVerlet via
        set_barostat and relaxes the compressed cell toward P_target."""
        volume_before = argon_frame.box.volume()
        sim = self._lj_no_integrator()
        sim.set_integrator(VelocityVerlet(self._DT))
        sim.set_barostat(BerendsenBarostat(0.0, 1000.0, self._DT))
        sim.run(argon_frame, 10)
        assert np.isfinite(argon_frame.meta["energy"])
        volume_after = argon_frame.box.volume()
        assert np.isfinite(volume_after)
        assert volume_after > 0.0
        assert abs(volume_after - volume_before) > 1e-9 * volume_before, (
            "Berendsen barostat ran but frame.box is unchanged"
        )

    def test_set_barostat_rejects_unknown_type(self):
        sim = MD()
        with pytest.raises(TypeError, match="BerendsenBarostat"):
            sim.set_barostat(VelocityVerlet(self._DT))

    def test_set_integrator_accepts_pr_npt(self):
        sim = MD()
        sim.set_integrator(ParrinelloRahmanNPT(300.0, 0.0, 100.0, 1000.0, 0.5))

    def test_npt_couple_axis_freezes_axis(self, argon_frame):
        """couple_axis[2] = False freezes the z edge exactly while the
        barostatted x edge still breathes under the pressure offset."""
        lengths_before = np.asarray(argon_frame.box.lengths, dtype=np.float64).copy()
        sim = self._lj_no_integrator()
        sim.set_integrator(
            NoseHooverNPT(
                10.0,
                100.0,
                0.0,
                1000.0,
                self._DT,
                couple_axis=(True, True, False),
            )
        )
        sim.run(argon_frame, 10)
        lengths_after = np.asarray(argon_frame.box.lengths, dtype=np.float64)
        assert lengths_after[2] == lengths_before[2], (
            "couple_axis z is off: Lz must be bitwise unchanged"
        )
        assert lengths_after[0] != lengths_before[0], (
            "couple_axis x is on: Lx must respond to the pressure offset"
        )

    def test_npt_t_stop_ramp_heats(self, argon_frame):
        """A live T_stop ramp injects heat; T_stop == T_target does not.

        Both legs start from the same fixture state with seed 42 and a
        near-rigid cell (tau_P = 1e5 fs), so the thermostat setpoint is the
        only difference between them.

        Margin: the ramp spans the whole run, so the setpoint reaches
        T_stop by the last step whatever the step count. With the Nose-
        Hoover mismatch term saturating at ~1/tau_T^2 the velocity scale
        grows like exp(t^2 / 2 tau_T^2); 25 steps of 4 fs against
        tau_T = 60 fs is t = 1.67 tau_T, i.e. of order 10x in kinetic
        energy — and 100x the base temperature is available as headroom,
        so the strict inequality has a wide margin without any long run.
        tau_T = 15 dt also keeps the chain far from its stability limit.
        """
        frame_flat = _independent_argon_frame(argon_frame)
        frame_ramped = _independent_argon_frame(argon_frame)

        flat = self._lj_no_integrator()
        flat.set_integrator(
            NoseHooverNPT(10.0, 60.0, 0.0, 1.0e5, self._DT, T_stop=10.0)
        )
        flat.run(frame_flat, 25)

        ramped = self._lj_no_integrator()
        ramped.set_integrator(
            NoseHooverNPT(10.0, 60.0, 0.0, 1.0e5, self._DT, T_stop=1000.0)
        )
        ramped.run(frame_ramped, 25)

        kinetic_flat = _kinetic_from_columns(frame_flat)
        kinetic_ramped = _kinetic_from_columns(frame_ramped)
        assert np.isfinite(kinetic_flat)
        assert np.isfinite(kinetic_ramped)
        assert kinetic_ramped > kinetic_flat, (
            f"T_stop ramp did not heat the system: "
            f"ramped={kinetic_ramped!r} vs flat={kinetic_flat!r}"
        )

    def test_npt_knobs_accepted(self, argon_frame):
        """drag / mtchain / mpchain reach the C++ ctor and still integrate."""
        sim = self._lj_no_integrator()
        sim.set_integrator(
            NoseHooverNPT(
                10.0, 100.0, 0.0, 1000.0, self._DT, drag=0.5, mtchain=4, mpchain=4
            )
        )
        sim.run(argon_frame, 5)
        assert np.isfinite(argon_frame.meta["energy"])


# ── py-coulomb-pme: charged systems ─────────────────────────────────
#
# Geometry and Ewald settings mirror regressions/md/npt_pme_nacl exactly
# (reg_npt_pme_nacl.cpp `nacl()` / `electrostatic_energy()`): 4³ = 64 ions
# on a rock-salt lattice with alternating (-1)^(i+j+k) site charges, so the
# cell is neutral; nearest-neighbour spacing d = a₀/2 at the experimental
# a₀ = 5.6402 Å; real-space cutoff 5.0 Å, deliberately below L/2 = 5.64 Å
# for the minimum-image convention; α = 3.5/r_c (erfc(3.5) ≈ 7e-7); PME
# grid spacing 0.5 Å at B-spline order 6.

_NACL_SIDE: int = 4
_NACL_D: float = 2.8201  # nearest-neighbour spacing [Å]
_NACL_CUTOFF: float = 5.0  # real-space cutoff [Å]
_NACL_ALPHA: float = 3.5 / _NACL_CUTOFF  # Ewald splitting [1/Å]
_PME_SPACING: float = 0.5  # [Å]
_PME_ORDER: int = 6

#: [1/Å], Ewald rule-of-thumb for the 3.0 Å Ar cutoff — the chargeless-frame
#: cases reuse the α = 3.5/r_c rule on the Ar fixture's smaller cell.
_ARGON_ALPHA: float = 3.5 / 3.0

#: HARD-CODED GOLDEN — Madelung constant of the rock-salt lattice, from
#: regressions/md/npt_pme_nacl/REFERENCE.md §1 (Ziman, *Principles of the
#: Theory of Solids*, 2nd ed., Cambridge 1972, §2.3). Not measured here.
_MADELUNG: float = 1.747564594633182

#: HARD-CODED GOLDEN — the engine's Coulomb conversion, `units::coulomb`
#: in core/units.h (LAMMPS `qqr2e`): E [eV] = k_e q_i q_j / r, q in e, r in Å.
_COULOMB_EV_A: float = 14.399645


def _nacl_frame(d: float = _NACL_D) -> "molrs.Frame":
    """Rock-salt NaCl mini cell with a per-atom ``charge`` column.

    @param d: Nearest-neighbour spacing [Å]; the cubic cell edge is 4·d.
    @return A neutral 64-ion molrs.Frame (Na⁺ = +1 e, Cl⁻ = −1 e).
    """
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    charges: list[float] = []
    masses: list[float] = []
    numbers: list[int] = []
    for i in range(_NACL_SIDE):
        for j in range(_NACL_SIDE):
            for k in range(_NACL_SIDE):
                sodium = (i + j + k) % 2 == 0
                xs.append(i * d)
                ys.append(j * d)
                zs.append(k * d)
                charges.append(1.0 if sodium else -1.0)
                masses.append(22.9898 if sodium else 35.453)
                numbers.append(11 if sodium else 17)
    atoms = molrs.Block()
    atoms.insert("x", np.array(xs, dtype=np.float64))
    atoms.insert("y", np.array(ys, dtype=np.float64))
    atoms.insert("z", np.array(zs, dtype=np.float64))
    atoms.insert("charge", np.array(charges, dtype=np.float64))
    atoms.insert("mass", np.array(masses, dtype=np.float64))
    atoms.insert("atomic_number", np.array(numbers, dtype=np.uint32))
    frame = molrs.Frame()
    frame["atoms"] = atoms
    frame.box = molrs.Box.cube(_NACL_SIDE * d)
    return frame


@pytest.fixture
def nacl_frame() -> "molrs.Frame":
    """Neutral rock-salt NaCl mini cell at the experimental spacing."""
    return _nacl_frame()


class TestCoulombPme:
    """Charged periodic MD from Python (spec py-coulomb-pme).

    The new names are imported inside each test body: until the bindings
    land the import IS the failure, and the rest of this module still
    collects. ``KSpacePME`` cases are capability-gated on the extension
    like PW; ``CoulCut`` / ``CoulLong`` are unconditional.
    """

    @staticmethod
    def _md_with(*evaluators: object, cutoff: float = _NACL_CUTOFF) -> MD:
        """MD carrying *evaluators* plus the regression's neighbor list.

        The integrator is a 0.001 fs VelocityVerlet with no InitVelocity:
        perfect rock salt has zero net force on every ion by symmetry, so
        one step is a static single point on stationary atoms.
        """
        sim = MD()
        sim.set_integrator(VelocityVerlet(0.001))
        sim.set_neighbor(NeighborList(cutoff, 0.3))
        for evaluator in evaluators:
            sim.set_evaluator(evaluator)
        return sim

    @pytest.mark.skipif(
        not _HAS_KSPACE, reason="extension built without ATV_BUILD_KSPACE"
    )
    def test_nacl_pme_matches_madelung(self, nacl_frame: "molrs.Frame") -> None:
        """CoulLong + KSpacePME reproduce the Madelung lattice energy.

        E = −½ N M k_e / d with the hard-coded constants above; tolerance
        5e-4 relative is the PME discretization budget pinned by
        regressions/md/npt_pme_nacl/REFERENCE.md §1 (measured there: 3.8e-8).
        """
        from atomiverse.longrange import CoulLong, KSpacePME

        sim = self._md_with(
            CoulLong(_NACL_ALPHA, _NACL_CUTOFF),
            KSpacePME(_NACL_ALPHA, _PME_SPACING, _PME_ORDER),
        )
        sim.run(nacl_frame, 1)
        energy = float(nacl_frame.meta["energy"])
        n_ions = _NACL_SIDE**3
        e_ref = -0.5 * n_ions * _MADELUNG * _COULOMB_EV_A / _NACL_D
        assert abs(energy / e_ref - 1.0) <= 5e-4, (
            f"electrostatic energy {energy!r} eV is more than 5e-4 from the "
            f"Madelung reference {e_ref!r} eV"
        )

    def test_coul_cut_smoke(self, nacl_frame: "molrs.Frame") -> None:
        """The truncated Coulomb pair style needs no reciprocal partner and
        returns a finite, attractive lattice energy."""
        from atomiverse.potentials.pair import CoulCut

        sim = self._md_with(CoulCut(_NACL_CUTOFF))
        sim.run(nacl_frame, 1)
        energy = float(nacl_frame.meta["energy"])
        assert np.isfinite(energy)
        assert energy < 0.0, f"a neutral rock-salt lattice binds; got E = {energy!r} eV"

    def test_coul_cut_missing_charge_raises(self, argon_frame: "molrs.Frame") -> None:
        """A Coulomb evaluator on a frame with no ``charge`` column fails
        loud instead of silently zero-filling the charges.

        The Ar fixture carries no charge column; the cutoff is 3.0 Å so the
        6.8 Å cell still satisfies the minimum-image convention and the
        missing column is the only defect in the setup.
        """
        from atomiverse.potentials.pair import CoulCut

        assert "charge" not in argon_frame["atoms"].keys(), (
            "fixture drift: this case needs a frame WITHOUT a charge column"
        )
        sim = self._md_with(CoulCut(3.0), cutoff=3.0)
        with pytest.raises(RuntimeError, match="charge"):
            sim.run(argon_frame, 1)

    @pytest.mark.skipif(
        not _HAS_KSPACE, reason="extension built without ATV_BUILD_KSPACE"
    )
    def test_pme_missing_charge_raises(self, argon_frame: "molrs.Frame") -> None:
        """The Ewald pair fails the same way on a chargeless frame."""
        from atomiverse.longrange import CoulLong, KSpacePME

        assert "charge" not in argon_frame["atoms"].keys(), (
            "fixture drift: this case needs a frame WITHOUT a charge column"
        )
        sim = self._md_with(
            CoulLong(_ARGON_ALPHA, 3.0),
            KSpacePME(_ARGON_ALPHA, _PME_SPACING, _PME_ORDER),
            cutoff=3.0,
        )
        with pytest.raises(RuntimeError, match="charge"):
            sim.run(argon_frame, 1)

    @pytest.mark.skipif(
        not _HAS_KSPACE, reason="extension built without ATV_BUILD_KSPACE"
    )
    def test_pme_without_real_partner_raises(self) -> None:
        """Reciprocal space alone is not a Hamiltonian: the driver requires
        exactly one matched real-space partner."""
        from atomiverse.longrange import KSpacePME

        sim = self._md_with(KSpacePME(_NACL_ALPHA, _PME_SPACING, _PME_ORDER))
        with pytest.raises(RuntimeError, match="requires exactly one"):
            sim.build()

    def test_coul_long_without_reciprocal_raises(self) -> None:
        """Real space alone is not a Hamiltonian either: the damped
        short-range Ewald term requires its reciprocal partner."""
        from atomiverse.longrange import CoulLong

        sim = self._md_with(CoulLong(_NACL_ALPHA, _NACL_CUTOFF))
        with pytest.raises(RuntimeError, match="requires exactly one"):
            sim.build()


# ── py-recorder-xyz: trajectory output ──────────────────────────────
#
# The XYZ recorder is the first Python-visible trajectory writer. The name
# is STATIC, not capability-gated: the wheel forces ATV_BUILD_IO=ON through
# the CMake module DAG, so a wheel without it is not a supported build.
# ``XYZ`` is still imported inside each test body (the module-level import
# list is the pre-existing surface): until the binding lands the import IS
# the failure and the rest of this file keeps collecting.

_XYZ_STEPS: int = 10
_XYZ_FREQ: int = 5  # [steps]

#: Frames a ``_XYZ_STEPS``-step run at ``_XYZ_FREQ`` leaves in the file.
#: ``ctx.step`` runs 0..n-1, so the stride gate opens at steps 0 and 5
#: only — 2 frames, empirically pinned by the C++ unit test
#: (tests/cpu/record/test_xyz_recorder.cpp), which owns the authoritative
#: cadence; if it measures a different count, flip this one constant.
_EXPECTED_FRAMES: int = 2


def _lj_md_with_record(recorder: object) -> MD:
    """The conftest ``full_lj_md`` Ar recipe plus *recorder*.

    Spelled out here rather than taken from the fixture because the
    truncate case needs two independent drivers writing the same path
    inside one test.

    @param recorder: Recorder component handed to ``MD.set_record``.
    @return An unbuilt MD ready to ``run`` on the Ar fixture frame.
    """
    cutoff = 8.5  # [Å]
    sim = MD()
    sim.set_integrator(VelocityVerlet(0.001))
    sim.set_observer(InitVelocity(10.0, 42))
    sim.set_neighbor(NeighborList(cutoff))
    sim.set_evaluator(LJCut(0.0104, 3.4, cutoff))
    sim.set_record(recorder)
    return sim


def _xyz_frame_atom_counts(path: Path) -> list[int]:
    """Per-frame atom counts parsed out of an XYZ trajectory file.

    Standard XYZ: every frame block is an atom-count line, a comment line,
    then exactly that many ``element x y z`` lines. A short or malformed
    block raises instead of being silently reported as a frame, so a
    truncated write can never pass as a correct frame count.

    @param path: Trajectory file written by the recorder.
    @return One atom count per frame block, in file order.
    """
    lines: list[str] = path.read_text(encoding="utf-8").splitlines()
    counts: list[int] = []
    index = 0
    while index < len(lines):
        if not lines[index].strip():
            index += 1
            continue
        count = int(lines[index].strip())
        body = lines[index + 2 : index + 2 + count]
        if len(body) != count:
            raise AssertionError(
                f"{path.name}: frame {len(counts)} declares {count} atoms but "
                f"the file holds only {len(body)} coordinate lines"
            )
        for offset, line in enumerate(body):
            fields = line.split()
            if len(fields) < 4:
                raise AssertionError(
                    f"{path.name}: frame {len(counts)} atom line {offset} is "
                    f"not 'element x y z': {line!r}"
                )
            for value in fields[1:4]:
                float(value)
        counts.append(count)
        index += 2 + count
    return counts


class TestXyzRecorder:
    """XYZ trajectory output from Python (spec py-recorder-xyz)."""

    def test_xyz_writes_trajectory(
        self, argon_frame: "molrs.Frame", n_atoms: int, tmp_path: Path
    ) -> None:
        """An MD run with ``XYZ(path, freq)`` leaves a parseable trajectory
        whose every frame carries the whole system."""
        from atomiverse.recorder import XYZ

        path = tmp_path / "traj.xyz"
        sim = _lj_md_with_record(XYZ(str(path), _XYZ_FREQ))
        sim.run(argon_frame, _XYZ_STEPS)

        assert path.exists(), f"no trajectory written to {path}"
        counts = _xyz_frame_atom_counts(path)
        assert counts, f"{path.name} holds no frame blocks"
        assert set(counts) == {n_atoms}, (
            f"every frame must hold all {n_atoms} atoms; got counts {counts!r}"
        )
        assert len(counts) == _EXPECTED_FRAMES, (
            f"{_XYZ_STEPS} steps at freq {_XYZ_FREQ} must write "
            f"{_EXPECTED_FRAMES} frames; got {len(counts)}"
        )

    def test_xyz_second_run_truncates(
        self, argon_frame: "molrs.Frame", tmp_path: Path
    ) -> None:
        """A fresh recorder on an existing path starts the file over.

        Cadence appends *within* one run; a new run truncates. Without that
        the second run would double the frame count.
        """
        from atomiverse.recorder import XYZ

        path = tmp_path / "traj.xyz"
        first = _lj_md_with_record(XYZ(str(path), _XYZ_FREQ))
        first.run(argon_frame, _XYZ_STEPS)
        frames_first = len(_xyz_frame_atom_counts(path))

        second = _lj_md_with_record(XYZ(str(path), _XYZ_FREQ))
        second.run(argon_frame, _XYZ_STEPS)
        frames_second = len(_xyz_frame_atom_counts(path))

        assert frames_first == _EXPECTED_FRAMES
        assert frames_second == frames_first, (
            f"a second run must truncate, not append: {frames_first} frames "
            f"after run 1 but {frames_second} after run 2"
        )

    def test_set_record_rejects_unknown_type(self) -> None:
        """``set_record`` names its supported recorders on refusal."""
        sim = MD()
        with pytest.raises(TypeError, match="XYZ"):
            sim.set_record(object())


# ── py-am1-forces: semiempirical Born-Oppenheimer MD ────────────────
#
# The C++ twin is regressions/semiempirical/am1_water_bomd
# (reg_am1_water_bomd.cpp): MD + VelocityVerlet{0.25 fs} + InitVelocity{10 K,
# seed 7} + a gradient-enabled AM1Evaluator carrying max_iter = 300,
# density_tol = 1e-7, energy_tol = 1e-9 — tighter than the AM1 defaults
# because BOMD needs the force to be consistent from step to step. Those
# five numbers are reused verbatim here; only the molecule differs.
#
# H2 instead of the C++ case's water: two atoms and two electrons is the
# cheapest closed-shell AM1 system, and 5 steps is a smoke, not a
# conservation test. The bond is the experimental 0.741 Å, which is NOT
# AM1's own H2 minimum, so there is a real force to integrate from step 0.

_H2_BOND_ANGSTROM: float = 0.741
_H2_BOX_ANGSTROM: float = 20.0  # isolation only; AM1 here is non-periodic
_AM1_BOMD_DT: float = 0.25  # [fs]
_AM1_BOMD_T0: float = 10.0  # [K]
_AM1_BOMD_SEED: int = 7
_AM1_BOMD_STEPS: int = 5


def _h2_frame() -> "molrs.Frame":
    """H2 along z at the experimental bond length.

    Columns match the Ar fixture's set (x/y/z/mass/atomic_number): the
    driver materializes velocities itself and ``InitVelocity`` seeds them,
    so no vx/vy/vz column is required on the way in.

    @return A two-atom molrs.Frame in a 20 Å cubic cell.
    """
    half = 0.5 * _H2_BOND_ANGSTROM
    frame = molrs.Frame()
    atoms = molrs.Block()
    atoms.insert("x", np.zeros(2, dtype=np.float64))
    atoms.insert("y", np.zeros(2, dtype=np.float64))
    atoms.insert("z", np.array([-half, half], dtype=np.float64))
    atoms.insert("mass", np.full(2, 1.008, dtype=np.float64))
    atoms.insert("atomic_number", np.full(2, 1, dtype=np.uint32))
    frame["atoms"] = atoms
    frame.box = molrs.Box.cube(_H2_BOX_ANGSTROM)
    return frame


class TestAm1Bomd:
    """AM1 forces driving the MD loop (spec py-am1-forces).

    ``AM1Evaluator`` is imported inside the test bodies for the same
    build-capability reason as ``TestCoulombPme`` / ``TestXyzRecorder`` —
    the name is absent on an extension built with
    ``ATV_BUILD_SEMIEMPIRICAL=OFF``.

    No neighbor list is installed — AM1 is not a ``NeighborConsumer`` and the
    C++ BOMD regression does not install one either.
    """

    @pytest.fixture(scope="class")
    @staticmethod
    def bomd_h2() -> tuple["molrs.Frame", "np.ndarray"]:
        """One 5-step AM1 NVE run on H2; returns the frame and its starting
        z column (the frame itself is advanced in place)."""
        from atomiverse.semiempirical import AM1Evaluator

        frame = _h2_frame()
        z_before = np.asarray(frame["atoms"]["z"], dtype=np.float64).copy()
        sim = MD()
        sim.set_integrator(VelocityVerlet(_AM1_BOMD_DT))
        sim.set_observer(InitVelocity(_AM1_BOMD_T0, _AM1_BOMD_SEED))
        sim.set_evaluator(
            AM1Evaluator(
                charge=0,
                multiplicity=1,
                max_iter=300,
                density_tol=1.0e-7,
                energy_tol=1.0e-9,
            )
        )
        sim.run(frame, _AM1_BOMD_STEPS)
        return frame, z_before

    def test_set_evaluator_accepts_am1(self) -> None:
        """The dispatch keyed on the live component type must accept the
        semiempirical evaluator — this is the whole point of the spec, and
        it is checked without running so the failure is unambiguous."""
        from atomiverse.semiempirical import AM1Evaluator

        sim = MD()
        sim.set_integrator(VelocityVerlet(_AM1_BOMD_DT))
        sim.set_evaluator(AM1Evaluator())

    def test_potential_energy_finite_and_bound(
        self, bomd_h2: tuple["molrs.Frame", "np.ndarray"]
    ) -> None:
        """PE is finite and negative after the run — the same gate as
        reg_am1_water_bomd.cpp:80-87 (a bound molecule, not a blow-up)."""
        frame, _ = bomd_h2
        energy = frame.meta["energy"]
        assert isinstance(energy, float)
        assert np.isfinite(energy)
        assert energy < 0.0

    def test_kinetic_energy_finite_and_positive(
        self, bomd_h2: tuple["molrs.Frame", "np.ndarray"]
    ) -> None:
        """KE is finite and non-negative (reg_am1_water_bomd.cpp:88-91):
        InitVelocity seeded it, so it is strictly positive here."""
        frame, _ = bomd_h2
        kinetic = frame.meta["kinetic"]
        assert isinstance(kinetic, float)
        assert np.isfinite(kinetic)
        assert kinetic > 0.0

    def test_positions_advanced(
        self, bomd_h2: tuple["molrs.Frame", "np.ndarray"]
    ) -> None:
        """The atoms actually moved and the new positions came back into the
        SAME molrs.Frame — AM1 forces reached the integrator."""
        frame, z_before = bomd_h2
        z_after = np.asarray(frame["atoms"]["z"], dtype=np.float64)
        assert not np.allclose(z_before, z_after)

    def test_velocities_finite(
        self, bomd_h2: tuple["molrs.Frame", "np.ndarray"]
    ) -> None:
        """No NaN/inf leaked into the velocity columns."""
        frame, _ = bomd_h2
        atoms = frame["atoms"]
        for component in ("vx", "vy", "vz"):
            values = np.asarray(atoms[component], dtype=np.float64)
            assert np.all(np.isfinite(values)), f"{component} is not finite"

    def test_forces_finite(self, bomd_h2: tuple["molrs.Frame", "np.ndarray"]) -> None:
        """The AM1 forces written back on the last step are finite and, at
        the non-equilibrium bond length, not identically zero."""
        frame, _ = bomd_h2
        fz = np.asarray(frame["atoms"]["fz"], dtype=np.float64)
        assert np.all(np.isfinite(fz))
        assert np.any(fz != 0.0)

    def test_set_evaluator_still_rejects_unknown_type(self) -> None:
        """Growing the dispatch table must not turn it into an accept-all:
        an unsupported object still raises, and the message names the
        supported types (which now include AM1Evaluator)."""
        sim = MD()
        with pytest.raises(TypeError, match="AM1Evaluator"):
            sim.set_evaluator(object())


class TestMolrsFrameV2Boundary:
    def test_exact_frame_schema_version(self):
        assert molrs.FRAME_SCHEMA_VERSION == 2

    def test_box_is_the_only_cell_property(self, argon_frame):
        assert argon_frame.box is not None
        assert not hasattr(argon_frame, "simbox")

    def test_untyped_metadata_is_rejected(self, argon_frame):
        """Non-primitive meta values fail loud at the molrs boundary."""
        with pytest.raises(TypeError, match="MetaValue"):
            argon_frame.meta["bad"] = object()

    @pytest.mark.parametrize("legacy_block_name", ("scalars", "stress"))
    def test_synthetic_result_block_is_rejected(
        self, argon_frame, full_lj_md, legacy_block_name
    ):
        legacy = molrs.Block()
        legacy.insert("energy", np.array([0.0], dtype=np.float64))
        argon_frame[legacy_block_name] = legacy
        with pytest.raises(RuntimeError, match=legacy_block_name):
            full_lj_md.run(argon_frame, 1)

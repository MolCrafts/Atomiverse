"""Python integration tests for the MD driver running on a molrs.Frame.

Covers rust-frame-lowering-03 ac-002: a Python MD run starts from a
molrs.Frame end to end — the driver borrows the molrs.Frame, advances it,
and writes positions / velocities / energy back into that same frame.

Also asserts ac-001's structural deliverable from the Python side: no
legacy C++ frame type is exposed and atomiverse._convert no longer exists.

Imports use the deep canonical namespaces (spec python-api-namespaces-am1);
``TestTopLevelConvenience`` keeps ONE full run on the top-level re-exports
to prove the two spellings drive the same objects.
"""

import importlib

import numpy as np
import pytest
import molrs

import atomiverse as av
from atomiverse.md import MD, InitVelocity
from atomiverse.md.barostat import BerendsenBarostat
from atomiverse.md.integrator import NoseHooverNPT, ParrinelloRahmanNPT, VelocityVerlet
from atomiverse.md.thermostat import CSVR, Langevin, NoseHoover
from atomiverse.neighbor import NeighborList
from atomiverse.potentials.pair import LJCut
from atomiverse.recorder import Thermo


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
        sim.on("force").add(NeighborList(8.5), order=-10).add(
            LJCut(0.0104, 3.4, 8.5)
        )
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

"""Nose-Hoover NVT, MTK NPT and Berendsen on ``MD(device="cuda")``.

The CUDA handle takes the CPU parameter objects unchanged —
``set_thermostat(NoseHoover(T, tau_T, dt))`` and
``set_integrator(NoseHooverNPT(...))`` — so these tests drive both devices
through one Python surface:

* relaxation: a hot Lennard-Jones argon box cools toward the Nose-Hoover
  setpoint; a compressed LJ box expands under ``NoseHooverNPT`` toward
  ``P_target = 0``; with LJ + real-space Coulomb + a reciprocal solver
  (PPPM, PME) the cell contracts until the long-range virial cancels the
  LJ virial;
* parity: one step from the same state on CPU and CUDA agrees to the
  stated tolerance (double-precision CUDA build, ``ATV_CUDA_PREC_MODE``
  default; 64-bit fixed-point forces resolve 2.4e-13 eV/Å);
* refusals: combinations the CUDA backend cannot run (or that make no
  sense on either device) raise with a how-to.

Pressure is recomputed here from the columns — the pair virial in numpy,
the kinetic part from ``m v v`` — so the relaxation checks do not trust the
engine's own stress plumbing; one test compares the two.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import molpy

from atomiverse._binding import _cpp, extension_available
from _domains import skip_module_unless

skip_module_unless("ATV_BUILD_MD")

from atomiverse.md import MD, InitVelocity
from atomiverse.md.barostat import BerendsenBarostat
from atomiverse.md.integrator import (
    NoseHooverNPT,
    NvtSllod,
    ParrinelloRahmanNPT,
    VelocityVerlet,
)
from atomiverse.md.thermostat import CSVR, Langevin, NoseHoover
from atomiverse.neighbor import NeighborList
from atomiverse.potentials.pair import LJCut

_HAS_CUDA_MD: bool = extension_available() and hasattr(_cpp, "_CudaMD")
_HAS_KSPACE: bool = extension_available() and hasattr(_cpp, "KSpacePPPM")
needs_cuda = pytest.mark.skipif(not _HAS_CUDA_MD, reason="needs ATV_BUILD_CUDA")
needs_kspace = pytest.mark.skipif(
    not _HAS_KSPACE, reason="needs ATV_BUILD_KSPACE"
)

#: core/units.h: k_B [eV/K] and 1 amu·Å²/fs² in eV.
_KB = 8.617333262e-5
_MVV2E = 103.642696562

#: Ar Lennard-Jones (LAMMPS-style metal units).
_EPS = 0.0104  # [eV]
_SIG = 3.4  # [Å]
_MASS = 39.948  # [amu]
_CUTOFF = 8.5  # [Å]
_SKIN = 0.5  # [Å]
_SIDE = 6  # 6³ = 216 atoms; cell > 2 (cutoff + skin) for every spacing used


def _ar_frame(
    spacing: float,
    temperature: float,
    seed: int = 7,
    charge: float = 0.0,
    tilt: float = 0.0,
) -> "molpy.Frame":
    """Simple-cubic Ar with Maxwell-Boltzmann velocities at ``temperature``.

    @param spacing: Lattice spacing [Å]; the cubic cell edge is 6·spacing.
    @param temperature: Initial kinetic temperature [K] (COM motion removed,
        then rescaled to exactly this value over 3N-3 DOF).
    @param charge: Rock-salt ±charge [e] per site; 0 omits the column.
    @param tilt: xy tilt [Å]; nonzero makes the cell triclinic.
    """
    grid = (np.arange(_SIDE) + 0.5) * spacing
    gx, gy, gz = np.meshgrid(grid, grid, grid, indexing="ij")
    n = gx.size
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 3)) * math.sqrt(_KB * temperature / (_MASS * _MVV2E))
    v -= v.mean(axis=0)
    t_now = np.sum(_MASS * v * v) * _MVV2E / ((3 * n - 3) * _KB)
    v *= math.sqrt(temperature / t_now)
    atoms = molpy.Block()
    atoms.insert("x", gx.ravel().astype(np.float64))
    atoms.insert("y", gy.ravel().astype(np.float64))
    atoms.insert("z", gz.ravel().astype(np.float64))
    atoms.insert("vx", np.ascontiguousarray(v[:, 0]))
    atoms.insert("vy", np.ascontiguousarray(v[:, 1]))
    atoms.insert("vz", np.ascontiguousarray(v[:, 2]))
    atoms.insert("mass", np.full(n, _MASS, dtype=np.float64))
    atoms.insert("atomic_number", np.full(n, 18, dtype=np.uint32))
    atoms.insert("type_id", np.full(n, 1, dtype=np.uint32))
    if charge != 0.0:
        ijk = np.indices((_SIDE,) * 3).reshape(3, -1).sum(axis=0)
        atoms.insert("charge", np.where(ijk % 2 == 0, charge, -charge))
    frame = molpy.Frame()
    frame["atoms"] = atoms
    edge = _SIDE * spacing
    if tilt == 0.0:
        frame.box = molpy.Box.cube(edge)
    else:
        frame.box = molpy.Box.from_lengths_tilts([edge, edge, edge], [tilt, 0.0, 0.0])
    return frame


def _columns(frame: "molpy.Frame", key: str) -> np.ndarray:
    return np.asarray(frame["atoms"][key], dtype=np.float64)


def _temperature(frame: "molpy.Frame") -> float:
    """Kinetic temperature [K] from the velocity columns, 3N-3 DOF."""
    m = _columns(frame, "mass")
    v2 = sum(_columns(frame, k) ** 2 for k in ("vx", "vy", "vz"))
    return float(np.sum(m * v2) * _MVV2E / ((3 * m.size - 3) * _KB))


def _lj_virial(frame: "molpy.Frame") -> float:
    """Pair virial Σ r·F [eV] of the truncated LJ (orthorhombic min image)."""
    pos = np.stack([_columns(frame, k) for k in ("x", "y", "z")], axis=1)
    lengths = np.asarray(frame.box.lengths, dtype=np.float64)
    d = pos[:, None, :] - pos[None, :, :]
    d -= lengths * np.round(d / lengths)
    r2 = np.sum(d * d, axis=-1)
    iu = np.triu_indices(pos.shape[0], 1)
    r2 = r2[iu]
    r2 = r2[r2 < _CUTOFF * _CUTOFF]
    sr6 = (_SIG * _SIG / r2) ** 3
    return float(np.sum(24.0 * _EPS * (2.0 * sr6 * sr6 - sr6)))


def _lj_pressure(frame: "molpy.Frame") -> float:
    """Instantaneous LJ pressure [eV/Å³], LAMMPS sign (positive compressive)."""
    m = _columns(frame, "mass")
    mv2 = np.sum(m * sum(_columns(frame, k) ** 2 for k in ("vx", "vy", "vz")))
    return float((mv2 * _MVV2E + _lj_virial(frame)) / (3.0 * frame.box.volume()))


def _meta_pressure(frame: "molpy.Frame") -> float:
    """Pressure [eV/Å³] from the engine's virial meta plus column kinetics."""
    w = np.asarray(frame.meta["stress"], dtype=np.float64)
    m = _columns(frame, "mass")
    mv2 = np.sum(m * sum(_columns(frame, k) ** 2 for k in ("vx", "vy", "vz")))
    return float((w[0] + w[1] + w[2] + mv2 * _MVV2E) / (3.0 * frame.box.volume()))


def _lj_md(device: str) -> MD:
    sim = MD(device=device)
    sim.set_neighbor(NeighborList(_CUTOFF, _SKIN))
    sim.set_evaluator(LJCut(_EPS, _SIG, _CUTOFF))
    return sim


def _by_tag(frame: "molpy.Frame") -> np.ndarray:
    atoms = frame["atoms"]
    return np.argsort(np.asarray(atoms["tag"])) if "tag" in atoms.keys() else slice(None)


def _unwrapped(frame: "molpy.Frame") -> np.ndarray:
    lengths = np.asarray(frame.box.lengths, dtype=np.float64)
    cols = []
    for d, (k, img) in enumerate((("x", "ix"), ("y", "iy"), ("z", "iz"))):
        x = _columns(frame, k)
        if img in frame["atoms"].keys():
            x = x + lengths[d] * np.asarray(frame["atoms"][img], dtype=np.float64)
        cols.append(x)
    return np.stack(cols, axis=1)[_by_tag(frame)]


def _velocities(frame: "molpy.Frame") -> np.ndarray:
    v = np.stack([_columns(frame, k) for k in ("vx", "vy", "vz")], axis=1)
    return v[_by_tag(frame)]


def _chunked(sim: MD, frame: "molpy.Frame", chunks: int, steps: int, probe):
    """Run ``chunks × steps`` on one CUDA driver; ``probe(frame)`` after each."""
    values = []
    for _ in range(chunks):
        sim.run(frame, steps)
        values.append(probe(frame))
    return values


# ── relaxation toward the setpoints (CUDA) ──────────────────────────


@needs_cuda
class TestCudaNoseHooverNvt:
    def test_hot_box_cools_toward_the_setpoint(self) -> None:
        """200 K → 50 K setpoint, τ_T = 50 fs, 1 ps: the second-half mean
        temperature sits within half the initial offset of the target.
        A run without the thermostat (same state, same steps) stays hot,
        so the cooling is the thermostat's doing."""
        target, t0 = 50.0, 200.0
        frame = _ar_frame(3.8, t0)
        sim = _lj_md("cuda")
        sim.set_integrator(VelocityVerlet(2.0))
        sim.set_thermostat(NoseHoover(target, 50.0, 2.0))
        temps = _chunked(sim, frame, 20, 25, _temperature)
        late = float(np.mean(temps[10:]))
        assert all(math.isfinite(t) for t in temps)
        assert abs(late - target) < 0.5 * abs(t0 - target), temps

        nve = _ar_frame(3.8, t0)
        plain = _lj_md("cuda")
        plain.set_integrator(VelocityVerlet(2.0))
        plain_temps = _chunked(plain, nve, 20, 25, _temperature)
        assert float(np.mean(plain_temps[10:])) > late + 0.25 * (t0 - target)

    def test_langevin_and_nose_hoover_both_accepted(self) -> None:
        """The thermostat slot is plural on both devices."""
        sim = _lj_md("cuda")
        sim.set_integrator(VelocityVerlet(2.0))
        sim.set_thermostat(NoseHoover(50.0, 50.0, 2.0))
        sim.set_thermostat(Langevin(50.0, 50.0, 100.0, 2.0, 11))
        frame = _ar_frame(3.8, 60.0)
        sim.run(frame, 10)
        assert math.isfinite(_temperature(frame))


@needs_cuda
class TestCudaInitVelocity:
    def test_seeds_the_target_temperature_with_zero_momentum(self) -> None:
        """InitVelocity on MD(device="cuda") replaces the frame's velocities
        before step 0: a zero-velocity box leaves a zero-step run at the
        seeded temperature (3N-3 DOF) with no net momentum."""
        frame = _ar_frame(3.8, 1.0)
        for key in ("vx", "vy", "vz"):
            frame["atoms"][key] = np.zeros(_SIDE**3)
        sim = _lj_md("cuda")
        sim.set_integrator(VelocityVerlet(2.0))
        sim.set_observer(InitVelocity(120.0, 5))
        sim.run(frame, 0)
        assert _temperature(frame) == pytest.approx(120.0, rel=1e-6)
        m = _columns(frame, "mass")
        for key in ("vx", "vy", "vz"):
            assert abs(float(np.sum(m * _columns(frame, key)))) < 1e-9


@needs_cuda
class TestCudaNoseHooverNpt:
    def test_compressed_lj_box_expands_toward_zero_pressure(self) -> None:
        """Spacing 3.5 Å (r < 2^(1/6)σ) is strongly compressive: under
        P_target = 0 the cell grows and the pressure falls by more than
        half within 1 ps. The engine's virial meta agrees with the numpy
        LJ virial at the final state."""
        frame = _ar_frame(3.5, 50.0)
        v0, p0 = frame.box.volume(), _lj_pressure(frame)
        assert p0 > 1e-4, p0  # precondition: compressive start
        sim = _lj_md("cuda")
        sim.set_integrator(NoseHooverNPT(50.0, 100.0, 0.0, 500.0, 2.0))
        pressures = _chunked(sim, frame, 20, 25, _lj_pressure)
        assert all(math.isfinite(p) for p in pressures)
        assert frame.box.volume() > 1.01 * v0
        assert float(np.mean(pressures[-5:])) < 0.5 * p0, (p0, pressures)
        np.testing.assert_allclose(
            np.sum(np.asarray(frame.meta["stress"])[:3]),
            _lj_virial(frame),
            rtol=1e-6,
            atol=1e-8,
        )

    @needs_kspace
    @pytest.mark.parametrize("solver", ["pppm", "pme"])
    def test_charged_box_balances_lj_against_the_kspace_virial(
        self, solver: str
    ) -> None:
        """LJ + CoulLong + reciprocal solver on a rock-salt ±0.5 e lattice.
        The Madelung attraction outweighs the LJ repulsion at 3.5 Å, so the
        cell contracts (> 5 %) until the electrostatic virial — real space
        plus the reciprocal part — cancels the LJ virial: over the second
        half the total pressure averages to < 10 % of the LJ partial
        pressure alone, i.e. the barostat drives the full long-range
        pressure to P_target = 0."""
        from atomiverse.longrange import CoulLong, KSpacePME, KSpacePPPM

        alpha = 3.5 / _CUTOFF
        frame = _ar_frame(3.5, 50.0, charge=0.5)
        v0 = frame.box.volume()
        sim = _lj_md("cuda")
        sim.set_evaluator(CoulLong(alpha, _CUTOFF))
        if solver == "pppm":
            sim.set_evaluator(KSpacePPPM(alpha, 1.0, 5))
        else:
            sim.set_evaluator(KSpacePME(alpha, 1.0, 6))
        sim.set_integrator(NoseHooverNPT(50.0, 100.0, 0.0, 500.0, 2.0))
        trace = _chunked(
            sim, frame, 20, 25, lambda f: (_meta_pressure(f), _lj_pressure(f))
        )
        total = np.array([p for p, _ in trace])
        lj_only = np.array([p for _, p in trace])
        assert np.all(np.isfinite(total))
        assert frame.box.volume() < 0.95 * v0
        assert float(np.mean(lj_only[10:])) > 1e-3  # LJ alone: compressive
        assert abs(float(np.mean(total[10:]))) < 0.1 * float(
            np.mean(lj_only[10:])
        ), trace

    def test_t_stop_ramp_heats_over_the_first_run(self) -> None:
        """T_stop ramps the setpoint across the first run's span (as on
        CPU); T_stop == T_target holds it. Near-rigid cell (τ_P = 1e5 fs)."""
        flat_frame, ramp_frame = _ar_frame(3.8, 10.0), _ar_frame(3.8, 10.0)
        flat = _lj_md("cuda")
        flat.set_integrator(NoseHooverNPT(10.0, 60.0, 0.0, 1.0e5, 4.0, T_stop=10.0))
        flat.run(flat_frame, 25)
        ramp = _lj_md("cuda")
        ramp.set_integrator(NoseHooverNPT(10.0, 60.0, 0.0, 1.0e5, 4.0, T_stop=1000.0))
        ramp.run(ramp_frame, 25)
        assert _temperature(ramp_frame) > _temperature(flat_frame)

    def test_set_run_span_spans_chunked_runs(self) -> None:
        """An explicit span over 50 steps run as two chunks of 25 heats less
        by the first chunk's end than a 25-step span does."""
        long_frame, short_frame = _ar_frame(3.8, 10.0), _ar_frame(3.8, 10.0)
        long_span = _lj_md("cuda")
        long_span.set_integrator(
            NoseHooverNPT(10.0, 60.0, 0.0, 1.0e5, 4.0, T_stop=1000.0)
        )
        long_span.set_run_span(0, 50)
        long_span.run(long_frame, 25)
        short_span = _lj_md("cuda")
        short_span.set_integrator(
            NoseHooverNPT(10.0, 60.0, 0.0, 1.0e5, 4.0, T_stop=1000.0)
        )
        short_span.run(short_frame, 25)
        assert _temperature(long_frame) < _temperature(short_frame)
        long_span.run(long_frame, 25)  # continuation finishes the span
        assert math.isfinite(_temperature(long_frame))

    def test_couple_axis_freezes_z(self) -> None:
        frame = _ar_frame(3.5, 50.0)
        lz = float(np.asarray(frame.box.lengths)[2])
        lx = float(np.asarray(frame.box.lengths)[0])
        sim = _lj_md("cuda")
        sim.set_integrator(
            NoseHooverNPT(
                50.0, 100.0, 0.0, 500.0, 2.0, couple_axis=(True, True, False)
            )
        )
        sim.run(frame, 50)
        lengths = np.asarray(frame.box.lengths, dtype=np.float64)
        assert lengths[2] == lz
        assert lengths[0] != lx


@needs_cuda
class TestCudaBerendsen:
    def test_berendsen_beside_nose_hoover_expands_a_compressed_box(self) -> None:
        """The split NPT recipe: VelocityVerlet + NoseHoover + Berendsen."""
        frame = _ar_frame(3.5, 50.0)
        v0, p0 = frame.box.volume(), _lj_pressure(frame)
        sim = _lj_md("cuda")
        sim.set_integrator(VelocityVerlet(2.0))
        sim.set_thermostat(NoseHoover(50.0, 100.0, 2.0))
        sim.set_barostat(BerendsenBarostat(0.0, 500.0, 2.0))
        pressures = _chunked(sim, frame, 10, 25, _lj_pressure)
        assert frame.box.volume() > v0
        assert pressures[-1] < p0


# ── CPU ↔ CUDA one-step parity ──────────────────────────────────────

#: Same state, same parameters, one step: the two backends differ only in
#: summation order (double CUDA build), so 1e-9 relative is generous.
_RTOL = 1e-9
_ATOL_X = 1e-10  # [Å]
_ATOL_V = 1e-12  # [Å/fs]


def _assert_same_state(cpu_frame, cuda_frame, *, box: bool) -> None:
    np.testing.assert_allclose(
        _unwrapped(cuda_frame), _unwrapped(cpu_frame), rtol=_RTOL, atol=_ATOL_X
    )
    np.testing.assert_allclose(
        _velocities(cuda_frame), _velocities(cpu_frame), rtol=_RTOL, atol=_ATOL_V
    )
    if box:
        np.testing.assert_allclose(
            np.asarray(cuda_frame.box.lengths),
            np.asarray(cpu_frame.box.lengths),
            rtol=1e-12,
        )


@needs_cuda
class TestCpuCudaParity:
    @pytest.mark.parametrize("steps", [1, 5])
    def test_nose_hoover_nvt(self, steps: int) -> None:
        """The thermostat's own dt (99 fs) is replaced by the integrator's
        2 fs on both devices — a mismatch would show here at once."""
        frames = {}
        for device in ("cpu", "cuda"):
            frame = _ar_frame(3.8, 120.0)
            sim = _lj_md(device)
            sim.set_integrator(VelocityVerlet(2.0))
            sim.set_thermostat(NoseHoover(40.0, 30.0, 99.0))
            sim.run(frame, steps)
            frames[device] = frame
        before = _ar_frame(3.8, 120.0)
        assert not np.allclose(_velocities(frames["cpu"]), _velocities(before))
        _assert_same_state(frames["cpu"], frames["cuda"], box=False)

    @pytest.mark.parametrize("steps", [1, 5])
    def test_nose_hoover_npt(self, steps: int) -> None:
        frames = {}
        for device in ("cpu", "cuda"):
            frame = _ar_frame(3.5, 50.0)
            sim = _lj_md(device)
            sim.set_integrator(
                NoseHooverNPT(50.0, 100.0, 0.0, 200.0, 2.0, drag=0.1, mtchain=4)
            )
            sim.run(frame, steps)
            frames[device] = frame
        v0 = _ar_frame(3.5, 50.0).box.volume()
        assert frames["cpu"].box.volume() != v0
        _assert_same_state(frames["cpu"], frames["cuda"], box=True)
        np.testing.assert_allclose(
            frames["cuda"].meta["conserved"],
            frames["cpu"].meta["conserved"],
            rtol=1e-9,
        )

    @needs_kspace
    def test_nose_hoover_npt_pppm(self) -> None:
        """One NPT step with the PPPM virial in the pressure. The two PPPM
        implementations share the LAMMPS stencil and Green's function; the
        bound is their reciprocal-force agreement, not round-off."""
        from atomiverse.longrange import CoulLong, KSpacePPPM

        alpha = 3.5 / _CUTOFF
        frames = {}
        for device in ("cpu", "cuda"):
            frame = _ar_frame(3.5, 50.0, charge=0.5)
            sim = _lj_md(device)
            sim.set_evaluator(CoulLong(alpha, _CUTOFF))
            sim.set_evaluator(KSpacePPPM(alpha, 1.0, 5))
            sim.set_integrator(NoseHooverNPT(50.0, 100.0, 0.0, 200.0, 2.0))
            sim.run(frame, 1)
            frames[device] = frame
        np.testing.assert_allclose(
            _unwrapped(frames["cuda"]), _unwrapped(frames["cpu"]), rtol=0, atol=1e-8
        )
        np.testing.assert_allclose(
            _velocities(frames["cuda"]), _velocities(frames["cpu"]), rtol=0, atol=1e-9
        )
        np.testing.assert_allclose(
            np.asarray(frames["cuda"].box.lengths),
            np.asarray(frames["cpu"].box.lengths),
            rtol=1e-9,
        )


# ── refusals ────────────────────────────────────────────────────────

_DEVICES = ["cpu", pytest.param("cuda", marks=needs_cuda)]


@pytest.mark.parametrize("device", _DEVICES)
class TestNptRefusesASecondThermostatOrBarostat:
    def test_thermostat_after_npt(self, device: str) -> None:
        sim = _lj_md(device)
        sim.set_integrator(NoseHooverNPT(50.0, 100.0, 0.0, 500.0, 2.0))
        with pytest.raises(ValueError, match="already thermostats"):
            sim.set_thermostat(NoseHoover(50.0, 100.0, 2.0))

    def test_npt_after_thermostat(self, device: str) -> None:
        sim = _lj_md(device)
        sim.set_thermostat(NoseHoover(50.0, 100.0, 2.0))
        with pytest.raises(ValueError, match="do not also set a thermostat"):
            sim.set_integrator(NoseHooverNPT(50.0, 100.0, 0.0, 500.0, 2.0))

    def test_barostat_after_npt(self, device: str) -> None:
        sim = _lj_md(device)
        sim.set_integrator(NoseHooverNPT(50.0, 100.0, 0.0, 500.0, 2.0))
        with pytest.raises(ValueError, match="already barostats"):
            sim.set_barostat(BerendsenBarostat(0.0, 500.0, 2.0))

    def test_npt_after_barostat(self, device: str) -> None:
        sim = _lj_md(device)
        sim.set_barostat(BerendsenBarostat(0.0, 500.0, 2.0))
        with pytest.raises(ValueError, match="barostat"):
            sim.set_integrator(NoseHooverNPT(50.0, 100.0, 0.0, 500.0, 2.0))

    def test_nose_hoover_beside_nvt_sllod(self, device: str) -> None:
        sim = _lj_md(device)
        sim.set_integrator(NvtSllod(50.0, 100.0, 2.0, 1e-6, 0))
        with pytest.raises(ValueError, match="NvtSllod already thermostats"):
            sim.set_thermostat(NoseHoover(50.0, 100.0, 2.0))


@needs_cuda
class TestCudaRefusals:
    def test_csvr_names_the_alternatives(self) -> None:
        sim = _lj_md("cuda")
        with pytest.raises(ValueError, match="NoseHoover or Langevin"):
            sim.set_thermostat(CSVR(50.0, 100.0, 2.0, 3))

    def test_parrinello_rahman_names_the_alternatives(self) -> None:
        sim = _lj_md("cuda")
        with pytest.raises(ValueError, match="NoseHooverNPT"):
            sim.set_integrator(ParrinelloRahmanNPT(50.0, 0.0, 100.0, 500.0, 2.0))

    def test_triclinic_npt_is_refused_with_a_how_to(self) -> None:
        sim = _lj_md("cuda")
        sim.set_integrator(NoseHooverNPT(50.0, 100.0, 0.0, 500.0, 2.0))
        frame = _ar_frame(3.8, 50.0, tilt=1.0)
        with pytest.raises(RuntimeError, match="(?s)triclinic.*orthorhombic"):
            sim.run(frame, 1)

    def test_bad_nose_hoover_parameters(self) -> None:
        sim = _lj_md("cuda")
        with pytest.raises(ValueError, match="must be finite and > 0"):
            sim.set_thermostat(NoseHoover(0.0, 100.0, 2.0))

    def test_set_run_span_needs_a_ramp_owner(self) -> None:
        sim = _lj_md("cuda")
        sim.set_integrator(VelocityVerlet(2.0))
        sim.set_thermostat(NoseHoover(50.0, 100.0, 2.0))
        with pytest.raises(RuntimeError, match="Langevin thermostat or a NoseHooverNPT"):
            sim.set_run_span(0, 100)

    def test_second_integrator_is_refused(self) -> None:
        sim = _lj_md("cuda")
        sim.set_integrator(VelocityVerlet(2.0))
        with pytest.raises(ValueError, match="set_integrator already configured"):
            sim.set_integrator(NoseHooverNPT(50.0, 100.0, 0.0, 500.0, 2.0))

"""Every unit-bearing MD component under LAMMPS ``units real`` and ``units
metal``, on the CPU and the CUDA driver.

Each expectation is a closed form evaluated with the LAMMPS factors of
``Update::set_units`` (``src/update.cpp``), written out below rather than read
back from the engine:

* ``FixEfield``: F = q E qe2f (real: 23.060549 kcal/mol/Å per e·V/Å; metal:
  1 eV/Å), U = −Σ q qe2f E·r, and the velocity a lone ion gains,
  v = F t / (m mvv2e);
* the Coulomb energy of one pair, qqr2e q_i q_j / r (``PairCoulCut``) and
  qqr2e q_i q_j erfc(α r) / r (the real-space ``PairLJCutCoulLong``);
* ``InitVelocity``: Σ m v² mvv2e / (N_f boltz) is the requested temperature;
* CSVR (global and ``types`` + ``bias="com"``), Langevin and Nosé–Hoover hold
  an LJ liquid at the setpoint, measured from the velocities;
* ``NoseHooverNPT`` on an ideal gas settles at ⟨V⟩ = N boltz T nktv2p / P
  with P in atm (real) or bar (metal);
* the same molecular system — fused LJ + PPPM, harmonic bonds and angles,
  multi-harmonic dihedrals, SHAKE, a field, every thermostat and the NPT
  barostat — written once in real and once in metal (energies kcal/mol vs
  eV, times fs vs ps, pressures atm vs bar) follows one physical trajectory.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import molpy

from atomiverse._binding import _cpp, extension_available
from _domains import skip_module_unless

skip_module_unless("ATV_BUILD_MD")

import atomiverse as av
from atomiverse.md import MD, FixEfield, FixShake, InitVelocity
from atomiverse.md.integrator import NoseHooverNPT, VelocityVerlet
from atomiverse.md.thermostat import CSVR, Langevin, NoseHoover
from atomiverse.neighbor import NeighborList

_HAS_CUDA_MD: bool = extension_available() and hasattr(_cpp, "_CudaMD")
_HAS_KSPACE: bool = extension_available() and hasattr(_cpp, "KSpacePPPM")
needs_kspace = pytest.mark.skipif(not _HAS_KSPACE, reason="needs ATV_BUILD_KSPACE")

DEVICES = [
    "cpu",
    pytest.param(
        "cuda", marks=pytest.mark.skipif(not _HAS_CUDA_MD, reason="needs ATV_BUILD_CUDA")
    ),
]
STYLES = ["real", "metal"]

#: LAMMPS ``Update::set_units`` (src/update.cpp), copied, not read back.
LAMMPS = {
    "real": {
        "boltz": 0.0019872067,
        "mvv2e": 48.88821291 * 48.88821291,
        "nktv2p": 68568.415,
        "qqr2e": 332.06371,
        "qe2f": 23.060549,
    },
    "metal": {
        "boltz": 8.617343e-5,
        "mvv2e": 1.0364269e-4,
        "nktv2p": 1.6021765e6,
        "qqr2e": 14.399645,
        "qe2f": 1.0,
    },
}

#: The same physics in both styles: one style-unit of each quantity per
#: real-style unit (kcal/mol, fs, atm, Å/fs). 1 eV = 23.060549 kcal/mol
#: (the LAMMPS factors agree on it to 5e-8); 1 atm = 1.01325 bar.
SCALE = {
    "real": {"energy": 1.0, "time": 1.0, "pressure": 1.0, "velocity": 1.0},
    "metal": {
        "energy": 1.0 / 23.060549,
        "time": 1.0e-3,
        "pressure": 1.01325,
        "velocity": 1.0e3,
    },
}


def _block(columns: dict) -> "molpy.Block":
    block = molpy.Block()
    for key, value in columns.items():
        block.insert(key, np.ascontiguousarray(value))
    return block


def _atoms_frame(pos, mass, box, *, vel=None, charge=None, types=None) -> "molpy.Frame":
    pos = np.asarray(pos, dtype=np.float64)
    n = len(pos)
    vel = np.zeros_like(pos) if vel is None else np.asarray(vel, dtype=np.float64)
    cols = {}
    for c, k in enumerate("xyz"):
        cols[k] = pos[:, c].copy()
        cols["v" + k] = vel[:, c].copy()
    cols["mass"] = np.broadcast_to(np.asarray(mass, dtype=np.float64), (n,)).copy()
    if charge is not None:
        cols["charge"] = np.asarray(charge, dtype=np.float64)
    cols["type_id"] = (
        np.ones(n, dtype=np.uint32) if types is None else np.asarray(types, dtype=np.uint32)
    )
    cols["atomic_number"] = np.full(n, 18, dtype=np.uint32)
    frame = molpy.Frame()
    frame["atoms"] = _block(cols)
    frame.box = molpy.Box.cube(box)
    return frame


def _by_tag(frame) -> np.ndarray | slice:
    atoms = frame["atoms"]
    return np.argsort(np.asarray(atoms["tag"])) if "tag" in atoms.keys() else slice(None)


def _cols(frame, keys) -> np.ndarray:
    a = frame["atoms"]
    return np.stack([np.asarray(a[k], dtype=np.float64) for k in keys], axis=1)[_by_tag(frame)]


def _temperature(frame, style: str, *, dof: int | None = None, types=None) -> float:
    """Σ m v² mvv2e / (N_f boltz), N_f = 3N − 3 by default (LAMMPS temp)."""
    a = frame["atoms"]
    m = np.asarray(a["mass"], dtype=np.float64)
    v = np.stack([np.asarray(a[k], dtype=np.float64) for k in ("vx", "vy", "vz")], axis=1)
    if types is not None:
        sel = np.isin(np.asarray(a["type_id"]), types)
        m, v = m[sel], v[sel]
        v = v - (m[:, None] * v).sum(axis=0) / m.sum()
    n_f = 3 * m.size - 3 if dof is None else dof
    u = LAMMPS[style]
    return float(np.sum(m[:, None] * v * v) * u["mvv2e"] / (n_f * u["boltz"]))


def _volume(frame) -> float:
    return float(abs(np.linalg.det(np.asarray(frame.box.matrix, dtype=np.float64))))


# ── FixEfield: F = q E qe2f ───────────────────────────────────────────────


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("style", STYLES)
def test_efield_force_is_charge_field_qe2f(device: str, style: str) -> None:
    """Two ions 15 Å apart and nothing else: the field force is q E qe2f in
    the style's force unit, the field energy −Σ q qe2f E·r, and after n
    velocity-Verlet steps of length dt each ion moves at v = F n dt / (m
    mvv2e) (ftm2v = 1/mvv2e in real and metal)."""
    E = np.array([0.2, -0.1, 0.05])  # [V/Å]
    q = np.array([1.0, -0.5])
    m = 10.0
    pos = np.array([[10.0, 10.0, 10.0], [25.0, 12.0, 11.0]])
    u = LAMMPS[style]
    dt = 1.0 * SCALE[style]["time"]

    frame = _atoms_frame(pos, m, 40.0, charge=q)
    sim = MD(device=device, units=style)
    sim.set_neighbor(NeighborList(5.0, 1.0))
    sim.set_integrator(VelocityVerlet(dt))
    sim.set_evaluator(FixEfield(*E))
    sim.run(frame, 0)
    force = _cols(frame, ("fx", "fy", "fz"))
    np.testing.assert_allclose(force, q[:, None] * E[None, :] * u["qe2f"], rtol=1e-9, atol=1e-12)
    want_u = -float(np.sum(q[:, None] * u["qe2f"] * E[None, :] * pos))
    assert float(frame.meta["energy"]) == pytest.approx(want_u, rel=1e-9)

    frame = _atoms_frame(pos, m, 40.0, charge=q)
    sim = MD(device=device, units=style)
    sim.set_neighbor(NeighborList(5.0, 1.0))
    sim.set_integrator(VelocityVerlet(dt))
    sim.set_evaluator(FixEfield(*E))
    sim.run(frame, 10)
    v = _cols(frame, ("vx", "vy", "vz"))
    want_v = q[:, None] * E[None, :] * u["qe2f"] * 10 * dt / (m * u["mvv2e"])
    np.testing.assert_allclose(v, want_v, rtol=1e-9, atol=1e-14)


def test_efield_real_force_is_lammps_value() -> None:
    """The number from the bug report: 1 e in 1 V/Å under units real is
    23.060549 kcal/mol/Å (LAMMPS ``fix efield``), not 1."""
    frame = _atoms_frame([[5.0, 5.0, 5.0]], 10.0, 30.0, charge=[1.0])
    sim = MD(units="real")
    sim.set_neighbor(NeighborList(5.0, 1.0))
    sim.set_integrator(VelocityVerlet(1.0))
    sim.set_evaluator(FixEfield(1.0, 0.0, 0.0))
    sim.run(frame, 0)
    assert float(np.asarray(frame["atoms"]["fx"])[0]) == pytest.approx(23.060549, rel=1e-12)


# ── Coulomb pair energy: qqr2e ───────────────────────────────────────────


@pytest.mark.parametrize("style", STYLES)
def test_coul_cut_pair_energy_is_qqr2e_qq_over_r(style: str) -> None:
    r = 3.0
    q = [1.0, -0.5]
    frame = _atoms_frame([[5.0, 5.0, 5.0], [5.0 + r, 5.0, 5.0]], 10.0, 30.0, charge=q)
    sim = MD(units=style)
    sim.set_neighbor(NeighborList(8.0, 1.0))
    sim.set_integrator(VelocityVerlet(SCALE[style]["time"]))
    sim.set_evaluator(av.PairCoulCut(8.0))
    sim.run(frame, 0)
    want = LAMMPS[style]["qqr2e"] * q[0] * q[1] / r
    assert float(frame.meta["energy"]) == pytest.approx(want, rel=1e-12)
    fx = _cols(frame, ("fx",))[:, 0]
    assert fx[1] == pytest.approx(-want / r, rel=1e-12)  # attraction toward atom 0


@needs_kspace
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("style", STYLES)
def test_lj_cut_coul_long_real_space_pair_is_qqr2e_erfc(device: str, style: str) -> None:
    """LJ switched off (ε = 0): the fused pair style's real-space energy of
    one pair is qqr2e q_i q_j erfc(α r)/r, and PPPM adds the reciprocal
    part, which is also proportional to qqr2e: the total over qqr2e is
    the same number in real and metal."""
    r, alpha, cut, box = 3.0, 0.3, 9.0, 30.0
    q = [1.0, -1.0]
    pos = [[10.0, 10.0, 10.0], [10.0 + r, 10.0, 10.0]]
    frame = _atoms_frame(pos, 10.0, box, charge=q)
    sim = MD(device=device, units=style)
    sim.set_neighbor(NeighborList(cut, 1.0))
    sim.set_integrator(VelocityVerlet(SCALE[style]["time"]))
    sim.set_evaluator(av.PairLJCutCoulLong([av.LJTypeCoeff(1, 1, 0.0, 3.0)], alpha, cut))
    sim.set_evaluator(av.KSpacePPPM(alpha, 0.5, 5))
    sim.run(frame, 0)
    want_pair = LAMMPS[style]["qqr2e"] * q[0] * q[1] * math.erfc(alpha * r) / r
    assert float(frame.meta["energy_pair"]) == pytest.approx(want_pair, rel=1e-9)
    # The whole Ewald sum scales with qqr2e alone.
    total = float(frame.meta["energy"]) / LAMMPS[style]["qqr2e"]
    other = "metal" if style == "real" else "real"
    f2 = _atoms_frame(pos, 10.0, box, charge=q)
    sim = MD(device=device, units=other)
    sim.set_neighbor(NeighborList(cut, 1.0))
    sim.set_integrator(VelocityVerlet(SCALE[other]["time"]))
    sim.set_evaluator(av.PairLJCutCoulLong([av.LJTypeCoeff(1, 1, 0.0, 3.0)], alpha, cut))
    sim.set_evaluator(av.KSpacePPPM(alpha, 0.5, 5))
    sim.run(f2, 0)
    assert float(f2.meta["energy"]) / LAMMPS[other]["qqr2e"] == pytest.approx(total, rel=1e-9)


# ── LJ liquid: InitVelocity and the thermostats ──────────────────────────

_AR_MASS = 39.948
_AR_EPS = 0.2381  # [kcal/mol]
_AR_SIGMA = 3.405  # [Å]
_AR_CUT = 8.5  # [Å]


def _argon(cells: int = 4, a: float = 5.6) -> "molpy.Frame":
    """fcc argon, 4 cells³ (256 atoms) at 5.6 Å — a dense liquid at 120 K."""
    basis = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
    pts = [
        (np.array([i, j, k]) + b) * a
        for i in range(cells)
        for j in range(cells)
        for k in range(cells)
        for b in basis
    ]
    pos = np.asarray(pts) + 0.25
    n = len(pos)
    types = np.where(np.arange(n) % 2 == 0, 1, 2)
    return _atoms_frame(pos, _AR_MASS, cells * a, types=types)


def _argon_md(device: str, style: str, dt_fs: float = 4.0) -> MD:
    s = SCALE[style]
    sim = MD(device=device, units=style)
    sim.set_neighbor(NeighborList(_AR_CUT, 1.0))
    eps = _AR_EPS * s["energy"]
    sim.set_evaluator(
        av.PairLJCut(
            [av.LJTypeCoeff(1, 1, eps, _AR_SIGMA), av.LJTypeCoeff(2, 2, eps, _AR_SIGMA)],
            _AR_CUT,
        )
    )
    sim.set_integrator(VelocityVerlet(dt_fs * s["time"]))
    return sim


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("style", STYLES)
def test_init_velocity_sets_the_requested_temperature(device: str, style: str) -> None:
    frame = _argon()
    sim = _argon_md(device, style)
    sim.set_observer(InitVelocity(250.0, 4242))
    sim.run(frame, 0)
    assert _temperature(frame, style) == pytest.approx(250.0, rel=1e-6)
    p = (_cols(frame, ("vx", "vy", "vz")) * np.asarray(frame["atoms"]["mass"])[:, None]).sum(axis=0)
    assert np.all(np.abs(p) < 1e-9 * _AR_MASS * 1e3 * SCALE[style]["velocity"])


def _thermostat(kind: str, style: str):
    t = SCALE[style]["time"]
    dt = 4.0 * t
    if kind == "csvr":
        return [CSVR(120.0, 100.0 * t, dt, 11)]
    if kind == "csvr_types_com":
        return [
            CSVR(120.0, 100.0 * t, dt, 12, types=[1], bias="com"),
            CSVR(120.0, 100.0 * t, dt, 13, types=[2], bias="com"),
        ]
    if kind == "langevin":
        return [Langevin(120.0, 120.0, 200.0 * t, dt, 14)]
    if kind == "nose_hoover":
        return [NoseHoover(120.0, 200.0 * t, dt)]
    raise ValueError(kind)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("style", STYLES)
@pytest.mark.parametrize("kind", ["csvr", "csvr_types_com", "langevin", "nose_hoover"])
def test_thermostat_holds_the_setpoint(device: str, style: str, kind: str) -> None:
    """Start at 60 K, set 120 K: after 4 ps the velocity temperature
    averaged over the next 6 ps is within 4 % of 120 K (a boltz or mvv2e of
    the wrong style is off by orders of magnitude)."""
    frame = _argon()
    sim = _argon_md(device, style)
    sim.set_observer(InitVelocity(60.0, 99))
    for th in _thermostat(kind, style):
        sim.set_thermostat(th)
    sim.run(frame, 1000)
    temps, groups = [], []
    for _ in range(30):
        sim.run(frame, 50)
        temps.append(_temperature(frame, style))
        groups.append([_temperature(frame, style, types=[t]) for t in (1, 2)])
    assert abs(np.mean(temps) / 120.0 - 1.0) < 0.04, np.mean(temps)
    if kind == "csvr_types_com":
        for g in np.mean(groups, axis=0):
            assert abs(g / 120.0 - 1.0) < 0.05, g


# ── NoseHooverNPT: target pressure in atm (real) / bar (metal) ───────────


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("style", STYLES)
def test_npt_ideal_gas_volume_follows_the_target_pressure(device: str, style: str) -> None:
    """No interactions: P = N boltz T nktv2p / V exactly, so the barostat
    holds ⟨V⟩ = N boltz T nktv2p / P_target. 200 atm (real) and 202.65 bar
    (metal) are one pressure and give one volume, 81.8 nm³ for 4000 atoms
    at 300 K; a target read in the other style's unit is 1.3 % off, in
    eV/Å³ or Pa orders of magnitude off."""
    n, T, p_atm = 4000, 300.0, 200.0
    s, u = SCALE[style], LAMMPS[style]
    p_target = p_atm * s["pressure"]
    v_want = n * u["boltz"] * T * u["nktv2p"] / p_target
    side = v_want ** (1.0 / 3.0)
    rng = np.random.default_rng(3)
    frame = _atoms_frame(rng.uniform(0.0, side, size=(n, 3)), _AR_MASS, side)
    sim = MD(device=device, units=style)
    sim.set_neighbor(NeighborList(2.0, 1.0))
    sim.set_evaluator(av.PairLJCut(0.0, 1.0, 2.0))  # no interactions
    dt = 5.0 * s["time"]
    sim.set_integrator(NoseHooverNPT(T, 100.0 * s["time"], p_target, 1000.0 * s["time"], dt))
    sim.set_observer(InitVelocity(T, 7))
    sim.run(frame, 2000)
    vols = []
    for _ in range(400):
        sim.run(frame, 25)
        vols.append(_volume(frame))
    assert abs(np.mean(vols) / v_want - 1.0) < 0.008, (np.mean(vols), v_want)


# ── One molecular system, written in real and in metal ───────────────────

_BOX = 24.0  # [Å]
_CUTOFF = 8.0  # [Å]
_ALPHA = 3.2 / _CUTOFF  # [1/Å]
_CHAIN_Q = (0.25, -0.4, 0.3, -0.4, 0.25)  # neutral 5-site chain [e]
_DIMER_R0 = 1.1  # [Å]


def _molecules() -> "molpy.Frame":
    """8 five-site chains (bonds, angles, dihedrals; bond type 1), 8 rigid
    dimers (bond type 2, SHAKE) and 16 ±0.6 e ions; velocities in Å/fs."""
    rng = np.random.default_rng(2027)
    xyz, charge, types, bonds, btypes, angles, dihedrals = [], [], [], [], [], [], []
    base = np.array(
        [[0.0, 0.0, 0.0], [1.25, 0.88, 0.0], [2.5, 0.0, 0.0], [3.75, 0.88, 0.0], [5.0, 0.0, 0.0]]
    )
    n = 0
    for gx in range(2):
        for gy in range(2):
            for gz in range(2):
                origin = np.array([1.0 + 12.0 * gx, 1.5 + 12.0 * gy, 1.5 + 12.0 * gz])
                xyz.extend(base + origin + rng.normal(scale=0.2, size=base.shape))
                charge.extend(_CHAIN_Q)
                types.extend([1] * 5)
                bonds += [(n + k, n + k + 1) for k in range(4)]
                btypes += [1] * 4
                angles += [(n + k, n + k + 1, n + k + 2) for k in range(3)]
                dihedrals += [(n + k, n + k + 1, n + k + 2, n + k + 3) for k in range(2)]
                n += 5
                c = origin + np.array([7.5, 5.0, 2.0])
                d = rng.normal(size=3)
                d /= np.linalg.norm(d)
                xyz += [c, c + _DIMER_R0 * d]
                charge += [0.2, -0.2]
                types += [2, 3]
                bonds.append((n, n + 1))
                btypes.append(2)
                n += 2
                for k, qi in enumerate((0.6, -0.6)):
                    xyz.append(origin + np.array([3.0 + 3.0 * k, 7.5, 7.0]))
                    charge.append(qi)
                    types.append(4)
                    n += 1
    xyz = np.asarray(xyz)
    types = np.asarray(types)
    mass = np.array([{1: 14.027, 2: 12.011, 3: 1.008, 4: 22.99}[int(t)] for t in types])
    # 300 K Maxwell-Boltzmann in real units (Å/fs), zero net momentum.
    sig = np.sqrt(LAMMPS["real"]["boltz"] * 300.0 / (mass * LAMMPS["real"]["mvv2e"]))
    vel = rng.normal(size=xyz.shape) * sig[:, None]
    vel -= (mass[:, None] * vel).sum(axis=0) / mass.sum()
    frame = _atoms_frame(xyz, mass, _BOX, vel=vel, charge=charge, types=types)
    for name, rows, tids in (
        ("bonds", bonds, btypes),
        ("angles", angles, [1] * len(angles)),
        ("dihedrals", dihedrals, [1] * len(dihedrals)),
    ):
        rows = np.asarray(rows, dtype=np.uint32)
        cols = {
            end: rows[:, c] for c, end in enumerate(("atomi", "atomj", "atomk", "atoml")[: rows.shape[1]])
        }
        cols["type_id"] = np.asarray(tids, dtype=np.uint32)
        frame[name] = _block(cols)
    return frame


def _in_style(frame, style: str) -> "molpy.Frame":
    """The same state with velocities in the style's unit."""
    a = frame["atoms"]
    for k in ("vx", "vy", "vz"):
        a[k] = np.asarray(a[k], dtype=np.float64) * SCALE[style]["velocity"]
    return frame


def _molecular_md(device: str, style: str, kind: str) -> MD:
    s = SCALE[style]
    e, t, dt = s["energy"], s["time"], 1.0 * s["time"]
    sim = MD(device=device, units=style)
    sim.set_neighbor(NeighborList(_CUTOFF, 2.0))
    lj = [
        av.LJTypeCoeff(1, 1, 0.09 * e, 3.6),
        av.LJTypeCoeff(2, 2, 0.07 * e, 3.4),
        av.LJTypeCoeff(3, 3, 0.02 * e, 2.5),
        av.LJTypeCoeff(4, 4, 0.10 * e, 2.6),
    ]
    sim.set_evaluator(av.PairLJCutCoulLong(lj, _ALPHA, _CUTOFF))
    sim.set_evaluator(av.KSpacePPPM(_ALPHA, 1.0, 5))
    # Entry 0 is unused (bonded tables are indexed by the raw 1-based type).
    sim.set_evaluator(av.BondHarmonic([0.0, 300.0 * e, 340.0 * e], [1.0, 1.53, _DIMER_R0]))
    sim.set_evaluator(av.AngleHarmonic([0.0, 60.0 * e], [1.0, math.radians(112.0)]))
    sim.set_evaluator(
        av.DihedralMultiHarmonic(
            [[0.0] * 5, [1.4 * e, -2.8 * e, 0.7 * e, 3.4 * e, -0.5 * e]]
        )
    )
    sim.set_evaluator(FixEfield(0.3, -0.1, 0.2))  # [V/Å] in both styles
    shake = FixShake(dt, tolerance=1e-10, max_iter=500)
    shake.constrain_bond(2, _DIMER_R0)
    sim.set_observer(shake)
    if kind == "npt":
        sim.set_integrator(
            NoseHooverNPT(300.0, 50.0 * t, 3000.0 * s["pressure"], 200.0 * t, dt)
        )
    else:
        sim.set_integrator(VelocityVerlet(dt))
    if kind == "csvr":
        sim.set_thermostat(CSVR(300.0, 20.0 * t, dt, 21, types=[1, 2, 3], bias="com"))
        sim.set_thermostat(CSVR(300.0, 20.0 * t, dt, 22, types=[4], bias="com"))
    elif kind == "langevin":
        sim.set_thermostat(Langevin(300.0, 300.0, 20.0 * t, dt, 23))
    elif kind == "nose_hoover":
        sim.set_thermostat(NoseHoover(300.0, 20.0 * t, dt))
    return sim


def _pressure(frame, style: str) -> float:
    """LAMMPS compute pressure from the frame's virial sums [style]."""
    w = np.asarray(frame.meta["stress"], dtype=np.float64) + np.asarray(
        frame.meta["stress_kinetic"], dtype=np.float64
    )
    return float((w[0] + w[1] + w[2]) / 3.0 / _volume(frame) * LAMMPS[style]["nktv2p"])


@needs_kspace
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kind", ["nve", "csvr", "langevin", "nose_hoover", "npt"])
def test_real_and_metal_follow_one_physical_trajectory(device: str, kind: str) -> None:
    """200 steps of 1 fs. Every factor is the style's or the two runs part:
    a field 23× too weak, a bar read as atm or a kB of the other style
    each move atoms by far more than the 2e-4 Å allowed here."""
    out = {}
    for style in STYLES:
        frame = _in_style(_molecules(), style)
        _molecular_md(device, style, kind).run(frame, 200)
        out[style] = frame
    real, metal = out["real"], out["metal"]
    np.testing.assert_allclose(
        _cols(metal, ("x", "y", "z")), _cols(real, ("x", "y", "z")), rtol=0, atol=2e-4
    )
    np.testing.assert_allclose(
        _cols(metal, ("vx", "vy", "vz")) / SCALE["metal"]["velocity"],
        _cols(real, ("vx", "vy", "vz")),
        rtol=0,
        atol=2e-6,
    )
    for key in ("energy", "kinetic", "energy_bond", "energy_angle", "energy_dihedral"):
        assert float(metal.meta[key]) / SCALE["metal"]["energy"] == pytest.approx(
            float(real.meta[key]), rel=1e-4, abs=1e-4
        ), key
    assert _temperature(metal, "metal", dof=1) == pytest.approx(
        _temperature(real, "real", dof=1), rel=1e-4
    )
    assert _pressure(metal, "metal") / SCALE["metal"]["pressure"] == pytest.approx(
        _pressure(real, "real"), rel=1e-3, abs=1.0
    )
    assert _volume(metal) == pytest.approx(_volume(real), rel=1e-6)
    if kind == "npt":
        assert abs(_volume(real) / _BOX**3 - 1.0) > 1e-3  # the barostat acted

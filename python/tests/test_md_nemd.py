"""Thermostatted NEMD of molten salts from Python (spec nemd-thermostats).

J. Petravic and J. Delhommelle, J. Chem. Phys. 118, 7477 (2003): a
Tosi–Fumi NaCl melt under a uniform electric field, integrated by a
fifth-order Gear predictor-corrector with Gaussian kinetic thermostats
(eqs. 9–11) or a configurational Nosé–Hoover thermostat (eqs. 12a–c) that
holds the first-order configurational temperature (eq. 7).

Every case runs the CPU engine in LAMMPS ``units metal`` (eV, Å, g/mol,
ps, K, e, V/Å) and compares against a value computed in the test from the
frame's own columns or a closed form, never against a number read back off
the engine.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import molpy

from _domains import skip_module_unless

skip_module_unless("ATV_BUILD_MD")

from atomiverse.longrange import CoulLong, KSpaceEwald
from atomiverse.md import MD, ConfigurationalTemperature, FixEfield
from atomiverse.md.integrator import GearPC, VelocityVerlet
from atomiverse.md.thermostat import (
    ConfigurationalThermostat,
    GaussThermostat,
    NoseHoover,
)
from atomiverse.neighbor import NeighborList
from atomiverse.potentials.pair import Born

#: HARD-CODED GOLDEN — LAMMPS ``units metal`` (src/update.cpp): Boltzmann
#: constant [eV/K], m·v² → energy [eV per g/mol·Å²/ps²], k_e [eV·Å/e²].
_BOLTZ: float = 8.617343e-5
_MVV2E: float = 1.0364269e-4
_COULOMB_EV_A: float = 14.399645

#: HARD-CODED GOLDEN — Madelung constant of rock salt (Ziman, *Principles of
#: the Theory of Solids*, 2nd ed., Cambridge 1972, §2.3).
_MADELUNG: float = 1.747564594633182

_SIDE: int = 4
_N: int = _SIDE**3
_MASS_NA: float = 22.9898  # [g/mol]
_MASS_CL: float = 35.453  # [g/mol]
_TYPE_NA: int = 1  # force-field type id of Na+
_TYPE_CL: int = 2  # force-field type id of Cl-

#: The melt: spacing 2.82 Å (L = 11.28 Å), real-space cutoff 5.5 Å < L/2,
#: Ewald α = 3/r_c — the C++ fixture tests/cpu/dynamics/nacl_fixture.h.
_SPACING: float = 2.82  # [Å]
_CUT: float = 5.5  # [Å]
_ALPHA: float = 3.0 / _CUT  # [1/Å]
_DT: float = 0.001  # [ps]


def _tosi_fumi() -> Born:
    """Tosi–Fumi NaCl short-range table (Born–Mayer–Huggins).

    The Tosi–Fumi d/r⁸ term is attractive; ``Born`` adds +D/r⁸, so D is
    the negated dispersion coefficient.
    """
    return Born(
        type_i=[_TYPE_NA, _TYPE_NA, _TYPE_CL],
        type_j=[_TYPE_NA, _TYPE_CL, _TYPE_CL],
        A=[0.2637038, 0.2109630, 0.1582223],
        rho=[0.317, 0.317, 0.317],
        sigma=[2.340, 2.755, 3.170],
        C=[0.6553585, 4.3690564, 45.2509408],
        D=[-0.3114513, -5.4176299, -90.8139570],
        cutoff=_CUT,
    )


def _rock_salt(
    spacing: float,
    jitter: float = 0.0,
    temperature: float = 0.0,
    seed: int = 0,
) -> molpy.Frame:
    """Neutral 64-ion rock-salt NaCl cell (Na⁺ +1 e, Cl⁻ −1 e).

    @param spacing: Nearest-neighbour spacing [Å]; the cubic cell edge is
           4·spacing.
    @param jitter: Uniform displacement in [−jitter, jitter] per axis [Å].
    @param temperature: Maxwell–Boltzmann velocities at this temperature
           [K], total momentum removed; no velocity columns when 0.
    @param seed: RNG seed for the jitter and the velocities.
    """
    rng = np.random.default_rng(seed)
    idx = np.indices((_SIDE, _SIDE, _SIDE)).reshape(3, -1).T
    sodium = idx.sum(axis=1) % 2 == 0
    pos = idx * spacing + rng.uniform(-jitter, jitter, size=(_N, 3))
    mass = np.where(sodium, _MASS_NA, _MASS_CL)
    atoms = molpy.Block()
    atoms.insert("x", pos[:, 0].copy())
    atoms.insert("y", pos[:, 1].copy())
    atoms.insert("z", pos[:, 2].copy())
    atoms.insert("charge", np.where(sodium, 1.0, -1.0))
    atoms.insert("mass", mass)
    atoms.insert("atomic_number", np.where(sodium, 11, 17).astype(np.uint32))
    atoms.insert(
        "type_id", np.where(sodium, _TYPE_NA, _TYPE_CL).astype(np.uint32)
    )
    if temperature > 0.0:
        sd = np.sqrt(_BOLTZ * temperature / (mass * _MVV2E))
        vel = rng.normal(size=(_N, 3)) * sd[:, None]
        vel -= (mass[:, None] * vel).sum(axis=0) / mass.sum()
        atoms.insert("vx", vel[:, 0].copy())
        atoms.insert("vy", vel[:, 1].copy())
        atoms.insert("vz", vel[:, 2].copy())
    frame = molpy.Frame()
    frame["atoms"] = atoms
    frame.box = molpy.Box.cube(_SIDE * spacing)
    return frame


def _melt_md(integrator: GearPC, accuracy: float = 1e-6) -> MD:
    """Tosi–Fumi NaCl with Ewald electrostatics, integrated by *integrator*."""
    sim = MD(units="metal")
    sim.set_integrator(integrator)
    sim.set_neighbor(NeighborList(_CUT, 0.5))
    sim.set_evaluator(_tosi_fumi())
    sim.set_evaluator(CoulLong(_ALPHA, _CUT))
    sim.set_evaluator(KSpaceEwald(_ALPHA, accuracy))
    return sim


def _constrained_temperature(frame: molpy.Frame, kind: str) -> float:
    """The temperature a Gauss thermostat of *kind* holds (field along x),
    from the frame's own mass, type and velocity columns.

    kinetic: every axis relative to the centre-of-mass velocity, 3N − 3
    dof. flow: x relative to each species' mean velocity, y and z
    barycentric, 3N − 2 − n_species dof. transverse: y and z barycentric,
    2N − 2 dof.
    """
    atoms = frame["atoms"]
    mass = np.asarray(atoms["mass"], dtype=np.float64)
    species = np.asarray(atoms["type_id"])
    vel = np.stack(
        [np.asarray(atoms[c], dtype=np.float64) for c in ("vx", "vy", "vz")],
        axis=1,
    )
    w = vel - (mass[:, None] * vel).sum(axis=0) / mass.sum()
    n = len(mass)
    if kind == "kinetic":
        axes, dof = (0, 1, 2), 3 * n - 3
    elif kind == "flow":
        kinds = np.unique(species)
        for s in kinds:
            sel = species == s
            w[sel, 0] = vel[sel, 0] - vel[sel, 0].mean()
        axes, dof = (0, 1, 2), 3 * n - 2 - len(kinds)
    else:
        axes, dof = (1, 2), 2 * n - 2
    twice_k = sum(float(np.sum(mass * w[:, d] ** 2)) for d in axes)
    return _MVV2E * twice_k / (dof * _BOLTZ)


class TestPairBornCutoffTreatment:
    """``energy_shift`` and ``set_tail_correction`` on a Born dimer.

    The dimer is two type-1 atoms 2.5 Å apart in a 20 Å cubic cell, with
    the LAMMPS dimer coefficients of PairBorn.AnalyticEnergyAndForceLammpsDimer
    (tests/cpu/potentials/test_pair_born.cpp).
    """

    _A, _RHO, _SIGMA, _C, _D = 1000.0, 3.0, 1.0, 50.0, 20.0
    _CUTOFF, _R, _BOX = 8.0, 2.5, 20.0

    def _u(self, r: float) -> float:
        return (
            self._A * math.exp((self._SIGMA - r) / self._RHO)
            - self._C / r**6
            + self._D / r**8
        )

    def _born(self, energy_shift: bool = True) -> Born:
        return Born(
            type_i=[1],
            type_j=[1],
            A=[self._A],
            rho=[self._RHO],
            sigma=[self._SIGMA],
            C=[self._C],
            D=[self._D],
            cutoff=self._CUTOFF,
            energy_shift=energy_shift,
        )

    def _energy(self, born: Born) -> float:
        atoms = molpy.Block()
        atoms.insert("x", np.array([0.0, self._R]))
        atoms.insert("y", np.zeros(2))
        atoms.insert("z", np.zeros(2))
        atoms.insert("mass", np.full(2, 39.948))
        atoms.insert("atomic_number", np.full(2, 18, dtype=np.uint32))
        atoms.insert("type_id", np.full(2, 1, dtype=np.uint32))
        frame = molpy.Frame()
        frame["atoms"] = atoms
        frame.box = molpy.Box.cube(self._BOX)
        sim = MD(units="metal")
        sim.set_integrator(VelocityVerlet(_DT))
        sim.set_evaluator(born)
        sim.run(frame, 0)
        return float(frame.meta["energy"])

    def test_unshifted_energy_is_the_bare_potential(self) -> None:
        """``energy_shift=False`` (LAMMPS ``pair_modify shift no``) leaves
        U(r) itself; the default subtracts U(r_c) = −1.4e-4 eV."""
        assert self._energy(self._born(energy_shift=False)) == pytest.approx(
            self._u(self._R), abs=1e-9
        )
        assert self._energy(self._born()) == pytest.approx(
            self._u(self._R) - self._u(self._CUTOFF), abs=1e-9
        )

    def test_tail_correction_adds_the_homogeneous_fluid_tail(self) -> None:
        """``set_tail_correction(True)`` adds 2π N²/V ∫_{r_c}^∞ r² U(r) dr
        (LAMMPS ``pair_modify tail yes``, one type, N = 2), with the
        integral in closed form:
        A ρ e^{(σ−r_c)/ρ} (r_c² + 2ρ r_c + 2ρ²) − C/(3 r_c³) + D/(5 r_c⁵).
        """
        rc, rho = self._CUTOFF, self._RHO
        integral = (
            self._A
            * rho
            * math.exp((self._SIGMA - rc) / rho)
            * (rc**2 + 2.0 * rho * rc + 2.0 * rho**2)
            - self._C / (3.0 * rc**3)
            + self._D / (5.0 * rc**5)
        )
        tail = 2.0 * math.pi * 2**2 / self._BOX**3 * integral
        plain = self._energy(self._born())
        with_tail = self._energy(self._born().set_tail_correction(True))
        assert with_tail - plain == pytest.approx(tail, rel=1e-12)


class TestKSpaceEwald:
    def test_coul_long_with_ewald_matches_madelung(self) -> None:
        """CoulLong + KSpaceEwald reproduce the Madelung energy of perfect
        rock salt, E = −½ N M k_e / d, at d = 2.8201 Å.

        r_c = 5 Å and α = 3.5/r_c leave the real-space truncation
        (erfc 3.5 ≈ 7e-7) as the error: 3.8e-8 measured at accuracy 1e-8.
        At the default accuracy 1e-5 the reciprocal cutoff alone costs
        5.6e-6, so the bound 1e-6 also holds the solver to the accuracy
        passed in.
        """
        d, rc = 2.8201, 5.0
        alpha = 3.5 / rc
        sim = MD(units="metal")
        sim.set_integrator(VelocityVerlet(_DT))
        sim.set_neighbor(NeighborList(rc, 0.3))
        sim.set_evaluator(CoulLong(alpha, rc))
        sim.set_evaluator(KSpaceEwald(alpha, accuracy=1e-8))
        frame = _rock_salt(d)
        sim.run(frame, 0)
        energy = float(frame.meta["energy"])
        reference = -0.5 * _N * _MADELUNG * _COULOMB_EV_A / d
        assert abs(energy / reference - 1.0) < 1e-6


class TestConfigurationalTemperature:
    def test_born_dimer_matches_the_closed_form(self) -> None:
        """T_C1 = Σ|F|² / (k_B Σ∇²U) of a Born dimer:
        N = 2φ'², D = 2(φ'' + 2φ'/r), at the configuration the step leaves
        in the frame (the observer runs after the force evaluation of that
        configuration).

        Tosi–Fumi Na–Cl coefficients at r = 3.0 Å, where φ'' + 2φ'/r > 0.
        """
        a, rho, sigma, c, dd = 0.2109630, 0.317, 2.755, 4.3690564, -5.4176299
        atoms = molpy.Block()
        atoms.insert("x", np.array([5.0, 8.0]))
        atoms.insert("y", np.array([5.0, 5.0]))
        atoms.insert("z", np.array([5.0, 5.0]))
        atoms.insert("mass", np.array([_MASS_NA, _MASS_CL]))
        atoms.insert("atomic_number", np.array([11, 17], dtype=np.uint32))
        atoms.insert("type_id", np.array([_TYPE_NA, _TYPE_CL], dtype=np.uint32))
        frame = molpy.Frame()
        frame["atoms"] = atoms
        frame.box = molpy.Box.cube(20.0)
        observer = ConfigurationalTemperature()
        sim = MD(units="metal")
        sim.set_integrator(VelocityVerlet(_DT))
        sim.set_evaluator(
            Born(
                type_i=[_TYPE_NA],
                type_j=[_TYPE_CL],
                A=[a],
                rho=[rho],
                sigma=[sigma],
                C=[c],
                D=[dd],
                cutoff=8.0,
            )
        )
        sim.set_observer(observer)
        sim.run(frame, 1)

        x = np.asarray(frame["atoms"]["x"], dtype=np.float64)
        r = float(x[1] - x[0])
        e = a * math.exp((sigma - r) / rho)
        d1 = -e / rho + 6.0 * c / r**7 - 8.0 * dd / r**9
        d2 = e / rho**2 - 42.0 * c / r**8 + 72.0 * dd / r**10
        force_sq = 2.0 * d1**2
        laplacian = 2.0 * (d2 + 2.0 * d1 / r)
        assert observer.force_sq == pytest.approx(force_sq, rel=1e-12)
        assert observer.laplacian == pytest.approx(laplacian, rel=1e-12)
        assert observer.temperature == pytest.approx(
            force_sq / (_BOLTZ * laplacian), rel=1e-12
        )


class TestGearPC:
    """Each ``MD.run`` is an independent simulation (the molpy overload
    tears the run down), so every Gear history, ξ and run average below
    belongs to one ``run`` call."""

    @pytest.mark.parametrize("kind", ["kinetic", "flow", "transverse"])
    def test_gauss_thermostat_holds_its_temperature_under_a_field(
        self, kind: str
    ) -> None:
        """Gaussian thermostats of eqs. (9)–(11) keep their constrained
        temperature exactly while a 0.5 V/Å field along x drives the melt.

        The start is a hot (1800 K) jittered lattice and the target 1500 K:
        the first projection lands on the target, and after 60 steps the
        temperature measured from the frame, with each kind's own peculiar
        velocity and degrees of freedom, is still there. What the last
        projection removed is truncation error (≤ 3.3e-7 measured over
        40–100 steps), not the field's work: an isokinetic multiplier
        without the field term leaves 1e-2.
        """
        t0 = 1500.0
        gear = GearPC(_DT, GaussThermostat(kind, t0, axis=0))
        sim = _melt_md(gear)
        sim.set_evaluator(FixEfield(0.5, 0.0, 0.0))
        frame = _rock_salt(_SPACING, jitter=0.25, temperature=1800.0, seed=5)
        sim.run(frame, 60)
        assert abs(_constrained_temperature(frame, kind) / t0 - 1.0) < 1e-12
        assert gear.constraint_drift < 1e-6

    def test_configurational_thermostat_drives_tconf_to_the_target(
        self,
    ) -> None:
        """Eqs. (12a–c) with Q = 20 Å²/ps² cool the melt's configurational
        temperature to 700 K: after a 300-step run from the 1500 K start,
        T_CONF = <N>/(k_B <D>) of a second 300-step run lies within 10 % of
        the target (720 K measured; unthermostatted, the same protocol
        gives 1090 K). The second run averages its own 300 evaluations.
        """
        t0 = 700.0
        observer = ConfigurationalTemperature()
        sim = _melt_md(GearPC(_DT, ConfigurationalThermostat(t0, 20.0)), 1e-4)
        sim.set_observer(observer)
        frame = _rock_salt(_SPACING, jitter=0.25, temperature=1500.0, seed=7)
        sim.run(frame, 300)
        sim.run(frame, 300)
        assert observer.samples == 300
        assert abs(observer.mean_temperature / t0 - 1.0) < 0.1

    def test_a_second_thermostat_beside_gear_is_refused(self) -> None:
        """GearPC thermostats inside its own step; another thermostat would
        act on the same velocities twice."""
        sim = MD(units="metal")
        sim.set_integrator(GearPC(_DT, GaussThermostat("kinetic", 1500.0)))
        with pytest.raises(ValueError, match="GearPC"):
            sim.set_thermostat(NoseHoover(1500.0, 0.1, _DT))
        sim = MD(units="metal")
        sim.set_thermostat(NoseHoover(1500.0, 0.1, _DT))
        with pytest.raises(ValueError, match="GearPC"):
            sim.set_integrator(GearPC(_DT))

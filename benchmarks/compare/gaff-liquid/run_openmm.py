#!/usr/bin/env python3
"""OpenMM side of the general-GAFF benchmark.

Reads the same prmtop/inpcrd that ``prmtop_to_lammps.py`` converted for
LAMMPS and Atomiverse, and sets the same physics: 9 Å cutoff, no switching,
no dispersion correction, Amber 1-4 scaling (from the prmtop), explicit PME
alpha + grid (OpenMM's B-spline order is fixed at 5).

Modes
-----
``sp``  single point. Writes JSON with energy components in kcal/mol and
        per-atom forces in kcal/mol/Å, in prmtop atom order:
        bond, angle, torsion (proper + improper), lj (incl. scaled 1-4),
        coulomb (direct + reciprocal + exceptions), coulomb_recip.
``md``  NVT Nose-Hoover (chain 1, no MTS: LAMMPS ``fix nvt ... tchain 1`` and
        Atomiverse's single-chain NoseHoover),
        HBonds constraints optional; times ``--steps`` after ``--warmup``
        and reports steps/s and ns/day.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

KJ_PER_KCAL = 4.184


def snap_torsion_phases(force) -> None:
    """Snap prmtop phases to exact 0 / pi.

    A prmtop stores pi as ``3.14159400E+00``; molpy writes the LAMMPS
    ``fourier`` phase as exactly 180 degrees. Without this, OpenMM's torsion
    energy differs from LAMMPS/Atomiverse by ~1e-7 relative purely from the
    file's truncated pi, which would mask real disagreements.
    """
    import math

    import openmm.unit as u

    for k in range(force.getNumTorsions()):
        i, j, l, m, n, phase, kk = force.getTorsionParameters(k)
        rad = phase.value_in_unit(u.radian)
        for exact in (0.0, math.pi):
            if abs(rad - exact) < 1.0e-5:
                force.setTorsionParameters(k, i, j, l, m, n, exact, kk)
                break


def build_system(args: argparse.Namespace, constraints: bool):
    import openmm as mm
    import openmm.app as app
    import openmm.unit as u

    prmtop = app.AmberPrmtopFile(str(args.dir / "system.prmtop"))
    inpcrd = app.AmberInpcrdFile(str(args.dir / "system.inpcrd"))
    system = prmtop.createSystem(
        nonbondedMethod=app.PME,
        nonbondedCutoff=args.cut / 10.0 * u.nanometer,
        constraints=app.HBonds if constraints else None,
        rigidWater=False,
        removeCMMotion=False,
    )
    for index, force in enumerate(system.getForces()):
        force.setForceGroup(index)
        if isinstance(force, mm.PeriodicTorsionForce):
            snap_torsion_phases(force)
        if isinstance(force, mm.NonbondedForce):
            force.setUseDispersionCorrection(False)
            force.setUseSwitchingFunction(False)
            grid = args.mesh
            force.setPMEParameters(args.gewald * 10.0, grid[0], grid[1],
                                   grid[2])
            force.setReciprocalSpaceForceGroup(31)
        if isinstance(force, mm.CMMotionRemover):
            raise RuntimeError("unexpected CMMotionRemover")
    return prmtop, inpcrd, system


def platform(args: argparse.Namespace):
    import openmm as mm

    plat = mm.Platform.getPlatformByName(args.platform)
    props = {}
    if args.platform in ("CUDA", "OpenCL"):
        props["Precision"] = args.precision
    if args.platform == "CUDA":
        props["DeterministicForces"] = "false"
    return plat, props


def single_point(args: argparse.Namespace) -> dict:
    import numpy as np
    import openmm as mm
    import openmm.unit as u

    _, inpcrd, system = build_system(args, constraints=False)
    nb = next(f for f in system.getForces()
              if isinstance(f, mm.NonbondedForce))
    plat, props = platform(args)
    ctx = mm.Context(system, mm.VerletIntegrator(0.001), plat, props)
    ctx.setPeriodicBoxVectors(*inpcrd.boxVectors)
    ctx.setPositions(inpcrd.positions)

    def energy(groups: set[int]) -> float:
        state = ctx.getState(getEnergy=True, groups=groups)
        return state.getPotentialEnergy().value_in_unit(
            u.kilocalorie_per_mole)

    by_class: dict[str, set[int]] = {}
    for index, force in enumerate(system.getForces()):
        by_class.setdefault(type(force).__name__, set()).add(index)
    nb_group = by_class["NonbondedForce"]
    out = {
        "bond": energy(by_class.get("HarmonicBondForce", set())),
        "angle": energy(by_class.get("HarmonicAngleForce", set())),
        "torsion": energy(by_class.get("PeriodicTorsionForce", set())),
    }
    full_nb = energy(nb_group | {31})
    recip = energy({31})
    # Split LJ from Coulomb by zeroing charges (LJ-only nonbonded energy).
    charges = [nb.getParticleParameters(i) for i in range(nb.getNumParticles())]
    exceptions = [nb.getExceptionParameters(i)
                  for i in range(nb.getNumExceptions())]
    for i, (q, sig, eps) in enumerate(charges):
        nb.setParticleParameters(i, 0.0, sig, eps)
    for k, (i, j, qq, sig, eps) in enumerate(exceptions):
        nb.setExceptionParameters(k, i, j, 0.0, sig, eps)
    nb.updateParametersInContext(ctx)
    lj = energy(nb_group | {31})
    for i, (q, sig, eps) in enumerate(charges):
        nb.setParticleParameters(i, q, sig, eps)
    for k, (i, j, qq, sig, eps) in enumerate(exceptions):
        nb.setExceptionParameters(k, i, j, qq, sig, eps)
    nb.updateParametersInContext(ctx)
    out.update({"lj": lj, "coulomb": full_nb - lj, "coulomb_recip": recip})
    state = ctx.getState(getEnergy=True, getForces=True)
    out["total"] = state.getPotentialEnergy().value_in_unit(
        u.kilocalorie_per_mole)
    forces = state.getForces(asNumpy=True).value_in_unit(
        u.kilocalorie_per_mole / u.angstrom)
    np.save(args.out.with_suffix(".forces.npy"), np.asarray(forces))
    return {"energies_kcal_mol": out, "platform": plat.getName(),
            "properties": props, "openmm": mm.__version__}


def dynamics(args: argparse.Namespace) -> dict:
    import openmm as mm
    import openmm.unit as u

    _, inpcrd, system = build_system(args, constraints=args.shake)
    if args.ensemble == "nve":
        integrator = mm.VerletIntegrator(args.dt * u.femtosecond)
    else:
        integrator = mm.NoseHooverIntegrator(args.temp * u.kelvin,
                                             10.0 / u.picosecond,
                                             args.dt * u.femtosecond, 1, 1, 1)
    integrator.setConstraintTolerance(1.0e-5)
    plat, props = platform(args)
    ctx = mm.Context(system, integrator, plat, props)
    ctx.setPeriodicBoxVectors(*inpcrd.boxVectors)
    ctx.setPositions(inpcrd.positions)
    ctx.setVelocitiesToTemperature(args.temp * u.kelvin, args.seed)
    integrator.step(args.warmup)
    ctx.getState(getEnergy=True)  # drain the queue before timing
    start = time.perf_counter()
    done = 0
    series = []  # (step, total energy kcal/mol), thermo cadence
    while done < args.steps:
        chunk = min(args.thermo, args.steps - done)
        integrator.step(chunk)
        done += chunk
        state = ctx.getState(getEnergy=True)  # thermo readback, as LAMMPS
        series.append((done, (state.getPotentialEnergy()
                              + state.getKineticEnergy()).value_in_unit(
                                  u.kilocalorie_per_mole)))
    wall = time.perf_counter() - start
    steps_per_s = args.steps / wall
    return {"steps": args.steps, "wall_s": wall, "steps_per_s": steps_per_s,
            "ns_per_day": steps_per_s * args.dt * 86400.0 * 1.0e-6,
            "platform": plat.getName(), "properties": props,
            "openmm": mm.__version__, "atoms": system.getNumParticles(),
            "constraints": system.getNumConstraints(),
            "etotal_kcal_mol": series}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--mode", choices=("sp", "md"), default="sp")
    ap.add_argument("--out", type=Path, required=True,
                    help="JSON result path (sp also writes .forces.npy)")
    ap.add_argument("--platform", default="CUDA")
    ap.add_argument("--precision", default="mixed",
                    choices=("single", "mixed", "double"))
    ap.add_argument("--cut", type=float, default=9.0, help="Å")
    ap.add_argument("--gewald", type=float, default=0.35, help="1/Å")
    ap.add_argument("--mesh", type=int, nargs=3, default=(64, 64, 64))
    ap.add_argument("--shake", action=argparse.BooleanOptionalAction,
                    default=True)
    ap.add_argument("--dt", type=float, default=2.0, help="fs")
    ap.add_argument("--ensemble", choices=("nvt", "nve"), default="nvt")
    ap.add_argument("--temp", type=float, default=300.0)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--thermo", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20261020)
    args = ap.parse_args()
    result = single_point(args) if args.mode == "sp" else dynamics(args)
    result["settings"] = {k: (str(v) if isinstance(v, Path) else v)
                          for k, v in vars(args).items()}
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

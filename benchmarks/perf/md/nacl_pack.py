#!/usr/bin/env python3
"""Throughput of independent 512-ion molten-NaCl NEMD runs sharing one GPU.

The model is the one of ``bench_nacl_ewald`` (Petravic & Delhommelle, J.
Chem. Phys. 118, 7477 (2003)): Tosi-Fumi ``Born`` + ``CoulLong`` +
``KSpaceEwald`` at the paper's Ewald setting (kappa = 1.8 pi / L, cutoff
L/2, |k| <= 6 * 2 pi / L, skin 2 A), integrated by ``GearPC`` with the
Gauss isokinetic or the configurational thermostat, through the Python
drivers a production script uses.

    nacl_pack.py STATE THERMOSTAT [--device cuda|cpu] [--pack K]
                 [--steps N] [--warm N] [--melt N] [--field E]

``STATE`` is ``liquid`` (1500 K, L = 25.4904 A, 1 fs) or ``fluid`` (3200 K,
L = 49.8957 A, 0.5 fs); ``THERMOSTAT`` is ``gauss`` or ``config``.

One process melts the rock-salt start for ``--melt`` Gauss-thermostatted
steps (the expanded lattice has a negative Laplacian sum, which the
configurational thermostat refuses), runs ``--warm`` untimed steps of the
thermostat under test, then times ``--steps`` steps and prints one ``BENCH``
line. ``--pack K`` starts K such processes, each bound to its own core of
this process's affinity mask, releases their timed intervals together once
all are warm, and prints the aggregate rate (the sum of the processes'
steps/s over the common interval).

The GPU is shared by whatever the environment provides: start an NVIDIA MPS
daemon first (``nacl_pack.sh``) for concurrent execution; without one the
processes are time-sliced.
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass

os.environ.setdefault("ATV_TIMER", "0")
os.environ.setdefault("ATV_LOG_LEVEL", "warn")
os.environ.setdefault("OMP_NUM_THREADS", "1")

# Atomiverse reads the three switches above when it is imported.
import molpy
import numpy as np
from atomiverse.longrange import CoulLong, KSpaceEwald
from atomiverse.md import MD, FixEfield
from atomiverse.md.integrator import GearPC
from atomiverse.md.thermostat import (
    ConfigurationalThermostat,
    GaussThermostat,
)
from atomiverse.neighbor import NeighborList
from atomiverse.potentials.pair import Born

SIDE = 8  # 8^3 simple-cubic sites with alternating charge
MASS_NA, MASS_CL = 22.9898, 35.453  # [g/mol]
TYPE_NA, TYPE_CL = 1, 2
BOLTZ = 8.617343e-5  # [eV/K], LAMMPS units metal
MVV2E = 1.0364269e-4  # [eV per g/mol A^2/ps^2]
SKIN = 2.0  # [A]
EWALD_ACCURACY = 1.4947e-5  # e^(-100/9): |k| <= 6 * 2 pi / L
CONFIGURATIONAL_Q = 100.0  # [A^2/ps^2]
#: Tosi-Fumi NaCl rows (type_i, type_j, A [eV], rho [A], sigma [A],
#: C [eV A^6], D [eV A^8]); ``Born`` adds +D/r^8.
TOSI_FUMI = (
    (TYPE_NA, TYPE_NA, 0.263705848, 0.317, 2.340, 1.048583006, -0.49932529),
    (TYPE_NA, TYPE_CL, 0.210964679, 0.317, 2.755, 6.99055303, -8.675775756),
    (TYPE_CL, TYPE_CL, 0.158223509, 0.317, 3.170, 72.40215779, -145.4284714),
)


@dataclass(frozen=True)
class StatePoint:
    name: str
    length: float  # cubic edge [A]
    temperature: float  # [K]
    dt: float  # [ps]


STATES = {
    "liquid": StatePoint("liquid", 25.4904, 1500.0, 0.001),
    "fluid": StatePoint("fluid", 49.8957, 3200.0, 0.0005),
}


def rock_salt(state: StatePoint, seed: int) -> molpy.Frame:
    """Rock salt at the state's density, Maxwell velocities, zero momentum."""
    rng = np.random.default_rng(seed)
    idx = np.indices((SIDE, SIDE, SIDE)).reshape(3, -1).T
    sodium = idx.sum(axis=1) % 2 == 0
    pos = (idx + 0.25) * (state.length / SIDE)
    mass = np.where(sodium, MASS_NA, MASS_CL)
    vel = (
        rng.normal(size=pos.shape)
        * np.sqrt(BOLTZ * state.temperature / (mass * MVV2E))[:, None]
    )
    vel -= (mass[:, None] * vel).sum(axis=0) / mass.sum()
    atoms = molpy.Block()
    for axis, name in enumerate("xyz"):
        atoms.insert(name, pos[:, axis].copy())
        atoms.insert("v" + name, vel[:, axis].copy())
    atoms.insert("charge", np.where(sodium, 1.0, -1.0))
    atoms.insert("mass", mass)
    atoms.insert("atomic_number", np.where(sodium, 11, 17).astype(np.uint32))
    atoms.insert("type_id", np.where(sodium, TYPE_NA, TYPE_CL).astype(np.uint32))
    frame = molpy.Frame()
    frame["atoms"] = atoms
    frame.box = molpy.Box.cube(state.length)
    return frame


def make_md(state: StatePoint, thermostat: str, device: str, field: float) -> MD:
    """The model on *device* with the Gear integrator and *thermostat*."""
    if thermostat == "gauss":
        law = GaussThermostat("kinetic", state.temperature, 0)
    else:
        law = ConfigurationalThermostat(state.temperature, CONFIGURATIONAL_Q)
    cutoff = 0.5 * state.length
    alpha = 1.8 * math.pi / state.length
    columns = list(zip(*TOSI_FUMI))
    sim = MD(units="metal", device=device)
    sim.set_integrator(GearPC(state.dt, law, device=device))
    sim.set_neighbor(NeighborList(cutoff, SKIN))
    sim.set_evaluator(
        Born(
            type_i=list(columns[0]),
            type_j=list(columns[1]),
            A=list(columns[2]),
            rho=list(columns[3]),
            sigma=list(columns[4]),
            C=list(columns[5]),
            D=list(columns[6]),
            cutoff=cutoff,
        )
    )
    sim.set_evaluator(CoulLong(alpha, cutoff))
    sim.set_evaluator(KSpaceEwald(alpha, EWALD_ACCURACY))
    if field != 0.0:
        sim.set_evaluator(FixEfield(field, 0.0, 0.0))
    return sim


def run_worker(args: argparse.Namespace) -> None:
    """One simulation: melt, warm up, wait for the release, time."""
    if args.core is not None:
        os.sched_setaffinity(0, {args.core})
    state = STATES[args.state]
    frame = rock_salt(state, args.seed)
    if args.melt > 0:
        make_md(state, "gauss", args.device, 0.0).run(frame, args.melt)
    sim = make_md(state, args.thermostat, args.device, args.field)
    sim.run(frame, args.warm)
    if args.barrier:
        print("READY", flush=True)
        sys.stdin.readline()
    start = time.time()
    sim.run(frame, args.steps)
    stop = time.time()
    print(
        f"BENCH device={args.device} state={state.name} "
        f"thermostat={args.thermostat} field={args.field:g} steps={args.steps} "
        f"ms_per_step={1e3 * (stop - start) / args.steps:.6f} "
        f"start={start:.6f} stop={stop:.6f} "
        f"energy={float(frame.meta['energy']):.10g}",
        flush=True,
    )


def bench_fields(line: str) -> dict[str, str]:
    return dict(item.split("=", 1) for item in line.split()[1:])


def collect(workers: list[subprocess.Popen]) -> list[dict[str, str]]:
    """Release the workers together once all are warm; their BENCH fields."""
    for worker in workers:
        # The library logs on stdout too: skip to the worker's READY line.
        if not any(line.strip() == "READY" for line in worker.stdout):
            raise SystemExit(f"worker {worker.pid} exited before its timed run")
    for worker in workers:
        worker.stdin.write("\n")
        worker.stdin.flush()
    results = []
    for worker in workers:
        out, _ = worker.communicate()
        if worker.returncode != 0:
            raise SystemExit(f"worker {worker.pid} exited with {worker.returncode}")
        line = next(ln for ln in out.splitlines() if ln.startswith("BENCH "))
        results.append(bench_fields(line))
    return results


def run_pack(args: argparse.Namespace) -> None:
    """K workers on K cores, timed over one common interval."""
    cores = sorted(os.sched_getaffinity(0))
    if args.pack > len(cores):
        raise SystemExit(
            f"--pack {args.pack} needs one core per process; this process "
            f"may use {len(cores)}"
        )
    base = [sys.executable, os.path.abspath(__file__), args.state, args.thermostat]
    for key in ("device", "steps", "warm", "melt", "field"):
        base += [f"--{key}", str(getattr(args, key))]
    workers = [
        subprocess.Popen(
            base + ["--barrier", "--core", str(cores[k]), "--seed", str(args.seed + k)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )
        for k in range(args.pack)
    ]
    try:
        results = collect(workers)
    finally:
        for worker in workers:
            if worker.poll() is None:
                worker.kill()
    rates = [1e3 / float(r["ms_per_step"]) for r in results]
    # The interval every worker was inside its timed run.
    common = min(float(r["stop"]) for r in results) - max(
        float(r["start"]) for r in results
    )
    longest = max(float(r["stop"]) - float(r["start"]) for r in results)
    print(
        f"PACK device={args.device} state={args.state} "
        f"thermostat={args.thermostat} field={args.field:g} K={args.pack} "
        f"steps={args.steps} ms_per_step_mean={1e3 * args.pack / sum(rates):.6f} "
        f"ms_per_step_worst={1e3 / min(rates):.6f} "
        f"steps_per_s_total={sum(rates):.2f} overlap={common / longest:.3f}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("state", choices=sorted(STATES))
    parser.add_argument("thermostat", choices=("gauss", "config"))
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--pack", type=int, default=0, help="concurrent processes")
    parser.add_argument("--steps", type=int, default=20000, help="timed steps")
    parser.add_argument("--warm", type=int, default=500, help="untimed steps")
    parser.add_argument("--melt", type=int, default=2000, help="Gauss melt steps")
    parser.add_argument("--field", type=float, default=0.0, help="E_x [V/A]")
    parser.add_argument("--seed", type=int, default=20031)
    parser.add_argument("--core", type=int, help="bind to this core (worker)")
    parser.add_argument("--barrier", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.steps < 1 or args.warm < 1 or args.melt < 0 or args.pack < 0:
        parser.error("require --steps >= 1, --warm >= 1, --melt >= 0, --pack >= 0")
    if args.pack > 0:
        run_pack(args)
    else:
        run_worker(args)


if __name__ == "__main__":
    main()

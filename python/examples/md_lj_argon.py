#!/usr/bin/env python3
"""Lennard-Jones argon under a Langevin thermostat (NVT molecular dynamics).

Demonstrates the always-available classical-MD surface of the Python
package, end to end and with no optional build flag:

* building system state as a ``molrs.Frame`` — the only public currency
  across the binding boundary (x/y/z, ``mass``, ``atomic_number``, box);
* assembling the ``MD`` driver from components with the categorical
  setters (``set_integrator`` / ``set_observer`` / ``set_thermostat`` /
  ``set_neighbor`` / ``set_evaluator`` / ``set_thermo``) — the same shape
  as the C++ ``examples/07_md_lj.cpp``, whose Ar parameters this reuses;
* reading results back out of the *same* frame afterwards: potential and
  kinetic energy from ``frame.meta``, moved atoms from the x/y/z columns.

MolCrafts stack only (molrs + numpy) — no ASE, no matplotlib. Headless,
single process, well under a second of compute. It self-checks its own
result and **exits nonzero** when a check fails, so it can be smoke-run.

Run — development tree, extension built into ``build-py``::

    cd /path/to/Atomiverse
    PYTHONPATH="$PWD/build-py:$PWD/python/src" python python/examples/md_lj_argon.py

Run — installed wheel (pure-Python phase 1, then the native phase 2)::

    pip install atomiverse
    python -m atomiverse config md --src /path/to/Atomiverse
    python -m atomiverse compile -j 8
    python -m atomiverse install
    python md_lj_argon.py

Units: Å, eV, fs, amu, K.
"""

from __future__ import annotations

import sys

import molrs
import numpy as np

import atomiverse as av

# ── Lennard-Jones argon, the parameters of examples/07_md_lj.cpp ──
_EPS: float = 0.0104  # well depth [eV]
_SIGMA: float = 3.4  # collision diameter [Å]
_MASS_AR: float = 39.948  # [amu]
_Z_AR: int = 18  # atomic number

#: Simple-cubic 4×4×4 seed lattice — 64 atoms at 4.0 Å spacing, i.e. a
#: 16 Å cube at 0.0156 Å⁻³. The spacing sits just outside the LJ minimum
#: separation r* = 2^(1/6)σ = 3.816 Å, so the lattice starts on the
#: attractive side of the well rather than on the repulsive wall.
_N_SIDE: int = 4
_SPACING: float = 4.0  # [Å]
_BOX: float = _N_SIDE * _SPACING  # cubic cell edge [Å]

#: Pair cutoff 2σ with a 0.5 Å neighbor skin: cutoff + skin = 7.3 Å stays
#: below half the 16 Å cell edge, which minimum-image pair distances
#: require and nothing in the API checks for you.
_CUTOFF: float = 2.0 * _SIGMA  # [Å]
_SKIN: float = 0.5  # [Å]

# ── Thermostatted run ──
_DT: float = 1.0  # timestep [fs]
_TEMPERATURE: float = 94.4  # target temperature [K]
_DAMP: float = 100.0  # Langevin damping time [fs]
_SEED: int = 42
_N_STEPS: int = 200  # 0.2 ps
_THERMO_EVERY: int = 100  # LAMMPS-style print cadence

#: Boltzmann constant in engine units — the value core/units.h carries
#: (`kB_ev_per_k`, CODATA); only used to report a temperature.
_KB_EV_PER_K: float = 8.617333262e-5  # [eV/K]

#: Minimum motion a 0.2 ps run at 94 K must produce [Å]. Argon's thermal
#: speed there is ~2.4 Å/ps, so this leaves two orders of magnitude of
#: slack — it catches a frozen or unwritten position column, not a
#: physics change.
_MIN_DISPLACEMENT: float = 0.01  # [Å]


def build_argon_frame() -> "molrs.Frame":
    """Simple-cubic argon lattice as a ``molrs.Frame``.

    @return A 64-atom frame with x/y/z [Å], ``mass`` [amu] and
            ``atomic_number`` [Z] columns in a cubic cell.
    """
    grid = np.arange(_N_SIDE, dtype=np.float64) * _SPACING
    xs, ys, zs = (axis.ravel() for axis in np.meshgrid(grid, grid, grid, indexing="ij"))
    n_atoms = xs.size

    atoms = molrs.Block()
    atoms.insert("x", np.ascontiguousarray(xs))
    atoms.insert("y", np.ascontiguousarray(ys))
    atoms.insert("z", np.ascontiguousarray(zs))
    atoms.insert("mass", np.full(n_atoms, _MASS_AR, dtype=np.float64))
    atoms.insert("atomic_number", np.full(n_atoms, _Z_AR, dtype=np.uint32))

    frame = molrs.Frame()
    frame["atoms"] = atoms
    frame.box = molrs.Box.cube(_BOX)
    return frame


def positions(frame: "molrs.Frame") -> "np.ndarray":
    """Read the x/y/z columns of *frame* as an (N, 3) array [Å].

    @param frame: Any molrs.Frame carrying an ``atoms`` block.
    @return (N, 3) positions [Å], as stored — the integrator keeps them
            wrapped into the primary cell.
    """
    atoms = frame["atoms"]
    return np.stack(
        [np.asarray(atoms[key], dtype=np.float64) for key in ("x", "y", "z")],
        axis=1,
    )


def main() -> int:
    frame = build_argon_frame()
    start = positions(frame)
    n_atoms = start.shape[0]
    print(f"=== NVT Lennard-Jones argon: {n_atoms} atoms, {_N_STEPS} × {_DT} fs ===")

    # Categorical setters, same vocabulary as cpu::md::MD in C++.
    sim = av.MD()
    sim.set_integrator(av.VelocityVerlet(_DT))
    sim.set_observer(av.InitVelocity(_TEMPERATURE, _SEED))
    sim.set_thermostat(av.Langevin(_TEMPERATURE, _TEMPERATURE, _DAMP, _DT, _SEED))
    sim.set_neighbor(av.NeighborList(_CUTOFF, _SKIN))
    sim.set_evaluator(av.PairLJCut(_EPS, _SIGMA, _CUTOFF))
    sim.set_thermo(_THERMO_EVERY)

    sim.run(frame, _N_STEPS)

    # Results come back on the SAME molrs.Frame the driver borrowed.
    potential = frame.meta["energy"]  # [eV]
    kinetic = frame.meta["kinetic"]  # [eV]
    temperature = 2.0 * kinetic / (3.0 * n_atoms * _KB_EV_PER_K)  # [K]
    # Minimum-image displacement: the integrator wraps atoms back into the
    # primary cell, so a raw difference of the stored columns reads a
    # boundary crossing as a jump of nearly a cell edge.
    delta = positions(frame) - start
    delta -= _BOX * np.round(delta / _BOX)
    displacement = float(np.max(np.linalg.norm(delta, axis=1)))

    print(f"potential energy : {potential:12.6f} eV")
    print(f"kinetic energy   : {kinetic:12.6f} eV")
    print(f"total energy     : {potential + kinetic:12.6f} eV")
    print(f"temperature      : {temperature:12.4f} K   (target {_TEMPERATURE} K)")
    print(f"max displacement : {displacement:12.6f} Å   (minimum image)")

    failures: list[str] = []
    if not np.isfinite(potential):
        failures.append(f"potential energy is not finite: {potential!r} eV")
    if not np.isfinite(kinetic):
        failures.append(f"kinetic energy is not finite: {kinetic!r} eV")
    if not kinetic > 0.0:
        failures.append(f"kinetic energy is not positive: {kinetic!r} eV")
    if not displacement > _MIN_DISPLACEMENT:
        failures.append(
            f"atoms did not move: max displacement {displacement!r} Å "
            f"<= {_MIN_DISPLACEMENT} Å"
        )

    for failure in failures:
        print(f"FAIL: {failure}", file=sys.stderr)
    if failures:
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())

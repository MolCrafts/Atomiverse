"""LJ-MD parity leg for the MPI launcher (mpi-launcher-embedded-python-04).

The SAME script runs serial-python-hosted and launcher-hosted; the compare
step asserts the two runs' results agree. One single run() (not segmented):
each launcher dispatch materializes a fresh driver, so a segmented run would
re-fire InitVelocity per segment — parity is defined over one task.

Usage: <python|atomiverse> atv_lj.py <out.npz>
"""

import sys

import numpy as np

import atomiverse as atv
import molrs

# FCC argon in the regime the in-tree bruteforce oracle validates
# (tests/mpi/test_lj_energy_bruteforce.cpp): >= 3 neighbor cells per axis
# on every rank. Small boxes with 2 cells/axis at 2 ranks silently
# mis-count cross-PBC ghost pairs (~16% energy error observed) — that
# fail-loud gap is filed separately; parity runs in the supported regime.
N_CELLS = 6
A0 = 5.3  # [A] — FCC lattice constant; NN 3.75 A < cutoff 4.0 A
N_STEPS = 200
JITTER = 0.05  # [A] — deterministic lattice perturbation (drives dynamics)

out_path = sys.argv[1]

# Deterministic, decomposition-invariant jitter: cold-start dynamics
# from a perturbed FCC lattice. InitVelocity is NOT used here — its RNG
# streams are decomposition-DEPENDENT (each rank seeds over its local
# atoms), so serial vs 2-rank initial velocities would legitimately
# differ and parity would be meaningless. Zero velocities + perturbed
# positions are identical in both hostings by construction.
BASIS = [(0.0, 0.0, 0.0), (0.5, 0.5, 0.0), (0.5, 0.0, 0.5), (0.0, 0.5, 0.5)]
xs, ys, zs = [], [], []
m = 0
for i in range(N_CELLS):
    for j in range(N_CELLS):
        for k in range(N_CELLS):
            for bx, by, bz in BASIS:
                xs.append((i + bx) * A0 + JITTER * ((m % 5) - 2))
                ys.append((j + by) * A0 + JITTER * (((m + 1) % 5) - 2))
                zs.append((k + bz) * A0 + JITTER * (((m + 2) % 5) - 2))
                m += 1
n = len(xs)

frame = molrs.Frame()
atoms = molrs.Block()
atoms.insert("x", np.array(xs))
atoms.insert("y", np.array(ys))
atoms.insert("z", np.array(zs))
atoms.insert("mass", np.full(n, 39.948))  # Ar [amu]
atoms.insert("type", np.zeros(n, dtype=np.int32))
frame["atoms"] = atoms
frame.simbox = molrs.Box.cube(N_CELLS * A0)

sim = atv.md.MD()
sim.add(atv.VelocityVerlet(0.001))  # dt [fs]
sim.on("force").add(atv.NeighborList(4.0), order=-10).add(
    atv.PairLJCut(0.0104, 3.4, 4.0)  # [eV, A, A] — the bruteforce
    # oracle regime: halo 4.4 A << 15.9 A subdomain, >=3 cells per axis
)
sim.run(frame, N_STEPS)

np.savez(
    out_path,
    energy=np.asarray([frame.meta["energy"]], dtype=np.float64),
    x=np.asarray(frame["atoms"]["x"]),
    y=np.asarray(frame["atoms"]["y"]),
    z=np.asarray(frame["atoms"]["z"]),
)
print(f"lj leg done: E = {float(frame.meta['energy']):.12f} eV")

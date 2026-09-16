"""AM1 single-point parity leg for the MPI launcher
(mpi-launcher-embedded-python-04).

Deterministic whole-molecule solve — serial vs 2-rank launcher must agree
to the convergence floor (energy 1e-10 relative, charges 1e-10 e absolute).

Usage: <python|atomiverse> atv_am1.py <out.npz>
"""

import sys

import numpy as np

import atomiverse as atv
import molrs

H = 0.629118  # [A] — tetrahedral CH4, C-H along body diagonals

out_path = sys.argv[1]

frame = molrs.Frame()
atoms = molrs.Block()
atoms.insert("x", np.array([0.0, H, -H, H, -H]))
atoms.insert("y", np.array([0.0, H, -H, -H, H]))
atoms.insert("z", np.array([0.0, H, H, -H, -H]))
atoms.insert("mass", np.array([12.011, 1.008, 1.008, 1.008, 1.008]))
atoms.insert("type", np.array([6, 1, 1, 1, 1], dtype=np.int32))  # Z
frame["atoms"] = atoms
frame.box = molrs.Box.cube(20.0)

ev = atv.AM1Evaluator(charge=0)
atv.single_point(frame, ev)

np.savez(
    out_path,
    energy=np.asarray([frame.meta["energy"]], dtype=np.float64),
    charge=np.asarray(frame["atoms"]["charge"]),
)
print(f"am1 leg done: E = {float(frame.meta['energy']):.12f} eV")

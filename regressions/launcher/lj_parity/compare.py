"""Parity comparator for the launcher regression
(mpi-launcher-embedded-python-04).

Usage: python compare.py <lj_serial.npz> <lj_mpi.npz> <am1_serial.npz> <am1_mpi.npz>
Exits nonzero (assertion) on any tolerance violation.
"""

import sys

import numpy as np

lj_s = np.load(sys.argv[1])
lj_m = np.load(sys.argv[2])
am_s = np.load(sys.argv[3])
am_m = np.load(sys.argv[4])

# LJ MD: 2-rank domain decomposition changes force-summation order; the
# divergence over 200 steps stays within these bounds (spec Domain basis).
e_s, e_m = float(lj_s["energy"][0]), float(lj_m["energy"][0])
lj_rel = abs(e_m - e_s) / max(abs(e_s), 1e-300)
if lj_rel > 1e-9:
    sys.exit(f"LJ energy parity violated: rel diff {lj_rel:.3e}")
for ax in "xyz":
    d = np.max(np.abs(lj_m[ax] - lj_s[ax]))
    if d > 1e-8:
        sys.exit(f"LJ position parity violated on {ax}: {d:.3e} A")

# AM1 single point: deterministic — tolerances at the convergence floor.
e_s, e_m = float(am_s["energy"][0]), float(am_m["energy"][0])
am1_rel = abs(e_m - e_s) / max(abs(e_s), 1e-300)
if am1_rel > 1e-10:
    sys.exit(f"AM1 energy parity violated: rel diff {am1_rel:.3e}")
dq = np.max(np.abs(am_m["charge"] - am_s["charge"]))
if dq > 1e-10:
    sys.exit(f"AM1 charge parity violated: {dq:.3e} e")

print(f"PARITY OK: LJ energy rel {lj_rel:.2e}; AM1 charge max {dq:.2e} e")

#!/usr/bin/env python
"""Offline pymbd pins for gpw-mbd-kpoint-stress.

Pure-Python pymbd (libmbd src/pymbd/pymbd.py) mbd_energy_species on the
gpw-mbd-periodic water dimer, periodic 12 A cube, free-atom volumes,
beta=0.83, MBD k-grid 2x2x2 -- plus E_disp at orthorhombic diagonally
strained cells (axis scale 1 +/- h, fixed fractional coordinates, h=2e-3).

Prints numbers only; run offline at pin time, never at ctest.
"""
import numpy as np
from pymbd import mbd_energy_species

A2B = 1.0 / 0.52917720859  # ATV Angstrom -> Bohr
H = 2.0e-3
L_ANG = 12.0
K_GRID = (2, 2, 2)
BETA = 0.83

species = ['O', 'H', 'H', 'O', 'H', 'H']
coords_ang = np.array([
    [0.000000, 0.000000, 0.000000],
    [0.758602, 0.000000, 0.504284],
    [0.758602, 0.000000, -0.504284],
    [3.000000, 0.000000, 0.000000],
    [3.758602, 0.000000, 0.504284],
    [3.758602, 0.000000, -0.504284],
])
volume_ratios = [1.0] * 6


def e_disp(scale):
    """MBD energy [Ha] with per-axis scale (fixed fractional coords)."""
    s = np.asarray(scale)
    coords = coords_ang * A2B * s[None, :]
    lattice = np.diag([L_ANG * A2B * s[0], L_ANG * A2B * s[1],
                       L_ANG * A2B * s[2]])
    return mbd_energy_species(coords, species, volume_ratios, BETA,
                              lattice=lattice, k_grid=K_GRID)


e0 = e_disp([1.0, 1.0, 1.0])
print(f"E_disp unstrained            = {e0:.16e} Ha")

vol_bohr3 = (L_ANG * A2B) ** 3
for a, name in enumerate("xyz"):
    sp = [1.0, 1.0, 1.0]
    sm = [1.0, 1.0, 1.0]
    sp[a] = 1.0 + H
    sm[a] = 1.0 - H
    ep = e_disp(sp)
    em = e_disp(sm)
    sig = -(ep - em) / (2.0 * H) / vol_bohr3
    print(f"E_disp {name} +h              = {ep:.16e} Ha")
    print(f"E_disp {name} -h              = {em:.16e} Ha")
    print(f"sigma_MBD_{name}{name} (FD, pymbd)    = {sig:.10e} Ha/Bohr3")

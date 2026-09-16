#!/usr/bin/env python
"""Offline dual pure-Python XDM pins for gpw-xdm-stress.

Independent free-atom XDM (Becke-Johnson) implementation on the
gpw-xdm-ice water dimer, periodic 12 A cube -- E_disp at the unstrained
cell plus at orthorhombic diagonally strained cells (axis scale 1 +/- h,
fixed fractional coordinates, h = 2e-3, rcut fixed 50 A), and the
per-axis sigma_XDM strain-FD terms -(E+ - E-)/(2 h Omega0).

Inputs are re-derived from the cited literature, never copied from ATV
headers:
  - alpha_free: CRC Handbook 88th Ed. static polarizabilities
    (H 0.6668 A^3, O 0.802 A^3 -- the same list postg / QE alpha_free
    transcribe), converted A^3 -> Bohr^3 here.
  - M1..M3 inverted from homoatomic free-atom dispersion coefficients
    under the XDM combination rules:
      H: C6 = 6.4990267057, C8 = 124.39908358, C10 = 3285.828
         (Yan, Babb, Dalgarno, Drake, Phys. Rev. A 54, 2824 (1996),
          exact-H values at the precision the in-tree free-atom table
          was generated from)
      O: C6 = 15.55 (Chu & Dalgarno, J. Chem. Phys. 121, 4083 (2004));
         C8 = 235, C10 = 4500 (consistent free-atom scale)
    Identity-atom rules give M1 = 2 C6/alpha, M2 = 2 C8/(3 alpha),
    M3 = (C10 - (21/10) alpha M2^2 / M1) / (2 alpha).
  - BJ damping: QE NC-PBE a1 = 0.4283, a2 = 2.4690 A
    (Otero-de-la-Roza & Johnson / postg xdm.param).

Energy: E = -sum_{i<=j} sum_T' [ C6/(R^6+RvdW^6) + C8/(R^8+RvdW^8)
+ C10/(R^10+RvdW^10) ], real-space orthorhombic lattice sum, image range
covering the 50 A sphere, pairs pruned at R > rcut.

Prints numbers only; run offline at pin time, never at ctest.
"""
import math

A2B = 1.0 / 0.52917720859  # ATV Angstrom -> Bohr
H_STRAIN = 2.0e-3
L_ANG = 12.0
RCUT_BOHR = 50.0 * A2B
A1 = 0.4283
A2_BOHR = 2.4690 * A2B

# Literature free-atom data (see module docstring).
ALPHA_ANG3 = {1: 0.6668, 8: 0.802}  # CRC 88th Ed. [A^3]
CN_HOMO = {1: (6.4990267057, 124.39908358, 3285.828), 8: (15.55, 235.0, 4500.0)}


def moments(z):
    """(alpha, M1, M2, M3) inverted from homoatomic C6/C8/C10."""
    a = ALPHA_ANG3[z] * A2B**3
    c6, c8, c10 = CN_HOMO[z]
    m1 = 2.0 * c6 / a
    m2 = 2.0 * c8 / (3.0 * a)
    m3 = (c10 - 2.1 * a * m2 * m2 / m1) / (2.0 * a)
    return a, m1, m2, m3


SPECIES = [8, 1, 1, 8, 1, 1]
COORDS_ANG = [
    (0.000000, 0.000000, 0.000000),
    (0.758602, 0.000000, 0.504284),
    (0.758602, 0.000000, -0.504284),
    (3.000000, 0.000000, 0.000000),
    (3.758602, 0.000000, 0.504284),
    (3.758602, 0.000000, -0.504284),
]


def e_disp(scale):
    """Free-atom XDM energy [Ha]; per-axis scale, fixed fractional coords."""
    n = len(SPECIES)
    pos = [
        (x * A2B * scale[0], y * A2B * scale[1], z * A2B * scale[2])
        for (x, y, z) in COORDS_ANG
    ]
    cell = [L_ANG * A2B * scale[d] for d in range(3)]
    par = [moments(z) for z in SPECIES]
    # Image range: one shell beyond ceil(rcut/L) covers |R_i - R_j| + rcut.
    m = [int(math.ceil(RCUT_BOHR / cell[d])) + 1 for d in range(3)]
    rcut2 = RCUT_BOHR * RCUT_BOHR
    e = 0.0
    for i in range(n):
        ai, m1i, m2i, m3i = par[i]
        for j in range(i + 1):
            aj, m1j, m2j, m3j = par[j]
            fac = ai * aj / (m1i * aj + m1j * ai)
            c6 = fac * m1i * m1j
            c8 = 1.5 * fac * (m1i * m2j + m2i * m1j)
            c10 = 2.0 * fac * (m1i * m3j + m3i * m1j) + 4.2 * fac * m2i * m2j
            rc = (
                math.sqrt(c8 / c6)
                + (c10 / c6) ** 0.25
                + math.sqrt(c10 / c8)
            ) / 3.0
            rvdw = A1 * rc + A2_BOHR
            d6 = rvdw**6
            d8 = rvdw**8
            d10 = rvdw**10
            w_pair = 0.5 if i == j else 1.0
            for tx in range(-m[0], m[0] + 1):
                for ty in range(-m[1], m[1] + 1):
                    for tz in range(-m[2], m[2] + 1):
                        if tx == 0 and ty == 0 and tz == 0 and i == j:
                            continue
                        rx = pos[i][0] - pos[j][0] - tx * cell[0]
                        ry = pos[i][1] - pos[j][1] - ty * cell[1]
                        rz = pos[i][2] - pos[j][2] - tz * cell[2]
                        r2 = rx * rx + ry * ry + rz * rz
                        if r2 > rcut2:
                            continue
                        r6 = r2 * r2 * r2
                        r8 = r6 * r2
                        r10 = r8 * r2
                        e -= w_pair * (
                            c6 / (r6 + d6) + c8 / (r8 + d8) + c10 / (r10 + d10)
                        )
    return e


e0 = e_disp([1.0, 1.0, 1.0])
print(f"E_disp unstrained            = {e0:.16e} Ha")

vol_bohr3 = (L_ANG * A2B) ** 3
for a, name in enumerate("xyz"):
    sp = [1.0, 1.0, 1.0]
    sm = [1.0, 1.0, 1.0]
    sp[a] = 1.0 + H_STRAIN
    sm[a] = 1.0 - H_STRAIN
    ep = e_disp(sp)
    em = e_disp(sm)
    sig = -(ep - em) / (2.0 * H_STRAIN) / vol_bohr3
    print(f"E_disp {name} +h              = {ep:.16e} Ha")
    print(f"E_disp {name} -h              = {em:.16e} Ha")
    print(f"sigma_XDM_{name}{name} (FD, dual-py)  = {sig:.10e} Ha/Bohr3")

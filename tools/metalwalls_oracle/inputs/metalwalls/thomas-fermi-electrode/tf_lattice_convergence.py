#!/usr/bin/env python3
"""Why the thomas-fermi-electrode pin moved: xy-image truncation vs 2D Ewald.

Waves 1-5 pinned this regression from a Python script that *re-implemented*
`cpu::md::ConstPotBO`'s truncated xy-image lattice sum at `xy_image_range = 40`.
That was a self-golden: it agreed with Atomiverse by construction and could not
detect that the sum was 6 % away from the converged 2D-periodic answer.

This script is the diagnosis, not the pin. The pin is MetalWalls c8ad175d -
see `mw_twin/` and regressions/md/thomas-fermi-electrode/REFERENCE.md.

It prints, for the gold-capacitor toy:
  1. the ConstPotBO truncated square-image sum as a function of the half-range R
     (converges as O(1/R) because the missing tail of the in-plane lattice sum
      is (pi z^2 / R) / A_cell per pair, and z_gap / L_xy ~ 6 here);
  2. the exact 2D Ewald elastance (alpha-independent), which is what MetalWalls
     computes for num_pbc = 2 and what the truncated sum converges to.

Never launched from ctest. Needs numpy + scipy.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.special import erf, erfc, erfcx

KE = 14.399645  # units::coulomb [eV A / e^2]
ETA = 1.0 / 0.56  # [1/A] MetalWalls w = 0.56 A
BETA = ETA / math.sqrt(2.0)  # eta_ij for a Gaussian-Gaussian pair
A0 = 4.07  # Au lattice constant [A]
D_SITE = 4.0 / A0**3  # [1/A^3]
APL = A0 / 2.0  # plane spacing [A]
LGAP = 20.0  # first-plane gap [A]
LX = LY = A0
AREA = LX * LY
PSI = np.array([0.0] * 3 + [1.0] * 3)  # [V]


def positions() -> np.ndarray:
    z_bot = 5.0 + 2 * APL
    z_top = z_bot + LGAP
    z = [z_bot - 2 * APL, z_bot - APL, z_bot, z_top, z_top + APL, z_top + 2 * APL]
    return np.array([[0.0, 0.0, zz] for zz in z])


def elastance_truncated(pos: np.ndarray, half_range: int, l_tf: float) -> np.ndarray:
    """ConstPotBO::build_matrix_ transcribed: square xy image block, no Ewald."""
    m = len(pos)
    a_mat = np.zeros((m, m))
    ns = np.arange(-half_range, half_range + 1)
    nx, ny = np.meshgrid(ns, ns, indexing="ij")
    nx = nx.ravel() * LX
    ny = ny.ravel() * LY
    for i in range(m):
        a_mat[i, i] = KE * ETA * math.sqrt(2.0 / math.pi)
        for j in range(i + 1, m):
            dx = pos[i, 0] - pos[j, 0] + nx
            dy = pos[i, 1] - pos[j, 1] + ny
            dz = pos[i, 2] - pos[j, 2]
            r = np.sqrt(dx * dx + dy * dy + dz * dz)
            r = r[r > 1e-14]
            a_mat[i, j] = a_mat[j, i] = KE * np.sum(erf(BETA * r) / r)
        r0 = np.sqrt(nx * nx + ny * ny)
        r0 = r0[r0 > 1e-14]
        a_mat[i, i] += KE * np.sum(erf(BETA * r0) / r0)
    if l_tf > 0.0:
        a_mat[np.diag_indices(m)] += 4.0 * math.pi * KE * l_tf * l_tf * D_SITE
    return a_mat


def _phi_2d(rho_x: float, rho_y: float, z: float, alpha: float, nreal: int, nk: int):
    """Exact 2D-periodic 1/r lattice potential (uniform neutralizing background)."""
    ns = np.arange(-nreal, nreal + 1)
    nx, ny = np.meshgrid(ns, ns, indexing="ij")
    dx = rho_x + nx.ravel() * LX
    dy = rho_y + ny.ravel() * LY
    r = np.sqrt(dx * dx + dy * dy + z * z)
    real = np.sum(erfc(alpha * r) / r)

    ks = np.arange(-nk, nk + 1)
    kxi, kyi = np.meshgrid(ks, ks, indexing="ij")
    kx = 2 * math.pi * kxi.ravel() / LX
    ky = 2 * math.pi * kyi.ravel() / LY
    k = np.sqrt(kx * kx + ky * ky)
    sel = k > 1e-12
    kx, ky, k = kx[sel], ky[sel], k[sel]
    # e^{kz} erfc(k/2a + az) = exp(-k^2/4a^2 - a^2 z^2) erfcx(k/2a + az): no overflow
    pref = np.exp(-(k**2) / (4 * alpha**2) - (alpha * z) ** 2)
    term = pref * (
        erfcx(k / (2 * alpha) + alpha * z) + erfcx(k / (2 * alpha) - alpha * z)
    )
    recip = math.pi / AREA * np.sum(np.cos(kx * rho_x + ky * rho_y) / k * term)
    k0 = (
        -2
        * math.pi
        / AREA
        * (
            z * erf(alpha * z)
            + math.exp(-((alpha * z) ** 2)) / (alpha * math.sqrt(math.pi))
        )
    )
    return real + recip + k0


def _phi_2d_self(alpha: float, nreal: int, nk: int) -> float:
    ns = np.arange(-nreal, nreal + 1)
    nx, ny = np.meshgrid(ns, ns, indexing="ij")
    r = np.sqrt((nx.ravel() * LX) ** 2 + (ny.ravel() * LY) ** 2)
    r = r[r > 1e-14]
    real = np.sum(erfc(alpha * r) / r) - 2 * alpha / math.sqrt(math.pi)
    ks = np.arange(-nk, nk + 1)
    kxi, kyi = np.meshgrid(ks, ks, indexing="ij")
    k = np.sqrt(
        (2 * math.pi * kxi.ravel() / LX) ** 2 + (2 * math.pi * kyi.ravel() / LY) ** 2
    )
    k = k[k > 1e-12]
    recip = math.pi / AREA * np.sum(2.0 * erfc(k / (2 * alpha)) / k)
    return real + recip - 2 * math.pi / AREA / (alpha * math.sqrt(math.pi))


def elastance_ewald2d(
    pos: np.ndarray, l_tf: float, alpha: float = 0.6, nreal: int = 14, nk: int = 60
) -> np.ndarray:
    """Exact 2D Ewald Gaussian elastance - the MetalWalls num_pbc = 2 model."""
    m = len(pos)
    a_mat = np.zeros((m, m))
    ns = np.arange(-nreal, nreal + 1)
    nx, ny = np.meshgrid(ns, ns, indexing="ij")
    for i in range(m):
        for j in range(i, m):
            dx0 = pos[i, 0] - pos[j, 0]
            dy0 = pos[i, 1] - pos[j, 1]
            dz = pos[i, 2] - pos[j, 2]
            r = np.sqrt(
                (dx0 + nx.ravel() * LX) ** 2 + (dy0 + ny.ravel() * LY) ** 2 + dz * dz
            )
            r = r[r > 1e-14]
            # erf(beta r)/r = 1/r - erfc(beta r)/r, summed over the same lattice
            short = np.sum(erfc(BETA * r) / r)
            if i == j:
                val = (
                    _phi_2d_self(alpha, nreal, nk)
                    + 2 * BETA / math.sqrt(math.pi)
                    - short
                )
            else:
                val = _phi_2d(dx0, dy0, dz, alpha, nreal, nk) - short
            a_mat[i, j] = a_mat[j, i] = KE * val
    if l_tf > 0.0:
        a_mat[np.diag_indices(m)] += 4.0 * math.pi * KE * l_tf * l_tf * D_SITE
    return a_mat


def solve_neutral(a_mat: np.ndarray, psi: np.ndarray) -> np.ndarray:
    """Same projector as ConstPotBO and MetalWalls matrix_inversion."""
    inv = np.linalg.inv(a_mat)
    e = np.ones(len(psi))
    u = inv @ e
    v = inv @ psi
    return v - (e @ v) / (e @ u) * u


def main() -> None:
    pos = positions()
    mw = {0.0: 4.354124119406114e-03, 2.0: 3.707555065364343e-03}  # [e]
    print(f"# gold capacitor toy: eta={ETA:.12f} 1/A  d={D_SITE:.12f} 1/A^3")
    for l_tf in (0.0, 2.0):
        ref = mw[l_tf]
        print(f"\n## l_tf = {l_tf} A   MetalWalls c8ad175d Q_top = {ref:.12e} e")
        for half_range in (40, 160, 320, 640, 1280, 2560):
            q = solve_neutral(elastance_truncated(pos, half_range, l_tf), PSI)
            q_top = float(q[3:].sum())
            print(
                f"  truncated R={half_range:5d}  Q_top={q_top:.12e} e"
                f"  ({100.0 * (q_top - ref) / ref:+.4f} % vs MW)"
            )
        for alpha in (0.4, 0.6, 0.9):
            q = solve_neutral(elastance_ewald2d(pos, l_tf, alpha), PSI)
            q_top = float(q[3:].sum())
            print(
                f"  2D Ewald alpha={alpha}  Q_top={q_top:.12e} e"
                f"  ({100.0 * (q_top - ref) / ref:+.6f} % vs MW)"
            )


if __name__ == "__main__":
    main()

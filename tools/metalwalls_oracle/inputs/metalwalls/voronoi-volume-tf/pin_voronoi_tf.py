#!/usr/bin/env python3
"""Offline pin: TF gold capacitor with per-site Voronoi volumes (MetalWalls model).

MetalWalls exposes Thomas-Fermi metallicity as per-species electrode inputs
`thomas_fermi_length` + `voronoi_volume` (Scalfi et al. JCP 153, 174704
(2020)).  This pin mirrors Atomiverse cpu::md::ConstPotBO algebra:

  Gaussian electrode elastance + 2D xy lattice images
  + TF diagonal  A_ii += 4*pi*ke * l_tf**2 / v_i     (per-site volume)
  + electroneutrality projector.

The fixed-density foothold uses A_ii += 4*pi*ke * l_tf**2 * d with
d = 4/a0**3; supplying the FCC Voronoi volume v_i = a0**3/4 = 1/d must give
bit-identical algebra, so Q(voronoi bulk) == Q(fixed density) is itself a
pinned invariant.  A second pin halves v_i (denser sites -> harder diagonal
-> lower C) for the metallicity gate.

Never launched from ctest.  See regressions/md/voronoi-volume-tf/REFERENCE.md.
"""
from __future__ import annotations

import math

import numpy as np

KE = 14.399645  # units::coulomb [eV*A/e^2]
SQRT2 = math.sqrt(2.0)
SQRT2_OVER_PI = math.sqrt(2.0 / math.pi)

ETA = 1.0 / 0.56  # [1/A] MetalWalls w = 0.56 A
A0 = 4.07  # [A] Au lattice constant
V_FCC = A0**3 / 4.0  # [A^3] FCC Voronoi (rhombic dodecahedron) volume
A_PLANE = A0 / 2.0
L_GAP = 20.0
LX = LY = A0
R = 40
L_TF = 2.0  # [A]


def pair_a(r: float, eta: float) -> float:
    if r < 1e-14:
        return KE * eta * SQRT2_OVER_PI
    etaij = eta / math.sqrt(2.0)
    return KE * math.erf(etaij * r) / r


def gold_cap_atoms():
    z_surf_bot = 5.0 + 2 * A_PLANE
    z_surf_top = z_surf_bot + L_GAP
    z_bots = [z_surf_bot - 2 * A_PLANE, z_surf_bot - A_PLANE, z_surf_bot]
    z_tops = [z_surf_top, z_surf_top + A_PLANE, z_surf_top + 2 * A_PLANE]
    atoms = [(0.0, 0.0, z, 1) for z in z_bots] + [(0.0, 0.0, z, 2) for z in z_tops]
    return atoms


def build_a(atoms, l_tf: float, volumes: list[float]) -> np.ndarray:
    m = len(atoms)
    a_mat = np.zeros((m, m))
    for i in range(m):
        a_mat[i, i] = KE * ETA * SQRT2_OVER_PI
        xi, yi, zi, _ = atoms[i]
        for j in range(i + 1, m):
            xj, yj, zj, _ = atoms[j]
            aij = 0.0
            for nx in range(-R, R + 1):
                for ny in range(-R, R + 1):
                    dx = xi - xj + nx * LX
                    dy = yi - yj + ny * LY
                    dz = zi - zj
                    r = math.sqrt(dx * dx + dy * dy + dz * dz)
                    if r < 1e-14:
                        continue
                    aij += pair_a(r, ETA)
            a_mat[i, j] = a_mat[j, i] = aij
        self_img = 0.0
        for nx in range(-R, R + 1):
            for ny in range(-R, R + 1):
                if nx == 0 and ny == 0:
                    continue
                r = math.sqrt((nx * LX) ** 2 + (ny * LY) ** 2)
                self_img += KE * math.erf((ETA / SQRT2) * r) / r
        a_mat[i, i] += self_img
    if l_tf > 0.0:
        for i in range(m):
            a_mat[i, i] += 4.0 * math.pi * KE * l_tf * l_tf / volumes[i]
    return a_mat


def solve_neutral(a_mat: np.ndarray, psi: np.ndarray) -> np.ndarray:
    """Electroneutral BO solve: augmented [A 1; 1^T 0] [Q; mu] = [psi; 0]."""
    m = a_mat.shape[0]
    aug = np.zeros((m + 1, m + 1))
    aug[:m, :m] = a_mat
    aug[:m, m] = 1.0
    aug[m, :m] = 1.0
    rhs = np.zeros(m + 1)
    rhs[:m] = psi
    sol = np.linalg.solve(aug, rhs)
    return sol[:m]


def run_case(label: str, volumes: list[float]) -> None:
    atoms = gold_cap_atoms()
    psi = np.array([0.0 if t == 1 else 1.0 for (_, _, _, t) in atoms])
    a_mat = build_a(atoms, L_TF, volumes)
    q = solve_neutral(a_mat, psi)
    q_top = float(q[3:].sum())
    print(f"{label}")
    print(f"  v_i        = {volumes}")
    print(f"  Q_top      = {q_top:+.12e} e")
    print(f"  C          = {abs(q_top):.12e} e/V   (dPsi = 1 V)")
    print(f"  |sum Q|    = {abs(q.sum()):.3e} e")


def main() -> None:
    print("Voronoi/site-volume TF pin (MetalWalls model, gold capacitor toy)")
    print(f"  a0 = {A0} A;  FCC Voronoi volume v = a0^3/4 = {V_FCC:.12e} A^3")
    print(f"  fixed-density twin d = 4/a0^3 = {4.0 / A0**3:.12e} 1/A^3")
    print(f"  l_tf = {L_TF} A;  eta = {ETA:.12f} 1/A;  xy image range R = {R}")
    m = 6
    run_case("case A: bulk FCC Voronoi volume (== fixed-density twin)",
             [V_FCC] * m)
    run_case("case B: halved volume v/2 (denser sites, harder diagonal)",
             [V_FCC / 2.0] * m)


if __name__ == "__main__":
    main()

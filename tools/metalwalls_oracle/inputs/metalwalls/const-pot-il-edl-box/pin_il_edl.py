#!/usr/bin/env python3
"""Offline oracle pin for regressions/md/const-pot-il-edl-box.

Independent numpy implementation of the MetalWalls-form Gaussian-electrode
Born-Oppenheimer constant-potential solve (Scalfi et al. form; MW JOSS
10.21105/joss.02373) with the electrode-only electroneutrality projection,
evaluated on the EXACT deterministic initial geometry of the regression:

  - honeycomb graphene-like walls, a_CC = 1.42 A, 4-atom rectangular cell,
    4 x 7 cells -> 112 C per wall; left wall z = 12 A at Psi = -1 V, right
    wall z = 28 A at Psi = +1 V; box 17.04 x 17.216585027234639 x 40 A.
    Interior (IL-filled) gap 16 A < boundary vacuum path 24 A, so the EDL
    response dominates the periodic return path.
  - 12 + 12 single-site CG IL ions (+1/-1 e) on a 3x3 (minus centre) x 3
    grid between the walls: checkerboard parity, plus one deterministic
    charge swap per outer layer (5+/3- facing the -1 V wall, 3+/5- facing
    the +1 V wall) — a nascent-EDL polarization, so the step-0 solve shows
    |Q| screening enhancement over the empty capacitor

Model algebra (identical to ATV cpu::md::ConstPotBO with xy_image_range=0,
min-image Coulomb, eta shared; MW Gaussian-charge conventions):
  A_aa = ke * eta * sqrt(2/pi)                       (Gaussian self)
  A_ab = ke * erf((eta/sqrt(2)) r_ab) / r_ab         (Gaussian-Gaussian)
  b_a  = sum_j ke q_j erf(eta r_aj) / r_aj           (Gaussian-point)
  rhs  = Psi - b;  Q0 = A^-1 rhs;  u = A^-1 1
  Q    = Q0 - u (1.Q0 - target)/(1.u)   (electrode-only neutrality target 0)
eta = 1.805132 A^-1 = 0.955234 bohr^-1, the MetalWalls graphite electrode
Gaussian width (MW example inputs; Marin-Lafleche et al. JOSS 2020).

Run:  python3 pin_il_edl.py
Outputs Q_left / Q_right for (a) the empty capacitor (IL charges zeroed) and
(b) the full IL box. ctest NEVER runs this script; the numbers are hard-coded
in the regression and REFERENCE.md.
"""

import numpy as np
from scipy.special import erf

KE = 14.399645   # eV*A/e^2, ATV units::coulomb
ETA = 1.805132   # Gaussian width [1/A] (MW graphite: 0.955234 bohr^-1)
A_CC = 1.42
CX, CY = 3.0 * A_CC, np.sqrt(3.0) * A_CC
NX, NY = 4, 7
LX, LY, LZ = NX * CX, NY * CY, 40.0
Z_LEFT, Z_RIGHT = 12.0, 28.0
PSI_LEFT, PSI_RIGHT = -1.0, +1.0  # [V]


def wall(z):
    basis = [(0.0, 0.0), (A_CC, 0.0), (1.5 * A_CC, 0.5 * CY),
             (2.5 * A_CC, 0.5 * CY)]
    pts = []
    for ix in range(NX):
        for iy in range(NY):
            for bx, by in basis:
                pts.append((ix * CX + bx, iy * CY + by, z))
    return np.array(pts)


def ions():
    pts, q = [], []
    for iz in range(3):
        z = 15.5 + 4.5 * iz
        for ix in range(3):
            for iy in range(3):
                if ix == 1 and iy == 1:
                    continue
                qq = 1.0 if (ix + iy + iz) % 2 == 0 else -1.0
                # Nascent-EDL polarization: one swap per outer layer.
                if iz == 0 and ix == 0 and iy == 1:
                    qq = 1.0  # cation excess facing the -1 V wall
                if iz == 2 and ix == 0 and iy == 0:
                    qq = -1.0  # anion excess facing the +1 V wall
                pts.append(((ix + 0.5) * LX / 3.0, (iy + 0.5) * LY / 3.0, z))
                q.append(qq)
    return np.array(pts), np.array(q)


def min_image(d):
    box = np.array([LX, LY, LZ])
    return d - box * np.rint(d / box)


def solve(elec, psi, elyt, q_elyt):
    n = len(elec)
    d = min_image(elec[:, None, :] - elec[None, :, :])
    r = np.linalg.norm(d, axis=-1)
    rsafe = np.where(r < 1e-14, 1.0, r)
    # Gaussian-Gaussian pair kernel: erf((eta/sqrt(2)) r)/r; Gaussian self.
    A = KE * erf((ETA / np.sqrt(2.0)) * r) / rsafe
    np.fill_diagonal(A, KE * ETA * np.sqrt(2.0 / np.pi))

    phi = np.zeros(n)
    live = q_elyt != 0.0
    if live.any():
        d = min_image(elec[:, None, :] - elyt[None, live, :])
        r = np.linalg.norm(d, axis=-1)
        # Gaussian-point kernel: erf(eta r)/r.
        phi = (KE * q_elyt[live][None, :] * erf(ETA * r) / r).sum(axis=1)
    # Electrode-only neutrality (ConstPotBO default): target sum(Q) = 0.
    Q0 = np.linalg.solve(A, psi - phi)
    u = np.linalg.solve(A, np.ones(n))
    lam = (Q0.sum() - 0.0) / u.sum()
    return Q0 - lam * u


def main():
    wl, wr = wall(Z_LEFT), wall(Z_RIGHT)
    elec = np.vstack([wl, wr])
    psi = np.concatenate([np.full(len(wl), PSI_LEFT),
                          np.full(len(wr), PSI_RIGHT)])
    ipos, iq = ions()
    assert len(elec) == 224 and len(ipos) == 24 and iq.sum() == 0.0

    for tag, q_elyt in (("empty", np.zeros(len(ipos))), ("with-IL", iq)):
        Q = solve(elec, psi, ipos, q_elyt)
        ql, qr = Q[:len(wl)].sum(), Q[len(wl):].sum()
        print(f"{tag:8s} Q_left = {ql:+.12e} e   Q_right = {qr:+.12e} e   "
              f"sum = {Q.sum():+.3e}")


if __name__ == "__main__":
    main()

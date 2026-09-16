#!/usr/bin/env python3
"""Offline oracle pin for regressions/md/pim-npt-polarizable-il.

Independent (numpy) implementation of the published bulk-PIM Thole
multipole form — MetalWalls JOSS 5, 2373 (2020) bulk PIM; Thole,
Chem. Phys. 59, 341 (1981) exponential damping — evaluated on the
deterministic start configuration of the regression:

  4x4x4 simple-cubic rock-salt toy polarizable IL, d = 2.35 A,
  alternating charges (-1)^(ix+iy+iz), cation type 0 (alpha = 0),
  anion type 1 (alpha = 3.0 A^3), deterministic sinusoidal
  displacement disp[a] = 0.15 * sin(1.7*(i+1) + 2.3*a) (i = atom
  index, a = 0,1,2 axis), cubic box L = 4*d, PIM cutoff 4.0 A,
  thole_a = 0.39, tt_b = 0, Jacobi SCC to max|dmu| < 1e-13 e*A.

Outputs the pinned observables hard-coded in the regression:
  E_pol [eV], PIM virial Voigt W [eV], P_pol = tr(W)/(3V) [eV/A^3],
  force on atom 0 [eV/A].

Run (x86_64 venv):
  ~/work/.x86_64/bin/python3 pim_npt_polarizable_il_pin.py

This script is documentation of the pin provenance. It is NEVER run at
build/ctest/regression time (offline-pin contract, tools/metalwalls_oracle/README.md).
"""

import numpy as np

K_COULOMB = 14.399645  # eV*A/e^2 — matches atomiverse/core/units.h exactly

# ── deterministic start configuration (must match the regression) ─────────
NC = 4
D = 2.35
L = NC * D
RC = 4.0
THOLE_A = 0.39
ALPHA_ANION = 3.0
AMP, C1, C2 = 0.15, 1.7, 2.3
TOL = 1e-13
MAX_IT = 1000

n = NC * NC * NC
pos = np.zeros((n, 3))
q = np.zeros(n)
alpha = np.zeros(n)
idx = 0
for ix in range(NC):
    for iy in range(NC):
        for iz in range(NC):
            base = np.array([ix * D, iy * D, iz * D])
            disp = np.array(
                [AMP * np.sin(C1 * (idx + 1) + C2 * a) for a in range(3)])
            pos[idx] = base + disp
            if (ix + iy + iz) % 2 == 0:
                q[idx] = 1.0
                alpha[idx] = 0.0  # cation, type 0
            else:
                q[idx] = -1.0
                alpha[idx] = ALPHA_ANION  # anion, type 1
            idx += 1

# ── pair geometry (min image, round-half-even == std::nearbyint) ──────────
dr = pos[:, None, :] - pos[None, :, :]
dr -= L * np.round(dr / L)  # numpy round == banker's rounding == nearbyint
r2 = np.einsum('ijk,ijk->ij', dr, dr)
np.fill_diagonal(r2, np.inf)
mask = r2 < RC * RC
r = np.sqrt(np.where(mask, r2, 1.0))

# ── Thole screening (both alpha > 0 only; van Duijnen/Swart exponential) ──
l3 = np.ones((n, n))
l5 = np.ones((n, n))
l7 = np.ones((n, n))
both = (alpha[:, None] > 0) & (alpha[None, :] > 0)
aa = np.where(both, alpha[:, None] * alpha[None, :], 1.0)
u = np.where(both, THOLE_A * r / aa ** (1.0 / 6.0), 0.0)
u3 = u ** 3
u6 = u3 * u3
expu = np.exp(-u3)
l3 = np.where(both, 1.0 - expu, 1.0)
l5 = np.where(both, 1.0 - (1.0 + u3) * expu, 1.0)
l7 = np.where(both, 1.0 - (1.0 + u3 + 0.6 * u6) * expu, 1.0)
# tt_b = 0 → Tang–Toennies factors all 1
s3 = np.where(mask, l3, 0.0)
s5 = np.where(mask, l5, 0.0)
s7 = np.where(mask, l7, 0.0)

r2m = np.where(mask, r2, 1.0)
r3 = r ** 3
r5 = r3 * r2m
r7 = r5 * r2m

# ── permanent field E^perm_i = sum_j s3 q_j r_ij / r^3  [e/A^2] ──────────
fac = s3 * q[None, :] / r3
e_perm = np.einsum('ij,ijk->ik', fac, dr)

# ── Jacobi SCC: mu = alpha (E^perm + T mu) ───────────────────────────────
mu = np.zeros((n, 3))
it = 0
for it in range(1, MAX_IT + 1):
    mdotr = np.einsum('jk,ijk->ij', mu, dr)  # mu_j · r_ij
    tmu = (3.0 * (s5 / r5 * mdotr)[:, :, None] * dr
           - (s3 / r3)[:, :, None] * mu[None, :, :])
    # only dipole-dipole coupling between polarizable sites contributes,
    # but alpha_j = 0 → mu_j = 0 makes those terms vanish automatically.
    e_tot = e_perm + tmu.sum(axis=1)
    mu_new = alpha[:, None] * e_tot
    dmax = np.max(np.abs(mu_new - mu))
    mu = mu_new
    if dmax < TOL:
        break

e_pol = K_COULOMB * (-0.5 * np.einsum('ik,ik->', mu, e_perm))

# ── Hellmann–Feynman pair forces + virial ────────────────────────────────
mjr = np.einsum('jk,ijk->ij', mu, dr)  # mu_j · r_ij
mir = np.einsum('ik,ijk->ij', mu, dr)  # mu_i · r_ij
mimj = np.einsum('ik,jk->ij', mu, mu)  # mu_i · mu_j

# q_i E^mu_j  +  q_j (L3 mu_i/r^3 − 3 L5 (mu_i·r) r/r^5)  +  dip-dip
em = (3.0 * (s5 / r5 * mjr)[:, :, None] * dr
      - (s3 / r3)[:, :, None] * mu[None, :, :])
fp = q[:, None, None] * em
fp += q[None, :, None] * ((s3 / r3)[:, :, None] * mu[:, None, :]
                          - 3.0 * (s5 / r5 * mir)[:, :, None] * dr)
fp += (3.0 * (s5 / r5)[:, :, None]
       * (mimj[:, :, None] * dr
          + mjr[:, :, None] * mu[:, None, :]
          + mir[:, :, None] * mu[None, :, :])
       - 15.0 * (s7 / r7 * mir * mjr)[:, :, None] * dr)
fp *= mask[:, :, None]

force = K_COULOMB * fp.sum(axis=1)  # [eV/A]

# W = 1/2 sum_{i != j} r_ij (x) f_ij  (Voigt xx yy zz xy xz yz) [eV]
w = np.zeros(6)
w[0] = 0.5 * np.sum(dr[:, :, 0] * fp[:, :, 0])
w[1] = 0.5 * np.sum(dr[:, :, 1] * fp[:, :, 1])
w[2] = 0.5 * np.sum(dr[:, :, 2] * fp[:, :, 2])
w[3] = 0.5 * np.sum(dr[:, :, 0] * fp[:, :, 1])
w[4] = 0.5 * np.sum(dr[:, :, 0] * fp[:, :, 2])
w[5] = 0.5 * np.sum(dr[:, :, 1] * fp[:, :, 2])
w *= K_COULOMB

vol = L ** 3
p_pol = (w[0] + w[1] + w[2]) / (3.0 * vol)

print(f"n            = {n}")
print(f"L            = {L!r} A   V = {vol!r} A^3")
print(f"SCC iters    = {it}   (tol {TOL:g} e*A)")
print(f"E_pol  [eV]  = {e_pol!r}")
print(f"W_xx   [eV]  = {w[0]!r}")
print(f"W_yy   [eV]  = {w[1]!r}")
print(f"W_zz   [eV]  = {w[2]!r}")
print(f"W_xy   [eV]  = {w[3]!r}")
print(f"W_xz   [eV]  = {w[4]!r}")
print(f"W_yz   [eV]  = {w[5]!r}")
print(f"P_pol [eV/A^3] = {p_pol!r}")
print(f"F0 [eV/A]    = ({force[0, 0]!r}, {force[0, 1]!r}, {force[0, 2]!r})")

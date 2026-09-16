#!/usr/bin/env python3
"""Decompose the ATV-vs-MetalWalls gap for regressions/md/dipole-electrode-joint.

Same kernels as ATV's DipoleElectrodeJoint, but the real-space *summation
convention* is switchable:

    sum_mode = "minimage"  -> exactly what ATV does (xy_image_range = 0)
    sum_mode = "ewald3d"   -> full 3D-periodic Ewald lattice sum

If the "ewald3d" run reproduces MetalWalls' num_pbc 3 number, then the ATV
FORMULAS are right and the whole discrepancy is the boundary condition.
"""
from __future__ import annotations

import math
import cmath

K = 14.399645  # ATV units::coulomb [eV*A/e^2]  == 1 Ha*Bohr
SQRT2 = math.sqrt(2.0)
SQRT2_OVER_PI = math.sqrt(2.0 / math.pi)

ETA = 2.0            # [1/A]
PSI = [0.0, 1.0]     # bot, top [V]
LX, LY, LZ = 10.0, 10.0, 20.0
VOL = LX * LY * LZ

# x, y, z, q_perm, is_elec, eid, alpha[A^3]
ATOMS = [
    (0.0, 0.0, 2.0, 0.0, True, 0, 0.0),
    (0.0, 0.0, 18.0, 0.0, True, 1, 0.0),
    (0.0, 0.0, 8.0, 1.0, False, -1, 1.0),
    (0.0, 0.0, 12.0, -1.0, False, -1, 1.5),
]

# ---------------------------------------------------------------- Ewald 1/r
ALPHA = 0.35              # Ewald splitting [1/A]
NREAL = 12
NK = 22


def ewald_phi(dx, dy, dz, self_term=False):
    """Sum_L 1/|d+L| over the 3D lattice with neutralising background.

    self_term=True returns the regularised Sum_{L!=0} 1/|L| (Wigner/Madelung).
    """
    tot = 0.0
    for nx in range(-NREAL, NREAL + 1):
        for ny in range(-NREAL, NREAL + 1):
            for nz in range(-NREAL, NREAL + 1):
                if self_term and nx == 0 and ny == 0 and nz == 0:
                    continue
                rx = dx + nx * LX
                ry = dy + ny * LY
                rz = dz + nz * LZ
                r = math.sqrt(rx * rx + ry * ry + rz * rz)
                if r < 1e-14:
                    continue
                tot += math.erfc(ALPHA * r) / r
    rec = 0.0
    for kx in range(-NK, NK + 1):
        for ky in range(-NK, NK + 1):
            for kz in range(-NK, NK + 1):
                if kx == 0 and ky == 0 and kz == 0:
                    continue
                gx = 2.0 * math.pi * kx / LX
                gy = 2.0 * math.pi * ky / LY
                gz = 2.0 * math.pi * kz / LZ
                k2 = gx * gx + gy * gy + gz * gz
                rec += (math.exp(-k2 / (4.0 * ALPHA * ALPHA)) / k2
                        * math.cos(gx * dx + gy * dy + gz * dz))
    tot += 4.0 * math.pi / VOL * rec
    tot -= math.pi / (ALPHA * ALPHA * VOL)
    if self_term:
        tot -= 2.0 * ALPHA / math.sqrt(math.pi)
    return tot


def min_image(dx, dy, dz):
    # C++ Box::min_image uses std::nearbyint (round-half-to-EVEN, default mode);
    # python round() is also round-half-to-even -> identical on the +-L/2 ties.
    return (dx - LX * round(dx / LX),
            dy - LY * round(dy / LY),
            dz - LZ * round(dz / LZ))


# ------------------------------------------------------- ATV kernel pieces
def thole_u(r, ai, aj, a):
    if a <= 0 or ai <= 0 or aj <= 0 or r <= 0:
        return 0.0
    return a * r / (ai * aj) ** (1.0 / 6.0)


def thole_lambda(u):
    if u <= 0:
        return 1.0, 1.0, 1.0
    u3 = u ** 3
    e = math.exp(-u3)
    return 1.0 - e, 1.0 - (1.0 + u3) * e, 1.0 - (1.0 + u3 + 0.6 * u3 * u3) * e


def cholesky_solve(A_in, b):
    n = len(b)
    A = [row[:] for row in A_in]
    x = b[:]
    for i in range(n):
        for j in range(i + 1):
            s = A[i][j] - sum(A[i][k] * A[j][k] for k in range(j))
            if i == j:
                A[i][j] = math.sqrt(s)
            else:
                A[i][j] = s / A[j][j]
    for i in range(n):
        x[i] = (x[i] - sum(A[i][k] * x[k] for k in range(i))) / A[i][i]
    for i in range(n - 1, -1, -1):
        x[i] = (x[i] - sum(A[k][i] * x[k] for k in range(i + 1, n))) / A[i][i]
    return x


def solve_neutral(A, rhs):
    Q0 = cholesky_solve(A, rhs)
    m = len(rhs)
    inv = [[0.0] * m for _ in range(m)]
    for j in range(m):
        e = [0.0] * m
        e[j] = 1.0
        col = cholesky_solve(A, e)
        for i in range(m):
            inv[i][j] = col[i]
    u = [sum(inv[i][j] for j in range(m)) for i in range(m)]
    return [Q0[i] - (sum(Q0) / sum(u)) * u[i] for i in range(m)]


def solve(sum_mode="minimage", thole_a=0.39, use_alpha=True,
          joint_tol=1e-13, max_outer=200):
    n = len(ATOMS)
    elec = [i for i, a in enumerate(ATOMS) if a[4]]
    m = len(elec)
    alphas = [a[6] if use_alpha else 0.0 for a in ATOMS]
    charges = [a[3] for a in ATOMS]
    eta_ij = ETA / SQRT2

    # ---- elastance A
    A = [[0.0] * m for _ in range(m)]
    for a in range(m):
        ia = elec[a]
        if sum_mode == "minimage":
            A[a][a] = K * ETA * SQRT2_OVER_PI
        else:
            # Sum_{L!=0} erf(eta_ij |L|)/|L|  +  self (2 eta_ij/sqrt(pi))
            lat = ewald_phi(0.0, 0.0, 0.0, self_term=True)
            corr = 0.0
            for nx in range(-3, 4):
                for ny in range(-3, 4):
                    for nz in range(-3, 4):
                        if nx == ny == nz == 0:
                            continue
                        r = math.sqrt((nx * LX) ** 2 + (ny * LY) ** 2
                                      + (nz * LZ) ** 2)
                        corr += math.erfc(eta_ij * r) / r
            A[a][a] = K * (lat - corr + ETA * SQRT2_OVER_PI)
        for b in range(a + 1, m):
            ib = elec[b]
            dx, dy, dz = (ATOMS[ia][0] - ATOMS[ib][0],
                          ATOMS[ia][1] - ATOMS[ib][1],
                          ATOMS[ia][2] - ATOMS[ib][2])
            if sum_mode == "minimage":
                mx, my, mz = min_image(dx, dy, dz)
                r = math.sqrt(mx * mx + my * my + mz * mz)
                v = K * math.erf(eta_ij * r) / r
            else:
                lat = ewald_phi(dx, dy, dz)
                corr = 0.0
                for nx in range(-3, 4):
                    for ny in range(-3, 4):
                        for nz in range(-3, 4):
                            rx, ry, rz = dx + nx * LX, dy + ny * LY, dz + nz * LZ
                            r = math.sqrt(rx * rx + ry * ry + rz * rz)
                            if r > 1e-14:
                                corr += math.erfc(eta_ij * r) / r
                v = K * (lat - corr)
            A[a][b] = A[b][a] = v

    Q = [0.0] * m
    mu = [[0.0, 0.0, 0.0] for _ in range(n)]

    for outer in range(1, max_outer + 1):
        # ---- rhs b_a = Phi_perm + Phi_mu at each electrode
        b = []
        for a in range(m):
            ia = elec[a]
            bi = 0.0
            for j in range(n):
                if ATOMS[j][4]:
                    continue
                qj = charges[j]
                dx, dy, dz = (ATOMS[ia][0] - ATOMS[j][0],
                              ATOMS[ia][1] - ATOMS[j][1],
                              ATOMS[ia][2] - ATOMS[j][2])
                if qj != 0.0:
                    if sum_mode == "minimage":
                        mx, my, mz = min_image(dx, dy, dz)
                        r = math.sqrt(mx * mx + my * my + mz * mz)
                        bi += K * qj * math.erf(ETA * r) / r
                    else:
                        lat = ewald_phi(dx, dy, dz)
                        corr = 0.0
                        for nx in range(-3, 4):
                            for ny in range(-3, 4):
                                for nz in range(-3, 4):
                                    rx, ry, rz = (dx + nx * LX, dy + ny * LY,
                                                  dz + nz * LZ)
                                    r = math.sqrt(rx * rx + ry * ry + rz * rz)
                                    if r > 1e-14:
                                        corr += math.erfc(ETA * r) / r
                        bi += K * qj * (lat - corr)
                if alphas[j] > 0.0:
                    mj = mu[j]
                    if sum_mode == "minimage":
                        mx, my, mz = min_image(dx, dy, dz)
                        r2 = mx * mx + my * my + mz * mz
                        r = math.sqrt(r2)
                        bi += K * (mj[0] * mx + mj[1] * my + mj[2] * mz) / (r2 * r)
                    else:
                        # d/d(r_j) of the Ewald 1/r potential, by central diff
                        h = 1e-5
                        g = [0.0, 0.0, 0.0]
                        for c in range(3):
                            dp = [dx, dy, dz]
                            dm = [dx, dy, dz]
                            dp[c] += h
                            dm[c] -= h
                            g[c] = (ewald_phi(*dp) - ewald_phi(*dm)) / (2 * h)
                        bi += K * (mj[0] * g[0] + mj[1] * g[1] + mj[2] * g[2])
            b.append(bi)

        rhs = [PSI[ATOMS[elec[a]][5]] - b[a] for a in range(m)]
        Qn = solve_neutral(A, rhs)
        dQ = max(abs(Qn[i] - Q[i]) for i in range(m))
        Q = Qn
        for a in range(m):
            charges[elec[a]] = Q[a]

        # ---- inner PIM SCC (always min-image; ATV has no Ewald here)
        mu_old = [row[:] for row in mu]
        e_perm = [[0.0] * 3 for _ in range(n)]
        for i in range(n):
            if alphas[i] <= 0:
                continue
            for j in range(n):
                if i == j or charges[j] == 0.0:
                    continue
                dx, dy, dz = min_image(ATOMS[i][0] - ATOMS[j][0],
                                       ATOMS[i][1] - ATOMS[j][1],
                                       ATOMS[i][2] - ATOMS[j][2])
                r2 = dx * dx + dy * dy + dz * dz
                r = math.sqrt(r2)
                l3 = 1.0
                if thole_a > 0 and alphas[i] > 0 and alphas[j] > 0:
                    l3 = thole_lambda(thole_u(r, alphas[i], alphas[j], thole_a))[0]
                f = l3 * charges[j] / (r2 * r)
                e_perm[i][0] += f * dx
                e_perm[i][1] += f * dy
                e_perm[i][2] += f * dz
        for _ in range(500):
            e = [row[:] for row in e_perm]
            for i in range(n):
                if alphas[i] <= 0:
                    continue
                for j in range(n):
                    if i == j or alphas[j] <= 0:
                        continue
                    dx, dy, dz = min_image(ATOMS[i][0] - ATOMS[j][0],
                                           ATOMS[i][1] - ATOMS[j][1],
                                           ATOMS[i][2] - ATOMS[j][2])
                    r2 = dx * dx + dy * dy + dz * dz
                    r = math.sqrt(r2)
                    l3, l5, _ = (1.0, 1.0, 1.0)
                    if thole_a > 0:
                        l3, l5, _ = thole_lambda(
                            thole_u(r, alphas[i], alphas[j], thole_a))
                    f3 = l3 / (r2 * r)
                    f5 = l5 / (r2 * r2 * r)
                    md = mu[j][0] * dx + mu[j][1] * dy + mu[j][2] * dz
                    e[i][0] += 3.0 * f5 * md * dx - f3 * mu[j][0]
                    e[i][1] += 3.0 * f5 * md * dy - f3 * mu[j][1]
                    e[i][2] += 3.0 * f5 * md * dz - f3 * mu[j][2]
            new = [[alphas[i] * e[i][c] for c in range(3)] for i in range(n)]
            d = max(abs(new[i][c] - mu[i][c]) for i in range(n) for c in range(3))
            mu = new
            if d < 1e-14:
                break
        dmu = max(abs(mu[i][c] - mu_old[i][c]) for i in range(n) for c in range(3))
        if dQ < joint_tol and dmu < joint_tol:
            break
    return Q, mu, outer


B2A = 0.52917721067
MW = {
    ("pbc2", "joint"): (+1.302179493406125e-01, 1.499868533452496e-01, 2.213337450494945e-01),
    ("pbc3", "joint"): (+6.707261457445883e-02, 1.326729243171588e-01, 1.957837940515414e-01),
    ("pbc2", "a0"): (+1.362412229597930e-01, None, None),
    ("pbc3", "a0"): (+6.913534940031944e-02, None, None),
}

print("ATV pins (regression):  Q_top=+7.3285008958883610e-02  "
      "mu_cat_z=+5.8951916148990660e-02  mu_an_z=+8.7432701391015569e-02  "
      "Q_top(a=0)=+7.5339627935767609e-02\n")

for mode in ("minimage", "ewald3d"):
    for th, ua, tag in ((0.39, True, "joint thole=0.39"),
                        (0.0, True, "joint thole=0   "),
                        (0.0, False, "alpha=0         ")):
        Q, mu, it = solve(sum_mode=mode, thole_a=th, use_alpha=ua)
        print(f"{mode:9s} {tag}  Q_top={Q[1]:+.16e}  "
              f"mu_cat_z={mu[2][2]:+.10e}  mu_an_z={mu[3][2]:+.10e}  outer={it}")
    print()

print("MetalWalls c8ad175d (converted e*Bohr -> e*A with 0.52917721067):")
for (p, t), (q, mc, ma) in MW.items():
    if mc is None:
        print(f"  {p} {t:5s}  Q_top={q:+.16e}")
    else:
        print(f"  {p} {t:5s}  Q_top={q:+.16e}  mu_cat_z={mc*B2A:+.10e}  "
              f"mu_an_z={ma*B2A:+.10e}")

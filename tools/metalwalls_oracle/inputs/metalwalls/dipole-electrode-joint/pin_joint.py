#!/usr/bin/env python3
"""Offline pin — coupled PIM dipoles ⊕ ConstPot electrode Q (MetalWalls E5 form).

No MetalWalls binary required. Algebra matches Atomiverse foothold kernels:
  - Gaussian elastance A (ConstPotBO, min-image R=0)
  - Electrolyte permanent → electrode Φ via erf Gaussian–point
  - Dipole → electrode Φ via Thole-scaled μ·r/r³
  - PIM SCC with Thole for polarizable non-electrode sites
  - Outer alternating joint solve

Oracle: metalwalls (JCP 2022 / MW multipole + Gaussian electrode BO form).
ctest never runs this script; pins are hard-coded in REFERENCE.md / regression.
"""
from __future__ import annotations

import math
from typing import List, Tuple

K = 14.399645  # ATV units::coulomb [eV·Å/e²]
SQRT2 = math.sqrt(2.0)
SQRT2_OVER_PI = math.sqrt(2.0 / math.pi)

# Geometry (IL + graphene parallel-plate toy)
ETA = 2.0  # [1/Å]
PSI = [0.0, 1.0]  # bot, top [V]
THOLE_A = 0.39
LX, LY, LZ = 10.0, 10.0, 20.0

# atoms: x,y,z, q_perm, type, is_elec, electrode_id, alpha
# types: 1 bot, 2 top, 3 cation, 4 anion
ATOMS = [
    # x, y, z, q0, typ, is_elec, eid, alpha
    (0.0, 0.0, 2.0, 0.0, 1, True, 0, 0.0),
    (0.0, 0.0, 18.0, 0.0, 2, True, 1, 0.0),
    (0.0, 0.0, 8.0, 1.0, 3, False, -1, 1.0),
    (0.0, 0.0, 12.0, -1.0, 4, False, -1, 1.5),
]


def min_image(dx: float, dy: float, dz: float) -> Tuple[float, float, float]:
    dx -= LX * round(dx / LX)
    dy -= LY * round(dy / LY)
    dz -= LZ * round(dz / LZ)
    return dx, dy, dz


def thole_u(r: float, ai: float, aj: float, a: float) -> float:
    if a <= 0 or ai <= 0 or aj <= 0 or r <= 0:
        return 0.0
    return a * r / (ai * aj) ** (1.0 / 6.0)


def thole_lambda(u: float) -> Tuple[float, float, float]:
    if u <= 0:
        return 1.0, 1.0, 1.0
    u3 = u * u * u
    u6 = u3 * u3
    expu = math.exp(-u3)
    l3 = 1.0 - expu
    l5 = 1.0 - (1.0 + u3) * expu
    l7 = 1.0 - (1.0 + u3 + 0.6 * u6) * expu
    return l3, l5, l7


def pair_A(r: float, eta: float) -> float:
    if r < 1e-14:
        return K * eta * SQRT2_OVER_PI
    etaij = eta * eta / math.sqrt(eta * eta + eta * eta)  # = eta/√2
    return K * math.erf(etaij * r) / r


def pair_b(r: float, q: float, eta: float) -> float:
    if r < 1e-14:
        return 0.0
    return K * q * math.erf(eta * r) / r


def build_A(elec_idx: List[int]) -> List[List[float]]:
    m = len(elec_idx)
    A = [[0.0] * m for _ in range(m)]
    for a in range(m):
        A[a][a] = K * ETA * SQRT2_OVER_PI
        ia = elec_idx[a]
        for b in range(a + 1, m):
            ib = elec_idx[b]
            dx, dy, dz = min_image(
                ATOMS[ia][0] - ATOMS[ib][0],
                ATOMS[ia][1] - ATOMS[ib][1],
                ATOMS[ia][2] - ATOMS[ib][2],
            )
            r = math.sqrt(dx * dx + dy * dy + dz * dz)
            aij = pair_A(r, ETA)
            A[a][b] = A[b][a] = aij
    return A


def cholesky_solve(A_in: List[List[float]], b: List[float]) -> List[float]:
    n = len(b)
    A = [row[:] for row in A_in]
    x = b[:]
    for i in range(n):
        for j in range(i + 1):
            s = A[i][j] - sum(A[i][k] * A[j][k] for k in range(j))
            if i == j:
                if s <= 0:
                    raise RuntimeError("not SPD")
                A[i][j] = math.sqrt(s)
            else:
                A[i][j] = s / A[j][j]
    # forward
    for i in range(n):
        x[i] = (x[i] - sum(A[i][k] * x[k] for k in range(i))) / A[i][i]
    # back
    for i in range(n - 1, -1, -1):
        x[i] = (x[i] - sum(A[k][i] * x[k] for k in range(i + 1, n))) / A[i][i]
    return x


def invert_spd(A: List[List[float]]) -> List[List[float]]:
    n = len(A)
    inv = [[0.0] * n for _ in range(n)]
    for j in range(n):
        e = [0.0] * n
        e[j] = 1.0
        col = cholesky_solve(A, e)
        for i in range(n):
            inv[i][j] = col[i]
    return inv


def solve_neutral(A: List[List[float]], rhs: List[float]) -> List[float]:
    Q0 = cholesky_solve(A, rhs)
    inv = invert_spd(A)
    m = len(rhs)
    u = [sum(inv[i][j] for j in range(m)) for i in range(m)]
    eTu = sum(u)
    eTQ = sum(Q0)
    scale = eTQ / eTu
    return [Q0[i] - scale * u[i] for i in range(m)]


def dipole_potential_at(
    i: int, mu: List[Tuple[float, float, float]], alphas: List[float]
) -> float:
    """Φ at atom i from dipoles on all polarizable sites j≠i (Thole s3)."""
    xi, yi, zi = ATOMS[i][0], ATOMS[i][1], ATOMS[i][2]
    ai = alphas[i]
    phi = 0.0
    for j, (mx, my, mz) in enumerate(mu):
        if j == i:
            continue
        if alphas[j] <= 0:
            continue
        dx, dy, dz = min_image(xi - ATOMS[j][0], yi - ATOMS[j][1], zi - ATOMS[j][2])
        r2 = dx * dx + dy * dy + dz * dz
        if r2 < 1e-28:
            continue
        r = math.sqrt(r2)
        l3, _, _ = thole_lambda(thole_u(r, ai if ai > 0 else alphas[j], alphas[j], THOLE_A))
        # electrode sites have α=0 → use undamped-with-aj Thole via aj only: u=0 if ai=0
        # Match C++ make_damp(r, ai, aj): if ai=0, thole off → λ=1
        if ai <= 0 or alphas[j] <= 0:
            l3 = 1.0
        else:
            l3, _, _ = thole_lambda(thole_u(r, ai, alphas[j], THOLE_A))
        mdotr = mx * dx + my * dy + mz * dz
        phi += K * l3 * mdotr / (r2 * r)
    return phi


def permanent_potential_at_elec(
    ia: int, charges: List[float], elec_set: set
) -> float:
    phi = 0.0
    for j, q in enumerate(charges):
        if j in elec_set or q == 0.0:
            continue
        dx, dy, dz = min_image(
            ATOMS[ia][0] - ATOMS[j][0],
            ATOMS[ia][1] - ATOMS[j][1],
            ATOMS[ia][2] - ATOMS[j][2],
        )
        r = math.sqrt(dx * dx + dy * dy + dz * dz)
        phi += pair_b(r, q, ETA)
    return phi


def scc_mu(charges: List[float], alphas: List[float], pol_tol: float = 1e-14, maxit: int = 200):
    n = len(ATOMS)
    # E_perm from all charges (point, Thole with α)
    e_perm = [[0.0, 0.0, 0.0] for _ in range(n)]
    for i in range(n):
        ai = alphas[i]
        if ai <= 0:
            continue
        for j in range(n):
            if i == j:
                continue
            qj = charges[j]
            if qj == 0.0:
                continue
            dx, dy, dz = min_image(
                ATOMS[i][0] - ATOMS[j][0],
                ATOMS[i][1] - ATOMS[j][1],
                ATOMS[i][2] - ATOMS[j][2],
            )
            r2 = dx * dx + dy * dy + dz * dz
            if r2 < 1e-28:
                continue
            r = math.sqrt(r2)
            aj = alphas[j]
            if THOLE_A > 0 and ai > 0 and aj > 0:
                l3, _, _ = thole_lambda(thole_u(r, ai, aj, THOLE_A))
            else:
                l3 = 1.0
            fac = l3 * qj / (r2 * r)
            e_perm[i][0] += fac * dx
            e_perm[i][1] += fac * dy
            e_perm[i][2] += fac * dz

    mu = [[0.0, 0.0, 0.0] for _ in range(n)]
    for it in range(maxit):
        e_tot = [row[:] for row in e_perm]
        for i in range(n):
            ai = alphas[i]
            if ai <= 0:
                continue
            for j in range(n):
                if i == j:
                    continue
                aj = alphas[j]
                if aj <= 0:
                    continue
                dx, dy, dz = min_image(
                    ATOMS[i][0] - ATOMS[j][0],
                    ATOMS[i][1] - ATOMS[j][1],
                    ATOMS[i][2] - ATOMS[j][2],
                )
                r2 = dx * dx + dy * dy + dz * dz
                if r2 < 1e-28:
                    continue
                r = math.sqrt(r2)
                r3 = r2 * r
                r5 = r3 * r2
                if THOLE_A > 0:
                    l3, l5, _ = thole_lambda(thole_u(r, ai, aj, THOLE_A))
                else:
                    l3 = l5 = 1.0
                f3 = l3 / r3
                f5 = l5 / r5
                mj = mu[j]
                mdotr = mj[0] * dx + mj[1] * dy + mj[2] * dz
                e_tot[i][0] += 3.0 * f5 * mdotr * dx - f3 * mj[0]
                e_tot[i][1] += 3.0 * f5 * mdotr * dy - f3 * mj[1]
                e_tot[i][2] += 3.0 * f5 * mdotr * dz - f3 * mj[2]
        dmax = 0.0
        mu_new = [[0.0, 0.0, 0.0] for _ in range(n)]
        for i in range(n):
            ai = alphas[i]
            for c in range(3):
                m = ai * e_tot[i][c]
                mu_new[i][c] = m
                dmax = max(dmax, abs(m - mu[i][c]))
        mu = mu_new
        if dmax < pol_tol:
            return mu, it + 1, True
    return mu, maxit, False


def joint_solve(use_alpha: bool = True, joint_tol: float = 1e-12, max_outer: int = 100):
    n = len(ATOMS)
    elec_idx = [i for i, a in enumerate(ATOMS) if a[5]]
    elec_set = set(elec_idx)
    alphas = [a[7] if use_alpha else 0.0 for a in ATOMS]
    charges = [a[3] for a in ATOMS]  # permanent; electrode overwritten
    A = build_A(elec_idx)
    m = len(elec_idx)
    Q = [0.0] * m
    mu = [(0.0, 0.0, 0.0)] * n
    outer = 0
    for outer in range(1, max_outer + 1):
        # b = Φ_perm + Φ_μ
        mu_list = [list(m_) for m_ in mu]
        b = []
        for a, ia in enumerate(elec_idx):
            phi_p = permanent_potential_at_elec(ia, charges, elec_set)
            # Φ_μ: electrode α=0 → undamped λ=1
            phi_m = 0.0
            for j in range(n):
                if alphas[j] <= 0:
                    continue
                dx, dy, dz = min_image(
                    ATOMS[ia][0] - ATOMS[j][0],
                    ATOMS[ia][1] - ATOMS[j][1],
                    ATOMS[ia][2] - ATOMS[j][2],
                )
                r2 = dx * dx + dy * dy + dz * dz
                if r2 < 1e-28:
                    continue
                r = math.sqrt(r2)
                mx, my, mz = mu_list[j]
                mdotr = mx * dx + my * dy + mz * dz
                phi_m += K * mdotr / (r2 * r)  # undamped (elec α=0)
            b.append(phi_p + phi_m)
        psi = [PSI[ATOMS[ia][6]] for ia in elec_idx]
        rhs = [psi[i] - b[i] for i in range(m)]
        Q_new = solve_neutral(A, rhs)
        dQ = max(abs(Q_new[i] - Q[i]) for i in range(m))
        Q = Q_new
        # write electrode charges
        for a, ia in enumerate(elec_idx):
            charges[ia] = Q[a]
        # PIM SCC
        mu_arr, pit, ok = scc_mu(charges, alphas)
        if not ok:
            raise RuntimeError("SCC failed")
        dmu = 0.0
        for i in range(n):
            for c in range(3):
                dmu = max(dmu, abs(mu_arr[i][c] - mu[i][c]))
        mu = [tuple(m_) for m_ in mu_arr]
        if dQ < joint_tol and dmu < joint_tol:
            break
    return {
        "Q": Q,
        "mu": mu,
        "outer": outer,
        "charges": charges,
        "sumQ": sum(Q),
        "alphas": alphas,
    }


def main() -> None:
    r0 = joint_solve(use_alpha=False)
    r1 = joint_solve(use_alpha=True)
    print("=== alpha=0 (ConstPot reduction) ===")
    print(f"Q_bot={r0['Q'][0]:.16e}")
    print(f"Q_top={r0['Q'][1]:.16e}")
    print(f"sumQ={r0['sumQ']:.3e} outer={r0['outer']}")
    print("=== alpha>0 joint ===")
    print(f"Q_bot={r1['Q'][0]:.16e}")
    print(f"Q_top={r1['Q'][1]:.16e}")
    print(f"sumQ={r1['sumQ']:.3e} outer={r1['outer']}")
    for label, idx in [("cation", 2), ("anion", 3)]:
        mx, my, mz = r1["mu"][idx]
        print(f"mu_{label}=({mx:.16e}, {my:.16e}, {mz:.16e})")
    dQ = abs(r1["Q"][1] - r0["Q"][1]) / max(abs(r0["Q"][1]), 1e-30)
    print(f"rel |Q_top(α)-Q_top(0)| = {dQ:.6e}")


if __name__ == "__main__":
    main()

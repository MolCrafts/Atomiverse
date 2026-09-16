#!/usr/bin/env python3
"""Is ATV's real-space xy image sum absolutely convergent (shape-independent)?

Same 8-site case-2 elastance, truncated on (a) the square |nx|,|ny| <= R that
ConstPotBO uses and (b) a circle |n| <= R.  If both Richardson-extrapolate to
the same limit the sum is absolutely convergent and the limit is the physical
2D answer; if they differ the sum is conditionally convergent and the
truncation shape (not the physics) sets the answer.
"""
import math
import sys

import numpy as np
from scipy.special import erf

K = 14.399645
S2PI = 0.79788456080286535588
SQRT2 = 1.41421356237309504880
LX = LY = 4.0

POS = np.array([(2.0 * ax, 2.0 * ay, z)
                for z in (8.0, 13.0) for ax in range(2) for ay in range(2)])
ETA = np.array([1.979] * 4 + [0.955] * 4)
PSI = np.array([0.0] * 4 + [1.0] * 4)


def lattice(R, shape):
    n = np.arange(-R, R + 1, dtype=np.float64)
    NX, NY = np.meshgrid(n, n, indexing="ij")
    NX, NY = NX.ravel(), NY.ravel()
    if shape == "circle":
        keep = (NX * NX + NY * NY) <= R * R
        NX, NY = NX[keep], NY[keep]
    return NX, NY


def solve(R, shape="square"):
    NX, NY = lattice(R, shape)
    m = 8
    A = np.zeros((m, m))
    r0 = np.hypot(NX * LX, NY * LY)
    sel0 = r0 > 1e-14
    r0s = r0[sel0]
    for a in range(m):
        A[a, a] = K * ETA[a] * S2PI + np.sum(K * erf((ETA[a] / SQRT2) * r0s) / r0s)
        for b in range(a + 1, m):
            eab = ETA[a] * ETA[b] / math.sqrt(ETA[a] ** 2 + ETA[b] ** 2)
            dx = POS[a, 0] - POS[b, 0] + NX * LX
            dy = POS[a, 1] - POS[b, 1] + NY * LY
            dz = POS[a, 2] - POS[b, 2]
            r = np.sqrt(dx * dx + dy * dy + dz * dz)
            A[a, b] = A[b, a] = np.sum(K * erf(eab * r) / r)
    inv = np.linalg.inv(A)
    Qu = inv @ PSI
    u = inv.sum(axis=1)
    Q = Qu - (Qu.sum() / u.sum()) * u
    return Q[4:].sum()


if __name__ == "__main__":
    MW = 4 * 4.908309464985498e-03
    print("MW 2D-Ewald Q_top = %.15e\n" % MW)
    for shape in ("square", "circle"):
        vals = []
        for R in (200, 400, 800, 1600, 3200):
            q = solve(R, shape)
            vals.append((R, q))
            print("%-7s R=%5d  Q_top=%.15e  rel vs MW=%+.6e"
                  % (shape, R, q, (q - MW) / MW))
        # Richardson on an assumed c/R tail
        for i in range(len(vals) - 1):
            lim = 2 * vals[i + 1][1] - vals[i][1]
            print("        Richardson(%d,%d) -> %.15e   rel vs MW=%+.6e"
                  % (vals[i][0], vals[i + 1][0], lim, (lim - MW) / MW))
        print()

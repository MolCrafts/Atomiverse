#!/usr/bin/env python3
"""Compare the real MetalWalls constant-potential charges against the ATV
regression pin for md/per-electrode-eta case 1.

ATV case 1 is the *isolated* (image-free) two-site capacitor: ConstPotBO with
xy_image_range = 0 sums a single minimum image, i.e. exactly one pair at
r = 4 A.  MetalWalls is an Ewald code and is unconditionally periodic, so the
MW twin carries the 2D image lattice.  For the electroneutral two-site solve
the elastance combination

    S(L) = A11 + A22 - 2 A12  = DeltaPsi / Q          [eV/e^2]

is background-free, and its periodic value differs from the isolated one by an
absolutely convergent dipole-lattice sum

    dS(L) = 2 k sum_{n != 0} [ 1/R_n - 1/sqrt(R_n^2 + d^2) ],  R_n = L |n|

(all erf factors are 1 to <1e-300 at these separations).  Subtracting dS(L)
from the MW result must reproduce the pinned isolated S for every L.
"""
import math
import os
import sys

import numpy as np

BOHR = 0.529177210903
HA_EV = 27.211386245988
K_COUL = HA_EV * BOHR            # eV*A/e^2  == ATV units::coulomb

D_A = 4.0
ETA1, ETA2 = 1.979, 0.955

Q_PIN = 0.037722100829787        # e, regression hard-coded pin


def isolated_S():
    s2pi = math.sqrt(2.0 / math.pi)
    a11 = K_COUL * s2pi * ETA1
    a22 = K_COUL * s2pi * ETA2
    e12 = ETA1 * ETA2 / math.sqrt(ETA1**2 + ETA2**2)
    a12 = K_COUL * math.erf(e12 * D_A) / D_A
    return a11, a22, a12, e12, a11 + a22 - 2.0 * a12


def image_dS(L, nmax=20000):
    """2 k sum_{n!=0} [1/R - 1/sqrt(R^2+d^2)] for a square lattice, edge L [A]."""
    i = np.arange(-nmax, nmax + 1, dtype=np.float64)
    tot = 0.0
    d2 = D_A * D_A
    for ix in i:
        s2 = ix * ix + i * i
        if ix == 0.0:
            s2 = s2[s2 > 0.0]
        R = L * np.sqrt(s2)
        tot += np.sum(1.0 / R - 1.0 / np.sqrt(R * R + d2))
    # continuum tail beyond the square |i|,|j| <= nmax (radius >= nmax*L)
    Rc = nmax * L
    tail = math.pi * d2 / (L * L * Rc)
    return 2.0 * K_COUL * (tot + tail)


def read_q(path):
    with open(path) as f:
        rows = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    return -float(rows[0])          # rows[0] = Q_bot; Q_top = -Q_bot


if __name__ == "__main__":
    root = sys.argv[1]
    a11, a22, a12, e12, S_iso = isolated_S()
    print("k (units::coulomb) = %.15f eV*A/e^2  [= %.15f Ha*bohr/e^2 * %.12f * %.12f]"
          % (K_COUL, 1.0, HA_EV, BOHR))
    print("A11 = %.15f  A22 = %.15f  eta12 = %.15f  A12 = %.15f"
          % (a11, a22, e12, a12))
    print("S_iso (closed form) = %.15f eV/e^2 -> Q = %.15f e" % (S_iso, 1.0 / S_iso))
    print("pin                                        Q = %.15f e" % Q_PIN)
    print()
    hdr = ("  L[A]   Q_MW[e]              S_MW=1/Q          dS_img(L)"
           "       S_MW-dS            Q_iso(MW)          rel vs pin")
    print(hdr)
    rows = []
    for name in sorted(os.listdir(root), key=lambda s: int(s[1:])):
        L = float(name[1:])
        q = read_q(os.path.join(root, name, "charges.out"))
        S_mw = 1.0 / q
        dS = image_dS(L)
        S_corr = S_mw - dS
        q_iso = 1.0 / S_corr
        rel = (q_iso - Q_PIN) / Q_PIN
        rows.append((L, q, S_mw, dS, S_corr, q_iso, rel))
        print("%6.0f  %.15f  %.12f  %.4e  %.12f  %.15f  %+.3e"
              % (L, q, S_mw, dS, S_corr, q_iso, rel))
    print()
    print("raw (uncorrected) MW vs pin:")
    for L, q, *_ in rows:
        print("  L=%6.0f A : Q_MW = %.15f e   rel = %+.6e" % (L, q, (q - Q_PIN) / Q_PIN))

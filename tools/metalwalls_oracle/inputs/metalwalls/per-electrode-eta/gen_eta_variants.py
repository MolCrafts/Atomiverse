#!/usr/bin/env python3
"""eta-variant MW twins: (mixed, shared-1.979, shared-0.955) at one large box.

Validates A_ii(eta) and A_ij(eta_i, eta_j) separately, not only the mixed
combination: the two shared-eta runs fix each diagonal on its own.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_case1 as gen

if __name__ == "__main__":
    root = sys.argv[1]
    L = float(sys.argv[2]) if len(sys.argv) > 2 else 320.0
    lzf = float(sys.argv[3]) if len(sys.argv) > 3 else 20.0
    for tag, (e1, e2) in {
        "mixed": (1.979, 0.955),
        "hi": (1.979, 1.979),
        "lo": (0.955, 0.955),
    }.items():
        gen.ETA1_A, gen.ETA2_A = e1, e2
        gen.ETA1, gen.ETA2 = e1 * gen.BOHR, e2 * gen.BOHR
        gen.write_case(os.path.join(root, tag), L, Lz_A=lzf * L)
    print("written")

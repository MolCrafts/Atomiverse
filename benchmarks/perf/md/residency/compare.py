#!/usr/bin/env python3
"""Compare two md-residency final states (hex floats).

Prints whether positions/velocities are bitwise identical, how many atom rows
differ and the largest absolute difference, plus the scalar header lines.
"""
import sys


def load(path):
    head, rows = {}, []
    for line in open(path).read().splitlines():
        parts = line.split()
        if len(parts) == 2:
            head[parts[0]] = parts[1]
        elif parts:
            rows.append([float.fromhex(v) for v in parts])
    return head, rows


(ha, ra), (hb, rb) = load(sys.argv[1]), load(sys.argv[2])
differ = sum(1 for x, y in zip(ra, rb) if x != y)
worst = max((abs(u - v) for x, y in zip(ra, rb) for u, v in zip(x, y)), default=0.0)
rebuilds = ha.get("rebuilds") == hb.get("rebuilds")
print(f"{'IDENTICAL' if differ == 0 and len(ra) == len(rb) else 'DIFFERS'} "
      f"rows={differ}/{len(ra)} max_abs={worst:.3e} same_rebuilds={rebuilds} "
      f"energy {float.fromhex(ha['energy']):.12g} vs {float.fromhex(hb['energy']):.12g}")

#!/usr/bin/env python3
"""Offline dftd4 3.7.0 strained-cell pins for gpw-d4-kpoint-stress.

Never launched at ctest time. Requires `dftd4` on PATH (site EasyBuild
3.7.0-gfbf-2025b). Writes POSCARs and prints E_disp + FD sigma terms.
"""
from __future__ import annotations

import json
import subprocess
import sys

H = 2.0e-3
L = 12.0
ATOMS = [
    ("O", 0.000000, 0.000000, 0.000000),
    ("O", 3.000000, 0.000000, 0.000000),
    ("H", 0.758602, 0.000000, 0.504284),
    ("H", 0.758602, 0.000000, -0.504284),
    ("H", 3.758602, 0.000000, 0.504284),
    ("H", 3.758602, 0.000000, -0.504284),
]
ANG2BOHR = 1.0 / 0.52917720859


def write_poscar(path: str, scale_axis: int | None = None, scale: float = 1.0) -> None:
    cell = [[L, 0.0, 0.0], [0.0, L, 0.0], [0.0, 0.0, L]]
    coords = []
    for el, x, y, z in ATOMS:
        p = [x, y, z]
        if scale_axis is not None:
            p[scale_axis] *= scale
            cell[scale_axis][scale_axis] = L * scale
        coords.append((el, p))
    with open(path, "w", encoding="utf-8") as f:
        f.write("water dimer D4 strain pin\n1.0\n")
        for row in cell:
            f.write(f"{row[0]:.12f} {row[1]:.12f} {row[2]:.12f}\n")
        f.write("O H\n2 4\nCartesian\n")
        for el, p in coords:
            f.write(f"{p[0]:.12f} {p[1]:.12f} {p[2]:.12f}\n")


def run_dftd4(poscar: str) -> float:
    outj = poscar + ".json"
    cmd = ["dftd4", "-f", "pbe", "--mbdscale", "0.0", "--json", outj, poscar]
    r = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if r.returncode != 0:
        sys.stderr.write(r.stderr + r.stdout)
        raise SystemExit(f"dftd4 failed on {poscar}")
    with open(outj, encoding="utf-8") as f:
        data = json.load(f)
    e = data.get("energy")
    if e is None:
        raise SystemExit(f"no energy in {outj}: {data!r}")
    return float(e)


def main() -> None:
    write_poscar("POSCAR")
    e0 = run_dftd4("POSCAR")
    print(f"E_disp unstrained            = {e0:.16e} Ha")
    vol = (L * ANG2BOHR) ** 3
    for a, name in enumerate("xyz"):
        ep = {}
        for s, tag in ((H, "+h"), (-H, "-h")):
            scale = 1.0 + s
            path = f"strain_{name}_{tag}.poscar"
            write_poscar(path, scale_axis=a, scale=scale)
            ep[tag] = run_dftd4(path)
            print(f"E_disp {name} {tag:2s}              = {ep[tag]:.16e} Ha")
        sigma = -(ep["+h"] - ep["-h"]) / (2.0 * H) / vol
        print(f"sigma_D4_{name}{name} (FD, dftd4)    = {sigma:.10e} Ha/Bohr3")


if __name__ == "__main__":
    main()

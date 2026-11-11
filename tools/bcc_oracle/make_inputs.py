#!/usr/bin/env python3
"""Write the fixed oracle geometries for the AM1-BCC pins as Tripos mol2.

Idealised textbook geometries (standard bond lengths and angles, no
optimisation by any code). They are inputs only: both antechamber/sqm and
Atomiverse are run at exactly these coordinates (`maxcyc=0`). Coordinates
are rounded to 1e-3 A because antechamber stores them at that precision
before writing `sqm.in`; rounding here makes the two runs bit-identical
in geometry.

    python3 tools/bcc_oracle/make_inputs.py tools/bcc_oracle/inputs
"""
import math
import pathlib
import sys


def phenol():
    # Planar ring in the xy plane, regular hexagon, C-C 1.39 A.
    cc, ch, co, oh = 1.39, 1.08, 1.36, 0.96
    atoms, bonds = [], []
    for k in range(6):
        a = math.radians(60.0 * k)
        atoms.append(("C", "C.ar", cc * math.cos(a), cc * math.sin(a), 0.0))
    for k in range(6):
        bonds.append((k, (k + 1) % 6, "ar"))
    # C0 carries O; C1..C5 carry H, all radial.
    o = (cc + co, 0.0, 0.0)
    atoms.append(("O", "O.3", *o))
    bonds.append((0, 6, "1"))
    for k in range(1, 6):
        a = math.radians(60.0 * k)
        r = cc + ch
        atoms.append(("H", "H", r * math.cos(a), r * math.sin(a), 0.0))
        bonds.append((k, len(atoms) - 1, "1"))
    # Hydroxyl H in plane, C-O-H 109.0 deg.
    t = math.radians(180.0 - 109.0)
    atoms.append(("H", "H", o[0] + oh * math.cos(t), oh * math.sin(t), 0.0))
    bonds.append((6, len(atoms) - 1, "1"))
    return "phenol", atoms, bonds


def acetate():
    cc, co, ch = 1.52, 1.25, 1.09
    atoms = [("C", "C.3", 0.0, 0.0, 0.0), ("C", "C.2", cc, 0.0, 0.0)]
    bonds = [(0, 1, "1")]
    # Carboxylate O at +-120 deg from the C-C axis, in the xy plane.
    for sgn in (+1, -1):
        t = math.radians(sgn * 60.0)
        atoms.append(("O", "O.co2", cc + co * math.cos(t), co * math.sin(t), 0.0))
        bonds.append((1, len(atoms) - 1, "2" if sgn > 0 else "1"))
    # Methyl H, tetrahedral about C0 pointing away from C1.
    tet = math.radians(109.47)
    for k in range(3):
        p = math.radians(120.0 * k + 90.0)
        x = ch * math.cos(tet)
        r = ch * math.sin(tet)
        atoms.append(("H", "H", x, r * math.cos(p), r * math.sin(p)))
        bonds.append((0, len(atoms) - 1, "1"))
    return "acetate", atoms, bonds


def write_mol2(path, name, atoms, bonds, charge_note):
    with open(path, "w") as f:
        f.write(f"@<TRIPOS>MOLECULE\n{name}\n{len(atoms)} {len(bonds)} 1 0 0\n")
        f.write(f"SMALL\nNO_CHARGES\n{charge_note}\n\n@<TRIPOS>ATOM\n")
        counts = {}
        for i, (el, sy, x, y, z) in enumerate(atoms, 1):
            counts[el] = counts.get(el, 0) + 1
            x, y, z = (round(v, 3) + 0.0 for v in (x, y, z))
            f.write(f"{i:7d} {el}{counts[el]:<4d} {x:12.6f} {y:12.6f} {z:12.6f} "
                    f"{sy:<6s} 1 MOL 0.0000\n")
        f.write("@<TRIPOS>BOND\n")
        for i, (a, b, t) in enumerate(bonds, 1):
            f.write(f"{i:6d} {a + 1:5d} {b + 1:5d} {t}\n")


if __name__ == "__main__":
    out = pathlib.Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    for build, note in ((phenol, "net charge 0"), (acetate, "net charge -1")):
        name, atoms, bonds = build()
        write_mol2(out / f"{name}.mol2", name, atoms, bonds, note)

#!/usr/bin/env python3
"""Gate G1: compare one single point across LAMMPS, OpenMM and Atomiverse.

Energies are compared per component in kcal/mol; forces as the RMS of the
per-atom difference relative to the RMS force of the reference (LAMMPS).

Components every code reports the same way:

=============  ==============================  =====================
component      LAMMPS thermo                   OpenMM / Atomiverse
=============  ==============================  =====================
bond           ebond                           bond / bond
angle          eangle                          angle / angle
torsion        edihed + eimp                   torsion / dihedral+improper
nonbonded      evdwl + ecoul + elong           lj+coulomb / nonbonded
lj             evdwl                           lj / --
coulomb        ecoul + elong                   coulomb / --
=============  ==============================  =====================

Usage::

    python compare_sp.py --lammps-log lmp_sp.log --lammps-forces forces.dump \\
        [--openmm omm_sp.json] [--atv-log atv_sp.log --atv-forces atv.forces]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


def lammps_energies(log: Path) -> dict[str, float]:
    lines = log.read_text().splitlines()
    for k, line in enumerate(lines):
        if line.split()[:2] == ["Step", "Temp"]:
            head = line.split()
            vals = [float(v) for v in lines[k + 1].split()]
            row = dict(zip(head, vals))
            break
    else:
        raise ValueError(f"{log}: no thermo header")
    return {
        "bond": row["E_bond"], "angle": row["E_angle"],
        "torsion": row["E_dihed"] + row["E_impro"],
        "lj": row["E_vdwl"], "coulomb": row["E_coul"] + row["E_long"],
        "nonbonded": row["E_vdwl"] + row["E_coul"] + row["E_long"],
        "total": row["PotEng"],
    }


def lammps_forces(dump: Path) -> np.ndarray:
    lines = dump.read_text().splitlines()
    start = next(i for i, l in enumerate(lines)
                 if l.startswith("ITEM: ATOMS")) + 1
    rows = np.array([[float(x) for x in l.split()] for l in lines[start:]])
    return rows[np.argsort(rows[:, 0])][:, 1:4]


def openmm_result(path: Path) -> tuple[dict[str, float], np.ndarray]:
    data = json.loads(path.read_text())["energies_kcal_mol"]
    energies = {
        "bond": data["bond"], "angle": data["angle"],
        "torsion": data["torsion"], "lj": data["lj"],
        "coulomb": data["coulomb"],
        "nonbonded": data["lj"] + data["coulomb"], "total": data["total"],
    }
    forces = np.load(path.with_suffix(".forces.npy"))
    return energies, forces


def atv_result(log: Path, forces: Path) -> tuple[dict[str, float],
                                                 np.ndarray]:
    match = re.search(r"SP_KCAL (.*)", log.read_text())
    if not match:
        raise ValueError(f"{log}: no SP_KCAL line")
    row = {k: float(v) for k, v in
           (kv.split("=") for kv in match.group(1).split())}
    energies = {
        "bond": row["bond"], "angle": row["angle"],
        "torsion": row["dihedral"] + row["improper"],
        "nonbonded": row["nonbonded"], "total": row["total"],
    }
    data = np.loadtxt(forces)
    return energies, data[np.argsort(data[:, 0])][:, 1:4]


def report(name: str, ref: dict[str, float], other: dict[str, float],
           f_ref: np.ndarray, f_other: np.ndarray) -> dict:
    out: dict = {"energies": {}}
    print(f"\n== {name} vs LAMMPS")
    print(f"{'component':<11}{'LAMMPS':>22}{name:>22}{'abs diff':>12}"
          f"{'rel diff':>11}")
    for key in ("bond", "angle", "torsion", "lj", "coulomb", "nonbonded",
                "total"):
        if key not in other:
            continue
        a, b = ref[key], other[key]
        rel = abs(b - a) / max(abs(a), 1e-300)
        out["energies"][key] = {"lammps": a, name: b, "abs": b - a,
                                "rel": rel}
        print(f"{key:<11}{a:22.12f}{b:22.12f}{b - a:12.3e}{rel:11.2e}")
    diff = f_other - f_ref
    rms_ref = float(np.sqrt(np.mean(f_ref ** 2)))
    rel_rms = float(np.sqrt(np.mean(diff ** 2))) / rms_ref
    rel_max = float(np.max(np.abs(diff))) / rms_ref
    out["forces"] = {"rms_ref_kcal_mol_A": rms_ref, "rel_rms": rel_rms,
                     "rel_max": rel_max}
    print(f"forces: |F| rms {rms_ref:.4f} kcal/mol/Å, diff rms/ref "
          f"{rel_rms:.2e}, max/ref {rel_max:.2e}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lammps-log", type=Path, required=True)
    ap.add_argument("--lammps-forces", type=Path, required=True)
    ap.add_argument("--openmm", type=Path)
    ap.add_argument("--atv-log", type=Path)
    ap.add_argument("--atv-forces", type=Path)
    ap.add_argument("--json", type=Path, help="write the comparison here")
    args = ap.parse_args()

    ref = lammps_energies(args.lammps_log)
    f_ref = lammps_forces(args.lammps_forces)
    result: dict = {}
    if args.openmm:
        e, f = openmm_result(args.openmm)
        result["openmm"] = report("OpenMM", ref, e, f_ref, f)
    if args.atv_log:
        e, f = atv_result(args.atv_log, args.atv_forces)
        result["atv"] = report("ATV", ref, e, f_ref, f)
    if args.json:
        args.json.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Check shared-state LJ forces and total energies; requires only Python's stdlib."""
import argparse
import csv
import glob
import json
import math

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("prefix", help="same OUTPUT_PREFIX / LAMMPS PREFIX used by the run")
parser.add_argument("--stage", choices=("initial", "final"), default="initial")
parser.add_argument("--force-atol", type=float, default=2e-10, help="eV/Angstrom")
parser.add_argument("--energy-atol", type=float, default=2e-10, help="eV per atom")
args = parser.parse_args()

atv = {}
energies = []
paths = sorted(glob.glob(f"{args.prefix}.{args.stage}.rank*.csv"))
if not paths:
    raise SystemExit("No ATV owner snapshots found")
for path in paths:
    with open(path) as stream:
        energies.append(float(stream.readline().removeprefix("# energy ")))
        for row in csv.DictReader(stream):
            tag = int(row.pop("id"))
            if tag in atv:
                raise SystemExit(f"Duplicate ATV owner tag {tag}; use a fresh prefix per run")
            atv[tag] = {key: float(value) for key, value in row.items()}
with open(f"{args.prefix}.lmp.{args.stage}") as stream:
    lines = iter(stream)
    for line in lines:
        if line.startswith("ITEM: ATOMS "):
            columns = line.split()[2:]
            break
    else:
        raise SystemExit("Missing LAMMPS ATOMS section")
    lmp = {}
    for line in lines:
        row = dict(zip(columns, line.split(), strict=True))
        tag = int(row.pop("id"))
        if tag in lmp:
            raise SystemExit(f"Duplicate LAMMPS tag {tag}")
        lmp[tag] = {key: float(value) for key, value in row.items()}
if atv.keys() != lmp.keys() or not atv:
    raise SystemExit(f"Owner tag mismatch: ATV={len(atv)}, LAMMPS={len(lmp)}")
with open(f"{args.prefix}.lmp.energy") as stream:
    oracle = dict(line.split() for line in stream if line.strip())
lmp_energy = float(oracle[f"{args.stage.upper()}_ENERGY"])
if not all(math.isfinite(value) for engine in (atv, lmp) for row in engine.values() for value in row.values()):
    raise SystemExit("Non-finite position or force")
if not all(math.isfinite(value) for value in energies + [lmp_energy]):
    raise SystemExit("Non-finite energy")
force_max = max(abs(atv[tag][key] - lmp[tag][key]) for tag in atv for key in ("fx", "fy", "fz"))
energy_error = max(abs(energy - lmp_energy) for energy in energies) / len(atv)
passed = force_max <= args.force_atol and energy_error <= args.energy_atol
print(json.dumps({"stage": args.stage, "atoms": len(atv), "max_force_error_eV_A": force_max,
                  "energy_error_eV_atom": energy_error, "atv_energy_eV": energies[0],
                  "lammps_energy_eV": lmp_energy, "passed": passed}, indent=2))
raise SystemExit(0 if passed else 1)

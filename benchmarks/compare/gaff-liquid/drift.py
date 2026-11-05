#!/usr/bin/env python3
"""Gate G2: NVE total-energy drift from LAMMPS, OpenMM and Atomiverse runs.

Fits E_total(t) with a straight line and reports the slope in kcal/mol/ns
and normalised as k_B T per degree of freedom per ns, the usual way drift is
compared across system sizes. Ndof = 3N - N_constraints - 3.

Inputs (any subset):
  --lammps LOG   in.gaff with ``-var mode md -var ensemble nve``
  --openmm JSON  run_openmm.py ``--mode md --ensemble nve``
  --atv LOG      run_atv_gaff ``--nve --thermo N``

Usage::

    python drift.py --atoms 24840 --constraints 13500 --dt 2.0 \\
        --lammps lmp_nve.log --openmm omm_nve.json --atv atv_nve.log
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

KCAL_PER_EV = 23.060547830619026  # same factor as cpu/io/lammps_real.h
KB_KCAL = 0.0019872043  # kcal/mol/K


def lammps_series(log: Path) -> np.ndarray:
    lines = log.read_text().splitlines()
    starts = [k for k, l in enumerate(lines) if l.split()[:2] == ["Step",
                                                                 "Temp"]]
    if not starts:
        raise ValueError(f"{log}: no thermo table")
    rows = []
    for k in starts:  # warmup and timed runs are one NVE trajectory
        head = lines[k].split()
        col = head.index("TotEng")
        for line in lines[k + 1:]:
            tok = line.split()
            if not tok or not tok[0].isdigit():
                break
            rows.append((int(tok[0]), float(tok[col])))
    return np.array(sorted(set(rows)))


def openmm_series(path: Path) -> np.ndarray:
    return np.array(json.loads(path.read_text())["etotal_kcal_mol"])


def atv_series(log: Path) -> np.ndarray:
    # Thermo rows: "  <step>  <PE> <KE> <T> <TotalE> <Conserved>" [eV].
    row = re.compile(r"\]\s+(\d+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s*$")
    ansi = re.compile(r"\x1b\[[0-9;]*m")  # spdlog colours the level tag
    out = []
    for line in log.read_text().splitlines():
        match = row.search(ansi.sub("", line))
        if match:
            out.append((int(match.group(1)),
                        float(match.group(5)) * KCAL_PER_EV))
    if not out:
        raise ValueError(f"{log}: no Thermo rows (run with --thermo N)")
    return np.array(out)


def drift(series: np.ndarray, dt_fs: float, ndof: int, temp: float,
          skip_ps: float) -> dict:
    t_ns = series[:, 0] * dt_fs * 1.0e-6
    keep = t_ns >= skip_ps * 1.0e-3
    series, t_ns = series[keep], t_ns[keep]
    slope, _ = np.polyfit(t_ns, series[:, 1], 1)
    return {"points": int(len(series)), "span_ns": float(t_ns[-1] - t_ns[0]),
            "slope_kcal_mol_ns": float(slope),
            "kT_per_dof_per_ns": float(slope / (ndof * KB_KCAL * temp)),
            "fluctuation_kcal_mol": float(np.std(series[:, 1]))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--atoms", type=int, required=True)
    ap.add_argument("--constraints", type=int, default=0)
    ap.add_argument("--dt", type=float, default=2.0, help="fs")
    ap.add_argument("--temp", type=float, default=300.0)
    ap.add_argument("--skip-ps", type=float, default=1.0,
                    help="leading transient to drop: LAMMPS `velocity create`"
                         " seeds unconstrained velocities that RATTLE then "
                         "projects, a step drop in E_total")
    ap.add_argument("--lammps", type=Path)
    ap.add_argument("--openmm", type=Path)
    ap.add_argument("--atv", type=Path)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    ndof = 3 * args.atoms - args.constraints - 3
    result = {}
    for name, path, reader in (("lammps", args.lammps, lammps_series),
                               ("openmm", args.openmm, openmm_series),
                               ("atv", args.atv, atv_series)):
        if path:
            result[name] = drift(reader(path), args.dt, ndof, args.temp,
                                 args.skip_ps)
    for name, r in result.items():
        print(f"{name:>7}: slope {r['slope_kcal_mol_ns']:+.4e} kcal/mol/ns "
              f"= {r['kT_per_dof_per_ns']:+.3e} kT/dof/ns over "
              f"{r['span_ns']:.3f} ns, sd {r['fluctuation_kcal_mol']:.3f}")
    if args.json:
        args.json.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()

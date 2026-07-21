#!/usr/bin/env python3
"""compare.py — diff LAMMPS vs Atomiverse thermo logs for amorphous PE.

Reads:
  log.PE_nc10_cl1000.txt   (LAMMPS, thermo_style custom
                             step temp press pe ke etotal density
                             vol lx ly lz)
  log.atomiverse.txt       (Atomiverse, ThermoPrinter's matching columns)

Output:
  - prints a side-by-side text table (sampled at LAMMPS-print intervals)
  - writes compare.png with one row per observable, one column per
    LAMMPS stage. Observables plotted:

        * Temperature [K]
        * Pressure    [atm]
        * Potential, kinetic and total energy [eV]
        * Density     [g/cc]
        * RMS velocity v_rms = sqrt(2·KE / M_total) [Å/fs]
          (derived identically from KE on both engines, so it is a
          one-to-one visualisation of the KE column in Å/fs units)

Sign / unit conventions:
  * Pressure: LAMMPS and Atomiverse both report in atm, compression-positive.
    `frame.stress` stores the LAMMPS-style virial directly; ThermoPrinter
    prints `+trace/(3V)` so the sign matches LAMMPS without any flip.
  * Energy: LAMMPS reports in kcal/mol ('real' units), Atomiverse in eV.
    We convert LAMMPS to eV (`kKcalPerMolToEv = 1 / 23.060547830619026`)
    so both axes are in eV.
  * Velocity: derived from KE and total mass, so consistent across both
    engines without needing an additional log column.

Usage:
    python compare.py [--lammps log.PE_nc10_cl1000.txt]
                      [--atomiverse log.atomiverse.txt]
                      [--out compare.png]
"""
from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

# ── Physical constants (CODATA-consistent with include/atomiverse/core/units.h)

KCAL_PER_MOL_TO_EV = 1.0 / 23.060547830619026
# units::mass_v2_to_energy: 1 amu · (Å/fs)² in eV. Matches units.h `mass_v2_to_energy`.
KMVV2E = 103.6427

# ────────────────────────────────────────────────────────────────────


@dataclass
class Stage:
    label: str
    step: list[int] = field(default_factory=list)
    temp: list[float] = field(default_factory=list)
    press: list[float] = field(default_factory=list)
    pot_e: list[float] = field(default_factory=list)   # eV
    kin_e: list[float] = field(default_factory=list)   # eV
    tot_e: list[float] = field(default_factory=list)   # eV
    density: list[float] = field(default_factory=list) # g/cc
    vol: list[float] = field(default_factory=list)     # Å³
    lx: list[float] = field(default_factory=list)
    ly: list[float] = field(default_factory=list)
    lz: list[float] = field(default_factory=list)

    def v_rms(self, total_mass_amu: float) -> list[float]:
        """RMS velocity [Å/fs] derived from KE: sqrt(2 KE / (M_total × units::mass_v2_to_energy))."""
        if total_mass_amu <= 0.0:
            return []
        out = []
        for ke in self.kin_e:
            v2 = 2.0 * ke / (total_mass_amu * KMVV2E) if ke > 0 else 0.0
            out.append(math.sqrt(max(v2, 0.0)))
        return out


# ── LAMMPS log parser ──────────────────────────────────────────────


# Thermo header: "Step Temp Press PotEng KinEng TotEng Density Volume Lx Ly Lz"
# (LAMMPS columns: pe=PotEng, ke=KinEng, etotal=TotEng).
_LAMMPS_HEADER = re.compile(
    r"^\s*Step\s+Temp\s+Press\s+PotEng\s+KinEng\s+TotEng\s+Density\s+"
    r"Volume\s+Lx\s+Ly\s+Lz\b",
    re.IGNORECASE,
)
# Data line: 11 whitespace-separated numeric tokens.
_LAMMPS_RUN_LINE = re.compile(
    r"^\s*\d+(?:\s+[-+0-9.eE]+){10}\s*$")


def parse_lammps(path: Path) -> list[Stage]:
    stages: list[Stage] = []
    cur: Stage | None = None
    in_thermo = False
    stage_idx = 0
    with path.open() as f:
        for line in f:
            if _LAMMPS_HEADER.search(line):
                stage_idx += 1
                cur = Stage(label=f"LAMMPS Stage {stage_idx}")
                stages.append(cur)
                in_thermo = True
                continue
            if in_thermo and _LAMMPS_RUN_LINE.match(line):
                tok = line.split()
                assert cur is not None
                cur.step.append(int(tok[0]))
                cur.temp.append(float(tok[1]))
                cur.press.append(float(tok[2]))
                # pe / ke / etotal: kcal/mol → eV
                cur.pot_e.append(float(tok[3]) * KCAL_PER_MOL_TO_EV)
                cur.kin_e.append(float(tok[4]) * KCAL_PER_MOL_TO_EV)
                cur.tot_e.append(float(tok[5]) * KCAL_PER_MOL_TO_EV)
                cur.density.append(float(tok[6]))
                cur.vol.append(float(tok[7]))
                cur.lx.append(float(tok[8]))
                cur.ly.append(float(tok[9]))
                cur.lz.append(float(tok[10]))
                continue
            if in_thermo and line.strip() and not _LAMMPS_RUN_LINE.match(line):
                in_thermo = False
    return stages


# ── Atomiverse log parser ─────────────────────────────────────────


def parse_atomiverse(path: Path) -> list[Stage]:
    stages: list[Stage] = []
    cur: Stage | None = None
    with path.open() as f:
        for line in f:
            ls = line.strip()
            if ls.startswith("# Stage"):
                cur = Stage(label=ls.lstrip("# ").strip())
                stages.append(cur)
                continue
            if ls.startswith("#") or not ls:
                continue
            tok = ls.split()
            # Expected 11 columns: step, T, P, PE, KE, Etot, density, vol, lx, ly, lz
            if len(tok) < 11 or cur is None:
                continue
            try:
                cur.step.append(int(tok[0]))
                cur.temp.append(float(tok[1]))
                cur.press.append(float(tok[2]))
                cur.pot_e.append(float(tok[3]))
                cur.kin_e.append(float(tok[4]))
                cur.tot_e.append(float(tok[5]))
                cur.density.append(float(tok[6]))
                cur.vol.append(float(tok[7]))
                cur.lx.append(float(tok[8]))
                cur.ly.append(float(tok[9]))
                cur.lz.append(float(tok[10]))
            except ValueError:
                continue
    return stages


# ── Reporting ─────────────────────────────────────────────────────


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def text_summary(lammps: list[Stage], atv: list[Stage],
                 total_mass_amu: float) -> None:
    n_stages = max(len(lammps), len(atv))
    for i in range(n_stages):
        L = lammps[i] if i < len(lammps) else None
        A = atv[i]    if i < len(atv)    else None
        print(f"\n── Stage {i + 1} ─────────────────────────────────")
        print(f"  LAMMPS   : {L.label if L else '(no log)':32s}"
              f"  -- {len(L.step) if L else 0:5d} samples")
        print(f"  Atomiverse: {A.label if A else '(no log)':31s}"
              f"  -- {len(A.step) if A else 0:5d} samples")

        def row(tag: str, s: Stage | None) -> None:
            if s is None or not s.step:
                print(f"     {tag}  (no samples)")
                return
            vrms = s.v_rms(total_mass_amu)
            print(
                f"     {tag}  "
                f"⟨T⟩={_mean(s.temp):8.2f} K   "
                f"⟨P⟩={_mean(s.press):11.3e} atm   "
                f"⟨ρ⟩={_mean(s.density):7.4f} g/cc   "
                f"⟨PE⟩={_mean(s.pot_e):10.2f} eV   "
                f"⟨KE⟩={_mean(s.kin_e):9.2f} eV   "
                f"⟨vrms⟩={_mean(vrms):.4e} Å/fs"
            )

        row("LAMMPS   ", L)
        row("Atomivse ", A)


# ── Plot ─────────────────────────────────────────────────────────


_OBSERVABLES = [
    # (attr, ylabel, display_name)
    ("temp",    "T [K]",          "Temperature"),
    ("press",   "P [atm]",        "Pressure"),
    ("pot_e",   "PE [eV]",        "Potential energy"),
    ("kin_e",   "KE [eV]",        "Kinetic energy"),
    ("tot_e",   "E_total [eV]",   "Total energy"),
    ("density", "ρ [g/cc]",       "Density"),
]


def plot_overlay(lammps: list[Stage], atv: list[Stage], out: Path,
                 total_mass_amu: float) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[compare] matplotlib not installed — skipping plot.")
        return

    n_stages = max(len(lammps), len(atv))
    # +1 row for v_rms derived from KE.
    n_rows = len(_OBSERVABLES) + 1
    fig, axes = plt.subplots(
        n_rows, n_stages,
        figsize=(6 * n_stages, 2.3 * n_rows),
        sharex="col", squeeze=False,
    )

    for si in range(n_stages):
        L = lammps[si] if si < len(lammps) else None
        A = atv[si]    if si < len(atv)    else None
        col = axes[:, si]
        col[0].set_title(f"Stage {si + 1}")
        for ri, (attr, ylabel, _name) in enumerate(_OBSERVABLES):
            ax = col[ri]
            if L and getattr(L, attr):
                ax.plot(L.step, getattr(L, attr), label="LAMMPS", color="C0")
            if A and getattr(A, attr):
                ax.plot(A.step, getattr(A, attr), label="Atomiverse",
                        color="C1")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
            # Symlog for pressure / energy when spanning many orders.
            if attr in ("press", "pot_e", "kin_e", "tot_e"):
                vals = []
                if L is not None: vals += list(getattr(L, attr))
                if A is not None: vals += list(getattr(A, attr))
                if vals and max(abs(v) for v in vals) > 1e6:
                    ax.set_yscale("symlog", linthresh=1e3)

        # v_rms row (last).
        ax = col[-1]
        if L:
            vrms_L = L.v_rms(total_mass_amu)
            if vrms_L:
                ax.plot(L.step, vrms_L, label="LAMMPS", color="C0")
        if A:
            vrms_A = A.v_rms(total_mass_amu)
            if vrms_A:
                ax.plot(A.step, vrms_A, label="Atomiverse", color="C1")
        ax.set_ylabel("v_rms [Å/fs]")
        ax.set_xlabel("step")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"[compare] wrote {out}")


# ── total mass sniffer ────────────────────────────────────────────


_MASSES_HEADER = re.compile(r"^\s*Masses\s*$", re.IGNORECASE)


def read_total_mass(data_path: Path) -> float:
    """Read `Masses` block + atom count from LAMMPS data file and return
    Σ m_i [amu]. Assumes each atom's line is `id mol type ...` with type
    in column 2 (LAMMPS `atom_style molecular`)."""
    masses: dict[int, float] = {}
    n_atoms = 0

    with data_path.open() as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i]
        s = line.strip()
        if s.endswith("atoms") and s.split()[0].isdigit():
            n_atoms = int(s.split()[0])
        if _MASSES_HEADER.match(line):
            j = i + 1
            while j < len(lines):
                tok = lines[j].split()
                if len(tok) >= 2 and tok[0].isdigit():
                    try:
                        masses[int(tok[0])] = float(tok[1])
                    except ValueError:
                        pass
                    j += 1
                elif not lines[j].strip():
                    j += 1
                else:
                    break
            i = j
            continue
        i += 1

    # Read each atom line to get per-atom type and sum masses.
    # But for single-type systems the short-cut is just N × m_1. Use it
    # when we have exactly one mass to avoid parsing the whole Atoms
    # section (10 k lines).
    if len(masses) == 1 and n_atoms > 0:
        (only_mass,) = masses.values()
        return n_atoms * only_mass

    # Otherwise parse the Atoms section.
    total = 0.0
    for i, line in enumerate(lines):
        if line.strip().lower().startswith("atoms"):
            if line.strip().lower() == "atoms":
                for j in range(i + 1, len(lines)):
                    tok = lines[j].split()
                    if len(tok) >= 3 and tok[0].isdigit():
                        try:
                            total += masses.get(int(tok[2]), 0.0)
                        except ValueError:
                            pass
                    elif not lines[j].strip():
                        if total > 0:
                            break
                break
    return total


# ── main ──────────────────────────────────────────────────────────


def main() -> int:
    here = Path(__file__).parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--lammps",
                    default=str(here / "log.PE_nc10_cl1000.txt"))
    ap.add_argument("--atomiverse",
                    default=str(here / "log.atomiverse.txt"))
    ap.add_argument("--data",
                    default=str(here / "PE_nc10_cl1000.dat"),
                    help="LAMMPS data file (used for total mass)")
    ap.add_argument("--out",
                    default=str(here / "compare.png"))
    args = ap.parse_args()

    total_mass_amu = read_total_mass(Path(args.data))
    print(f"[compare] total mass: {total_mass_amu:.4f} amu")

    lammps = parse_lammps(Path(args.lammps))
    atv    = parse_atomiverse(Path(args.atomiverse))

    print(f"LAMMPS    : {args.lammps}  -> {len(lammps)} stages")
    print(f"Atomiverse: {args.atomiverse}  -> {len(atv)} stages")

    text_summary(lammps, atv, total_mass_amu)
    plot_overlay(lammps, atv, Path(args.out), total_mass_amu)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

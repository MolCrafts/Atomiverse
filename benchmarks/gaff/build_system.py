#!/usr/bin/env python3
"""Build the general-GAFF benchmark liquid (system B) as an Amber prmtop.

The prmtop is the single topology source for every code in the comparison:
OpenMM reads it directly, ``prmtop_to_lammps.py`` derives the LAMMPS data +
coefficient files that LAMMPS and Atomiverse read.

Pipeline (all AmberTools25 + RDKit + OpenMM, seeds fixed):

1. RDKit: SMILES -> 3D (ETKDG + MMFF) -> PDB with unique atom names.
2. ``antechamber -c bcc -at gaff2``: GAFF2 atom types + AM1-BCC charges.
3. ``parmchk2 -s gaff2``: missing parameters.
4. ``packmol``: equimolar mixture in a cubic box at a low start density.
5. ``tleap``: prmtop / inpcrd.
6. OpenMM: minimise, NVT, NPT (1 bar) -> equilibrated coordinates + box.
7. Optional ``--replicate nx ny nz``: tile the equilibrated cell.

The mixture covers what "general GAFF" has to exercise: aromatic and
carbonyl/amide impropers, multi-term proper torsions, X-H bonds for SHAKE.

Usage::

    python build_system.py --out DIR [--sets 540] [--npt-ps 100]
    python build_system.py --out DIR --from DIR0 --replicate 2 2 1

Requires: ``antechamber``, ``parmchk2``, ``tleap``, ``packmol`` on PATH,
python modules ``rdkit``, ``parmed``, ``openmm``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

# name -> (SMILES, 3-letter residue name). Equimolar.
MOLECULES = {
    "toluene": ("Cc1ccccc1", "TOL"),
    "nma": ("CNC(C)=O", "NMA"),
    "ethanol": ("CCO", "ETH"),
    "ec": ("O=C1OCCO1", "ECA"),
}
SEED = 20261020
START_DENSITY = 0.85  # g/cm^3, packmol start (NPT compresses)
AVOGADRO_A3 = 0.602214076  # (g/mol) / (g/cm^3) -> Å^3 conversion factor
TEMPERATURE_K = 300.0


def run(cmd: list[str], cwd: Path, stdin: Path | None = None) -> None:
    """Run one external tool; fail loudly with its output.

    `stdin` is a file, not a pipe: packmol's Fortran runtime seeks on it.
    """
    with open(stdin) if stdin else open("/dev/null") as fin:
        res = subprocess.run(cmd, cwd=cwd, stdin=fin, text=True,
                             capture_output=True)
    if res.returncode != 0:
        sys.stderr.write(res.stdout + res.stderr)
        raise RuntimeError(f"{' '.join(cmd)} failed ({res.returncode})")


def require_tools() -> None:
    missing = [t for t in ("antechamber", "parmchk2", "tleap", "packmol")
               if shutil.which(t) is None]
    if missing:
        raise RuntimeError(f"missing on PATH: {', '.join(missing)}")


def embed_sdf(smiles: str, path: Path) -> float:
    """SMILES -> 3D MDL/SDF (bond orders kept); returns molar mass [g/mol].

    SDF, not PDB: antechamber re-perceives bond orders from a PDB and gets
    amides and carbonates wrong (C=O typed c3/os).
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors

    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    params = AllChem.ETKDGv3()
    params.randomSeed = SEED
    if AllChem.EmbedMolecule(mol, params) != 0:
        raise RuntimeError(f"RDKit embedding failed for {smiles}")
    AllChem.MMFFOptimizeMolecule(mol)
    Chem.MolToMolFile(mol, str(path))
    return Descriptors.MolWt(mol)


def neutralise_mol2(path: Path) -> None:
    """Make a neutral molecule's mol2 charges sum to exactly zero.

    antechamber prints AM1-BCC charges to 6 decimals, leaving a per-molecule
    residual of ~1e-4 e; over 2,160 molecules that is a net -0.54 e cell,
    which PPPM and PME treat with different neutralising-background terms.
    The residual is spread over all atoms in 1e-6 e units (the format's
    resolution), with the integer remainder on the first atoms.
    """
    lines = path.read_text().splitlines()
    start = lines.index("@<TRIPOS>ATOM") + 1
    end = next(i for i in range(start, len(lines))
               if lines[i].startswith("@<TRIPOS>"))
    micro = [round(float(lines[i].split()[8]) * 1_000_000)
             for i in range(start, end)]
    residual = -sum(micro)
    count = len(micro)
    base, extra = divmod(abs(residual), count)
    sign = 1 if residual >= 0 else -1
    for k in range(count):
        micro[k] += sign * (base + (1 if k < extra else 0))
    if sum(micro) != 0:
        raise RuntimeError(f"{path}: charge neutralisation failed")
    for k, i in enumerate(range(start, end)):
        fields = lines[i].split()
        fields[8] = f"{micro[k] / 1_000_000:.6f}"
        lines[i] = (f"{int(fields[0]):7d} {fields[1]:<8s}"
                    f"{float(fields[2]):11.4f}{float(fields[3]):11.4f}"
                    f"{float(fields[4]):11.4f} {fields[5]:<6s}"
                    f"{int(fields[6]):3d} {fields[7]:<8s}{fields[8]:>11s}")
    path.write_text("\n".join(lines) + "\n")


def parametrise(name: str, smiles: str, resname: str, work: Path) -> float:
    mass = embed_sdf(smiles, work / f"{name}_rdkit.sdf")
    run(["antechamber", "-i", f"{name}_rdkit.sdf", "-fi", "mdl",
         "-o", f"{name}.mol2", "-fo", "mol2", "-c", "bcc", "-at", "gaff2",
         "-rn", resname, "-nc", "0", "-pf", "y"], work)
    neutralise_mol2(work / f"{name}.mol2")
    run(["parmchk2", "-i", f"{name}.mol2", "-f", "mol2",
         "-o", f"{name}.frcmod", "-s", "gaff2"], work)
    run(["antechamber", "-i", f"{name}.mol2", "-fi", "mol2",
         "-o", f"{name}.pdb", "-fo", "pdb", "-pf", "y"], work)
    return mass


def pack(work: Path, sets: int, masses: dict[str, float]) -> float:
    total_mass = sets * sum(masses.values())
    length = (total_mass / AVOGADRO_A3 / START_DENSITY) ** (1.0 / 3.0)
    inner = length - 2.0
    lines = ["tolerance 2.0", "filetype pdb", "output packed.pdb",
             f"seed {SEED}", "add_amber_ter"]
    for name in MOLECULES:
        lines += [f"structure {name}.pdb", f"  number {sets}",
                  f"  inside box 1.0 1.0 1.0 {inner:.3f} {inner:.3f} "
                  f"{inner:.3f}", "end structure"]
    (work / "pack.inp").write_text("\n".join(lines) + "\n")
    run(["packmol"], work, stdin=work / "pack.inp")
    return length


def leap(work: Path, length: float) -> None:
    lines = ["source leaprc.gaff2"]
    for name, (_, res) in MOLECULES.items():
        lines += [f"loadamberparams {name}.frcmod",
                  f"{res} = loadmol2 {name}.mol2"]
    lines += ["sys = loadpdb packed.pdb",
              f"set sys box {{{length:.4f} {length:.4f} {length:.4f}}}",
              "saveamberparm sys packed.prmtop packed.inpcrd", "quit"]
    (work / "leap.in").write_text("\n".join(lines) + "\n")
    run(["tleap", "-f", "leap.in"], work)


def equilibrate(work: Path, out: Path, npt_ps: float) -> dict:
    """Minimise + NVT 20 ps + NPT `npt_ps` at 300 K / 1 bar (HBonds, 2 fs)."""
    import openmm as mm
    import openmm.app as app
    import openmm.unit as u
    import parmed

    prmtop = app.AmberPrmtopFile(str(work / "packed.prmtop"))
    inpcrd = app.AmberInpcrdFile(str(work / "packed.inpcrd"))
    system = prmtop.createSystem(nonbondedMethod=app.PME,
                                 nonbondedCutoff=0.9 * u.nanometer,
                                 constraints=app.HBonds)
    integrator = mm.LangevinMiddleIntegrator(TEMPERATURE_K * u.kelvin,
                                             1.0 / u.picosecond,
                                             2.0 * u.femtosecond)
    integrator.setRandomNumberSeed(SEED)
    barostat = mm.MonteCarloBarostat(1.0 * u.bar, TEMPERATURE_K * u.kelvin)
    barostat.setRandomNumberSeed(SEED)
    barostat_index = system.addForce(barostat)
    platform = best_platform()
    sim = app.Simulation(prmtop.topology, system, integrator, platform)
    sim.context.setPositions(inpcrd.positions)
    sim.context.setPeriodicBoxVectors(*inpcrd.boxVectors)
    sim.minimizeEnergy(maxIterations=2000)
    sim.context.setVelocitiesToTemperature(TEMPERATURE_K * u.kelvin, SEED)
    system.getForce(barostat_index).setFrequency(0)  # NVT warm-up
    sim.context.reinitialize(preserveState=True)
    sim.step(10_000)
    system.getForce(barostat_index).setFrequency(25)
    sim.context.reinitialize(preserveState=True)
    sim.step(int(npt_ps * 500))
    state = sim.context.getState(getPositions=True, getEnergy=True,
                                 enforcePeriodicBox=True)
    struct = parmed.load_file(str(work / "packed.prmtop"))
    struct.coordinates = state.getPositions(asNumpy=True).value_in_unit(
        u.angstrom)
    a, b, c = state.getPeriodicBoxVectors()
    lengths = [a[0].value_in_unit(u.angstrom), b[1].value_in_unit(u.angstrom),
               c[2].value_in_unit(u.angstrom)]
    struct.box = lengths + [90.0, 90.0, 90.0]
    struct.save(str(out / "system.prmtop"), overwrite=True)
    struct.save(str(out / "system.inpcrd"), overwrite=True)
    mass = sum(at.mass for at in struct.atoms)
    volume = lengths[0] * lengths[1] * lengths[2]
    return {"atoms": len(struct.atoms), "box_A": lengths,
            "density_g_cm3": mass / AVOGADRO_A3 / volume,
            "equilibration": {"platform": platform.getName(),
                              "nvt_ps": 20.0, "npt_ps": npt_ps}}


def best_platform():
    import openmm as mm
    names = [mm.Platform.getPlatform(i).getName()
             for i in range(mm.Platform.getNumPlatforms())]
    for name in ("CUDA", "OpenCL", "CPU"):
        if name in names:
            return mm.Platform.getPlatformByName(name)
    raise RuntimeError("no OpenMM CUDA/OpenCL/CPU platform")


def replicate(src: Path, out: Path, reps: tuple[int, int, int]) -> dict:
    """Tile an equilibrated cell `reps` times along x, y, z."""
    import numpy as np
    import parmed

    base = parmed.load_file(str(src / "system.prmtop"),
                            xyz=str(src / "system.inpcrd"))
    lx, ly, lz = base.box[:3]
    xyz0 = np.asarray(base.coordinates)
    copies = reps[0] * reps[1] * reps[2]
    tiled = base * copies
    blocks = [xyz0 + np.array([i * lx, j * ly, k * lz])
              for i in range(reps[0]) for j in range(reps[1])
              for k in range(reps[2])]
    tiled.coordinates = np.concatenate(blocks)
    tiled.box = [lx * reps[0], ly * reps[1], lz * reps[2], 90.0, 90.0, 90.0]
    tiled.save(str(out / "system.prmtop"), overwrite=True)
    tiled.save(str(out / "system.inpcrd"), overwrite=True)
    return {"atoms": len(tiled.atoms), "box_A": list(tiled.box[:3]),
            "replicate": list(reps), "source": str(src)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sets", type=int, default=540,
                    help="molecules of each species (540 -> 24,840 atoms)")
    ap.add_argument("--npt-ps", type=float, default=100.0)
    ap.add_argument("--from", dest="src", type=Path,
                    help="equilibrated system dir to replicate")
    ap.add_argument("--replicate", type=int, nargs=3, metavar=("NX", "NY",
                                                                "NZ"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if args.replicate:
        if args.src is None:
            raise SystemExit("--replicate needs --from")
        meta = replicate(args.src, args.out, tuple(args.replicate))
    else:
        require_tools()
        work = args.out / "build"
        work.mkdir(exist_ok=True)
        masses = {name: parametrise(name, smi, res, work)
                  for name, (smi, res) in MOLECULES.items()}
        length = pack(work, args.sets, masses)
        leap(work, length)
        meta = equilibrate(work, args.out, args.npt_ps)
        meta.update({"molecules": {k: v[0] for k, v in MOLECULES.items()},
                     "sets": args.sets, "forcefield": "gaff2 + AM1-BCC",
                     "seed": SEED, "temperature_K": TEMPERATURE_K})
    (args.out / "system.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()

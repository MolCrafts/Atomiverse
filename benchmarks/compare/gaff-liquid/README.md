# General GAFF liquid: Atomiverse vs LAMMPS (CPU) and OpenMM (GPU)

This is a three-code comparison on one organic liquid. One topology source
feeds all three codes, every code runs the same physics, and correctness
gates must pass before any timing counts.

| Code | Entry point | Reads |
|---|---|---|
| Atomiverse (CPU, MPI) | `run_atv_gaff` | `system.data` + `system.ff` |
| LAMMPS | `in.gaff` | `system.data` + `system.ff` |
| OpenMM | `run_openmm.py` | `system.prmtop` + `system.inpcrd` |

There is no Atomiverse CUDA arm yet. It joins `run_gpu_matrix.sh` only after
CUDA GAFF passes G1 against the CPU result.

## System

`build_system.py` builds an equimolar toluene / N-methylacetamide / ethanol /
ethylene carbonate mixture with GAFF2 + AM1-BCC charges. It uses 540 molecules
of each species, 24,840 atoms in total. The pipeline is RDKit → antechamber →
parmchk2 → packmol → tleap, then OpenMM equilibration: minimize, 20 ps NVT,
100 ps NPT at 300 K and 1 bar. The result is a cubic cell of about 65 Å.
Each molecule's charges are made to sum to exactly zero, because PPPM and PME
neutralize a net cell charge differently. The topology has X–H constraints
(RATTLE), Fourier proper torsions and cvff impropers.

Larger cells are tiled copies: `--from DIR --replicate N N N`, with N = 2, 3, 4.
`prmtop_to_lammps.py` (molpy) writes the LAMMPS data and ff files that LAMMPS
and Atomiverse read. It needs a molpy whose prmtop reader handles fixed-width
`%FORMAT(10I8)` fields. Large cells also need that reader, or else use LAMMPS
`replicate` + `write_data nocoeff` on the base cell instead.

## Prerequisites

- AmberTools (`antechamber`, `parmchk2`, `tleap`), `packmol`, and Python with
  `rdkit`, `parmed`, `openmm`, `numpy` and `molcrafts-molpy`.
- LAMMPS with MOLECULE, EXTRA-MOLECULE, KSPACE, RIGID and OPENMP:
  `LAMMPS_SRC=<checkout> [LAMMPS_MODULE=<module>] bash build_lammps.sh`.
- OpenMM with the CUDA platform, for the GPU arm:
  `OMM_ENV=<prefix> [OMM_CONDA_MODULE=<module>] bash setup_openmm.sh`,
  run on the GPU node.

The `*_MODULE` variables are the only cluster-specific inputs. Leave them
unset to use the tools already on `PATH`.

## Build the Atomiverse side

```sh
cmake -B "$BUILD" -DCMAKE_BUILD_TYPE=Release -DATV_BUILD_BENCHMARKS=ON \
      -DATV_USE_MPI=ON -DATV_BUILD_KSPACE=ON
cmake --build "$BUILD" --target run_atv_gaff        # -> $BUILD/bin/run_atv_gaff
```

On a Slurm CPU node, `tools/cpu_job.sh` builds and runs in one job:

```sh
sbatch -A <account> -n 256 --exclusive -t 03:00:00 \
  --export=ALL,ATV_CMAKE_FLAGS="-DATV_BUILD_BENCHMARKS=ON -DATV_USE_MPI=ON -DATV_BUILD_HF=OFF -DATV_BUILD_PW=OFF -DATV_BUILD_GPW=OFF",ATV_TARGETS=run_atv_gaff,ATV_CTEST_RE='^$',ATV_RUN='GAFF_DATA_DIR=<dir> LMP=<lmp> benchmarks/compare/gaff-liquid/run_cpu_matrix.sh' \
  tools/cpu_job.sh
```

## Same-settings protocol (primary arm)

| Knob | LAMMPS `in.gaff` | OpenMM `run_openmm.py` | Atomiverse `run_atv_gaff` |
|---|---|---|---|
| real space | `lj/cut/coul/long 9`, shift no, tail no | cutoff 0.9 nm, no switch, no dispersion correction | `--cutoff 9` |
| 1-4 | `special_bonds amber` (5/6 exact) | prmtop exceptions | `special_bonds amber` from the ff |
| mixing | arithmetic | Lorentz–Berthelot | Lorentz–Berthelot |
| long range | `pppm`, mesh 64³, order 5, g_ewald 0.35, `diff ad` | SPME 64³ (order 5), α 3.5 nm⁻¹ | `KSpacePPPM` α 0.35, `--mesh 64`, order 5 |
| neighbor | skin 2.0, every 1, delay 0, check yes | (internal) | skin 2.0, every 1, delay 0, check |
| constraints | `fix rattle 1e-5 100 0 m 1.008` | HBonds, tolerance 1e-5 | `--shake` (FixRattle, X–H, 1e-5) |
| integrator | `fix nvt ... 100.0 tchain 1`, dt 2 fs | `NoseHooverIntegrator` chain 1, dt 2 fs | VelocityVerlet + single-chain NoseHoover, dt 2 fs |
| output | thermo every 1000 | energy readback every 1000 | — |
| precision | double | double and mixed | double |

LAMMPS uses RATTLE, not `fix shake`. Atomiverse and OpenMM constrain
velocities as well as positions, so timing `fix shake` would compare unequal
work. A secondary arm is allowed: each code's fastest legal settings at the
same measured force error (LAMMPS `diff ik` / `-sf omp`, or OpenMM's own
`ewaldErrorTolerance` mesh).

## Gates

Neither script below enforces a pass/fail threshold. Each prints the
quantities the gate is judged on, and `--json` saves them.

- **G1, single point** (`compare_sp.py`, base cell, kcal/mol). Run
  `in.gaff -var mode sp`, `run_openmm.py --mode sp` and
  `run_atv_gaff --single-point --alpha 0.35 --mesh 64 --forces-out atv.forces`.
  `compare_sp.py` compares the bond, angle, torsion+improper, LJ, Coulomb and
  nonbonded energy components against LAMMPS, plus forces as RMS/max
  difference relative to the reference RMS force. Bonded terms must agree to
  round-off. For nonbonded terms, measure each code's mesh error against a
  converged Ewald reference (`in.gaff -var kstyle ewald`, same α and
  cutoff). All three codes must reach the same force accuracy at these
  settings; that is what makes the primary arm an equal-accuracy comparison.
- **G2, NVE drift** (`drift.py`). Run with X–H RATTLE and a 2 fs step:
  `in.gaff -var ensemble nve`, `run_openmm.py --ensemble nve`,
  `run_atv_gaff --nve --thermo N`. `drift.py` reports the slope in
  k_BT per degree of freedom per ns, with Ndof = 3N − N_constraints − 3.
  Pass means every code's drift lies within the same bound. The gate run is
  100 ps. The first 1 ps is dropped (`--skip-ps`) because LAMMPS
  `velocity create` seeds unconstrained velocities that RATTLE then projects.

## Timing

- **CPU** (`run_cpu_matrix.sh`). Needs `GAFF_DATA_DIR` (system.data and
  system.ff) and `BUILD`. Optional: `LMP`, `CONFIGS` ("ranks x threads"),
  `REPS`, `STEPS`, `LAUNCH`, `OUT`, and the physics knobs in the script
  header. ATV and LAMMPS run paired per configuration and repeat. Threaded
  points give LAMMPS `-sf omp -pk omp N`. `summarize_cpu.py OUT` reports the
  median ATV/LAMMPS wall ratio and the ratio of each LAMMPS timer section.
  ATV timing tables are cumulative, so the timed run is its last table minus
  the warmup table.
- **GPU** (`run_gpu_matrix.sh`). Needs `GAFF_SIZES` (system directories) and
  optionally `PYTHON` (the OpenMM env interpreter), `PRECISIONS`, `REPS`,
  `STEPS`, `OUT`. Each size uses the FFT-friendly mesh nearest
  box / 1.0166 Å, the base cell's mesh spacing.

## Files

| File | Role |
|---|---|
| `build_system.py` | system generator (and `--replicate`) |
| `prmtop_to_lammps.py` | prmtop → LAMMPS data + ff (molpy), numeric types, `special_bonds amber` |
| `in.gaff` | LAMMPS deck: `mode sp\|md`, `ensemble nvt\|nve`, `kstyle pppm\|ewald` |
| `run_openmm.py` | OpenMM single point / timed MD |
| `compare_sp.py`, `drift.py` | gates G1, G2 |
| `run_cpu_matrix.sh`, `summarize_cpu.py` | CPU timing arm |
| `run_gpu_matrix.sh` | GPU timing arm (OpenMM) |
| `build_lammps.sh`, `setup_openmm.sh` | reference-engine setup |

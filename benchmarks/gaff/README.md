# benchmarks/gaff — general GAFF: Atomiverse vs LAMMPS (CPU) and OpenMM (GPU)

Standing harness for the classical GAFF milestone. One topology source, the
same physics in every code, correctness gates before any timing counts.
`tools/bench_gaff_lammps/` (PEO/LiTFSI, "system A") stays the continuity
harness; this directory adds the general-GAFF liquid ("system B").

## System B

`build_system.py` — equimolar toluene / N-methylacetamide / ethanol /
ethylene carbonate, GAFF2 + AM1-BCC (AmberTools25), 540 of each = 24,840
atoms, packmol + 20 ps NVT + 100 ps NPT (OpenMM, 300 K, 1 bar) → 65.06 Å
cubic cell, 0.975 g/cm³. Each molecule's charges sum to exactly zero (the
AM1-BCC 6-decimal residual would otherwise leave the cell at −0.54 e, which
PPPM and PME neutralise differently).

| Topology | Count |
|---|---|
| bonds / angles | 23,760 / 38,880 |
| proper torsions (`dihedral_style fourier`) | 41,580 |
| impropers (`improper_style cvff`) | 4,860 |
| X–H constraints (RATTLE) | 13,500 |

GPU sizes are tiled copies: `--replicate 2 2 2` (198,720), `3 3 3`
(670,680), `4 4 4` (1,589,760). The prmtop is the single source:
OpenMM reads it; `prmtop_to_lammps.py` (molrs) writes the LAMMPS data + ff
that LAMMPS and Atomiverse read (large cells need the molrs fix listed under
*Known issues*; LAMMPS `replicate` + `write_data nocoeff` of the base cell is
the equivalent alternative).

## Same-settings protocol (primary arm)

| Knob | LAMMPS `in.gaff` | OpenMM `run_openmm.py` | Atomiverse `run_atv_gaff` |
|---|---|---|---|
| real space | `lj/cut/coul/long 9`, shift no, tail no | cutoff 0.9 nm, no switch, no dispersion correction | `--cutoff 9` (shift from ff, LAMMPS default off) |
| 1-4 | `special_bonds amber` (5/6 exact) | prmtop exceptions | `special_bonds amber` from the ff |
| mixing | arithmetic | Lorentz–Berthelot | Lorentz–Berthelot |
| long range | `pppm`, mesh 64³, order 5, g_ewald 0.35, `diff ad` | SPME 64³ (order 5), α 3.5 nm⁻¹ | `KSpacePPPM` α 0.35, `--mesh 64`, order 5 |
| neighbor | skin 2.0, every 1, delay 0, check yes | (internal) | skin 2.0, every 1, delay 0, check |
| constraints | `fix rattle 1e-5 100 0 m 1.008` | HBonds, tolerance 1e-5 | `--shake` (FixRattle, X–H, 1e-5) |
| integrator | `fix nvt ... 100.0 tchain 1`, dt 2 fs | `NoseHooverIntegrator` chain 1, dt 2 fs | VelocityVerlet + single-chain NoseHoover, dt 2 fs |
| output | thermo every 1000 | energy readback every 1000 | — |
| precision | double | double and mixed | double |

RATTLE, not `fix shake`, on the LAMMPS side: Atomiverse and OpenMM both
constrain velocities as well as positions, and LAMMPS `fix shake` has no
velocity stage, so timing it would compare unequal work.

Each code's mesh method reaches the same force accuracy at these settings
(gate G1), so the primary arm compares implementations at equal accuracy.
A secondary arm ("each code's fastest legal settings at the same measured
force error") is allowed: LAMMPS `diff ik` / `-sf omp`, OpenMM's own
`ewaldErrorTolerance` mesh.

## Gates

**G1 — single point** (`compare_sp.py`), base cell, kcal/mol:

| Component | ATV − LAMMPS (rel) | OpenMM − LAMMPS (rel) |
|---|---|---|
| bond | 1.0e-14 | 3.5e-15 |
| angle | 6.6e-15 | 9.3e-13 |
| torsion + improper | 9.2e-15 | 3.5e-15 |
| LJ | — | 3.8e-10 |
| nonbonded total | 3.1e-6 | 2.1e-5 |

Mesh error against a converged Ewald reference (`in.gaff -var kstyle
ewald`, same α and real-space cutoff), relative to the RMS force:

| Code | force RMS error | Coulomb energy error |
|---|---|---|
| LAMMPS pppm (ad) | 4.0e-5 | −0.04 kcal/mol |
| OpenMM SPME | 3.3e-5 | +0.95 kcal/mol |
| Atomiverse KSpacePPPM | 4.2e-5 | +0.11 kcal/mol |

**G2 — NVE drift** (`drift.py`): X–H RATTLE, 2 fs, `-var ensemble nve` /
`--ensemble nve` / `--nve --thermo N`. Pass: every code's drift within the
same bound, stated in k_BT per degree of freedom per ns. The first 1 ps is
dropped (`--skip-ps`): LAMMPS `velocity create` seeds unconstrained
velocities that RATTLE projects in the first steps, a ~115 kcal/mol step.

Preliminary, 10 ps on the M4 workbench (the gate run is 100 ps on
Arrhenius); slope standard error is about 7e-3 kT/dof/ns at this length:

| Code | drift (kT/dof/ns) | E_total sd (kcal/mol) |
|---|---|---|
| LAMMPS | +5.2e-2 | 6.3 |
| OpenMM (CPU platform) | −1.5e-2 | 3.9 |
| Atomiverse | −3.5e-3 | 3.0 |

**OpenMM reference on one GH200** (`results/openmm-gh200-baseline.json`,
OpenMM 8.6.1 CUDA, same-settings arm, X–H constraints, 2 fs, NVT, 10,000
timed steps, median of 3; spread < 1 %), ns/day:

| atoms | double | mixed |
|---|---|---|
| 24,840 | 387 | 679 |
| 198,720 | 87.1 | 196 |
| 670,680 | 26.2 | 58.8 |
| 1,589,760 | 9.42 | 19.1 |

**CUDA** — no Atomiverse CUDA timing counts until the cluster-vs-brute-force
force red in `.claude/notes/cuda-openmm-md-perf.md` is green and CUDA GAFF
passes G1 against the CPU result.

## Running on Arrhenius

CPU node (x86, EPYC-9005, 256 cores):

```bash
LAMMPS_SRC=/path/to/lammps bash benchmarks/gaff/build_lammps.sh
sbatch -A <account> -n 256 --exclusive -t 03:00:00 \
  --export=ALL,ATV_CMAKE_FLAGS="-DATV_BUILD_BENCH_GAFF=ON -DATV_BUILD_HF=OFF -DATV_BUILD_PW=OFF -DATV_BUILD_GPW=OFF",ATV_TARGETS=run_atv_gaff,ATV_CTEST_RE='^$',ATV_RUN='GAFF_DATA_DIR=<dir> LMP=<lmp> benchmarks/gaff/run_cpu_matrix.sh' \
  tools/cpu_job.sh
```

GPU node (GH200): `OMM_ENV=<prefix> benchmarks/gaff/setup_openmm.sh` once
(conda-forge OpenMM with CUDA, built on the aarch64 node), then
`GAFF_SIZES="<base> <rep2> <rep3> <rep4>" PYTHON=<prefix>/bin/python
benchmarks/gaff/run_gpu_matrix.sh`.

`run_cpu_matrix.sh` runs ATV and LAMMPS paired per configuration and repeat;
`summarize_cpu.py` reports the median ATV/LAMMPS wall ratio and the ratio of
each LAMMPS timer section (Atomiverse prints the same labels; its tables are
cumulative per driver, so the timed run is the last table minus the warmup).

## Files

| File | Role |
|---|---|
| `build_system.py` | system B generator (and `--replicate`) |
| `prmtop_to_lammps.py` | prmtop → LAMMPS data + ff (molrs), numeric types |
| `in.gaff` | LAMMPS deck: `mode sp|md`, `ensemble nvt|nve`, `kstyle pppm|ewald` |
| `run_openmm.py` | OpenMM single point / timed MD |
| `compare_sp.py` | gate G1 |
| `drift.py` | gate G2 |
| `run_cpu_matrix.sh`, `summarize_cpu.py` | CPU timing arm |
| `run_gpu_matrix.sh` | GPU timing arm (OpenMM; ATV CUDA after its gate) |
| `build_lammps.sh` | LAMMPS with MOLECULE, EXTRA-MOLECULE, KSPACE, RIGID, OPENMP |
| `setup_openmm.sh` | OpenMM + CUDA conda env on the GH200 node |
| `results/` | recorded runs (JSON), one file per arm and machine |

## Known issues

- molrs before the fix in `molrs/src/io/amber/prmtop.rs` (`integer_field_width` /
  `split_integer_fields`, standalone molrs checkout, not yet in the pinned
  submodule) split `%FORMAT(10I8)` sections on whitespace; above ~3×10⁵
  atoms adjacent 8-wide fields touch (`-1000011`) and the read failed. With
  the fixed molrs the 670,680-atom cell converts directly; the 3³ / 4³ data
  in the first runs were made with LAMMPS `replicate` instead.

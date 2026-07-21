# bench_gaff_lammps — GAFF PEO head-to-head vs LAMMPS

Aligned with the paper **flexible PEO** protocol (`figures/flexible` /
`figures/relaxation/ref_lammps`):

| Knob | LAMMPS | Atomiverse |
|------|--------|------------|
| data | `melt.data` (`atom_style full`) | `load_lammps_data` |
| specials | `special_bonds amber` | `set_special_bonds_amber()` (LJ 0/0/0.5, coul 0/0/5/6) |
| pair | `lj/cut/coul/long 10 10` | **`PairLJCutCoulLong`**: multi-type LJ (lj1–4 SIMD) + tabulated coul, one neighbor walk, Amber specials |
| kspace | `kspace_style pppm 1e-4` | **`KSpacePPPM`**: real charge brick + r2c/c2r + SPME influence (B-spline consistent). Science: Madelung / Ewald / FD in `test_pppm_coul` |
| dihedral | `charmm` | cos(nφ) recurrence (no atan2); n=1,2,3 special-cased |
| neighbor | `neighbor 2.0 bin` + `delay 0 every 1 check yes` | `NeighborList{cut, 2.0, 1, 0, true}` |
| thermostat | `fix nvt 400 400 100` | `NoseHoover{400, 100, 1}` |
| dt | 1.0 fs | 1.0 fs |
| timing sections | Pair Bond Kspace Neigh Comm Output Modify Other | **same labels** |

## Build

```bash
export PATH=/software/.../MPICH/.../bin:$PATH
cmake -S . -B build-kspace \
  -DATV_BUILD_CUDA=OFF -DATV_USE_MPI=ON \
  -DATV_BUILD_KSPACE=ON -DATV_BUILD_BENCH_GAFF=ON \
  -DFFTW3_INC=... -DFFTW3_LIB=...
cmake --build build-kspace --target run_atv_gaff -j
```

## Run

```bash
# Default melt: packmol_n100_r2 (17700 atoms) — paper flexible/packmol arm
BUILD=build-kspace NP=1 STEPS=100 tools/bench_gaff_lammps/run.sh
```

Or manually:

```bash
mpirun -np 1 build-kspace/tools/bench_gaff_lammps/run_atv_gaff \
  --data $MELT/melt.data --warmup 10 --steps 100
```

## Snapshot (2026-07-21, np=1, 50 steps, packmol_n100_r2, OMP=1)

### After mesh/α alignment + PME caches (current)

| | ATV | LAMMPS | ATV/LAMMPS |
|--|-----|--------|-----------|
| Loop wall (50 step) | **1.68 s** | 1.01 s | **1.66×** |
| steps/s | 29.8 | 49.4 | 0.60× |
| ns/day (dt=1 fs) | 2.57 | 4.27 | 0.60× |

Earlier baseline (α≈0.30, mesh ~49³): wall **2.43 s** (2.4× vs LAMMPS).  
Optimizations recovered **~31%** wall (2.43 → 1.68).

### What moved the needle

1. **Mesh / α match LAMMPS** — was ~49³ (`rcut/5`), now ~25³ (`rcut/2.5`); LAMMPS uses 24³, g_ewald≈0.196 (we use α≈0.273 first-order heuristic).
2. **Cached influence tables** (no per-step `exp` over the mesh).
3. **`FFTW_MEASURE`** plans.
4. **MPI**: only rank 0 runs FFT+influence+iFFT, then Bcast force grid.
5. **Faster spread/interpolate wrap**; erfc table indexed by **r²** (no `sqrt`).

### Remaining gap (pair still dominates ATV)

ATV pair ≈ LAMMPS pair × ~1.7 (multi-type scalar + fused still not as tight as LAMMPS `lj/cut/coul/long`).  
Kspace gap shrinks a lot once mesh density matches; further wins = true PPPM / r2c FFT / SIMD multi-type LJ.

**Multi-rank (np=4, pre-FFT-rank0):** Kspace was ~95% of wall. With rank-0 FFT + Bcast, multi-rank is usable for correctness; Allreduce of the full charge grid is still the next scaling fix.

## Data path note

`figures/flexible/data/` holds **harvested thermo CSVs** for plots.
Full `melt.data` + `melt.ff` live under the MD job dirs that feed that figure:

`figures/relaxation/ref_lammps/runs/packmol_n100_r2/melt/`

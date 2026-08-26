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

## Snapshot (2026-08-25, MIL0PEGM18 n108 eq.data, 24048 atoms, single node EPYC-9005 256c)

NVT 400 K, lj/cut/coul/long 9.0 + SPME 32³ (α=0.25232811, spacing 1.98),
dt=1 fs, no SHAKE, 5000 steps after 200 warmup, `srun --cpu-bind=cores`.
These numbers are a record, not an acceptance criterion.

| config | steps/s | note |
|---|---|---|
| LAMMPS np=16 / 64 / 128 / 256 | 219 / 766 / 1075 / **1523** | pppm 1e-4 (24³); kspace saturates at 256 (1.57 s / 5000) |
| ATV np=16 / 64 / 125 (scalar kernel) | 127 / 281 / **381** | single-hop halo caps ranks at 5/axis (63 Å box, 11 Å shell) |
| ATV hybrid 125×2 / 64×4 / 32×8 | 337 / 338 / 318 | pure MPI wins even at half the cores — per-thread force-buffer zero+reduce and unthreaded kspace/comm eat the gain |

Section splits at np=64 (s / 5000 steps): ATV Pair 6.0, Kspace 9.4 (≈1.3 ms
of it is the density Allreduce *wait* behind pair imbalance; fft 0.10,
spread 0.04, interp 0.09 per step), Comm 2.4, Neigh 0.9 — LAMMPS 3.8 /
1.3 / 0.5 / 0.45. Kernel history: staged-SIMD ≈ scalar with restored
min-image (±3%); the min-image restore fixed a slow thermal runaway from
silently dropped wrapped pairs (VelocityVerlet wraps every step; LAMMPS
wraps only at reneighbor — a future integrator-side slug).

## Snapshot (2026-08-26, MIL0PEGM18 n108, 24048 atoms, multi-hop 256 ranks, single node EPYC-9005 256c)

First full-node run unlocked by the swap-list multi-hop halo
(comm-topology chain): 24048 atoms, box 63.0016 Å cubic, processor grid
**8×8×4** ⇒ subdomain span 7.875×7.875×15.750 Å vs halo shell
cutoff+skin = 11.0 Å (hops 2,2,1 — x/y ghosts cross two subdomains).
Same protocol as the 2026-08-25 block: NVT 400 K, lj/cut/coul/long 9.0 +
SPME 32³ (α=0.25232811, spacing 1.98), dt=1 fs, no SHAKE, 5000 steps
after 200 warmup, OMP=1. These numbers are a record, not an acceptance
criterion.

| config | steps/s | note |
|---|---|---|
| ATV np=64 / 125 / 200 / **256** | 393 / 419 / 384 / **343** | multi-hop; peak at 125, kspace Allreduce turns over beyond |
| LAMMPS np=256 (same node, double) | **1523** | pppm 1e-4 (24³); Loop 3.28 s / 5000 steps |

ATV MPI task timing breakdown at np=256 (s / 5000 steps; LAMMPS same-run
reference in parentheses):

| Section | min | avg | max | %total |
|---|---|---|---|---|
| Pair | 0.8378 | 1.1005 | 1.4336 | 7.2 (LMP 0.936, 28.5%) |
| Bond | 0.1124 | 0.1439 | 0.1757 | 0.9 (LMP 0.104, 3.2%) |
| Kspace | 9.4300 | 9.8740 | 10.2181 | 64.9 (LMP 1.563, 47.6%) |
| Neigh | 0.3627 | 0.4364 | 0.5419 | 2.9 (LMP 0.145, 4.4%) |
| Comm | 3.4121 | 3.4989 | 3.6144 | 23.0 (LMP 0.441, 13.4%) |
| Output | 0.0000 | 0.0000 | 0.0000 | 0.0 |
| Modify | 0.1277 | 0.1498 | 0.1642 | 1.0 |
| Other | 0.0068 | 0.0086 | 0.0104 | 0.1 |
| Loop | 15.2125 | 15.2193 | 15.2234 | 100.0 |

Pair is now *ahead* of LAMMPS per avg-time (1.10 vs 0.94 with 30% varavg
imbalance); the whole remaining gap is **Kspace (65%: every-rank 32³
replicated FFT + density Allreduce wait behind pair imbalance) and the
Comm it drags** — the next lever is a kspace subcommunicator
(brick-alltoallv to an FFT subgroup), not the pair kernel.

Correctness at this geometry: `widebox_parity.sh` np=256 (8,8,4) vs np=1
single point — bond/angle/dihedral/pair+kspace/E/fmax/fnorm all
dev=0.0 at every printed digit (job 1692800, logs
`atv-mil-bench/parity256/`).

Exact command (dataset external, env-passed, never committed):

```bash
DATA=/home/jicli594/work/salamander/MIL0PEGM18/aMIL0PEGM18_n20_LiTFSI_n108/eq2/eq.data
srun --mpi=pmi2 --ntasks=256 --cpus-per-task=1 --cpu-bind=cores \
  --distribution=block:block run_atv_gaff --data $DATA --ff system.ff \
  --alpha 0.25232811 --grid-spacing 1.98 --warmup 200 --steps 5000
# parity: NP=256 DIMS=8,8,4 tools/bench_gaff_lammps/widebox_parity.sh \
#   --alpha 0.25232811 --grid-spacing 1.98
```

## Snapshot (2026-08-26, kspace subcommunicator — three-arm matrix, same frozen binary)

`kspace-subcomm` landed the PPPM mesh-selector third arm: the FFTW-MPI slab
FFT runs on a derived subgroup of `pppm_fft_ranks(K0=32, np)` ranks instead
of the world communicator. One frozen binary (`run_atv_gaff.subcomm`), same
protocol as the blocks above (MIL0PEGM18 24048 atoms, NVT 400 K, SPME 32³
α=0.25232811 spacing 1.98, 5000 steps + 200 warmup, `srun
--cpu-bind=cores`, OMP=1). Arms forced per run: subgroup = default,
world-brick = `ATV_PPPM_BRICK=1`, replicated = `ATV_PPPM_ALLREDUCE=1`.
These numbers are a record, not an acceptance criterion.

steps/s (Kspace avg s / 5000 steps in parentheses):

| np | subgroup (default) | world brick | replicated |
|---|---|---|---|
| 64  | **445** (4.02) | 419 (4.77) | 384 (5.76) |
| 125 | **555** (3.74) | 453 (5.94) | 423 (6.78) |
| 200 | **605** (3.81) | 349 (9.82) | 390 (8.58) |
| 256 | **586** (3.72) | 339 (9.89) | 342 (9.91) |

The subgroup arm wins every cell; the two old arms hit the same ~9.9 s wall
at np≥200 (32 slab planes dragged through a world-sized FFT collective vs a
world-sized grid Allreduce). With the FFT pinned to 32 ranks, Kspace is
flat (~3.7–4.0 s) in world size; the machine peak moved 419 → **605**
steps/s (np=200) and np=256 went 343 → **586** (+71%). Same-node LAMMPS
np=256 reference stays 1523 steps/s (Kspace 1.56, Comm 0.44): the remaining
2.6× gap is Kspace 3.72 (Alltoallv remap over all ranks with mostly-zero
counts + density-wait behind pair imbalance) and Comm 3.54 — the remap
collective shape (reduce_scatter / neighborhood) is the registered next
spec.

Correctness at 256: `widebox_parity.sh` with the subgroup arm as default —
every window dev=0.0 vs np=1 (job 1693014, logs
`atv-mil-bench/parity256sub/`).

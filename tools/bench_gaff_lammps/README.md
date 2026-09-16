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

## Snapshot (2026-08-26, lammps-comm-port — CommBrick per-step discipline)

Port of LAMMPS `CommBrick` step-path discipline onto the CPU halo: no
collectives on the refresh path, `kColComm` is x/y/z (24 B/ghost), recv
buffers size-set without memset. Same protocol as the kspace-subcomm
block (MIL0PEGM18 24048 atoms, NVT 400 K, SPME 32³ α=0.25232811 spacing
1.98, 5000 steps + 200 warmup, `srun --cpu-bind=cores`, OMP=1, grid
8×8×4). These numbers are a record, **not** an acceptance criterion.

| config | steps/s | note |
|---|---|---|
| ATV np=256 (job 1702467) | **662** | wall 7.56 s / 5000 prod |
| LAMMPS np=256 (same node, double) | **1523** | Loop 3.28 s / 5000; Comm 0.441 s |

ATV MPI task timing at np=256 (s / 5200 launches = 200 warmup + 5000 prod;
LAMMPS Comm 0.441 s is the 5000-step reference):

| Section | min | avg | max | %total |
|---|---|---|---|---|
| Pair | 0.7838 | 1.0334 | 1.3738 | 12.9 |
| Bond | 0.1006 | 0.1346 | 0.1619 | 1.7 |
| Kspace | 3.1122 | 3.6573 | 3.9590 | 45.6 |
| Neigh | 0.3617 | 0.4344 | 0.5652 | 5.4 |
| Comm | 2.5262 | **2.6144** | 2.6927 | 32.6 |
| Modify | 0.1155 | 0.1278 | 0.1477 | 1.6 |
| Other | 0.0061 | 0.0070 | 0.0078 | 0.1 |
| Loop | 8.0090 | 8.0157 | 8.0213 | 100.0 |

Comm avg moved 3.50 → 2.61 s on the 5200-launch dump (halo_send 1.62,
halo_recv 0.72, migrate 0.42, reduce 0.047). No performance threshold is
asserted. Next registered lever remains kspace remap collective shape.

Correctness: `widebox_parity.sh` NP=256 DIMS=8,8,4 — every window
dev=0.0 vs np=1, hops=2,2,1 (job 1702467).

## Snapshot (2026-08-27, wt10 — multi-hop slab band + exchange collectives)

Same protocol as every block above (MIL0PEGM18 24048 atoms, NVT 400 K,
lj/cut/coul/long 9.0 + PPPM, dt = 1 fs, 5000 steps after 200 warmup,
`srun --cpu-bind=cores`, OMP=1, grid 8×8×4, hops 2,2,1). Best config
`--grid-spacing 2.63` (24³ mesh) `--fft-ranks 8`. Job 1734723; the LAMMPS
reference is the same job on the same node. These numbers are a record,
**not** an acceptance criterion.

Four changes, all in `cpu/communication/`:

1. **Forwarding rounds carry the LAMMPS `ineed >= 2` slab band again**
   (`halo.cpp`, `Halo::expand`). It had been switched off behind a
   `&& false` bisect flag, leaving hop ≥ 2 with an open band, so every
   forwarding round shipped the whole slab it had received instead of the
   part still inside the receiver's shell. The surplus compounds per hop:
   the per-axis ghost span was 39.4 Å where the shell only reaches
   29.9 Å, i.e. **1.76× the ghost rows** over three axes.
2. **`exchange` entry collectives 4 → 1.** The wire schema is anchored
   collectively on the first migration and compared locally after it (the
   discipline the halo plans already used); the "does anyone have an atom
   to home" flag and the global max native row width now ride one
   `MPI_Allreduce(MPI_MAX)` over a 2-element array.
3. **`exchange_along_dim` round shape.** The first round is unconditional
   (LAMMPS runs exactly one per dim and never asks the world); the
   Allreduce now only asks whether a *received* row is still out of slab,
   which is false on the common path, so a dim usually costs one
   reduction instead of two. Count and payload exchanges are nonblocking
   pairs rather than two serialized `MPI_Sendrecv`s.
4. **Slab test inclusive on both edges** (`>=`/`<=`), as
   `CommBrick::borders`. It had been strict; with (1) restoring a second
   *closed* edge at `sublo + cutoff` that became a real lost-pair risk.

| config | steps/s | note |
|---|---|---|
| ATV np=256 wt9 | 1307 | open forwarding band |
| ATV np=256 **wt10** | **1444 / 1448 / 1447** | 3 reps, wall 3.45 s / 5000 |
| LAMMPS np=256 (same node, same job) | **1521** | Loop 3.29 s / 5000 |

ATV is at **95%** of LAMMPS (was 86%). Section table (s / 5200 launches
for ATV, / 5000 steps for LAMMPS; both `avg` over ranks):

| Section | ATV wt9 | ATV wt10 | LAMMPS | wt10 / LMP |
|---|---|---|---|---|
| Pair | 0.958 | 0.951 | 0.931 | 1.02 |
| Bond | 0.124 | 0.122 | 0.103 | 1.18 |
| Kspace | 1.419 | 1.408 | 1.574 | **0.89** |
| Neigh | 0.126 | 0.106 | 0.144 | **0.74** |
| Comm | 1.142 | **0.842** | 0.438 | **1.92** |
| Modify | 0.131 | 0.120 | 0.062 | 1.94 |
| Loop | 4.058 | **3.656** | 3.287 | 1.11 |

Comm sub-ops (max over ranks, s / 5200 launches):

| op | calls | wt9 | wt10 |
|---|---|---|---|
| comm_forward | 4969 | 0.372 | 0.320 |
| comm_reverse | 5200 | 0.404 | 0.314 |
| comm_borders | 231 | 0.228 | 0.134 |
| comm_exchange | 231 | 0.201 | 0.122 |
| reduce | 5200 | 0.047 | 0.045 |

The ghost count is now reported directly in the load-balance table
(`nghost_min/avg/max` + `ghost/owned`), because that is where a slab-band
or hop-count regression shows up long before it shows up as Comm seconds.
It is also the sharpest available check that the port is faithful, and it
now passes: **ATV nghost_avg 3151.3 against LAMMPS `Nghost: 3147.24 ave`
on the same run** — 0.13% apart, the residue of the inclusive band edge
and of `ceil` vs LAMMPS' `int()+1` in the hop count. Before the band fix
ATV carried ~5530. Comm buffer allocation fell 0.200 → 0.112 GB over 256
ranks, the same 44%.

### Where the remaining Comm goes

Same binary plus per-phase `ATV_PROFILE_SCOPE`s in the halo paths (job
1734801, 1453 / 1460 steps/s — the scopes cost nothing measurable).
Rank-0 seconds over 5200 launches, 5 rounds per call:

| phase | s | phase | s |
|---|---|---|---|
| halo/fwd_pack (+`Isend` posting) | 0.090 | halo/rev_pack (+posting) | 0.061 |
| **halo/fwd_wait** | **0.188** | **halo/rev_wait** | **0.200** |
| halo/fwd_unpack | 0.008 | halo/rev_accum | 0.026 |

`MPI_Waitall` is **~63%** of forward + reverse. Unpack is negligible
(0.008 s), so receive-into-`x` — LAMMPS' `comm_x_only` trick, which ATV's
SoA Frame cannot do in one message anyway — would buy nothing here. In
`borders` (0.134 s total) the biggest item is brd_unpack 0.053 and the
ghost tag-map teardown is **0.000007 s**: `exchange` clears `ghost_tags`
immediately before, so that loop is a no-op on every migrating step. A
LAMMPS-style direct-index `map_array` would therefore buy nothing either.

So the remaining 1.92× is not volume (ghost counts match), not imbalance
(spreads match), not pack/unpack, and not the tag map. It is the
round-trip cost of the serialized dependency rounds themselves, and
ATV already posts *fewer* MPI calls per step than LAMMPS (26 vs 30 on the
forward path). Closing it needs a measurement the current instrumentation
cannot give: these scopes are **rank 0 only**, and rank 0 is one of the 8
`--fft-ranks` ranks, so it is a systematically *late* rank that waits less
than average. The next step is per-rank sub-phase statistics, or a
barrier-bracketed run, to separate transport from residual skew — not
another guess at the pack path.

Note the section bookkeeping is not identical to LAMMPS': ATV charges
`zero_ghost_forces` (ghost force rows) to Comm, where LAMMPS clears the
whole force array in `Verlet::force_clear` and charges it to Other. That
is worth ~0.02 s of ATV's Comm.

Correctness: `widebox_parity.sh` at NP=256, every window dev=0.0 vs np=1,
on three grids — 8,8,4 at 32³ and at 24³ (hops 2,2,1) and **16,4,4**
(3.94 Å subdomain vs 11 Å shell ⇒ **hops 3,1,1**). The 3-hop arm is the
one that matters for change (1): it is the only grid here where a
forwarding round has to hand its rows on to yet another hop, so an
over-tight depth clip would drop pairs there first. 237/237 unit + MPI
tests green.

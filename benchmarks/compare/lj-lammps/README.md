# LJ / EAM / SW+LJ: Atomiverse vs LAMMPS

Two independent head-to-heads against LAMMPS:

| Arm | Atomiverse | LAMMPS | Driver |
|---|---|---|---|
| CPU oracle + timing | `bench_lj_cpu` (MD, `PairLJCut` / EAM / SW+LJ) | `in.cpu` on the same initial state | `compare.py` |
| CUDA throughput | `bench_nonbonded_style` (cluster/tile LJ) | `in.lj_nve.kk`, KOKKOS CUDA | `run_compare_cuda.py` |

No results are kept here. A performance claim is the pair of numbers both
binaries print on the same idle node, recorded with compiler flags, CPU/GPU
model, LAMMPS version, MPI/OpenMP versions and thread binding.

## Prerequisites

- A LAMMPS binary. CPU arm: default styles, plus MANYBODY for `eam`/`sw-lj`
  and OPENMP for threaded points. CUDA arm: a KOKKOS CUDA build, with
  `KOKKOS_PREC` matching the Atomiverse precision mode.
- Python 3 (standard library only).
- `eam` and `sw-lj` need molpy in the Atomiverse build (`ATV_BUILD_MOLPY`,
  derived by the default product set; set `ATV_MOLPY_PATH`).

## Build the Atomiverse side

```sh
# CPU (MPI is optional and only needed for multi-rank points)
cmake -B "$BUILD" -DCMAKE_BUILD_TYPE=Release -DATV_BUILD_BENCHMARKS=ON -DATV_USE_MPI=ON
cmake --build "$BUILD" --target bench_lj_cpu

# CUDA; one build tree per precision mode
cmake -B "$BUILD" -DCMAKE_BUILD_TYPE=Release -DATV_BUILD_BENCHMARKS=ON \
      -DATV_BUILD_CUDA=ON -DATV_CUDA_PREC_MODE=double   # or mixed | single
cmake --build "$BUILD" --target bench_nonbonded_style
```

Binaries land in `$BUILD/bin/`.

## CPU arm

`bench_lj_cpu NSIDE WARM STEPS THREADS PREFIX [nested] [lj|eam|sw-lj] [--tilt XY XZ YZ]`
writes `PREFIX.data` (plus `.eam` / `.sw` tables) and owner snapshots, then
runs. `in.cpu` reads that same state, so both engines start from identical
positions and velocities. Use a fresh `PREFIX` per run.

```sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
P="$OUT/lj-r1-t1-a"
mpiexec -n 1 "$BUILD/bin/bench_lj_cpu" 22 20 200 1 "$P" > "$P.atv.log" 2>&1
mpiexec -n 1 "$LMP" -in in.cpu -var PREFIX "$P" -var WARM 20 -var STEPS 200 \
    -log "$P.lmp.log" -screen none          # add -var MODEL eam|swlj for many-body
python3 compare.py "$P"                     # --stage final for the end state
```

For one thread, use the plain LAMMPS style. For more threads, add
`-sf omp -pk omp THREADS`.

Systems:
- `lj`: distorted simple cubic lattice at 0.8 Å⁻³. ε = 1 eV, σ = 1 Å,
  m = 1 amu, cutoff 2.5 Å (shifted), dt = 0.005·√103.6427 fs (reduced
  0.005). Sides 22 and 47 give 10,648 and 103,823 atoms.
- `eam`: FCC Cu with Sutton–Chen, a = 3.61 Å, cutoff 5.5 Å, dt = 1 fs.
  Side 14 gives 10,976 atoms. LAMMPS reads the same functions as `eam/alloy`
  setfl tables, so interpolation error is part of what the gate bounds.
- `sw-lj`: diamond Si with Stillinger–Weber plus a shifted LJ overlay
  (ε = 0.01 eV, σ = 2 Å, cutoff 3.77118 Å), a = 5.43 Å, dt = 0.5 fs. Side
  11 gives 10,648 atoms. It exercises Full and Half neighbor consumers
  together; it is not a fitted model.

All models use skin 0.3 Å, `every 1 delay 0 check yes`, Newton on, sorting
every 1000 steps, and energy plus full virial every step. LJ starts at reduced
T ≈ 1; EAM and SW+LJ start from 300 K Maxwell velocities. Centre-of-mass
motion is removed in all three.

Timing protocol:
- Matrix: `(ranks, threads)` ∈ (1,1) (1,2) (1,4) (2,1) (4,1), with at least
  three repeats. Run the engines one after the other, never concurrently.
- ATV's final `BENCH` line is the maximum over ranks of the summed step-loop
  wall time. Compare it with the last LAMMPS `Loop time`, the measured run.
- With `pre no`, LAMMPS keeps counting neighbor builds across warmup. Subtract
  the warmup count before comparing with ATV's count.
- Never compare force-only LAMMPS throughput with ATV's
  force+energy+virial loop.

Modes: `nested` runs the same MD inside a generic outer Driver and checks
correctness only (prints `CHECK`). `--tilt` builds a triclinic cell; the
data file carries the same `xy xz yz`.

## CPU acceptance (`compare.py`)

| Check | Tolerance |
|---|---|
| `--stage initial`, max force-component error | ≤ 2e-10 eV/Å |
| `--stage initial`, total-energy error per atom | ≤ 2e-10 eV/atom |
| one-step check (`NSIDE=6 WARM=0 STEPS=1`, `--stage final`) | `--force-atol 1e-8`; energy stays 2e-10 |

`compare.py` also rejects non-finite values, missing owners and duplicate
tags. Run the short check at 1–4 ranks and 1–2 threads, orthorhombic,
`--tilt`, and `nested`, before you take any timings. The looser one-step force
window comes from the mass/velocity constant: ATV uses 103.6427 and LAMMPS
metal implies 103.64269. **Do not change masses to hide it.** Longer
`--stage final` comparisons need an explicitly stated tolerance; a chaotic
trajectory is not a force oracle.

## CUDA arm

```sh
# single point by hand
"$BUILD/bin/bench_nonbonded_style" 10000 80 15     # N STEPS WARM
"$LMP" -k on g 1 -sf kk -in in.lj_nve.kk -var N 10648 -var WARM 15 -var STEPS 80

# size sweep, both engines, on one GPU
python3 run_compare_cuda.py --atv-bench "$BUILD/bin/bench_nonbonded_style" \
    --lmp-bin "$LMP" --out "$OUT" --tag double [--lmp-neigh full --lmp-newton off]
python3 run_compare_cuda.py --lmp-bin "$LMP" --out "$OUT" --sweep-lmp   # LAMMPS package grid only
```

Both sides use an LJ liquid in reduced units: ρ = 0.8, cutoff 2.5, skin 0.3,
dt = 0.005, T = 1. The side is `ceil(N^(1/3))`. ATV prints
`path=... N=... ms/step=... Matom*step/s=...`, and LAMMPS is read from its
last `Loop time`. Only compare equal precision modes. There is no correctness
gate in this arm.

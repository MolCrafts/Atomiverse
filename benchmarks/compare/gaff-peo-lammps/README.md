# GAFF PEO melt: Atomiverse (CPU, MPI) vs LAMMPS

`run_atv_gaff` runs MPI-decomposed GAFF MD with PPPM on a PEO / PEO-salt melt
and is timed against LAMMPS on the same node and the same data file. Before any
multi-rank timing, `widebox_parity.sh` checks that a decomposed run
reproduces the single-rank result.

The melt (LAMMPS `atom_style full` data + coefficient file) is external and
never committed. Pass it through `DATA` / `FF`. No LAMMPS deck ships here:
use one that sets exactly the knobs below.

## Prerequisites

- MPI and FFTW3 (FFTW-MPI enables the PPPM FFT subgroup).
- A LAMMPS binary with MOLECULE and KSPACE, run on the same node.
- `gawk`, for `widebox_parity.sh`.

## Build the Atomiverse side

```sh
cmake -B "$BUILD" -DCMAKE_BUILD_TYPE=Release -DATV_BUILD_BENCHMARKS=ON \
      -DATV_USE_MPI=ON -DATV_BUILD_KSPACE=ON
cmake --build "$BUILD" --target run_atv_gaff     # -> $BUILD/bin/run_atv_gaff
```

## Matched protocol

| Knob | LAMMPS | `run_atv_gaff` |
|---|---|---|
| data | `units real`, `atom_style full`, `read_data` | `--data DATA --ff FF` |
| 1-4 | `special_bonds amber` | Amber default (LJ 0/0/½, Coulomb 0/0/⅚); a `special_bonds` line in the ff overrides |
| pair | `lj/cut/coul/long RC`, `pair_modify mix arithmetic tail no` | `--cutoff RC` (fused `PairLJCutCoulLong`); a `pair_style` cutoff in the ff overrides |
| long range | `kspace_style pppm 1e-4` | `--accuracy 1e-4` (or the ff's `kspace_style`), or pin `--alpha` / `--grid-spacing` / `--mesh` |
| neighbor | `neighbor 2.0 bin`, `neigh_modify delay 0 every 1 check yes` | same skin and cadence |
| ensemble | `fix nvt temp 400 400 100`, dt 1 fs | `--T 400 --tau-T 100 --dt 1` (single-chain NoseHoover) |
| constraints | none | none |
| timing | `timer full`, 200 warmup + 5000 steps | `--warmup 200 --steps 5000` |

For equal work, PPPM needs the same α and mesh on both sides. The accuracy
target alone does not ensure that, because each code picks its own α from
it. Pin both (LAMMPS `kspace_modify gewald α mesh K K K`; ATV `--alpha α`
plus `--mesh K` or `--grid-spacing`), or report the meshes each code chose.

```sh
export OMP_NUM_THREADS=1
srun --mpi=pmi2 --ntasks="$NP" --cpus-per-task=1 --cpu-bind=cores \
  "$BUILD/bin/run_atv_gaff" --data "$DATA" --ff "$FF" --cutoff 9.0 \
  --alpha 0.25232811 --grid-spacing 1.98 --warmup 200 --steps 5000
```

Diagnostic switches (no physics change):
- `--fft-ranks N` sets the size of the PPPM FFT subgroup.
- `ATV_PPPM_BRICK=1` and `ATV_PPPM_ALLREDUCE=1` force the world-brick and
  replicated PPPM arms.
- `ATV_FORCE_DIMS=x,y,z` forces the processor grid.

Reading the timings:
- ATV prints the LAMMPS section labels (Pair, Bond, Kspace, Neigh, Comm,
  Output, Modify, Other).
- Its tables are cumulative over the driver, including warmup, so subtract
  the warmup table. `../gaff-liquid/summarize_cpu.py` does this for logs
  named `{atv,lmp}_np<R>_nt<T>_r<K>.log`.
- ATV charges ghost-force zeroing to Comm; LAMMPS charges `force_clear` to
  Other.
- The ATV load-balance table's `nghost_avg` should match LAMMPS's `Nghost`
  average for the same grid. A gap there means a halo-volume difference,
  not a kernel difference.

## Acceptance: `widebox_parity.sh`

This script runs `run_atv_gaff --single-point` twice with identical physics
flags: first at np = 1, then at np = `NP` with `ATV_FORCE_DIMS=DIMS`. It
compares every published quantity. The reference is ATV's own np = 1 result,
so this checks cross-rank self-consistency; it is not an external oracle.

| Quantity | Window (strict `<`, relative to max(1, \|ref\|)) | Flag |
|---|---|---|
| bond, angle, dihedral energy | 1e-9 | `--tol-bonded` |
| pair+kspace energy | 1e-8 | `--tol-pair` |
| total energy | 1e-8, and \|ΔE\|/N < 1e-6 eV/atom | `--tol-total` |
| fmax, fnorm | 1e-6 | `--tol-force` |

For NP > 1 the script also asserts two geometry conditions:
- The reported grid equals `DIMS`.
- At least one `hops` component is ≥ 2, i.e. a genuine multi-hop halo. A
  single-hop PASS is the dangerous false green.

`--tol-total 0` must FAIL, which proves the gate bites. All other arguments
pass through to both runs unchanged.

```sh
BUILD=... DATA=... FF=... OUT=<log dir> NP=256 DIMS=8,8,4 \
  ./widebox_parity.sh --cutoff 9.0 --alpha 0.25232811 --grid-spacing 1.98
```

Required: `BUILD`, `DATA`, `FF`, `OUT`. Optional: `NP` [256], `DIMS`
[8,8,4], `LAUNCHER` [`srun --mpi=pmi2 --cpu-bind=cores` if `srun` exists,
else `mpirun`]. The exit status is 0 only when every component passes.

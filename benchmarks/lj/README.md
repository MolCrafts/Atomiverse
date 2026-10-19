# LJ 12-6 standing benchmark

Not registered in ctest. Build with
`-DATV_BUILD_BENCHMARKS=ON -DATV_BUILD_CUDA=ON`; the binary lands in
`<build>/bin` (see `benchmarks/cuda/CMakeLists.txt`).

## CUDA tile engine (GH200)

```
# on GH200 after a CUDA build
./bin/bench_nonbonded_style 10000 80 15
```

Recorded GH200 job **1332540** (n68, `GPU/buildenv-gcccuda/2026.03-cu13.0`):

```
path=openmm_cluster n_side=22 N=10648 steps=80  ms/step=0.1098  Matom*step/s=96.997  energy=-39062.893592
```

Job **1331961** (same node class): `96.740 Matom*step/s`.

## LAMMPS twin

Deck: `tools/bench_lj_vs_lammps/in.lj_nve.kk` (same density / cut / dt).
A perf pin is the number `lmp` prints — never a reimplementation.

```
lmp -in tools/bench_lj_vs_lammps/in.lj_nve.kk -var N 10648 -var WARM 15 -var STEPS 80
```

Record binary version + the `Loop time` / `Performance` line next to the
ATV `Matom*step/s` line. Keep the roadmap ⚡ only when that same-node ratio
supports it.

## CPU xsimd

The login-node CPU build of `PairLJCut` is the xsimd path (`-march=native`).
A same-size CPU wall is optional supporting evidence; the ⚡ claim is the
CUDA tile engine vs LAMMPS Kokkos on GH200.

## Reproducible CPU comparison

Build with `-DATV_BUILD_BENCHMARKS=ON -DATV_BUILD_MD=ON
-DATV_BUILD_CUDA=OFF`, then build target
`bench_lj_cpu`. This harness uses public `MD`, `PairLJCut`,
`NeighborList`, and the existing `timing_report()`/`reset_timing()` API.
It is not a ctest and does not reuse the historical CUDA timing numbers.

Both engines consume **exactly the same positions and velocities**: ATV writes
a LAMMPS data file before running. Coordinates are a slightly distorted simple
cubic lattice at number density 0.8 Angstrom^-3. The deterministic thermal
velocities have reduced temperature approximately 1, with centre-of-mass motion
removed. Mass = 1 amu, epsilon = 1 eV, sigma = 1 Angstrom, cutoff = 2.5
Angstrom, skin = 0.3 Angstrom, shifted energy, Newton on, periodic box.
The timestep is `0.005*sqrt(103.6427)` fs (reduced dt = 0.005), converted to ps
for LAMMPS `units metal`. Both use `every 1 delay 0 check yes` and spatial
sorting during setup and every 1000 steps. The 20 warmup + 200 measurement steps cause actual
skin-triggered neighbor rebuilds at this temperature.

ATV initializes forces through native `MD.run(frame, 0)`. Both initial and final
snapshots contain owners only and are joined by tag by `compare.py`.

Append `nested` to the ATV command to exercise native nested MD inside a generic
outer workflow. This mode is for correctness checks only: a zero-step child run
initializes the initial snapshot, then a single child run executes `WARM + STEPS`
continuous steps. It prints `CHECK` rather than performance measurements. For a
short MPI check use `NSIDE=6`, `WARM=0`, `STEPS=1`, repeat with 1/2/3/4 ranks,
and compare both `--stage initial` and `--stage final` against the same LAMMPS
deck (allow `--force-atol 1e-8` for the one-step integration constant difference).

Append `--tilt XY XZ YZ` to generate a triclinic cell, with absolute tilts
in Angstrom. The harness affinely maps the initial positions through the cell
and writes the same `xy xz yz` geometry in the LAMMPS data file. For example,
`6 0 1 2 /tmp/tilted-lj lj --tilt 1.7 -0.8 0.6` exercises a short two-thread
LJ run; the same option composes with `nested`, `eam` and `sw-lj`.

Example (replace binary paths; use a **fresh prefix** for each rank/thread/repeat):

```sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mpiexec -n 1 build/bin/bench_lj_cpu 22 20 200 1 /tmp/lj-r1-t1-a > /tmp/lj-r1-t1-a.atv.log 2>&1
mpiexec -n 1 /path/to/lmp \
  -in benchmarks/lj/in.cpu -var PREFIX /tmp/lj-r1-t1-a -var WARM 20 -var STEPS 200 \
  -log /tmp/lj-r1-t1-a.lmp.log -screen none
python3 benchmarks/lj/compare.py /tmp/lj-r1-t1-a
```

For one thread use the ordinary CPU `lj/cut` style, as above; for multiple
threads add `-sf omp -pk omp THREADS` to LAMMPS. Do not force a potentially
slower one-thread OPENMP variant as the CPU baseline.

Use sides 22 (10,648 atoms) and 47 (103,823 atoms), with `(ranks, threads)`
`(1,1), (1,2), (1,4), (2,1), (4,1)` and at least three repeats per point.
Run engines sequentially on the same idle machine; record compiler flags,
CPU model, LAMMPS commit, MPI/OpenMP versions, and thread binding. Do not run
multiple benchmark points concurrently. For a full hybrid matrix, additionally
run the remaining rank/thread combinations when physical cores permit it.
The historical CUDA deck used LJ `lattice` scale as a length; that is incorrect
because LAMMPS interprets it as density in LJ units. Its density is now corrected,
but CPU evidence comes exclusively from this shared-data harness.

The final ATV `BENCH` line reports the **maximum rank's summed step-loop wall**,
excluding setup, output snapshots and the end-of-run report. Its accompanying
standard timing table exposes Pair, Neigh, Comm and Other. Compare with the
**last LAMMPS Loop time** (the measurement run), retaining the LAMMPS timing
breakdown. `thermo 1` requests energy and all six virial tensor components on
every LAMMPS step, matching ATV's per-step tallies. LAMMPS formats this output
into its log; quote its Output cost separately when diagnosing loop differences.
Final dumps occur after the timed run. With `pre no`, LAMMPS retains its
neighbor-build counter across warmup and measurement: subtract the warmup count
from the final count before comparing with ATV's reset timing call count. Never
compare force-only LAMMPS throughput with force+energy+virial ATV throughput.

`compare.py` enforces initial max-component force error <= 2e-10 eV/Angstrom
and total-energy error per atom <= 2e-10 eV. It also rejects non-finite values,
missing owners and duplicate tags. `--stage final` checks the moving trajectory,
but needs explicitly stated looser tolerances: ATV's mass/velocity conversion
constant is 103.6427, while LAMMPS metal corresponds to 103.64269 in fs units.
This tiny difference, plus floating-point reductions, accumulates during the
trajectory. **Do not change masses to hide it.** The initial force/energy gate
is independent of that integration-constant difference.

## CPU MD architecture: Apple M4, 2026-10-19

[All 30 samples, component timings, oracle checks and frozen source/binary
hashes](cpu-md-architecture-m4.json) cover the CPU MD ownership refactor.
The build enables MD, neighbor and potentials only, with MPI/OpenMP; it does
not compile electronic methods, Monte Carlo, minimization, Evaluate or molrs.
Both engines run sequentially, with no task build/test overlap, for 20 warmup
and 200 measured steps. Values below are medians of three paired ATV/LAMMPS
loop-time ratios; below 1 means less wall time for ATV.

| MPI ranks × threads | 10,648 atoms | 103,823 atoms |
|---|---:|---:|
| 1 × 1 | 0.572 | 0.560 |
| 1 × 2 | 0.516 | 0.473 |
| 1 × 4 | 0.604 | 0.512 |
| 2 × 1 | 0.589 | 0.569 |
| 4 × 1 | 0.719 | 0.620 |

All 30 initial-state comparisons pass the strict force/energy tolerances above.
Neighbor rebuild counts match in every sample: 28 for 10,648 atoms, 31 for
103,823. Fifty initial/one-step checks also pass across orthorhombic, tilted,
small-box multiple-image and nested MD cases at 1/2/3/4 ranks and a two-thread
single-rank point. Maximum initial force error is 7.35e-13 eV/Angstrom;
maximum one-step force error is 3.72e-9, within the stated 1e-8 allowance.
The small-box case uses `NSIDE=4`, so the cutoff exceeds half the box length.

Frame now owns all row extents; decomposition uses fractional partitions;
CommBrick owns explicit periodic images for every rank count. Runtime alone
orders wrap, exchange, sort, borders and neighbor build. NeighborList owns
cadence/displacement history; CellList owns spatial indexing; pair kernels
consume raw Cartesian row differences. Interaction cutoff, skin and consumer
halo requirements are validated independently. These are shared contracts,
not LJ-specific routing. Related verification passes 75 selected checks in the
MPI build, 5 in the native non-MPI build and 5 MD/communication regressions.

The measured LJ loop meets the comparison target on this Apple M4. Component
parity is not complete: median Neigh category ratios are 0.89–1.07, while Comm
ratios are 1.16–4.27. Comm includes ATV's reductions; stage maxima and raw logs
are retained for diagnosis. LAMMPS log-output cost is at most 2.26% of a sample,
so it does not explain the total-loop difference. Neighbor construction takes
26–52% of ATV loop time across the measured medians and remains the largest
compute cost. These shared-desktop results do not establish x86/cluster,
other-potential, long-trajectory or pressure-value equivalence.

### Post-fix confirmation (same day)

After that measurement the Runtime gained a set of contract fixes found by
the full test gate and review (neighbor cadence outside MD, a no-provider
reneighbor default, the cell-revision stamp after `end_of_step`, communicator
wire recompilation on every contract refold, tag-ordered gathers, no row
sorting under Minimize, the trial-replay resolution window, and frame binding
that keeps ghost rows). [The rerun](cpu-md-architecture-m4-postfix.json) uses
the same runner, LAMMPS binary and deck with ATV rebuilt from the final
sources; medians of three paired ratios, below 1 means less wall for ATV.

| MPI ranks × threads | 10,648 atoms | 103,823 atoms |
|---|---:|---:|
| 1 × 1 | 0.567 | 0.526 |
| 1 × 2 | 0.501 | 0.523 |
| 1 × 4 | 0.568 | 0.529 |
| 2 × 1 | 0.592 | 0.592 |
| 4 × 1 | 0.607 | 0.641 |

The load average was still decaying from the preceding test gate during this
run, so it confirms the band (0.50–0.64) rather than refining it. All 30
initial-state checks passed; the 46 short LJ/EAM/SW+LJ oracle checks (1–4
ranks, tilted, nested) also pass on the final sources with the same maxima
(1.81e-11 eV/Å initial, 3.70e-9 eV/Å one-step, 2.31e-11 eV/atom).

## CPU measurement: Apple M4, 2026-10-18

[Raw samples and build metadata](cpu-results-m4.json) record three repeats per
point, 20 warmup steps and 200 measured steps. Both engines were run sequentially.
These are **provisional measurements under substantial unrelated host load**,
not an idle-machine performance acceptance: one LAMMPS 2-rank large-system point
varied from 14 to 68 seconds. Do not derive a before/after speedup by comparing
these samples with earlier runs under different load, or claim general CPU
backend parity from this LJ benchmark.

The table reports the median of the three paired ATV/LAMMPS wall-time ratios;
below 1 means ATV used less wall time. All samples, including outliers, are kept.

| MPI ranks × threads | 10,648 atoms | 103,823 atoms |
|---|---:|---:|
| 1 × 1 | 1.11 | 0.83 |
| 1 × 2 | 1.02 | 0.87 |
| 1 × 4 | 0.58 | 0.94 |
| 2 × 1 | 1.15 | 0.64 |
| 4 × 1 | 0.68 | 0.84 |

The samples above precede the thread-owned contiguous CSR update below;
their exact source hashes are retained in the raw artifact.

Validation passed for the neighbor/row-reuse unit tests, MD default installation,
LJ active-thread scratch reduction, and MPI EAM/SW checks. Eight short LJ oracle
runs (1/2/3/4 ranks × 1/2 threads) passed both initial and one-step force/energy
checks; maximum initial force error was 1.36e-13 eV/Angstrom, and one-step error
was 3.71e-9 eV/Angstrom. All 30 matrix runs passed the strict initial-state check.
No pressure-value or long-trajectory oracle equivalence is claimed. These historical samples used explicit Evaluate seeding; the current harness
uses native MD initialization, including the nested correctness mode above.

### Thread-owned neighbor rows

[Recorded samples and source/binary hashes](cpu-neighbor-architecture-m4.json)
record three repeats at 10,648 atoms, 20 warmup and 200 measured steps. Each
sample runs updated ATV, LAMMPS, then the saved pre-change ATV binary,
sequentially. The median paired updated/previous wall-time ratios are 0.959
at two threads and 0.972 at four; updated ATV/LAMMPS ratios are 0.735 and
0.750 respectively. This shared-host measurement suggests a modest improvement,
but does not establish a reproducible speedup or general LAMMPS parity.

CellList retains the existing PairList CSR while storing only j indices in
thread scratch and copying each thread's rows contiguously. Full counting and
ghost-centred row requirements are independent: SW builds owned rows. EAM now
uses half rows and intermediate density/embedding communication; the historical
samples in this subsection predate that change. The LJ timings do not measure
the SW or EAM saving.
Five affected unit executables passed, as did EAM/SW at 2/3/4 ranks with two
threads and initial/one-step LJ comparisons at 1/2/3/4 ranks with 1/2 threads.


### Native MD initialization and nested execution

[Raw samples, oracle checks and source/binary hashes](cpu-lifecycle-m4.json)
record the current lifecycle implementation on the same Apple M4. Native
`MD.run(frame, 0)` supplies initial forces. Each point below has three paired
samples, 20 warmup and 200 measured steps; ATV and LAMMPS run sequentially with
no task build or test concurrent. Values are median paired ATV/LAMMPS wall-time
ratios (below 1 means less wall time for ATV).

| MPI ranks × threads | 10,648 atoms | 103,823 atoms |
|---|---:|---:|
| 1 × 1 | 0.885 | 0.875 |
| 1 × 2 | 0.681 | 0.724 |
| 1 × 4 | 0.666 | 0.717 |
| 2 × 1 | 0.835 | 0.904 |
| 4 × 1 | 0.880 | 0.966 |

All 30 strict initial-state checks passed. Direct and native nested MD also
passed initial and one-step comparisons at 1/2/3/4 ranks × 1/2 threads: maximum
initial force error 1.36e-13 eV/Angstrom, one-step error 3.71e-9 eV/Angstrom,
energy error 2.32e-11 eV/atom. The one-step force allowance is 1e-8; the energy
allowance remains 2e-10 eV/atom.

The artifact separately retains six paired current/LAMMPS/previous-binary
samples at 10,648 atoms. Current/previous median ratios are 1.004 at two
threads and 0.836 at four, but the four-thread previous binary varies from
0.307 to 0.523 seconds. These short shared-desktop samples do not establish a
stable before/after speedup. The matrix supports LJ parity at these tested
points; it does not establish backend-wide parity or measure EAM's reduced
halo and mixed Half/Full search savings. The measured binary precedes the
final Metadynamics-only initialization guard; its LJ and Runtime sources are
identical, and its exact hash is retained.

## EAM and mixed Full/Half comparison

The same executable, shared-state writer, native MD loop, owner snapshots and
`compare.py` support two additional models when built with molrs enabled:

```sh
# FCC Cu, 4 * 14^3 = 10,976 atoms; final argument selects the potential.
mpiexec -n 1 build/bin/bench_lj_cpu 14 20 200 1 /tmp/cu-a eam
mpiexec -n 1 /path/to/lmp -in benchmarks/lj/in.cpu \
  -var MODEL eam -var PREFIX /tmp/cu-a -var WARM 20 -var STEPS 200 \
  -log /tmp/cu-a.lmp.log -screen none
python3 benchmarks/lj/compare.py /tmp/cu-a

# Diamond Si, 8 * 11^3 = 10,648 atoms; SW Full + LJ Half neighbor consumers.
mpiexec -n 1 build/bin/bench_lj_cpu 11 20 200 1 /tmp/si-a sw-lj
mpiexec -n 1 /path/to/lmp -in benchmarks/lj/in.cpu \
  -var MODEL swlj -var PREFIX /tmp/si-a -var WARM 20 -var STEPS 200 \
  -log /tmp/si-a.lmp.log -screen none
python3 benchmarks/lj/compare.py /tmp/si-a
```

LAMMPS requires MANYBODY and OPENMP. As for LJ, use the ordinary style for one
thread and add `-sf omp -pk omp THREADS` for multiple threads. Both models use
300 K deterministic Maxwell velocities with centre-of-mass motion removed,
0.3 A skin, Newton on, energy and virial each step, and the same sorting and
neighbor rebuild settings as LJ. Input generation and table loading are outside
the timed step loop.

- **Cu EAM:** the in-tree Sutton–Chen parameters, lattice constant 3.61 A,
  mass 63.546 amu, cutoff 5.5 A, pair-energy shift at the cutoff, timestep 1 fs.
  ATV evaluates the analytic functions. The harness samples those same
  `EAMPotential` functions into LAMMPS `eam/alloy` setfl with radial spacing
  0.0001 A and density spacing 0.005 up to density 1000. The unused interval
  below 1 A is regularized only for the table. This compares the same physical
  Hamiltonian with analytic versus interpolated implementations; interpolation
  error must pass the force/energy check and is not silently treated as zero.
- **Si SW+LJ:** original Stillinger–Weber silicon parameters read from the same
  in-tree table by both engines; lattice constant 5.43 A, mass 28.0855 amu,
  timestep 0.5 fs. A shifted LJ overlay has epsilon 0.01 eV, sigma 2 A and the
  same 3.77118 A cutoff as SW. LAMMPS uses `hybrid/overlay sw lj/cut`. The
  overlay deliberately exercises simultaneous Full/Half consumers; it is a
  benchmark composition, not a fitted silicon potential.

For numerical validation first run side 6, zero warmup and one measured step,
checking both snapshots across ranks and threads before taking timings. Use
fresh prefixes. Keep the raw output and state the selected force/energy
allowances; a long chaotic trajectory is not an initial force oracle.

### Historical first-pass many-body measurement: Apple M4, 2026-10-19

[Raw paired samples, oracle checks and exact source/binary hashes](cpu-manybody-m4.json)
record three samples per point, 20 warmup and 200 measured steps. Each sample
runs updated ATV, LAMMPS, then the saved pre-optimization ATV binary,
sequentially. Both models request energy, forces and virial every step. The
updated executable is the frozen `bench_manybody.contracts` binary, SHA256
`c2dbfaddd09a2f058f9d8bede0039b35c2c29193be47769e7573a3d698309768`;
later unrelated optimizer/PW/DIIS edits are outside this measurement.

Values below are medians of paired wall-time ratios; below 1 means less time.
Cu has 10,976 atoms and Si SW+LJ has 10,648 atoms.

| MPI ranks × threads | EAM / LAMMPS | EAM / previous ATV | SW+LJ / LAMMPS | SW+LJ / previous ATV |
|---|---:|---:|---:|---:|
| 1 × 1 | 1.022 | 0.528 | 1.126 | 0.590 |
| 1 × 2 | 0.850 | 0.292 | 1.456 | 0.490 |
| 1 × 4 | 0.787 | 0.189 | 1.436 | 0.310 |
| 2 × 1 | 1.002 | 0.526 | 1.172 | 0.573 |
| 4 × 1 | 1.035 | 0.553 | 1.043 | 0.632 |

EAM's one-rank median wall is 1.187 s at one thread and 0.395 s at four
threads (3.01×); SW+LJ is 0.812 s and 0.475 s (1.71×). EAM is close to or
faster than the LAMMPS reference at these points. SW+LJ improves substantially
against previous ATV but remains about 44–46% slower than LAMMPS at the two
multithread points. Its remaining gap lies primarily in Pair computation,
not the outer Driver: at two threads, the middle sample spends 0.665 s in
Pair out of 0.710 s total, versus LAMMPS 0.463 s in Pair out of 0.487 s.

EAM follows LAMMPS's OpenMP density/embedding communication ordering and
SW uses thread-owned short neighbor lists and force accumulation. Both reuse
`BondedForceScratch`; EAM also removes general real-power evaluations for
integer exponents, and SW specializes its published p=4, q=0 powers. The
artifact retains separate Pair timings because loop ratios also include the
concurrent Runtime contract work.

All 30 initial-state checks passed at 2e-10 eV/A and 2e-10 eV/atom. Another
36 initial/one-step checks cover each model at one rank with 1/2/4 threads,
and 2/3/4 ranks with 1/2 threads. Maximum initial force error is 1.81e-11
for EAM and 4.93e-14 for SW+LJ; maximum one-step force error is 3.11e-11,
below the stated 1e-8 allowance. Energy errors stay below 9.43e-14 eV/atom.

All three executables use `OMPI_MCA_mpi_yield_when_idle=1`,
`OMP_DYNAMIC=FALSE`, and one BLAS thread. No task build/test or external
compiler process was active at the start. This remains a shared desktop:
load and individual wall times vary, every sample is retained, and no general
CPU-backend performance claim follows. EAM compares analytic ATV functions
with LAMMPS interpolation of the same parameters; the strict measured oracle
error bounds that representation difference. Neither long-trajectory agreement
nor pressure-value equivalence is asserted here.

## Orthorhombic contracts snapshot: Apple M4, 2026-10-19

[Exact final binary, all raw samples, checks and source hashes](cpu-contracts-m4.json)
record the completed pass after the neighbor-image, filtered-cache and nested
Driver lifecycle fixes, plus the second SW arithmetic/ownership optimization.
The frozen `bench_manybody.nested-contracts` SHA256 is
`4c07f35a6f9d0a5974cd1a3acdf77a1b2c960227f356de76d87c7f250154411f`.
The previous `bench_manybody.contracts` comparator already includes the first
EAM/SW OpenMP pass; it is not the original pre-optimization baseline.

Every point has three sequential paired samples, 20 warmup and 200 measured
steps. EAM/SW+LJ run current ATV, LAMMPS, then previous ATV; pure LJ runs
current ATV then LAMMPS. The table reports median paired wall-time ratios
(below 1 is faster). EAM has 10,976 atoms, SW+LJ 10,648, and LJ the two sizes shown.

| MPI ranks × threads | EAM / LAMMPS | EAM / previous ATV | SW+LJ / LAMMPS | SW+LJ / previous ATV | LJ 10,648 / LAMMPS | LJ 103,823 / LAMMPS |
|---|---:|---:|---:|---:|---:|---:|
| 1 × 1 | 1.125 | 1.091 | 1.142 | 1.191 | 0.988 | 0.641 |
| 1 × 2 | 0.920 | 1.018 | 1.231 | 0.886 | 0.777 | 0.828 |
| 1 × 4 | 0.817 | 1.000 | 1.031 | 0.683 | 0.958 | 0.651 |
| 2 × 1 | 0.946 | 0.976 | 1.078 | 0.951 | 0.830 | 0.889 |
| 4 × 1 | 0.881 | 0.978 | 0.936 | 0.625 | 1.053 | 1.004 |

The largest SW+LJ median gap is at 1 MPI rank × 2 threads: 1.231× LAMMPS,
with individual ratios 0.747–1.376.
SW+LJ's four-thread median ratio is 1.031; its three paired ratios span 0.894–1.126.
Pair consumes 87.3% of ATV's loop at that point (median sample fraction).
Median MPI aggregate Pair time is 0.385 s for ATV versus 0.400 s for LAMMPS.
At the largest-gap point, the median paired MPI Pair ratio is 1.214 and neighbor ratio 1.744.
Pair dominates absolute cost; neighbor construction also remains slower.
These timings do not isolate scratch reduction from SW/LJ arithmetic or establish
a stable backend-wide speedup. Current/previous ratios also include framework
and image-contract changes.

All 68 initial/one-step checks passed: 36 EAM/SW+LJ and 16 each for native
and nested LJ, including ranks 1–4. All 60 timed initial-state checks passed.
The short-check maxima are 3.7e-09 eV/A force and 2.31e-11 eV/atom energy.
Allowances remain 2e-10 eV/A and 2e-10 eV/atom initially, with 1e-8 eV/A
for the one-step force check. No numerical threshold was relaxed.

All engines use `OMPI_MCA_mpi_yield_when_idle=1`, `OMP_DYNAMIC=FALSE`, and
one BLAS thread. No task build/test overlapped this window. Ordinary desktop
applications remained active; the artifact records load and every raw sample.
No samples were dropped. EAM still compares analytic ATV functions with
LAMMPS interpolation of the same parameters; the oracle bounds that difference.
The SW+LJ overlay is a composition benchmark, not a fitted silicon model.
These orthorhombic results do not cover triclinic CellList, long trajectories
or pressure-value equivalence.

LAMMPS Pair timing here comes from the final **MPI task** aggregate, including
force reduction; MPI rank 0 OpenMP thread and Reduce timings are stored
separately. They are not MPI-global maxima.
Earlier artifacts selected the last thread Pair row at OMP points, so their
Pair figures should not be treated as MPI totals. Their loop-time ratios remain
unchanged. The aborted first final pass is retained as
[historical partial evidence](cpu-image-contracts-partial-m4.json), including
the nested lifecycle failure subsequently fixed; it is not current validation.


## Triclinic support and neighbor measurement: Apple M4, 2026-10-19

[Source hashes, all samples and validation](cpu-triclinic-m4.json) describe
`bench_manybody.triclinic-flat-stencil` (SHA256
`19fac430520a18950a3e083bfa39c3a36f78de79a8cd1fdc30f570c8d79dbdc0`).
Box uses column lattice vectors. The existing CellList and CommBrick now handle
general triclinic geometry, including self-rank images and full-cell changes.
This snapshot also fixes EAM self-image density aliasing, MPI replica halos and
cell-revision invalidation before the first Verlet half kick.

All 105 selected non-MPI executables, 18 MPI registrations, PR-NPT (800 steps),
MTK NPT and NPH LJ regressions passed. All 34 tilted initial/one-step comparisons
passed: LJ/EAM/SW+LJ at 1×1, 1×2, 2×1, 3×1, 4×1 plus nested LJ at 1×2 and
2×2. Tilts are `(1.7, -0.8, 0.6)` Angstrom. Maximum force error is
`3.174e-9` eV/Angstrom and energy error is `1.834e-11` eV/atom, within the same
initial/one-step allowances above. These external checks do not validate every
KSpace/barostat composition or long-trajectory pressure equivalence.

The measured optimization ports LAMMPS's precomputed flat stencil offsets for
interior grid rows, retains the boundary path, and removes redundant duplicate
cell checks for injective triclinic extended grids. Performance below uses the
**orthorhombic** 10,648-atom SW+LJ workload, 20 warmup / 200 measured steps,
three sequential paired samples per point. The baseline is the first triclinic
binary before these optimizations and final boundary fixes. Values are median
paired time ratios; below 1 is faster.

| Ranks × threads | Neighbor / before | Neighbor / LAMMPS | Loop / before | Loop / LAMMPS |
|---|---:|---:|---:|---:|
| 1 × 1 | 0.886 | 2.006 | 0.965 | 1.190 |
| 1 × 2 | 0.861 | 1.921 | 0.951 | 1.176 |
| 1 × 4 | 0.912 | 1.683 | 1.013 | 1.166 |
| 2 × 1 | 0.861 | 1.584 | 0.975 | 1.132 |
| 4 × 1 | 0.761 | 1.536 | 1.008 | 1.021 |

Neighbor time falls 9–24%; whole-loop improvements are limited to the first,
second and fourth rows in this measurement. The other loop medians are within
1.3% of the baseline. Neighbor construction remains 1.54–2.01 times LAMMPS,
and performance parity is not established. No samples were discarded; ordinary
desktop load remains a source of variation. No build/test overlapped timings.
The remaining orthorhombic full-list path still applies per-candidate minimum
images on unresolved axes. Any specialization must preserve tiny-grid aliasing
and unwrapped direct-CellList input contracts; interior bin indices alone do
not establish that coordinates are wrapped.

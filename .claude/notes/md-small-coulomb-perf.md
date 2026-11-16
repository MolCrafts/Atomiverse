# Small dense Coulomb MD — CPU cost and threads

Harness: `benchmarks/perf/md/bench_nacl_ewald.cpp` +
`benchmarks/compare/nacl-ewald-lammps` (512-ion NaCl, PairBorn + CoulLong +
KSpaceEwald, GearPC + Gauss). Numbers are not kept in git; rerun it. The
per-optimization gains are in the `perf(...)` commit messages after
`af1bb61cf`.

## Where a step goes

- Reciprocal Ewald is `EwaldSum` over phase tables e^{i m θ_a} per cell
  axis (one cos/sin per atom and axis, a complex recurrence for the rest);
  S(k), the forces and the curvature kernels are column sums over m₂. Never
  reintroduce cos/sin per (k, atom).
- With the paper setting (cutoff L/2, |k| ≤ 6·2π/L, 447 k-vectors at any L)
  the real-space pair walks dominate (Born ≈ 50 %, CoulLong ≈ 30 % of a
  1-thread step); with a fixed-Å accuracy target the k-vector count grows
  with the volume (3015 → 22776 from liquid to fluid) and k-space dominates
  in the fluid.
- PairBorn's per-type table walks the published half list only when the
  list declares coverage (`PairList::cutoff`); PairBorn is not a
  NeighborConsumer, so nothing else checks the radii. A perfect lattice with
  pairs at exactly r = L/2 makes list and all-pairs walks differ by those
  pairs (≈1e-7 eV/Å); a melt does not.
- CoulLong erfc and Born r / exp are xsimd batches in the PairLoop `prepare`
  hook; the curvature walks of both (configurational thermostat and
  `ConfigurationalTemperature`) have the same shape in `md::CurvatureLoop`
  (`pair_curvature.h`). One curvature evaluation costs about 1.7 force
  evaluations at 1 and at 8 threads (Born ≈ CoulLong, the reciprocal part
  smallest), so a configurational step scales with threads like the Gauss
  step. The serial scalar walk it replaced cost 3.7x the force walk of the
  same pairs: 85 % of a 4-thread step.
- Past 8 threads the curvature walks stop scaling on 512 ions: ≈ 0.10 ms
  each at 16 and at 32 threads, where the two force walks together take
  0.08. Not profiled further; each source flushes its own per-thread
  accumulators (rows x threads cache lines), the force walks share one flush.
- What stays serial in a curvature evaluation is `CurvatureAssembly`'s own
  bookkeeping (zeroing ten ghost-sized rows, the forward comm of u, the
  reverse comm, the copy-out): ≈ 0.04 ms, a tenth of the evaluation on 8
  threads. Under a single-process halo no source writes a ghost row any
  more (the walks fold, the reciprocal part never did).
- `KSpaceEwald::add_curvature` rebuilds the phase tables and S(k) the force
  call of the same positions just made (≈ 1/4 of the reciprocal curvature,
  a few % of the step); nothing tells it the positions are unchanged.
- `bench_nacl_ewald --thermostat config` starts from the lattice, where
  T_C1 = 0: keep it to a few hundred steps (the liquid's Laplacian turned
  negative within 500 and GearPC refuses). Time the thermostat from a melt
  configuration instead.
- `.cu` tests include the CPU styles' headers as their oracle, so those
  headers (and what they include, `pair_curvature.h`) must keep xsimd behind
  `!defined(__CUDACC__)`; put batch kernels in the `.cpp`.

## Threads

- libgomp keeps its pool at the size of the last team it opened: a region
  that opens fewer threads than the one before ends the surplus threads, and
  the next region that opens more starts new ones. On EPYC 9755 (GCC 14.3,
  bound) a steady 64-thread region costs 14 µs and a 16-thread one 5 µs;
  alternating 64- and 16-thread regions cost 0.9 ms each, 16 and 7 0.09 ms.
  `OMP_WAIT_POLICY=active` and `OPENBLAS_NUM_THREADS=1` change nothing. Regions
  that each sized their team to their own work paid this on most regions of
  every step (it was the whole "64 threads slower than 4" effect, in every
  CPU MD benchmark).
- So every work-sized region opens the whole team or runs serially
  (`cpu::device::team_for`); its work picks only which. Never size a team in
  between: one such region reintroduces the churn for the whole step.
  Thresholds are two threads' minimum shares: pair walks 2048 entries plus
  the chunk buffer rows, Ewald 2^16 (k, atom) pairs with 32 atoms or 4
  columns (column kernels also 8 pairs per phase-table entry they pull),
  chunk-force flush 8192 slot rows, bonded 2048 terms, list builds 16384
  candidates, binning and position wrap 8192 rows, GearPC / VelocityVerlet /
  Virial 4096 rows, kinetic energy 16384.
- Under a single-process halo the generic pair walk adds a ghost partner's
  force to its owner (`Context::md.image_owner`), so chunk buffers, their
  zeroing and the flush span owned rows only (a long cutoff in a small box
  has several ghosts per atom). With one chunk per thread, thread c runs
  chunk c. Keep the chunk body inline: a per-chunk closure took the scratch
  arrays' addresses, so the compiler had to assume the force stores alias
  them, and the 1-thread walk lost 11 %.
- The serial `KSpaceEwald` time moves by ±12 % with heap layout alone
  (`GLIBC_TUNABLES=glibc.malloc.top_pad=…` or `mmap_threshold=…` shift
  either build, and the baseline, between 3.8 and 4.4 ms in fluid
  accuracy). Do not read a 1-thread Ewald difference between two builds as
  a code effect without that check.
- Unbound OpenMP on a shared Arrhenius node stalls full-ish teams (≥16
  threads) for a scheduler slice per barrier; `OMP_PROC_BIND=close
  OMP_PLACES=cores` removes it. Classical MD may bind; BLAS-bound paths must
  not ([openblas-thread-binding.md](openblas-thread-binding.md)).
- Other integrators and thermostats (NoseHoover, NPT, Langevin, Berendsen,
  CSVR, SLLOD, constraints) use bare `omp parallel for` over rows: the whole
  team, so they keep the pool too, but a small system pays a full fork there.
- Header changes (team constants, kernel signatures) can leave stale objects
  in an incremental CMake build that segfault in `KSpaceEwald`; rebuild the
  library objects before timing.

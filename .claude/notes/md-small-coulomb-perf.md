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
  hook. Curvature (configurational thermostat) is still scalar, CoulLong
  curvature still `std::erfc`: ≈4.5 ms of a liquid-paper step.

## Threads

- Every region of a Gauss step sizes its team from its own work: pair walks
  `PairRowPartition::kMinPairsPerThread`, Ewald `EwaldSum::kPairsPerThread`
  (k, atom) pairs, GearPC 4096 rows, chunk-force flush 4096 rows, list build
  256 centre rows, binning and position wrap 8192 rows. A 512-atom step runs
  most regions on 1–3 threads whatever `OMP_NUM_THREADS` says, so 64 threads
  cost what 4 do.
- `kMinPairsPerThread` (49152) counts pairs, not work: a Born + erfc pair is
  ≈12 ns, an LJ pair a few ns. At 16384 the fluid paper case goes 1.43 → 0.85
  ms at 4 threads, but the serial chunk flush then grows with the team. A
  per-style cost weight is the open fix; measure the LJ/GAFF harnesses
  before changing the constant.
- Unbound OpenMP on a shared Arrhenius node stalls full-ish teams (≥16
  threads) for a scheduler slice per barrier; `OMP_PROC_BIND=close
  OMP_PLACES=cores` removes it. Classical MD may bind; BLAS-bound paths must
  not ([openblas-thread-binding.md](openblas-thread-binding.md)).
- Still full-team: other integrators and thermostats (NoseHoover, NPT,
  Langevin, Berendsen, CSVR, SLLOD, constraints) use bare `omp parallel for`
  over rows.
- Header changes (team constants, kernel signatures) can leave stale objects
  in an incremental CMake build that segfault in `KSpaceEwald`; rebuild the
  library objects before timing.

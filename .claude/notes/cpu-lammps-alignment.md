# CPU MD execution — mechanism and LAMMPS mapping

Binding summary: docs/developer/project-contract.md › CPU MD execution. This note holds the mechanism
and which LAMMPS source each piece follows. Numbers: `benchmarks/lj/README.md`.

## Rebuild boundary and cadence

- `Runtime` rebuild (`comm_rebuild_`): wrap → exchange → sort → borders →
  neighbor, after LAMMPS `Verlet::setup` / `Verlet::run`.
- Wrap only when a decomposition or a NeighborProvider exists, never during a
  line-search trial; integrators advance owned rows and leave positions
  unwrapped between rebuilds, preserving cached image shifts.
- Sort only when `atom_sort->due()`, including the setup rebuild. Components
  that cache rows re-resolve by tag at each neighbor build or call.
- MD applies the provider's delay/every/check. Linear workflows (Evaluate,
  Minimize, MC energy, FD Hessian/Phonon force) decide at their force slot
  after line-search updates with `Min::init` cadence (`every 1 delay 0 check
  yes`); MC energy always rebuilds; no provider ⇒ migrate every step. Cell/grid
  invalidation bypasses the displacement cadence.
- Minimize, MC and Hessian/Phonon forbid row reordering for the provider's life
  (CG/L-BFGS histories, trial patches and FD workspaces index rows; there is no
  FixMinimize to carry them through a sort).
- Binding a Frame leaves its row extents alone; `Comm::exchange` /
  `Comm::borders` reset and rebuild ghosts. The Gaussian SCF node binds the
  shared Frame to its private backend mid-pass and must not drop halo rows the
  hosting Runtime's reverse pass owns.
- Halo reach = max(provider enumeration reach, widest consumer halo + provider
  skin). Consumers must fit the provider's interaction cutoff, excluding skin.

## Neighbor lists

- Half/full and ghost-centred rows are independent requests
  (`src/neigh_request.h`); Runtime aggregates `neighbor_list_mode()` /
  `requires_ghost_neighbor_rows()` and calls `configure_lists`.
- Full + half requested ⇒ CellList builds full CSR once and derives half with
  `src/npair_halffull.cpp` Newton ownership (owned pairs by row order, ghost
  pairs by shifted z/y/x; triclinic by tag/parity, equal-tag images kept).
  Tag-only or min-image filtering would collapse distinct periodic images.
- Thread-owned j-only scratch follows `src/OPENMP/npair_bin_omp.cpp`; PairList
  stays the canonical CSR of explicit atom/image rows (no axis bookkeeping).
- CellList bins general column-lattice cells by Cartesian bounding box and never
  wraps coordinates or manufactures images; images come from CommBrick.
- Filtered caches (LJ, CoulCut, TIP4P, fused LJ/CoulLong) key on source
  identity and rebuild version; version 0 is a valid first list.
- Force consumers account for self-rank ghosts through `force_count`.

## Many-body

- EAM follows `src/MANYBODY/pair_eam.cpp`: half-list density, `ReversePacker`
  density to owners, owner embedding derivative, `ForwardPacker` derivative to
  ghosts, half-list force/energy/virial, driver force reverse. Coordinate reach
  is cutoff + skin.
- SW consumes full owned-centred rows with ghosts as candidates; two-body terms
  are counted once per pair.

## Triclinic geometry

- Box: `r = H s`; mutations compute and validate inverse, face heights and
  reduced-lattice caches before publishing. Exact nearest image searches a
  lattice-reduced QR basis — fractional rounding is not a nearest-image
  algorithm for general skew. Positive diagonal cells keep the scalar/SIMD path.
- DomainDecomposition classifies owners in fractional coordinates; initial
  partition, migration and Rebalance share that. Halo converts reach R to
  R / face_height per slab and shifts images by full lattice vectors.
- Barostat/deformation transforms agree on full H rows.
- Still orthorhombic-only: GPW and slab Ewald. PPPM distributed mesh bricks
  reject nonuniform spatial cuts. General cells are not evidence for every
  tilted KSpace/barostat composition.

## Initial forces and invalidation

- `Runtime::prepare_run` (from `Verlet::setup`): after setup comm, evaluate
  `pre_force`..`end_of_step` skipping `final_integrate`, reverse comm before
  `post_force`; force slot via `launch`, others via `initialize_forces`. Setup
  scalar reductions bypass production cadence; setup is excluded from step
  timing.
- Forces are re-prepared when `pre_force`..`post_force` bindings change or the
  box revision differs from the stamp taken after `end_of_step` (a barostat or
  Deform inside the loop is the loop's own update, as LAMMPS
  `fix press/berendsen`; a cell edit between `run()` calls re-initializes).
- Setup failures invalidate communication state so a retry clears stale ghost
  accumulators.
- Langevin/DPD setup use the preceding stochastic step: reserved −1 on a fresh
  trajectory, N−1 when re-preparing at cumulative step N.

## Nesting and lifecycle

- `ScopedPlan` saves the active Plan, MD services, force validity, neighbor
  cache and timing handles; re-entry re-runs `refresh_plan_` →
  `setup_verlet_comm_`. A child MD keeps its cumulative clock across
  invocations; SCF iterations restart at zero per solve. Force-veto requests
  cannot contain MD.
- Registration-private stage adapters keep their own budgets; nested restoration
  happens before cleanup; shared components are destroyed once.
- Cell optimizers update owned positions only, refresh DD slab bounds, and
  publish force/displacement metrics in eV/Å and Å; pressure is a separate
  criterion. Nonfinite pressure/forces/scale parameters fail collectively before
  updates; nonfinite selected stress/pressure observations yield `Stop`.

## Electronic reuse

- GPW fixed-density CP reuses potentials only from the matching density result;
  XL shadow forces reuse geometry grids but recompute density-dependent
  potentials (a direct-kernel test poisons auxiliary potentials to enforce it).
- PW energy/force/stress requests reuse one solved `PWProblem`.

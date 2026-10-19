# CPU MD alignment with LAMMPS

Reference inspected: LAMMPS commit
`a0e2bd6efb3d67a29c0c4e4de3b19556ffdb9d5b`.

## Execution contracts

`Runtime::prepare_run` implements the initial force boundary from
LAMMPS `src/verlet.cpp::setup`. After wrapping, exchange, borders and neighbor
construction, it evaluates pre-force hooks, forces, reverse communication,
post-force hooks and opted-in scalar finalization. It skips integration and
ordinary observation. `MD.run(frame, 0)` exposes the initialized Frame.
Force assembly changes invalidate the initial force; observer-only changes do
not. The cell revision is stamped after `end_of_step`, so a barostat or Deform
rescaling the cell inside the loop is the loop's own update (the next step
integrates on the forces it has, as LAMMPS `fix press/berendsen` does), while a
cell edit between `run()` calls still re-initializes forces.
Setup failures invalidate communication state so a retry clears stale
ghost accumulators. Setup is excluded from production step timing. Rebinding the same timed
identity preserves its handle and samples instead of growing the registry.

The component-author `initialize_forces` hook is opt-in outside the force
stage. Force evaluators inspect `ExecServices::initializing_forces` to preserve
CP/XL/ASPC, Metadynamics hill/cadence, and auxiliary dipole/electrode histories.
Langevin and DPD setup use
the preceding stochastic step: reserved -1 on a fresh trajectory, N-1 on
re-preparation at cumulative step N. Setup scalar reductions bypass production
cadence. Production RNG indexing and scalar cadence retain their contracts.

Nested MD uses the same Verlet scheduler as top-level MD. ScopedPlan saves the
active Plan, MD services, force validity, neighbor cache and timing handles.
A child MD retains its cumulative clock across invocations; SCF iterations
start at zero per solve. Electronic children borrow parent communication.
Physical children rebuild their own communication and invalidate the parent's
halo/list/forces on exit. Physical nesting is supported in a generic outer
workflow or at MD end_of_step; it is rejected during an active parent force or
integration pipeline because migration and accumulator replacement would
corrupt the enclosing operation. Force-veto requests cannot contain MD.

## Neighbor and intermediate communication

Full counting and ghost-centred rows are independent requirements, following
LAMMPS `src/neigh_request.h`. SW consumes full owned-centred rows, with ghosts
as candidates. If both full and half are requested, CellList builds full CSR
once and derives half CSR using `src/npair_halffull.cpp` Newton ownership:
owned pairs use row order, ghost pairs use shifted z/y/x coordinates. Tag-only
or minimum-image filtering would collapse distinct periodic ghost images.
Thread-owned j-only scratch follows `src/OPENMP/npair_bin_omp.cpp`; PairList
remains the canonical CSR.

EAM follows `src/MANYBODY/pair_eam.cpp`: half-list density accumulation,
reverse density to owners, owner embedding derivative, forward derivative to
ghosts, half-list force/energy/virial, driver force reverse. Coordinate halo
reach is cutoff plus the existing skin/safety policy, replacing the old 2rc
shell and ghost-centred density reconstruction. ReversePacker is independent
of ForwardPacker; CommBrick reuses Halo's swap lists and buffers in reverse
order. No second communicator or public configuration bag is introduced.

## Image and invalidation contracts

PairList records which axes already contain halo-selected periodic images.
Every neighbor-list force consumer retains raw displacements on these axes;
only unresolved axes use minimum images. A far ghost inside the neighbor skin
must not be folded back inside the force cutoff. Full-to-half derivation and
filtered consumers preserve this policy. LJ, CoulCut, TIP4P and fused LJ/CoulLong
also key filtered caches by source identity and rebuild version; version zero
is a valid first list, not an invalid-cache sentinel.

CellList supports general column-lattice cells using Cartesian bounding-box
bins and full-cell invalidation. Triclinic DD lists carry explicit periodic
images on all axes, including self-rank images. Direct CellList consumers
without CommBrick enumerate translated center images through the same bins;
PairList preserves each displacement separately. Full-to-half derivation uses
LAMMPS tag/parity ownership for triclinic lists and keeps equal-tag images.
Box owns wrapping, Cartesian/fractional transforms and exact nearest-image
search in a cached lattice-reduced QR basis; fractional rounding alone is not
a nearest-image algorithm for general skew.

Linear workflows make the neighbor decision at the force boundary, after line
search updates, with LAMMPS `Min::init` cadence (`every 1 delay 0 check yes`):
only an MD plan applies the provider's delay/every/check. Minimize also forbids
row reordering for its provider's life, because CG / L-BFGS histories index
rows and there is no FixMinimize to carry them through a sort. A plan without
a NeighborProvider migrates and rebuilds its ghosts every step.
`gather_to_root` returns owners in tag order at every rank count; the row
order after `atom_modify sort` is never the API's view. Cell/grid
invalidation bypasses the displacement delay/every cadence. Cell optimizers update only owned positions, refresh DD slab bounds,
and publish force/displacement metrics in eV/Å and Å; pressure stays a separate
criterion. Nonfinite pressure, forces or scale parameters fail collectively
before updates. Nonfinite selected stress/pressure observations yield Stop,
which the Driver combines across ranks.

Binding a Frame to a Runtime leaves its row extents alone: `Comm::exchange`
and `Comm::borders` reset and rebuild the ghost rows, and an electronic method
that binds the shared Frame to its private backend mid-pass (the Gaussian SCF
node) must not drop the halo rows the hosting Runtime's reverse pass owns.
Every refold of the ghost contract (build, live mount, nested plan in or out)
marks the communicator stale; the next rebuild boundary re-runs `Comm::setup`
so the border/forward/reverse wires match the active plan's declarations.
Driver-owned Runtime cleans up successfully built components when an exception
escapes outer dispatch, then rebuilds on retry. Nested restoration happens before
cleanup, and shared components are destroyed once. Registration-private stage
adapters retain their own execution budgets, while Runtime counts nested
DriverNode identities separately so one child bound at multiple stages is
built and destroyed once. A different Runtime cannot acquire that live child.
Low-level Runtime users keep
manual lifecycle and borrowed-plan retry semantics. CPU convergence requires all
declared convergence criteria on all spatial ranks; stop-only guards are neutral.

GPW fixed-density CP reuses potentials only from the matching density result.
XL shadow forces reuse geometry grids but recompute density-dependent potentials;
a direct-kernel test poisons auxiliary potentials to enforce this distinction.
PW energy/force/stress requests reuse one solved PWProblem. These optimizations
reuse existing owners rather than introducing another solver facade.

## Triclinic ownership and lifetime

Box uses lattice vectors in columns, `r = H s`, consistently with molrs.
CellList, communication, PW reciprocal vectors, classical Ewald/mesh transforms,
DFTB lattice sums and vibrational supercell translations consume that convention.
Frame coordinates remain Cartesian throughout execution. Box mutations compute
and validate inverse, face heights and reduced-lattice caches before publishing
any state. Positive diagonal cells retain their scalar/SIMD fast path.

DomainDecomposition classifies owners in fractional coordinates for triclinic
cells. Existing cuts remain fractional; initial partition, migration and
Rebalance share this convention. Halo converts physical reach R to R/face_height
for each slab and shifts images by the full lattice vector. CommBrick reuses its
existing PackPlans, swap lists and buffers for both local copies and remote MPI
transport. LAMMPS source provenance is retained in the implementation.

Runtime wraps owned positions at the exchange/rebuild boundary, following
Verlet::setup and Verlet::run. Integrators leave positions unwrapped between
rebuilds, preserving cached image shifts. The condition still requires migration
or a neighbor consumer, so isolated cluster evaluators are not wrapped.
Barostat/deformation coordinate and cell transforms agree on full H rows;
all force consumers account for self-rank ghosts through force_count.

GPW and slab Ewald retain explicit orthorhombic restrictions; PPPM distributed
mesh bricks reject nonuniform spatial cuts. Supporting general three-dimensional
cell geometry is not evidence for every tilted KSpace/barostat composition.

## Validation boundary

Reviewed neighbor construction, EAM/SW consumers, direct/nested MD entry
points, initial force assembly, stochastic indexing and electronic history
handling. This is not an audit of every CPU numerical kernel or CUDA.
The wider approved migration remains in
[CPU orthogonality](../specs/cpu-orthogonality.md), including
live edits inside already-built nested loops.

Validation uses the expanded CPU build (`/tmp/atv-architecture-pw`), with
BLAS=1 and no overlapping build/test/performance jobs. OpenMPI tests and both
benchmark engines use `OMPI_MCA_mpi_yield_when_idle=1` on macOS.

- Box has 25 direct tests, including strongly skewed nearest-image counterexamples,
  wrapping, general H, reciprocal face heights and atomic mutation failure.
- All 105 selected non-MPI executables and 18 MPI registrations pass, covering
  rebuilt geometry, neighbor, potential, communication, runtime, MC, minimizer
  and affected electronic consumers. Four communication/neighbor sources also
  pass syntax compilation with MPI macros disabled; a separate no-MPI runtime
  suite and CUDA were not run.
- Unchanged PR-NPT (800 steps), MTK NPT and NPH LJ regressions pass. The
  materializer unit compares compiled plans without advancing a trajectory.
- The standing LJ harness accepts absolute `--tilt XY XZ YZ` values and writes
  the same tilted initial state to LAMMPS. Initial/one-step LJ, EAM and SW+LJ
  checks cover 1/2/3/4 ranks and a two-thread serial configuration; nested LJ
  at 1×2 and 2×2 also passes (34 checks total). General
  non-upper-triangular H and self-image multiplicity are direct unit-test cases.

The separate orthorhombic benchmark snapshot (`bench_manybody.nested-contracts`, SHA256
`4c07f35a6f9d0a5974cd1a3acdf77a1b2c960227f356de76d87c7f250154411f`)
passes all 68 short initial/one-step checks and all 60 timed initial-state
checks. Each timing point retains three paired samples. The standing
`benchmarks/lj/cpu-contracts-m4.json` records every sample, source/binary hashes,
host load and model limitations; `benchmarks/lj/README.md` summarizes them.
On the shared M4 desktop, measured ATV/LAMMPS median ratios range from
0.777–1.053 for 10,648-atom LJ, 0.641–1.004 for 103,823-atom LJ,
0.817–1.125 for EAM and 0.936–1.231 for SW+LJ. These are separate model and
configuration observations, not universal CPU backend parity. EAM uses
analytic versus interpolated representations of the same parameters.
LAMMPS stage comparisons use its final MPI task aggregate; rank-zero OpenMP
thread diagnostics are separate. Historical artifacts' OMP Pair fields used
the thread row, while their total loop ratios remain valid.


Current flat-stencil measurements are recorded separately in
`benchmarks/lj/cpu-triclinic-m4.json`. Three paired samples per point show
orthorhombic SW+LJ neighbor time at 0.761–0.912 of the pre-optimization
triclinic binary, but still 1.536–2.006 of LAMMPS. Whole-loop ratios to LAMMPS
are 1.021–1.190 in this pass; they are not universal CPU parity. The remaining
full-list unresolved-axis MIC path is a profiling target, not an approved
assumption that all direct CellList input is wrapped.

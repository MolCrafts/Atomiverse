# NVIDIA library ownership in CUDA classical MD

The review scope is the classical MD CUDA implementation and its support paths:
`dynamics/`, `potentials/`, `neighbor/`, `longrange/`, `communication/`,
`device/`, `frame/`, `driver/`, classical observers, and asynchronous recording.
Electronic structure is outside the review, except for the RHF/UHF call sites
migrated when removing the shared handwritten block reductions. Minimizer
consumers of those same helpers also call CUB directly.

| Operation | Canonical implementation | Call sites |
| --- | --- | --- |
| Energy, virial, kinetic stress, momentum/mass and convergence sums/maxima | CUB `BlockReduce` | Dynamics, pair/bonded forces, PME/Ewald, minimizers; RHF/UHF nuclear energy |
| Component-wise sum operator | `cuda::SumComponents` in `device/launch.cuh` | CUB payloads with independent components; not a parallel algorithm or a forwarding wrapper |
| Cluster AABB/min-max, displacement max, packed column OR | Cooperative Groups `reduce` | Neighbor search/sort and fused LJ+Coulomb |
| Spatial/tag ordering, migration destination grouping | CUB `DeviceRadixSort` | Neighbor sorting and `MultiGpu` |
| CSR offsets | CUB `DeviceScan` | Cluster hierarchy and packed tiles |
| Interior/boundary partition | CUB `DevicePartition` | `MultiGpu` |
| Destination histogram | CUB `DeviceHistogram` | `MultiGpu::migrate` |
| Halo faces, leavers, holes, tail keepers | CUB `DeviceSelect` | `MultiGpu` |
| Generic permutation, inverse permutation, indices, fixed-field communication packing | Thrust gather/scatter/sequence/copy with zip iterators | `atom_cell_sort.cu`, `MultiGpu` |
| Sorted exclusion/tag/CSR lookup | Thrust binary_search/lower_bound/upper_bound with device `seq` | Pair exclusions, mask building, bonded rematerialization |
| Gaussian velocities and stochastic kicks | cuRAND Philox device API | InitVelocity, Brownian, FixLangevin |
| Real FFT and half spectrum | cuFFT D2Z/Z2D or R2C/C2R | PME |
| Zero/copy/peer transfer/events/graphs | CUDA Runtime | Initializers, Frame, driver, communication and asynchronous snapshots |

Philox takes the user seed, the full global tag as subsequence and an offset
of `domain * 2^62 + 8 * step`: two `curand_normal2_double` calls consume eight
32-bit outputs. Domains 0/1/2 belong to initialization/Brownian/Langevin;
identical user seeds therefore cannot reuse samples across components. The
32-bit MD step index fits inside each domain's window.
State is local to each invocation, so atom reordering and migration do not move
an RNG state buffer. Trajectories change relative to the retired SplitMix/Box–Muller
and XORWOW paths; the public statistical and fluctuation-dissipation contracts
are unchanged. Initial velocity draws are tagged too.

CUB compile-time block dimensions must match each kernel launch: dihedrals and
fused force tiles use 128 threads; ordinary reduction kernels use 256. Shared
CUB storage must be synchronized before reuse by another collective. Arrays of
components use one CUB reduction with a component-wise sum, not CUB's array
input overload (which would sum the components together).

The unused centroid grid/histogram/stencil setup and abandoned standalone
AoS conversion/scatter variants have been removed. The active neighbor source
of truth is the Hilbert cluster hierarchy with exact geometric masks.

## Domain kernels retained

- LJ, screened Coulomb, bonded forces, thermostat/integrator equations,
  PME charge spreading/interpolation and reciprocal influence: physics and
  sparse stencils, not BLAS/FFT/reduction replacements. Per-thread scalar
  accumulation inside a force expression stays fused with that expression.
- Hilbert encoding, AABB distance, periodic images, tiny 3x3 inverse and owner
  geometry: no semantics-matching CUDA Toolkit primitive. A cuSOLVER call for
  one cell inverse would require a different lifecycle and kernel boundary.
- Packed-mask expansion, CSR tile construction, topology rematerialization,
  variable-width topology packing and force scatter-add: variable output structure,
  duplicate force destinations or layout contracts. Their generic selection,
  lookup and reduction substeps use the libraries above. CUDA atomics, ballots,
  popcount and lane-to-lane broadcasts are NVIDIA primitives already.
- Device-to-device peer copies implement the existing local multi-GPU transport;
  adding NCCL would not eliminate the topology-dependent packing or replace a
  missing collective in this transport. Multi-GPU Ewald structure-factor
  collectives remain outside the existing single-GPU Ewald product envelope.
- Host bookkeeping uses the C++ standard library. Classical trajectory analysis
  remains owned by molrs, as required by the repository architecture.

No new third-party dependency is needed: these NVIDIA APIs ship with the CUDA
Toolkit. Selection/histogram/sort temporary buffers in migration use the existing
stream-ordered pool; step-time force/integrator CUB storage is static shared
memory. PME retains its build-captured CUDA graph.

## Contracts checked by the unit fixtures

`test_cuda_md_primitives` exercises nondefault streams, partial final warps and
blocks, all seven distinct kinetic/stress sums, initialization temperature/COM,
last-block thermostat scaling, high tag bits, permutation stability, noise
variance/cross-correlations across axes/tags/steps/seeds, and independent
component counter domains and initial-force/first-step Langevin samples. Tags in adjacent pairs deliberately share their low
32 bits, so truncating identity would produce correlated draws.

The existing multi-GPU fixture now reads coordinates in `cuda::real_t`, not
hardcoded float. CPU MD and CUDA MD use Plan::prepare_forces for initial force/virial preparation
before integration. NPT parity starts from raw Frames without external force
preparation. For mixed storage the fixture starts from
identically representable cell/position inputs and compares the final CPU cell
projected to the CUDA storage type. Its original 1e-8/1e-6 residual bounds remain
unchanged; the test does not ask a float cell to retain unrepresentable FP64 bits.

NVIDIA API contracts used here: [CUB block reductions](https://nvidia.github.io/cccl/unstable/cub/api/classcub_1_1BlockReduce.html),
[Cooperative Groups reductions](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html),
and [cuRAND device API](https://docs.nvidia.com/cuda/archive/13.0.2/curand/group__DEVICE.html).

## Verification coverage

GH200 with CUDA 13.0, `ATV_USE_MPI=OFF`, double and mixed storage/work modes:

- Twelve CTest targets per mode: MD primitives, zero momentum, NPT, neighbor
  skin gating, cluster contract/masks, atom cell sort, tile step-zero,
  harmonic coefficients, packed tiles, special-bond LJ and multi-GPU components.
- Five regression executables per mode: `reg_cuda-gaff-fragment`,
  `reg_cuda-pme-half-spectrum`, `reg_cuda-ewald-nacl`, `reg_cuda-pme-nacl`,
  `reg_cuda-multi-gpu-trajectory` (including ownership migration and bonded
  topology preservation).
- Four visible GPUs: all sixteen `test_cuda_multi_gpu` cases pass in both
  modes, including the 2x2 decomposition; the trajectory regression passes too.
- Compute Sanitizer memcheck/racecheck on the ten focused MD primitive cases,
  and synccheck on the cluster-mask cases: zero errors/hazards.
- RHF/UHF Fock translation units compile with `ATV_HAS_HF` after their shared
  helper migration. Full electronic-structure runtime validation is outside
  this classical MD review.
- Host-only Brownian tests, `git diff --check` and clang-format checks pass.

CPU MD initial force preparation is part of Runtime, including `run(frame, 0)`.
NPT parity needs only MD; users do not prepare initial forces or stress through
Evaluate. Force preparation preserves bias/electronic/electrode propagation
history, skips integration and ordinary observers, and executes required
pre/post-force and scalar-finalization hooks. Chunked runs reuse current forces;
a new Frame starts with a new force evaluation.

CPU validation includes the analytic harmonic-bond first kick, `run(0)`, stale
input force overwrite, chunking and a new Frame; runtime setup order and retry
state; metadynamics hill/history preservation; eight-rank NPT; AO CPMD and
XL-BOMD; and GPW CPMD and second-generation CPMD. The OpenMPI no-partner marker
is accepted independently of the MPI-free halo marker, so the CPU MPI build
and NPT validation run with the local OpenMPI transport.

## Shared lifecycle and scalar MD

`core/plan.h` owns unique-instance lifecycle enumeration and the initial-force
schedule. CPU Runtime and CUDA Backend share its lifecycle ledger, including
nested-plan references and failed-build rollback. Unmounting removes every
binding of one instance; distinct instances sharing a name remain ambiguous.

`core/dynamics_math.h` owns the NHC half-step, thermal DOF formula, pressure
normalization, Berendsen dilation and full-width temperature ramp. CUDA NPT
history uses Thrust device vectors and launch contains only device work; its
capture unit exercises that contract. Initial velocities reuse the CUB kinetic
reduction and cuBLAS scaling to reach the target temperature after COM removal.

The backend publishes a device Box cache after declared box mutations. Neighbor
skin checks compare the device cell to the rebuild snapshot; force/integrator
kernels resolve the published cell at entry. The XYZ snapshot freezes its own
cell alongside positions, avoiding a later mutable host-box read.

Full trajectory/Frame ownership and device-controlled MD looping are still
tracked in `.claude/specs/md-unification.md`; these scalar/lifecycle changes do
not establish those contracts.

The MTK atom-count denominator is independent of thermal DOF, including when
constraints remove degrees of freedom. The analytical initial-propagation unit
covers 0/1/2 constraints. CPU InitVelocity removes the global COM under spatial
MPI decomposition and uses distinct standard-library rank streams; its 2/4-rank
single-owned-atom units establish nonzero thermal motion and exact temperature.
This CPU initializer is statistically, rather than bitwise, invariant to rank
count. CUDA noise retains its global-tag/step identities.

Current device/scalar acceptance: GH200 molq 4173953, double/mixed ten targets
per mode, NPT 80-step and 1/2-step regressions, initial-force regression,
primitive memcheck, NPT launch-capture racecheck. All passed; sanitizer errors
zero; mixed NPT relative volume difference 9.2962e-8. Local MD-only suite:
135 tests passed. MPI initializer: 2/4 ranks passed. Full CPU/Python builds and
three AIMD wiring checks passed. Full device-controlled MD/multi-GPU execution,
Frame editing/model ownership and throughput/profiling acceptance remain open.

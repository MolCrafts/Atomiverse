# NVIDIA library ownership in CUDA classical MD

Rule: docs/developer/project-contract.md › CUDA classical MD. Scope: `cuda/` dynamics, potentials,
neighbor, longrange, communication, device, frame, driver, classical observers,
asynchronous recording (plus the RHF/UHF and minimizer call sites of the same
reductions).

| Operation | Canonical implementation | Call sites |
| --- | --- | --- |
| Energy, virial, kinetic stress, momentum/mass, convergence sums/maxima | CUB `BlockReduce` | dynamics, pair/bonded, PME/Ewald, minimizers; RHF/UHF nuclear energy |
| Component-wise sum operator | `cuda::SumComponents` (`device/launch.cuh`) | CUB payloads with independent components |
| Cluster AABB/min-max, displacement max, packed column OR | Cooperative Groups `reduce` | neighbor search/sort, fused LJ+Coulomb |
| Spatial/tag ordering, migration grouping | CUB `DeviceRadixSort` | neighbor sort, `MultiGpu` |
| CSR offsets | CUB `DeviceScan` | cluster hierarchy, packed tiles |
| Interior/boundary partition, histogram, halo/leaver/hole selection | CUB `DevicePartition` / `DeviceHistogram` / `DeviceSelect` | `MultiGpu` |
| Permutation, inverse permutation, indices, fixed-field packing | Thrust gather/scatter/sequence/copy (zip iterators) | `atom_cell_sort.cu`, `MultiGpu` |
| Sorted exclusion/tag/CSR lookup | Thrust binary_search/lower_bound/upper_bound (device `seq`) | pair exclusions, masks, bonded rematerialization |
| Gaussian velocities, stochastic kicks | cuRAND Philox device API | InitVelocity, Brownian, FixLangevin |
| Real FFT, half spectrum | cuFFT D2Z/Z2D or R2C/C2R | PME |
| Zero/copy/peer transfer/events/graphs | CUDA Runtime | initializers, Frame, driver, comm, async snapshots |

## Contracts

- **Philox:** user seed, full global tag as subsequence, offset
  `domain * 2^62 + 8 * step` (two `curand_normal2_double` = eight 32-bit
  outputs per step). Domains 0/1/2 = initialization / Brownian / Langevin, so
  identical seeds never reuse samples across components; `BigInt` steps fit the
  window below 2^59. State is per invocation, so reordering/migration move no
  RNG buffer.
- **CUB block dims** must match the launch: dihedrals and fused force tiles use
  128 threads, ordinary reductions 256. Synchronize shared CUB storage before
  reuse. Arrays of components use one reduction with a component-wise sum, not
  CUB's array overload (which sums components together).
- Migration temporaries use the stream-ordered pool; step-time force/integrator
  CUB storage is static shared memory; PME keeps its build-captured CUDA graph.
- Neighbor source of truth: Hilbert cluster hierarchy with exact geometric masks.
- No new third-party dependency: all APIs ship with the CUDA Toolkit.

## Domain kernels retained (no matching library primitive)

- LJ, screened Coulomb, bonded forces, integrator/thermostat equations, PME
  spreading/interpolation and reciprocal influence; per-thread accumulation
  stays fused with its force expression.
- Hilbert encoding, AABB distance, periodic images, tiny 3×3 inverse, owner
  geometry.
- Packed-mask expansion, CSR tile construction, topology rematerialization,
  variable-width topology packing, force scatter-add (their generic substeps use
  the libraries above).
- Peer copies for local multi-GPU transport (NCCL would not remove the
  topology-dependent packing). Host bookkeeping uses the C++ standard library;
  trajectory analysis stays in molpy.

## Scalar MD and lifecycle (CUDA side)

- `core/dynamics_math.h` (NHC half-step, thermal DOF, pressure normalization,
  Berendsen dilation, temperature ramp) is shared with CPU. The MTK atom-count
  denominator is independent of thermal DOF under constraints.
- CUDA NPT history is device-resident (Thrust vectors); `launch` contains only
  device work (capture unit test). Initial velocities reuse the CUB kinetic
  reduction and cuBLAS scaling after COM removal.
- Backend publishes a device Box cache after declared box mutations; neighbor
  skin checks compare against the rebuild snapshot; kernels resolve the
  published cell at entry; the XYZ snapshot freezes its own cell.
- Lifecycle and initial forces run through core `Plan::prepare_forces` +
  `detail::ComponentLifecycle`, used by CUDA only (CPU has its own Runtime
  ledger) — docs/developer/project-contract.md › Known debt; plan: `.claude/specs/md-unification.md`.
- Unit fixture `test_cuda_md_primitives` covers nondefault streams, partial
  warps/blocks, all kinetic/stress sums, temperature/COM init, high tag bits
  (adjacent tags share low 32 bits on purpose), noise variance and
  cross-correlation across axes/tags/steps/seeds/domains.

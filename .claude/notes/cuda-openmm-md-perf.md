# CUDA GAFF / OpenMM performance methodology

Harness: `tools/bench_gaff_openmm/`, public consumer
`benchmarks/cuda/bench_gaff.cu`. Run only in an Arrhenius GPU batch
allocation; ten-second deadline per engine process; three short samples,
report the median. Build, installation and the instrumented profile are
excluded from throughput.

## Matched settings

- One GH200, same coordinates/topology/parameters, unconstrained NVE, 0.25 fs,
  zero full-step velocities, cutoff 9 Å, unshifted Lorentz–Berthelot LJ without
  tails, α 0.35 Å⁻¹, order-5 PME; mesh and precision explicit.
- OpenMM Verlet velocities are leapfrog half-step values: initialize them one
  half-kick behind the zero full-step velocity. ATV evaluates the initial force
  before its first half-kick.
- Neighbor padding matches OpenMM's 8 % of cutoff (do not shrink the force
  cutoff instead).

## Force path

- `PairLJCutCoulLong` fuses LJ and screened Coulomb, gathers live sorted atom
  parameters once, and spreads upper-triangle tiles across warps. At rebuild,
  nonempty J columns are packed across cluster boundaries with exact baked row
  masks; a globally unique rebuild generation invalidates the packed view even
  when the provider is replaced.
- The parameter gather computes an I-cluster center and a conservative
  single-image guard; tight orthorhombic clusters map positions before the pair
  loop, wide or skew clusters keep per-pair minimum images.
- Newton updates add 64-bit fixed-point words (OpenMM resolution/range);
  wraparound is checked after the block energy reduction. Energy/virial
  producers write disjoint build-reserved partials, reduced in a second pass.
- Fully excluded bonded pairs are baked into masks; a sparse pass restores Amber
  special interactions and removes their reciprocal part. Fourier terms with
  phase 0/π and n = 1–4 map exactly to `DihedralMultiHarmonic`.
- Library special functions stay in both precisions (a screened polynomial gave
  no tile-throughput gain). Never relax physics checks to accept a perf change.

## Neighbor build

- Directed cluster CSR is the source of truth. Hilbert-cluster AABBs can be much
  wider than the cutoff, so a centroid-cell stencil is not conservative; grouped
  AABB rejection then exact count/scan/fill covers wide clusters.
- All scratch and CSR capacity is reserved in build (default ≥ 2× the initial
  exact count with density headroom; explicit reserve overrides). Overflow
  throws before fill. Multi-GPU headroom is capacity: count actual owned+ghost
  rows.
- The skin gate checks current minimum-image displacements before force
  evaluation (a previous-step flag misses pairs). Mapped block flags avoid the
  per-step clear and copy; the host reads after the stream event. Cell changes
  force a rebuild. The decision still needs a sync — never hide it behind an
  unsafe delayed check. Rebuild-count probes use build-owned pinned scalars.

## PME / Ewald

- R2C/C2R transforms with a half spectrum; spline/FFT precision follows
  `force_t`; common spline orders specialized; energy/virial reductions FP64.
- Influence and virial coefficients are device tables keyed on `Box::revision`,
  refreshed inside the captured graph when the published cell changes; mesh
  fixed at build; atom set fixed per build (a change throws).
- Full-index conjugate averaging keeps full-grid energy and the real inverse,
  triclinic Nyquist modes included; only interior kz modes weigh two. A
  closed-polynomial spline and an explicit Fourier sum check energy and gradient
  on even/odd orthorhombic and skew meshes.
- Charge spreading assigns one stencil-z position to each of `order` adjacent
  threads per atom (coalesced writes, atomics per thread order² not order³).
- `KSpaceEwald` follows the cell the same way: integer mode set fixed at build;
  wave vectors, weights, volume and six-component virial refresh on revision.

## Reference data and acceptance

- Use an equilibrated GAFF PEO snapshot: the Arrhenius 400 K checkpoint
  `tg-proj/gaff_1atm_gpu_20261015/peo/r01/` (1,840 atoms; 3×3×3 = 49,680).
  Initial energy and every force component must pass before timings count.
- Raw `gaff-ready/cells/peo/r01/system.data` has severe overlaps; OpenMM CUDA
  fixed-point accumulation overflows above ~2.2e6 eV/Å while OpenMM Reference
  agrees with ATV. Never clamp ATV forces to match an invalid CUDA reference.
- Double compares double everywhere. Mixed inputs use a lattice exactly
  representable in float32 Å and nm (`5*k*2^e` Å, `|5*k| < 2^24`), spacing
  recorded. Both engines' mixed modes keep an FP32 position correction, FP64
  velocities and fixed-point forces.
- NVE drift (`run_drift.sh`, 1,840 atoms, six seeds, 4 ps at 0.25/0.125 fs):
  slopes do not shrink with dt in either engine (unshifted cutoff), so accept
  when ATV's mean slope is within two pooled standard errors of OpenMM's and the
  fluctuation RMS within 20 %.
- NaCl Ewald/PME regressions keep their literature tolerances; the eight-atom
  fragment pins the force pass to OpenMM CUDA.

OpenMM architecture reference: SHA `3a2326d0dd091229eaa4c0d7b5ad1001d5fe376e`
(cluster bounds, fused warp parameter exchange, packed J columns, exception
separation, real FFTs, compile-time PME order). ATV implements these in its own
Frame/Component/CSR structure; no OpenMM source was copied.

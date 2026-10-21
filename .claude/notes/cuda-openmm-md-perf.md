# CUDA GAFF / OpenMM performance methodology

The standing harness is `tools/bench_gaff_openmm/`, with public consumer
`benchmarks/cuda/bench_gaff.cu`. Run experiments only in an Arrhenius GPU batch
allocation. Each engine process has a ten-second deadline; use three short
samples and report their median. Build time, package installation and the
separate instrumented profile are excluded from throughput.

Both engines use one GH200, the same initial coordinates/topology/parameters,
unconstrained NVE, 0.25 fs, zero full-step velocities, cutoff 9 Å, unshifted
Lorentz–Berthelot LJ without tails, alpha 0.35 Å⁻¹ and order-5 PME. Match mesh
and precision explicitly. OpenMM Verlet velocities are leapfrog half-step
values, so initialize them one half-kick behind the zero full-step velocity.
ATV must evaluate the initial force before its first velocity-Verlet half-kick.

## Force path

`PairLJCutCoulLong` fuses LJ and screened Coulomb, gathers live sorted atom
parameters once, and distributes upper-triangle tiles across warps. At neighbor
rebuild, nonempty J columns are packed across the original cluster boundaries,
retaining exact baked row masks. A globally unique rebuild generation invalidates
the private packed view even when a provider is replaced. The live parameter
gather computes an I-cluster center and a conservative single-image guard once
per cluster. Tight orthorhombic clusters then map I/J positions before the
pair loop; wide or skew clusters retain per-pair minimum images. Newton
updates use floating atomics to preserve dynamic range. Energy and virial
producers write disjoint build-reserved partials; a second GPU reduction
separates the accumulator from the producers. This reduction alone did not
materially improve measured throughput. Fully excluded bonded
pairs are baked into geometric masks; a sparse pass restores Amber special
interactions and removes their reciprocal contribution. Fourier terms with
phase 0/pi and multiplicity 1–4 map exactly to `DihedralMultiHarmonic`.
CUDA CHARMM torsions and impropers remain outside this harness.

The directed cluster CSR remains the neighbor source of truth. Hilbert-cluster
AABBs can be much wider than the atom cutoff, so a centroid-cell stencil is
not conservative. Grouped AABB rejection followed by exact count/scan/fill
covers wide clusters and partial groups. A standalone single-GPU LJ force
check agrees with brute force to 3.6e-15 eV/Å on its fixture. Old throughput
claims made with the incomplete centroid stencil are not validated evidence.

All scratch and CSR capacity are reserved in build. The default reserves at
least twice the initial exact count, with density headroom; an explicit
constructor reserve overrides it. Overflow throws before fill. Multi-GPU
headroom is capacity, not initialized atom data: initial counting must use
actual owned+ghost rows.

The skin gate checks current minimum-image displacements before force
evaluation. A previous-step flag leaves an unbounded extra drift and can miss
pairs. Mapped block flags remove the per-step clear and copy-engine transfer;
the host reads only after the stream event. Cell changes also force a rebuild.
This host decision still requires synchronization; never hide its cost behind
an unsafe delayed check.
Rebuild count probes use build-owned pinned host scalars, avoiding pageable
stack destinations in the checked asynchronous readbacks. The benchmark matches
OpenMM's 8%-of-cutoff neighbor padding rather than selecting a smaller force cutoff.

PME uses real-to-complex / complex-to-real transforms and a half spectrum.
Spline/FFT precision follows CUDA work precision, specializes common
spline orders and keeps energy reductions in FP64. Fixed-cell influence
coefficients are prepared in build. Full-index conjugate averaging preserves
the full-grid energy and the real inverse, including triclinic Nyquist modes;
only interior kz modes have energy weight two. A closed-polynomial spline and
explicit full Fourier sum independently check energy and its analytic gradient
on even/odd orthorhombic and skew meshes. Its resident pipeline is captured in build and
replayed as a CUDA graph. Plans, workspaces and graph resources belong to the
build/destroy lifecycle. The current PME solver requires a fixed cell and atom
set; changing either must fail explicitly rather than reuse stale geometry.
Charge spreading assigns one stencil-z position to each of `order` adjacent
threads per atom, coalescing grid writes and reducing serial atomics per thread
from order cubed to order squared without changing the stencil.

Reciprocal-force interpolation uses eight adjacent lanes per atom for up to
16,384 atoms and one lane per atom above that size. Small systems benefit
from exposing more stencil parallelism; larger systems avoid duplicated
spline work. Both paths use the same stencil and force normalization.
CSR exclusion baking assigns a warp per tile, coalesces the column indices
and shares them across row lanes before writing the existing masks.
The packed force loop trusts those validated masks to exclude padding and
self pairs, avoiding a redundant J-index shuffle and predicates per pair.

## Reference data and acceptance

Use an equilibrated GAFF PEO snapshot, not the unrelaxed packing. The Arrhenius
400 K checkpoint under `tg-proj/gaff_1atm_gpu_20261015/peo/r01/` has 1,840 atoms;
3×3×3 replication has 49,680 atoms. Initial energy and every force component
must pass before accepting timings. The checked-in eight-atom fragment also
pins the public complete force pass to independently generated OpenMM CUDA
values. NaCl Ewald and PME regressions retain their literature tolerances.

Raw `gaff-ready/cells/peo/r01/system.data` contains severe overlaps. OpenMM CUDA
uses fixed-point force accumulation and can overflow for forces above about
2.2e6 eV/Å. OpenMM Reference agrees with ATV's unclipped nonbonded forces there
(maximum error 1.7e-6 eV/Å on forces around 2.8e7 eV/Å). Do not clamp ATV forces
to make that invalid CUDA reference pass.

Double compares double storage/work/accumulation on both engines. Mixed
inputs (coordinates and cell) use a binary lattice exactly representable
in both float32 Å and float32 nm: `5*k*2^e` Å, with `|5*k| < 2^24`.
Its spacing is chosen from the largest coordinate/cell magnitude and recorded.
This avoids an additional unit-conversion rounding in the initial static
comparison. Mixed CUDA motion uses a float coordinate plus residual, double
velocity/mass/integration and a canonical double cell. Force/FFT work stays
float and energy reductions stay double. Import/export, every position writer,
permutation, communication and async coordinate snapshots must preserve the
residual. Force geometry retains the main float coordinate, matching OpenMM's
force-position tier. ATV floating-point force atomics still differ from
OpenMM's fixed-point force buffer; this is not bitwise trajectory parity.
PME's fixed-cell guard compares the canonical motion cell, so a small box
change cannot disappear in the force-box float cache.

Throughput and resource sampling use separate engine runs/output destinations.
No memory monitor is active during accepted throughput samples. Plots require
one binary per precision, matching protocols and physics validation. Report
observed GPU peaks (including context, 100-ms PID-filtered NVML samples) and
host process VmHWM with their sampling limitations.

The pair pass retains library special functions in both precisions. A screened
polynomial did not improve measured tile throughput, so a less accurate
approximation adds no value to this kernel. Do not relax the physics checks
to accept a performance variation.

OpenMM architecture reference: source SHA
`3a2326d0dd091229eaa4c0d7b5ad1001d5fe376e`, cluster bounds, fused warp parameter
exchange, packed J columns, exception separation, real FFTs and compile-time PME stencil order. ATV code
implements these algorithms in its own Frame/Component/CSR structure; OpenMM
source was not copied.

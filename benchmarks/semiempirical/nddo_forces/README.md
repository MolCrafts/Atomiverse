# NDDO complete force-call benchmark

The current schema-2 harnesses cover AM1, RM1, MNDO and PM3 at 3/12/24/48
atoms (1/4/8/16 distorted waters). Native CUDA additionally accepts an explicit
96-atom selection; it is outside the default set. All use an isolated zero
cell, neutral closed shell, FP64, `max_iter=1000`, density tolerance `1e-10`
and energy tolerance `1e-12 eV`. Both residuals must converge.

The common water geometry is O `(0,0,0)`, H `(.96,0,0)`, H `(-.24,.93,.1)`,
with 4 Å grid spacing and masses 15.999/1.008/1.008. Each warm mode starts with
a fresh method and five evaluations at the original geometry, followed by
20 measured calls. Fixed mode preserves coordinates. Moving mode sets, from
the original geometry at step `s=repeat+1`, each water's H1.x offset to
`.002*sin(.2*s)`, H2.y to `.002*sin(.17*s)`, and H2.z to `.001*sin(.11*s)` Å.
This is a prescribed sequence, not integrated BOMD throughput.

Run serially on reserved resources without competing builds/tests; retain
source revision and dirty patch/archive, numerical-source hashes, build cache,
exact executable/module hashes, resource request and raw JSON. Use alternating
before/after unprofiled runs for latency. Raw `results/` files are local evidence,
not shipped datasets; benchmark outputs are consistency checks, not oracles.

## CPU allocation and Python boundaries

```sh
cmake --build "$BUILD" --target bench_nddo_allocations
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ATV_LOG_LEVEL=off \
  "$BUILD/benchmarks/bench_nddo_allocations" /tmp/nddo-allocations.json
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTHONPATH="$BUILD:$PWD/python/src" \
  python benchmarks/semiempirical/nddo_forces/run.py /tmp/nddo-python-forces.json
```

Both commands accept an optional uppercase `AM1|RM1|MNDO|PM3` selector after
the output path. Build allocations with `ATV_BUILD_BENCHMARKS=ON` and
`ATV_BUILD_SEMIEMPIRICAL=ON` in a static, non-MPI Release build. Python requires
the existing atomiverse/molpy environment. Five cold calls each include fresh
method construction plus `forces`; Frame creation, geometry edits, destruction,
validation and report serialization are outside each measurement. Warm modes
have independent method histories. Every measured call records iterations,
E/HOF, charges and the complete flat Cartesian force vector.

`count_allocations.cpp` counts successful C++ `new`/`new[]` calls, including
aligned/nothrow forms; its positive control must observe 8 calls/480 bytes.
It excludes direct `malloc/calloc/realloc`, including library calls using those
APIs. Bytes mean cumulative requested traffic, not live/peak memory. Separate
primitive rows warm once then measure 1,000 Hcore/ERI derivative calls for
H–H, C–C and N–O under every method, plus AM1 Br–I, in tilted and axial
geometries (52 rows). Parameter lookup, pair setup and logging are excluded.
A zero primitive count says nothing about complete solves or CUDA residency.

Run allocation counting separately from latency: its instrumentation changes
costs. `run.py` times the full Python/Frame boundary, including conversion and
force-return copies, and records the extension hash. Its peak RSS covers the
whole Python process/imports/all cases, not a per-method peak. Native CPU/CUDA
latency comparisons use the next harness rather than Python timings.

### Verified CPU density reuse (2026-10-25)

The local schema-2 comparison contains 720 complete calls across all four
methods, sizes and cold/fixed/moving modes. Every E/HOF/q/full-force value and
SCF iteration count is exactly unchanged. All 52 derivative rows are unchanged
and observe zero C++ allocations. Every complete call saves exactly
`2 * iterations` allocations and `16 * packed_size * iterations` requested
bytes, where `packed_size = nAO*(nAO+1)/2` and `nAO=6*waters`.

The change fills the existing spin-density buffer with the next density,
computes the unchanged residual, swaps buffers and overwrites the previous
density with half of the new one. It adds no third persistent packed matrix.
These are measured allocation savings, not a latency or peak-memory claim.
Evidence: `/tmp/atv-sqm-evidence/force-perf-20261025/`
`cpu-alloc-before.json`, `cpu-alloc-after.json` and
`cpu-allocation-comparison.txt`; the same directory retains the before binary
and source archive. Subsequent native timing/trace results require their own
recorded revision and measurements.

## Native CPU/CUDA force timing

`run.cu` builds as `bench_nddo_forces`, requiring semiempirical, CUDA, Evaluate
and benchmarks enabled, with CUDA/storage/work precision all `double`.
Use the repository GH200 toolchain; HF/PW/GPW are not prerequisites.

```sh
cmake --build "$BUILD" --target bench_nddo_forces
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ATV_LOG_LEVEL=off \
  "$BUILD/bin/bench_nddo_forces" "$RAW/forces.json" all 0 20 0 all
```

Arguments are `OUTPUT.json [am1|rm1|mndo|pm3|all] [waters:0|1|4|8|16|32]`
`[warm:1..100] [profile:0|1] [fixed|moving|all]`; defaults are `all 0 20 0 all`.
Water count zero selects the four default sizes. A profile requires one method,
one nonzero size and one geometry mode.

- `host_frame_energy_charge_force`: native CPU and CUDA each measure five
  cold constructor-plus-force calls and the independent warm series. Completion
  and host E/q/F publication are included; Frame setup, edits, destruction and
  result copies into the report are excluded. CUDA context initialization is
  recorded separately; cold means a fresh method, not a fresh process/context.
- `resident_evaluate_force`: a benchmark-only Evaluate author probe measures
  complete mounted steps using one device Frame. Every sample ends with a
  checked parent-stream synchronization, including completion overhead.
  Moving geometry is written by a device component inside the timed step.
  Setup/build, five warm-up steps, teardown and result downloads are excluded.
  Per-step times/iterations are retained, but full E/q/F is downloaded and
  validated **only at the final point** after the range.

Standalone CPU/CUDA results are checked for every cold/warm call at `1e-6 eV`
energy, `1e-7 e` maximum charge difference and `1e-5 eV/Å` maximum force-component
difference. Resident final results use the same gates; CPU/CUDA iteration counts
may differ. `cudaMemGetInfo` reports named whole-device free-memory snapshots,
not peak allocation or ownership. Process peak RSS likewise spans the entire
benchmark. No small-molecule GPU speedup is presumed.

For a controlled revision comparison, preserve before/after binaries built
with the same final harness. Run them serially in before/after/after/before/
before/after order on one allocation, using `all 0 20 0 all`; save outputs as
`1-paired-before.json`, `2-after.json`, `3-after.json`, `4-paired-before.json`,
`5-paired-before.json` and `6-after.json` in a paired directory. Then run:

```sh
python benchmarks/semiempirical/nddo_forces/analyze.py --paired "$PAIRED" \
  > "$PAIRED/summary.json"
```

This checks matching metadata, complete call counts and vectors, exact CPU
outputs/iterations for the density-buffer change, and CUDA agreement at the
stated tolerances. It retains each round's iteration sequence and median,
pooled timings and adjacent-pair ratios. A ratio alone does not distinguish
iteration changes from changes in per-iteration cost.

## Full-process residency trace and analysis

Keep ordinary timing in `forces.json`; collect traces in separate executions.
Capture the **whole process**, including setup and natural exit, while the
benchmark's profiler API markers delimit only the resident warm loop:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ATV_LOG_LEVEL=off \
  nsys profile --trace=cuda --sample=none --cuda-memory-usage=true \
    --cuda-trace-all-apis=true --cuda-um-cpu-page-faults=true \
    --cuda-um-gpu-page-faults=true \
    --output="$RAW/am1-48-moving-full" \
    "$BUILD/bin/bench_nddo_forces" "$RAW/am1-48-moving-full-profile.json" \
    am1 16 20 1 moving
nsys export --type=sqlite --output="$RAW/am1-48-moving-full.sqlite" \
  "$RAW/am1-48-moving-full.nsys-rep"
python benchmarks/semiempirical/nddo_forces/analyze.py "$RAW" > "$RAW/summary.json"
```

Set `RAW` to an existing evidence directory. Repeat method/mode selections as
needed. Do not substitute range-only capture: allocation lifetimes and the
closing marker require complete collection. Archive `provenance.json` with
actual revision/patch, node/resource request and collection commands, plus
`executable.sha256`, `CMakeCache.txt` and `numerical-sources.sha256`; the analyzer
requires these alongside `forces.json`. Each `*-full.sqlite` needs its matching
`*-full-profile.json` and `.nsys-rep`.

`analyze.py` reuses the energy harness's trace-window queries, adds memory-event
accounting, checks profiled final results against the ordinary run, and hashes
the evidence. Inspect warnings, API/transfer activity and memory events before
claiming residency or allocation behavior. Observed logical GPU allocations
are distinct from driver reservations/complete VRAM use; runtime and driver
API views overlap and must not be summed. Without allocation stacks, allocator
ownership is not established. CUDA traces do not measure host malloc/new.
Profiled timings include overhead and are excluded from latency summaries.

## GH200 four-method force acceptance (2026-10-25)

Measured against base `8e9b2daad` with the same final harness and only the
CPU density-buffer change in production. Six serial ABBAAB executions ran on
GH200 node `n481`: one GPU, eight allocated CPUs, one CPU/OpenBLAS thread,
Release GCC 14.3.0, CUDA 13.0, FP64. The executable SHA256s are
`87a61bf584ef7f1c8ce55a2b3519af9d19bb01030a6ac1677d78eee4b256ebff`
(before) and `acca376dda44a9ba5e984626fb4c5db7ef57654aba7b7101cb88dde723b9d21d`
(after). Current numerical sources match the measured after manifest exactly.
Raw binaries, manifests, six JSONs, captures and summaries remain in the local
`/tmp/atv-sqm-evidence/force-perf-20261025/` evidence directory.

The table gives pooled after medians in **milliseconds**, not speedup claims.
C/G/R mean CPU public force / CUDA public force / resident Evaluate force.
Cold is fresh method construction in an already initialized process; resident
setup has no cold timing. Moving SCF medians retain differing backend work;
all per-call sequences, including fixed-state iterations, remain in the JSONs.

| Method | Atoms | Cold C / G (ms) | Fixed C / G / R (ms) | Moving C / G / R (ms) | Moving SCF median C / G / R |
| --- | ---: | --- | --- | --- | --- |
| AM1 | 3 | 0.209 / 4.104 | 0.063 / 0.394 / 0.381 | 0.152 / 2.533 / 2.523 | 23.0 / 23.0 / 23.0 |
| AM1 | 12 | 2.316 / 8.045 | 0.457 / 0.786 / 0.773 | 3.037 / 5.922 / 5.894 | 23.0 / 23.0 / 23.0 |
| AM1 | 24 | 10.016 / 20.045 | 2.423 / 1.632 / 2.220 | 8.311 / 15.358 / 15.396 | 23.0 / 23.0 / 23.0 |
| AM1 | 48 | 55.297 / 38.635 | 13.660 / 2.968 / 2.967 | 40.736 / 29.790 / 29.801 | 28.5 / 23.0 / 23.0 |
| RM1 | 3 | 0.221 / 4.382 | 0.063 / 0.395 / 0.380 | 0.151 / 2.670 / 2.642 | 24.0 / 24.0 / 24.0 |
| RM1 | 12 | 2.516 / 8.942 | 0.456 / 0.799 / 0.779 | 1.955 / 6.129 / 6.076 | 24.0 / 24.0 / 24.0 |
| RM1 | 24 | 11.068 / 21.748 | 2.112 / 1.615 / 1.919 | 9.101 / 15.949 / 15.997 | 25.0 / 24.0 / 24.0 |
| RM1 | 48 | 55.892 / 44.556 | 12.387 / 3.574 / 4.157 | 40.071 / 31.692 / 31.753 | 28.0 / 24.0 / 24.0 |
| MNDO | 3 | 0.201 / 3.981 | 0.065 / 0.395 / 0.381 | 0.145 / 2.450 / 2.473 | 22.0 / 22.0 / 22.0 |
| MNDO | 12 | 2.219 / 7.868 | 0.451 / 0.786 / 0.770 | 1.891 / 5.727 / 5.688 | 22.5 / 22.0 / 22.0 |
| MNDO | 24 | 9.989 / 19.007 | 2.406 / 1.905 / 2.178 | 8.222 / 14.604 / 14.655 | 23.0 / 22.0 / 22.0 |
| MNDO | 48 | 43.562 / 36.777 | 11.059 / 2.951 / 2.936 | 39.328 / 29.801 / 29.462 | 27.5 / 23.0 / 22.5 |
| PM3 | 3 | 0.197 / 4.128 | 0.065 / 0.396 / 0.380 | 0.270 / 2.935 / 2.904 | 26.0 / 26.0 / 26.0 |
| PM3 | 12 | 2.457 / 8.229 | 0.459 / 0.777 / 0.764 | 2.532 / 6.642 / 6.654 | 26.5 / 26.5 / 26.5 |
| PM3 | 24 | 10.687 / 21.543 | 2.117 / 1.580 / 1.562 | 9.031 / 17.299 / 17.283 | 26.5 / 26.0 / 27.0 |
| PM3 | 48 | 66.203 / 43.285 | 8.646 / 2.956 / 2.943 | 39.985 / 34.994 / 35.612 | 28.0 / 27.0 / 27.0 |

CPU scientific results and iterations match exactly across all six rounds.
CPU/CUDA maximum discrepancies in the final after run are
`6.55e-11 eV`, `4.75e-11 e` and `2.58e-9 eV/Å` across the checked results,
including resident final vectors. Existing external pins remain a separate gate.
At 48 atoms, CPU moving pooled after/before ratios are AM1 0.9967,
RM1 1.0003, MNDO 0.9939 and PM3 0.9989. Adjacent-pair ratios straddle one;
smaller systems vary more. **No reliable end-to-end latency gain is established.**
CUDA production is unchanged, and changes in its timings/iteration counts are
not attributed to the CPU optimization. These clusters do not establish a
portable GPU advantage or a solver crossover size.

The separate GH200 allocation run confirms the local result: all 720 complete
calls and 52 primitive rows retain exact scientific values and iteration counts.
Each complete call saves `2I` C++ allocations and `16 * packed_size * I` bytes.
For moving 48-atom calls, cumulative requested bytes decrease by 43.5–44.8%.
After counts remain 1,232–2,688 allocations and 1.94–4.40 MB requested per call
across methods; complete CPU solves are **not allocation-free**. Direct library
malloc and a complete host heap peak remain outside this counter's coverage.

Full-process Nsight Systems 2025.3.2 traces cover a separate moving 48-atom run
for **each** method. AM1/RM1 were captured on `n481`, MNDO/PM3 on GH200 `n179`;
profiled timings are excluded above. The synchronized resident interval spans
20 complete force calls after five warm-ups, including the same geometry kernel.
All four intervals have zero observed GPU memory allocation/release events,
zero runtime CUDA allocation/free calls, no H2D copies, and only 4/8/24/56-byte
D2H status/energy blocks. Matrix copies remain D2D. No scientific arrays are
retrieved inside these intervals. With CPU/GPU unified-memory fault collection
enabled, no page-fault events were recorded. No separate driver-API activity
table was emitted; it is not reported as an independently measured zero.

| Method | SCF iterations in 20 profiled calls | D2H status bytes | GPU allocation/release events in interval |
| --- | ---: | ---: | ---: |
| AM1 | 465 | 34,160 | 0 |
| RM1 | 492 | 35,888 | 0 |
| MNDO | 446 | 32,944 | 0 |
| PM3 | 546 | 39,344 | 0 |

Each complete trace observes 238 matched dynamic-device allocations/releases,
with peak and resident live logical storage of **169,924,475 bytes**; seven
pinned allocations/releases peak at 56 bytes. These dynamic counts return to
zero at capture end. Device-static storage is a separate 3,044 bytes retained
through capture end. These include library allocations; neither ownership nor
complete reserved VRAM is inferred. Whole-process native RSS across the six
unprofiled runs is 377,487,360–452,984,832 bytes, spanning every case/runtime;
it does not establish a solver footprint reduction. The initial diagnostic
trace exposed 272 bytes of first-use trigonometric constants: warming the
actual moving kernel removed that initialization from the measured interval.

The analytic force kernel accounts for only 0.55–0.65% of summed GPU kernel
time in these traces; cuSOLVER dominates. A bounded Jacobi experiment passed
isolated eigensystem checks but failed the existing complete AM1 acceptance,
so it was removed before these final comparisons. The stable divide-and-conquer
solver remains; no accuracy or SCF stopping tolerance was relaxed.

Acceptance: local 1,301 unit tests, 16 affected regressions and 49 Python checks
(four CUDA-only skips); GH200 113 SE-only CUDA tests and 32 targeted combined
CUDA tests; all four external/native composition and 10 fs NVE refinement
cases; 53 Python checks and all four Python external/composition workflows.
Memcheck reports zero errors for AM1's combined AM1/RM1 force section and the
MNDO and PM3 cases. RM1's standalone legacy case is CPU-only: instrumenting it
with memcheck produced no CUDA calls, not a CUDA pass. Job `4379227` completed
the paired runs and AM1/RM1 traces but stopped on that selection error;
`4387715` successfully completed the missing traces and remaining acceptance.
No rejected candidate, notes or raw results are part of the shipped change.

## Historical measurement boundaries

The observations below used the earlier AM1-only harness, nominal 100 Å box,
and earlier warm-up/reporting procedures. They are retained as historical
slice evidence, not directly comparable to current schema-2 timings or proof
of current four-method performance. Earlier allocation reports retained a
force norm rather than full vectors; their stated validation limits remain.

## Historical analytic-force slice (2026-10-23)

Raw observations are in `results/2026-10-23-mac-{pair-fd,analytic}.json`.
Both runs used the same thread counts, geometry and requested tolerances.
The analytic revision additionally requires **both** density and energy
residuals to converge (the earlier revision accepted either). Cold iteration
counts therefore increased: 18/19/21/28 to 28/28/29/39. These are complete
revision comparisons, not an isolated kernel speedup or identical stopping
algorithms. The earlier uncommitted binary is identified by its SHA256, not
by an independently reproducible release tag. Re-run the harness to reproduce
current performance; these observations do not establish a portable speedup.

| Atoms | Prior cold ms | Analytic cold ms | Prior warm ms | Analytic warm ms |
| ---: | ---: | ---: | ---: | ---: |
| 3 | 0.700 | 0.174 | 0.415 | 0.076 |
| 12 | 8.909 | 1.678 | 2.024 | 0.309 |
| 24 | 29.281 | 7.158 | 6.924 | 1.246 |
| 48 | 140.730 | 31.765 | 21.568 | 7.052 |

Largest observed energy change: 3.64e-12 eV; force-component change:
3.86e-7 eV/Angstrom. The earlier force was a central difference with h=1e-4
Angstrom. Process peak RSS was 72.4/72.0 MB including Python imports. These
are not external scientific oracle values or an allocation/residency audit.


## Historical workspace and history slice (2026-10-24)

`results/2026-10-24-mac-{before-workspace,after-workspace,after-workspace-repeat}.json`
retain the same controls, cold/warm procedure and binary hashes. Energies,
charges, force components and the complete iteration sequences are unchanged.
The first after-run was much slower; an immediate repeat on the same binary
was substantially faster. The workstation had unrelated active applications.
This sampling does not establish a causal performance benefit or regression.

| Atoms | Before cold/warm ms | After cold/warm ms | Same-binary repeat cold/warm ms |
| ---: | ---: | ---: | ---: |
| 3 | 0.346 / 0.113 | 1.247 / 0.302 | 0.368 / 0.147 |
| 12 | 1.724 / 0.362 | 5.543 / 1.613 | 3.344 / 0.531 |
| 24 | 7.254 / 1.241 | 57.252 / 7.931 | 13.844 / 2.563 |
| 48 | 31.732 / 9.125 | 197.208 / 44.892 | 70.549 / 15.722 |

The force result/scratch double buffer adds 24 bytes per atom of persistent
storage. The pair derivative result has bounded stack storage. Neither fact
proves an allocation-free complete solve; SCF matrices, integral values and
Driver setup still allocate. Controlled paired timing on a reserved machine
remains necessary before choosing further performance changes from these data.

## Historical allocation baseline (2026-10-24)

Recorded allocation counts are in `results/2026-10-24-mac-allocations.json`.
The counter positive control observed 8 allocations / 480 requested bytes.
All 12 primitive combinations (H–H, C–C, Br–I, tilted/z-axis, Hcore/ERI),
each evaluated 1000 times after warming, observed **zero C++ new calls**.
This statement covers these derivative calls only.

| Atoms | C++ allocations per warm public force call | Requested bytes per call |
| ---: | ---: | ---: |
| 3 | 263 | 31,072 |
| 12 | 695 | 139,856 |
| 24 | 2,027–5,711 | 476,432–1,582,544 |
| 48 | 7,283–45,139 | 1,799,312–13,301,392 |

The varying counts track changing SCF iteration counts; the raw file records
each call. These totals show that complete calls still have substantial
allocation work. They are cumulative requested bytes, not peak/live memory,
and exclude direct library malloc and Python allocations. Instrumented runs
are not used as latency measurements.

## Historical persistent-stage allocation comparison

`results/2026-10-24-mac-before-persistent-stages.json` and
`results/2026-10-24-mac-after-persistent-stages.json` repeat the same allocation
benchmark around the NDDO stage/criterion lifetime change. The before executable
has the same SHA256 as the allocation report above. All 80 warm calls retain
exactly the same iteration counts, energies and squared force norms; all
primitive checksums and allocation counts are unchanged.

| Atoms | Before C++ allocations/call | After C++ allocations/call |
| ---: | ---: | ---: |
| 3 | 263 | 197 |
| 12 | 695 | 629 |
| 24 | 2,027–5,711 | 1,961–5,645 |
| 48 | 7,283–45,139 | 7,217–45,073 |

Every measured call saves 66 C++ allocations and 5,684 requested bytes. This
isolates a reduction in allocation traffic, not an end-to-end speedup or an
allocation-free SCF claim. The remaining matrix/integral allocation costs need
their own measured slice. Full scientific-vector validation comes from the
external-oracle regressions; the benchmark records a squared force norm.

## Historical Fock pair-buffer comparison

`results/2026-10-24-mac-{before,after}-fock-scratch.json` measure replacing
per-pair integral vectors and pair-index vectors with one 100-double buffer
per Fock assembly. All 80 calls preserve iterations, energy and squared force
norm exactly; the primitive measurements are unchanged. Full-vector oracle
regressions and the Python checks also pass.

| Atoms | Before C++ allocations/call | After C++ allocations/call |
| ---: | ---: | ---: |
| 3 | 197 | 170 |
| 12 | 629 | 197 |
| 24 | 1,961–5,645 | 233–461 |
| 48 | 7,217–45,073 | 305–1,297 |

The observed reduction is `(iterations + 1) * atoms²` calls. At 48 atoms,
requested allocation traffic drops from 1.79–13.30 MB to 1.25–9.88 MB per
force call. These are cumulative bytes, not peak memory. The earlier counter
binary hash was not captured before replacement; the report preserves that
limitation and hashes the saved source snapshots.

Six alternating timing runs (`results/2026-10-24-mac-fock-timing-*.json`) use
separately preserved and hashed Python modules in before/after/after/before/
before/after order. Full energies, forces, charges and iteration sequences
match exactly. At 48 atoms the warm medians range from 12.54–16.87 ms before
and 11.66–18.54 ms after. This workstation sampling establishes no reliable
end-to-end speedup; reserved-machine timing remains necessary. Matrix and
eigensolver allocations remain visible in complete calls.

## Historical LAPACK workspace comparison

`results/2026-10-24-mac-{before,after}-eigen-workspace.json` record the next
isolated change: `NDDODiagonalizer` retains the same LAPACK `syevd('V','L')`
matrix, eigenvalue and queried work buffers. Coefficients use the solver's
existing output capacity. No algorithm or stopping criterion changes.

| Atoms | Before C++ allocations/call | After C++ allocations/call |
| ---: | ---: | ---: |
| 3 | 170 | 152 |
| 12 | 197 | 179 |
| 24 | 233–461 | 215–389 |
| 48 | 305–1,297 | 287–1,135 |

All 80 calls preserve iteration counts, energies and squared force norms
exactly, with nine fewer C++ allocations per SCF iteration. For 48 atoms,
requested allocation traffic falls from 1.25–9.88 MB to 0.50–3.10 MB per
complete force call. The workspace adds 228,500 persistent bytes at 96 AOs
with this LAPACK build's queried sizes; library workspace sizes can differ.
Reports include binary and source hashes. The SE suite (711 checks), six
external full-vector fixtures and 26 Python checks pass. No latency or peak
memory improvement is inferred from these allocation measurements.

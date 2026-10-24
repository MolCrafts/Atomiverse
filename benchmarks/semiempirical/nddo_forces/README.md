# NDDO complete force-call benchmark

Raw files under `results/` are retained locally and are not shipped in the
repository. The commands below reproduce new observations.

`run.py` measures AM1 energy + charges + forces at 1/4/8/16 distorted waters.
It uses the public Python/Frame interface, including its conversion and output
copy costs. Five cold calls each construct a method; twenty warm calls reuse
its converged electronic state at the same geometry. Frame construction is
outside the timer. Record every call and iteration count; do not compare warm
latency to cold latency or call it a changing-geometry BOMD measurement.

Use the same Release build options and thread settings before and after a
change, run serially without other builds or tests, and retain both JSONs:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PYTHONPATH="$BUILD:$PWD/python/src" \
python benchmarks/semiempirical/nddo_forces/run.py /tmp/nddo-forces.json
```

The Python environment needs the existing `atomiverse`/`molpy` dependencies.
The report includes the actual extension's SHA256, machine, thread settings,
SCF controls, energies, full forces, charges and process peak RSS. RSS covers
the entire Python process and imports; it is not solver workspace memory.
No allocation count or CUDA residency measurement is inferred from this
report. The captured values are before/after checks, not external oracles.

## Recorded CPU slice (2026-10-23)

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


## Workspace and history slice (2026-10-24)

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

## C++ allocation counts

`count_allocations.cpp` is a separate native benchmark. It replaces the
executable's standard C++ `new`/`new[]` entry points, including aligned and
nothrow forms, and records successful calls and requested bytes while a
measurement scope is active. A positive control exercises all eight forms
before collecting scientific measurements. It does **not** count direct
`malloc`/`calloc`/`realloc` calls, notably any such allocations inside BLAS,
Rust, or other libraries. Requested bytes are cumulative allocation traffic,
not live or peak memory.

Use a static, non-MPI Release build so the native engine's C++ allocations
resolve to the executable's replacement operators. With the standing
`bench_nddo_allocations` target enabled:

```sh
cmake -S . -B "$BUILD" -DATV_BUILD_BENCHMARKS=ON
cmake --build "$BUILD" --target bench_nddo_allocations
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ATV_LOG_LEVEL=off \
"$BUILD/benchmarks/bench_nddo_allocations" /tmp/nddo-allocations.json
```

The report records the **observed** counts; it does not assume that a kernel
allocates nothing. It contains separate rows for:

- Hcore and two-electron pair derivatives: H–H, C–C and Br–I, both inclined
  and on the z axis; each kernel is warmed once, then called 1,000 times.
  Pair construction, parameter lookup and output logging occur outside the
  counted interval. A finite checksum consumes the returned derivatives.
- Complete public `AM1::forces(Frame&)`: the same 1/4/8/16-water geometries
  and SCF settings as `run.py`, five warm-up evaluations followed by twenty
  individually counted calls. Iterations, energy and squared force norm
  accompany each row. This includes native Driver and solver allocations,
  but excludes Python/Frame conversion costs.

Run this separately from the timing benchmark: allocation instrumentation
changes execution costs. Keep the JSON, exact executable hash, source
revision/patch and build options with any reported result. A zero count for
a derivative primitive would establish only zero observed C++ allocations
for those calls, not an allocation-free complete method or a CUDA residency
claim. A complete allocator audit additionally needs a platform profiler
(e.g. Instruments Allocations / `malloc_history` on macOS or heaptrack on
Linux), including direct library allocation stacks.


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

## Persistent-stage allocation comparison

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

## Reused Fock pair buffer

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

## Persistent LAPACK workspace

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

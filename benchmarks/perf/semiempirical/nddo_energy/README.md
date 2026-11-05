# NDDO energy and charge benchmark

`run.cu` (target `bench_nddo_energy`) times the public CPU and CUDA AM1 and
RM1 energy/charge calls at 3, 12, 24 and 48 atoms (1, 4, 8 and 16 waters). It
measures energy only, through the native API; it is not a force, BOMD or Python
benchmark. It is never registered with CTest.

## System and controls

The system uses the distorted water geometry and grid spacing from
`../nddo_forces/run.py`: O `(0,0,0)`, H `(.96,0,0)`, H `(-.24,.93,.1)`, with
4 Å between waters. Both backends use an isolated default Box, because CUDA
molecular NDDO rejects periodic cells. The charge is zero and the multiplicity
is one.

Coordinates stay fixed across repeats, and the accepted density is reused.
Both backends use the same controls: FP64, `max_iter=1000`, density change
`1e-10` and energy change `1e-12 eV`. Both residuals must converge. CPU and
CUDA iteration counts can differ, so keep them with the timings when you
interpret results.

## Timing boundaries

- `host_frame_energy_charge`: five cold calls, each a fresh method
  construction plus a complete `run(Frame&)`. These are followed by warm calls
  on the last method, twenty by default. Each sample includes completion and
  host energy/charge publication. The timer excludes Frame construction,
  result copies into the report, validation and method destruction.
- CUDA context initialization is measured separately, before either backend's
  method calls. "Cold" means a fresh method, not a fresh process or driver
  context, so library first-use costs can remain in the first call.
- `resident_evaluate_energy_intent`: a benchmark-only Evaluate author component
  uses the existing setup/step/teardown lifecycle. It uploads one Frame, runs
  five warm-up steps, then times complete Evaluate steps on that device Frame.
  Each step requests energy without forces or stress. A checked parent-stream
  synchronization ends every timed sample, so each sample includes serial
  completion overhead, not only kernel submission.
- The resident interval excludes setup/build, warm-up, Frame retrieval, the
  `last_result()` charge download and report writing. Only one final
  energy/charge vector is downloaded, after the interval. Standalone rows keep
  energy and the full charge vector for every call.

## Build

Configure with these options, then build the target:

- `ATV_BUILD_BENCHMARKS=ON`
- `ATV_BUILD_SEMIEMPIRICAL=ON`
- `ATV_BUILD_CUDA=ON`
- `ATV_BUILD_EVALUATE=ON`
- `ATV_CUDA_PREC_MODE=double`
- `ATV_REAL_STORAGE=double`
- `ATV_REAL_WORK=double`

HF, PW and GPW are not needed.

```sh
cmake --build "$BUILD" --target bench_nddo_energy
```

## Run

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ATV_LOG_LEVEL=off \
  "$BUILD/bin/bench_nddo_energy" "$OUT/energy.json"
```

The positional arguments are
`OUTPUT.json [am1|rm1|both] [waters:0|1|4|8|16] [warm:1..100] [profile:0|1]`.
The defaults are `both 0 20 0`, and `waters=0` selects all four sizes.
One-thread OMP/OpenBLAS is required. Run serially on a reserved GPU and CPU
core, with no builds or competing jobs, and bind the process through the
scheduler. Use unprofiled runs for latency. To claim a difference between
revisions, repeat the runs and alternate the revisions; a single run is not
enough.

The report records:

- GPU name, total device memory and CUDA runtime/driver versions;
- the SCF controls and iteration counts;
- every raw timing sample;
- the final physical results;
- a lower bound for CUDA storage, `6 * nao * nao * sizeof(double)`, for the
  six dense matrices. This bound is neither measured live memory nor a peak.

## Checks

The process exits nonzero, and writes no success report, if any of these
fails:

- a solve doesn't converge or gives a non-finite result;
- the Frame publication check fails;
- final CPU and CUDA energies differ by more than `1e-6 eV`;
- charges differ by more than `1e-7 e`.

These are internal consistency checks, not comparisons with an external oracle.

## Resident-interval trace

Set `profile=1` with exactly one method and one nonzero water count. Only the
resident warm loop is wrapped in `cudaProfilerStart`/`cudaProfilerStop`;
standalone calls and output reads stay outside the range. Capture the whole
process, so allocations and the closing marker are recorded, then export to
SQLite:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ATV_LOG_LEVEL=off \
  nsys profile --trace=cuda --sample=none --output="$OUT/am1-48-full" \
    "$BUILD/bin/bench_nddo_energy" "$OUT/am1-48-full-profile.json" am1 16 20 1
nsys export --type sqlite --output="$OUT/am1-48-full.sqlite" "$OUT/am1-48-full.nsys-rep"
```

Repeat the capture with `rm1`. Profiled timings include profiler overhead;
keep them separate from latency samples. To determine residency, inspect the
captured memory transfers and allocator calls, not the source. The harness
doesn't count allocations or certify residency itself.

`analyze.py` summarizes a capture directory using only the standard library.
The directory must contain:

- `energy.json` and `provenance.json`;
- `executable.sha256`, `CMakeCache.txt` and `numerical-sources.sha256`;
- `{am1,rm1}-48-full.sqlite`, `.nsys-rep` and `-profile.json`.

```sh
python3 benchmarks/perf/semiempirical/nddo_energy/analyze.py "$OUT" > "$OUT/summary.json"
```

The analyzer opens each SQLite file read-only and selects the window from the
return of `cudaProfilerStart` to entry into `cudaProfilerStop`. It asserts:

- every kernel, memcpy, memset and synchronization launched inside the window
  also completes inside it;
- the final stream synchronization returns after all of that activity;
- every runtime call in the window returns zero.

It then reports transfer sizes, allocator calls and the profiled-against-
unprofiled final-result deltas. Keep `$OUT` under `build/` or outside the
source tree.

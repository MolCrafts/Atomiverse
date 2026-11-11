# NDDO complete force-call benchmark

These harnesses time complete AM1, RM1, MNDO and PM3 force calls. None of them
is registered with CTest. Their outputs are consistency checks, not oracle
comparisons.

| Harness | Target / entry | Measures |
| --- | --- | --- |
| `count_allocations.cpp` | `bench_nddo_allocations` | C++ allocation counts for derivative primitives and complete CPU force calls |
| `run.py` | Python script | CPU force calls through the Python/Frame boundary |
| `run.cu` | `bench_nddo_forces` | Native CPU against CUDA force calls, including a resident Evaluate loop |
| `analyze.py` | Python script | Paired-revision timing summary and full-process Nsight trace summary |

## System and controls

The default sizes are 3, 12, 24 and 48 atoms (1, 4, 8 and 16 distorted waters).
Native CUDA also accepts an explicit 96-atom selection (32 waters), which is
outside the default set.

- Water geometry: O `(0,0,0)`, H `(.96,0,0)`, H `(-.24,.93,.1)`, with 4 Å
  grid spacing and masses 15.999, 1.008 and 1.008.
- Cell: isolated, with a zero cell.
- Electronic state: neutral closed shell.
- SCF: FP64, `max_iter=1000`, density tolerance `1e-10`, energy tolerance
  `1e-12 eV`. Both residuals must converge.

Each run starts with five cold calls; each one constructs a fresh method and
then calls `forces`. Each warm mode then starts a fresh method, runs five
evaluations at the original geometry, and times 20 calls:

- **Fixed** mode keeps the coordinates.
- **Moving** mode starts from the original geometry at step `s = repeat + 1`
  and offsets each water's atoms:
  - H1.x by `.002*sin(.2*s)`;
  - H2.y by `.002*sin(.17*s)`;
  - H2.z by `.001*sin(.11*s)` Å.

  This is a prescribed sequence, not integrated BOMD throughput.

Every call records iterations, E/HOF, charges and the complete flat Cartesian
force vector. Run serially on reserved resources, with no competing builds or
tests. Keep raw outputs under `build/` or outside the source tree.

## CPU allocation counts and Python boundary

Build `bench_nddo_allocations` in a static, non-MPI Release build with
`ATV_BUILD_BENCHMARKS=ON` and `ATV_BUILD_SEMIEMPIRICAL=ON`. `run.py` needs the
`atomiverse` package with the semiempirical module, plus `molpy`.

```sh
cmake --build "$BUILD" --target bench_nddo_allocations
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ATV_LOG_LEVEL=off \
  "$BUILD/bin/bench_nddo_allocations" "$OUT/allocations.json"
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTHONPATH="$BUILD:$PWD/python/src" \
  python benchmarks/perf/semiempirical/nddo_forces/run.py "$OUT/python-forces.json"
```

Both commands accept an optional `AM1|RM1|MNDO|PM3` selector, in uppercase,
after the output path.

`count_allocations.cpp` counts successful C++ `new`/`new[]` calls, including
the aligned and nothrow forms.

- Its positive control must observe 8 calls and 480 bytes.
- Direct `malloc`, `calloc` and `realloc` calls aren't counted, including
  library calls that use those APIs.
- Bytes are cumulative requested traffic, not live or peak memory.
- Primitive rows warm once, then time 1,000 Hcore/ERI derivative calls. They
  cover H–H, C–C and N–O under every method, plus AM1 Br–I, in tilted and
  axial geometries.
- The count excludes parameter lookup, pair setup and logging.

Run allocation counting separately from latency, because the instrumentation
changes costs.

`run.py` times the full Python/Frame boundary, including conversion and
force-return copies. It records the SHA256 of the loaded extension. Its peak
RSS covers the whole Python process, not one method. Use `run.cu` for native
CPU against CUDA latency.

## Native CPU/CUDA force timing

Build `bench_nddo_forces` with these options:

- `ATV_BUILD_BENCHMARKS=ON`
- `ATV_BUILD_SEMIEMPIRICAL=ON`
- `ATV_BUILD_CUDA=ON`
- `ATV_BUILD_EVALUATE=ON`
- `ATV_CUDA_PREC_MODE=double`
- `ATV_REAL_STORAGE=double`
- `ATV_REAL_WORK=double`

HF, PW and GPW are not needed.

```sh
cmake --build "$BUILD" --target bench_nddo_forces
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ATV_LOG_LEVEL=off \
  "$BUILD/bin/bench_nddo_forces" "$OUT/forces.json" all 0 20 0 all
```

The arguments are
`OUTPUT.json [am1|rm1|mndo|pm3|all] [waters:0|1|4|8|16|32] [warm:1..100] [profile:0|1] [fixed|moving|all]`.
The defaults are `all 0 20 0 all`, and `waters=0` selects the four default
sizes. A profiled run requires one method, one nonzero size and one geometry
mode.

The report has two timing scopes:

- `host_frame_energy_charge_force` covers native CPU and CUDA cold and warm
  series. Each sample includes completion and host publication of E, q and F.
  The timer excludes Frame setup, edits, destruction and result copies into the
  report. CUDA context initialization is recorded separately.
- `resident_evaluate_force` uses a benchmark-only Evaluate author component.
  It times complete mounted steps on one device Frame, and each sample ends
  with a checked parent-stream synchronization. In moving mode, a device
  component writes the geometry inside the timed step. The timer excludes
  setup/build, the five warm-up steps, teardown and result downloads. Full
  E, q and F are downloaded and validated only at the final point.

`cudaMemGetInfo` gives whole-device free-memory snapshots. These are not peak
allocations, and they don't show which code owns the memory.

## Checks

Standalone CPU and CUDA results are compared for every cold and warm call. The
resident final results use the same gates:

- energy within `1e-6 eV`;
- maximum charge difference within `1e-7 e`;
- maximum force-component difference within `1e-5 eV/Å`.

CPU and CUDA iteration counts may differ. The process exits nonzero on a failed
solve or comparison.

## Paired revision comparison

Build before and after binaries from the same harness source. Run them
serially on one allocation, with `all 0 20 0 all`, in this order:

| Order | Binary | Output file |
| --- | --- | --- |
| 1 | before | `1-paired-before.json` |
| 2 | after | `2-after.json` |
| 3 | after | `3-after.json` |
| 4 | before | `4-paired-before.json` |
| 5 | before | `5-paired-before.json` |
| 6 | after | `6-after.json` |

Save all six files in `$PAIRED`, then summarize them:

```sh
python benchmarks/perf/semiempirical/nddo_forces/analyze.py --paired "$PAIRED" \
  > "$PAIRED/summary.json"
```

The analyzer checks that metadata, call counts and vectors match. It also
checks that CUDA agrees with CPU within the stated tolerances. CPU outputs and
iteration counts must exactly equal those of the first before round, so this
mode applies only to changes that leave the CPU results unchanged. It reports each round's
iteration sequence and medians, pooled timings and adjacent-pair ratios. A
ratio alone can't separate a change in iteration count from a change in
per-iteration cost.

## Full-process residency trace

Keep ordinary timing in `forces.json` and collect traces in separate runs.
Capture the whole process, including setup and natural exit. The benchmark's
profiler markers delimit only the resident warm loop.

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
python benchmarks/perf/semiempirical/nddo_forces/analyze.py "$RAW" > "$RAW/summary.json"
```

`$RAW` must contain `forces.json`, `provenance.json`, `executable.sha256`,
`CMakeCache.txt` and `numerical-sources.sha256`. Each `*-full.sqlite` needs
its matching `*-full-profile.json` and `.nsys-rep`. A range-only capture
doesn't work: allocation lifetimes and the closing marker require complete
collection.

`analyze.py` reuses the trace-window queries in `../nddo_energy/analyze.py` and
adds:

- memory-event accounting;
- a check of profiled final results against the ordinary run;
- hashes of every input.

Logical GPU allocations are not the same as driver reservations or total VRAM
use. Runtime and driver API views overlap, so don't sum them. CUDA traces don't
record host `malloc`/`new`. Profiled timings include overhead and are excluded
from latency summaries.

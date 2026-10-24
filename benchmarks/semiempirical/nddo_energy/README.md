# Native NDDO energy and charge benchmark

Raw measurement files under `results/` are local artifacts and are not shipped
in the repository. Paths below identify the retained evidence; rerun the
benchmark and analysis commands to produce a fresh dataset.

`run.cu` compares the public CPU/CUDA AM1 and RM1 energy/charge calls at
3/12/24/48 atoms. It reuses the distorted water geometry and spacing from
`../nddo_forces/run.py`: O `(0,0,0)`, H `(.96,0,0)`, H `(-.24,.93,.1)`, with
4 Å spacing on the same grid. Both backends use an **isolated default Box**;
the force harness's nominal 100 Å box is not used because CUDA molecular
NDDO explicitly rejects periodic cells. Charge is zero and multiplicity is one.

This is an energy-only native benchmark, not a force, BOMD, or Python benchmark.
Coordinates are fixed across repeats and the accepted density is reused.
The controls are identical on both backends: FP64, `max_iter=1000`, maximum
density change `1e-10`, energy change `1e-12 eV`; both residuals must converge.
CPU and CUDA iteration counts need not be identical. Preserve them when
interpreting timings. The harness checks finite/converged results and Frame
publication, then compares final CPU/CUDA energies at `1e-6 eV` and charges
at `1e-7 e`. These are internal consistency checks, not external oracle evidence.

## Timing boundaries

- `host_frame_energy_charge`: five cold method constructions plus complete
  `run(Frame&)` calls; then twenty warm calls on the last method by default.
  Each sample includes completion and host energy/charge publication. Frame
  construction, result copies into the report, validation, and method destruction
  are outside the timer. CUDA context initialization is measured separately
  before either backend's method calls; cold means a fresh method, not a fresh
  process or driver context. Library first-use costs can remain in the first call.
- `resident_evaluate_energy_intent`: a benchmark-only Evaluate author probe
  composes the existing protected setup/step/teardown lifecycle. It uploads
  one Frame, warms the mounted method for five complete steps, then measures
  repeated complete Evaluate steps on that same device Frame. Each step runs
  the preset's output clearing and an author component that requests energy
  without forces/stress; the method still uses its normal nested public mount.
  A check-stage host counter records iterations without downloading results.
  A checked parent-stream synchronization ends every timed sample. This includes
  serial completion overhead, not merely kernel submission latency.

The resident interval excludes setup/build, warm-up, Frame retrieval, explicit
`last_result()` charge download and report writing. It has per-step timings and
iteration counts, but **only one final energy/charge vector**, downloaded after
the interval. It does not claim that each intermediate vector was inspected.
Standalone rows retain energy and the complete charge vector for every call.

## Build and run on the reserved GPU node

The standing `bench_nddo_energy` target requires CPU semiempirical plus CUDA;
it links the existing `atomiverse` target and is never registered with CTest.
Use the repository's normal GH200 toolchain and dependencies. Enable
`ATV_BUILD_BENCHMARKS=ON`, `ATV_BUILD_SEMIEMPIRICAL=ON`, `ATV_BUILD_CUDA=ON`,
`ATV_BUILD_EVALUATE=ON`,
`ATV_CUDA_PREC_MODE=double`, `ATV_REAL_STORAGE=double`, and
`ATV_REAL_WORK=double`. HF/PW/GPW are not prerequisites.

```sh
cmake --build "$BUILD" --target bench_nddo_energy
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ATV_LOG_LEVEL=off \
  "$BUILD/bin/bench_nddo_energy" /tmp/nddo-energy.json
```

Positional arguments are `OUTPUT.json [am1|rm1|both] [waters] [warm] [profile]`.
Defaults are `both 0 20 0`; waters `0` selects all of `1/4/8/16`. Warm counts
are bounded to `1..100`; no size beyond 48 atoms is accepted. The process exits
nonzero on a failed solve, publication check, or CPU/CUDA comparison, and does
not write a success report for a partially completed measurement set.

Run serially without builds or competing jobs on a reserved GPU and CPU core.
Use unprofiled measurements for latency; repeat and alternate revisions rather
than treating a single run as a speedup. The fixed one-thread OMP/OpenBLAS
settings are required; also bind the process using the scheduler and preserve
its resource request. This first small-size set makes no GPU speedup claim.

## Capture the actual resident interval

Set `profile=1` with exactly one method and one nonzero water count. Only that
resident warm loop calls `cudaProfilerStart`/`cudaProfilerStop`; all standalone
calls and complete output reads occur outside the range. For example:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ATV_LOG_LEVEL=off \
  nsys profile --trace=cuda --sample=none --capture-range=cudaProfilerApi \
    --capture-range-end=stop --output=/tmp/am1-48-resident \
    "$BUILD/bin/bench_nddo_energy" /tmp/am1-48-profile.json am1 16 20 1
```

`--capture-range-end=stop` lets the target finish retrieving and validating its
results after collection ends. These capture controls follow the
[Nsight Systems user guide](https://docs.nvidia.com/nsight-systems/UserGuide/).
Check the installed node's `nsys profile --help` before choosing additional
trace options. Profiler-enabled timings include observer overhead and must be
kept separate from ordinary latency samples.

Inspect the captured CUDA memory transfers and allocator calls, not source
searches, to determine residency and steady-state allocation behavior. The
method's convergence/error scalar transfers remain expected. Classify observed
transfer sizes against the full density/coefficient/Fock arrays and full charge
vector before asserting that scientific arrays remain resident. Keep the raw
`.nsys-rep` and exported trace/statistics alongside the JSON. This harness does
not itself count allocations or certify residency.

## Resources and provenance

The report includes GPU name, total device memory, CUDA runtime/driver versions,
SCF controls, iteration counts, raw timing samples, and final physical results.
For CUDA it reports the **lower bound** `6 * nao * nao * sizeof(double)` for the
current six dense matrices, with six AOs per water. This is 1,728 / 27,648 /
110,592 / 442,368 bytes for the four sizes. It excludes cuSOLVER/CUB workspace,
parameters, eigenvalues, charges, Frame storage, library/runtime allocations,
and host memory. It is neither measured live memory nor a peak estimate. Use
the runtime trace for observed allocation sizes and lifetimes.

Keep the exact executable SHA256, source revision and uncommitted patch,
`CMakeCache.txt`, compiler/BLAS/CUDA versions, node identity, scheduler CPU/GPU
allocation and affinity with each run. A build target or a source inspection
does not establish successful execution. The first executed measurements and
their evidence limits follow below.

## GH200 energy measurement on the public-force revision, 2026-10-24

The primary dataset below uses the current public analytic-force revision,
but measures **energy and charge calls only**. It establishes neither force
performance nor force residency. Historical pre-public-force observations remain
in the earlier aggregate (`results/2026-10-24-gh200-summary.json`); they are not
combined with this dataset or used as a controlled revision comparison.

The run used a NVIDIA GH200 120GB, one GPU and eight allocated CPU cores;
OMP and OpenBLAS each used one thread. The two full-process captures identify
`n519` in both their captured device-display metadata and SQLite export-host
metadata. The unprofiled benchmark JSON does not independently record a
hostname. CPU affinity was not recorded. The toolchain was
`GPU/buildenv-gcccuda/2026.03-cu13.0`, Release, `-march=native`, CUDA architecture
90 and OpenBLAS; CUDA runtime/driver API versions report 13000.

The executable SHA256 is
`3c15b5fce61a0071680c7d156f151f4585819abff60ca1050ad7a3616edf8eb7`.
The provenance record (`results/2026-10-24-gh200-public-force-provenance.json`)
identifies HEAD `9c2453abdfb89c8949b959ca160ef2c300b55552`, the archived working-tree
patch and untracked-source archive by SHA256, and the molrs HEAD and patch.
Those source archives remain in the external evidence directory named in the
record. The transferred numerical trees, CUDA driver/Evaluate/timer and
benchmark source comparison was empty. The contemporaneous
numerical-source hashes (`results/2026-10-24-gh200-public-force-numerical-sources.sha256`),
CMake cache (`results/2026-10-24-gh200-public-force-CMakeCache.txt`) and
executable hash (`results/2026-10-24-gh200-public-force-executable.sha256`)
are retained with the local measurement artifacts. Rebuilding this exact dirty snapshot requires the recorded
source archives as well as HEAD; the numeric hash list alone is not source code.

### Scientific results and unprofiled latency

The unprofiled observations (`results/2026-10-24-gh200-public-force-energy.json`)
contain 400 standalone cold/warm calls and 160 resident warm calls. All eight
method/size combinations completed. Across final CPU/CUDA standalone and
resident results, the maximum energy difference is `4.730e-11 eV` and the
maximum charge-component difference is `4.730e-14 e`. Across every matched
standalone cold/warm call, the maxima are `4.366e-11 eV` and `4.747e-11 e`.
These pass the unchanged `1e-6 eV` / `1e-7 e` benchmark tolerances. They are
internal comparisons, not additional external-oracle evidence.

Median complete-call wall time in **milliseconds**, unprofiled:

| Method | Atoms | CPU cold | CUDA cold | CPU warm | CUDA warm, host Frame | CUDA warm, resident |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| AM1 | 3 | 0.201 | 4.030 | 0.053 | 0.330 | 0.318 |
| AM1 | 12 | 2.103 | 7.988 | 0.241 | 0.639 | 0.625 |
| AM1 | 24 | 9.087 | 20.233 | 1.023 | 1.479 | 2.107 |
| AM1 | 48 | 51.564 | 38.070 | 9.870 | 4.108 | 4.088 |
| RM1 | 3 | 0.206 | 4.425 | 0.053 | 0.340 | 0.325 |
| RM1 | 12 | 2.276 | 8.809 | 0.236 | 0.644 | 0.629 |
| RM1 | 24 | 10.204 | 21.661 | 1.042 | 1.795 | 1.485 |
| RM1 | 48 | 47.545 | 43.465 | 9.209 | 2.732 | 2.715 |

Cold uses five samples; warm uses twenty. CUDA context initialization took
0.898 s outside these timers. All 3-atom warm calls took two SCF iterations.
Larger systems have different iteration sequences despite identical tolerances:

| Method, 48 atoms | CPU cold iterations | CUDA cold | CPU warm | CUDA warm, host Frame | CUDA warm, resident |
| --- | ---: | ---: | ---: | ---: | ---: |
| AM1 | 39 | 28–32 | 2–25 | 2–7 | 2–11 |
| RM1 | 36 | 31–34 | 2–18 | 2–8 | 2–5 |

All sequences and min/median/max times are in the
current aggregate (`results/2026-10-24-gh200-public-force-summary.json`).
This is one job at fixed geometries, not a repeated/reversed revision
experiment, isolated kernel speedup or crossover claim. The resident and
standalone sequences have different warm-up histories; their median difference
cannot be interpreted as transfer cost. No broad GPU performance advantage is
established by these small systems.

### Full-process traces: bounded resident interval verified

Separate complete-process captures use the same executable, with natural
process exit before SQLite export. Neither contains a diagnostic warning or
error. Their outputs are retained separately from the latency dataset:
AM1 (`results/2026-10-24-gh200-public-force-am1-48-full-profile.json`),
RM1 (`results/2026-10-24-gh200-public-force-rm1-48-full-profile.json`).
No range-only capture is used for the current conclusions.

The analysis selects the return of `cudaProfilerStart` through entry into
`cudaProfilerStop`. It joins GPU activity to interval-contained runtime calls
by `correlationId`, which is unique in each process trace. For kernels,
memcpy, memset and synchronization:

- All activity launched by interval-contained APIs completes inside the interval.
- No overlapping activity has a missing or out-of-interval API.
- The final successful `cudaStreamSynchronize` returns after all that activity
  completes and before profiler-stop entry.

Setup, warm-up and final result downloads therefore remain outside this
measurement. Exact endpoints, table schemas, SQL and boundary checks are
retained in the current aggregate. Every captured runtime call in the selected
interval has return value zero.

| Method | Calls / total SCF iterations | D2H records | D2H total | H2D records | D2D records | Visible allocator/free API calls |
| --- | --- | --- | ---: | ---: | --- | ---: |
| AM1 | 20 / 57 | 8 B × 57; 56 B × 117 | 7,008 B | 0 | 73,728 B × 174 | 0 |
| RM1 | 20 / 56 | 8 B × 56; 56 B × 116 | 6,944 B | 0 | 73,728 B × 172 | 0 |

Both windows contain one kernel stream: 13,404 AM1 kernels and 13,172 RM1
kernels. No host transfer matches a full 48-atom FP64 charge vector (384 B),
orbital-energy vector (768 B), or 96×96 electronic matrix (73,728 B); even the
total D2H volume is smaller than one matrix. Only 8/56-byte D2H blocks occur.
This verifies explicit-transfer residency and no visible CUDA allocation/free
calls **for these twenty warm 48-atom energy calls per method**. It does not
identify the payload of each small transfer or extend to cold calls, other
sizes/geometries, forces, BOMD, Python, host allocations or implicit managed
memory migration.

As a visibility control, each whole-process trace records 204 `cudaMalloc`,
492 `cudaFree`, six `cudaHostAlloc` and six `cudaFreeHost` calls outside the
warm interval. Free counts are API calls, not necessarily allocation releases
(for example, null pointers can be passed). There are no allocator callchains,
allocation arguments, separate driver-API allocation table or host `malloc`/`new`
trace. Thus application versus library allocation ownership, allocation bytes
and leaks cannot be inferred. GPU-memory-usage and UVM page-fault collection
were off, and the capture reports that some CUDA API calls are skipped.
**Measured live/peak memory is unavailable.** The six-matrix byte count above
is a storage lower bound, not an observed peak.

Profiled final results differ from unprofiled resident results by at most
`1.092e-11 eV` and `2.332e-14 e`; each process also passed its CPU/CUDA checks.
Different iteration sequences remain recorded. Profiled timings are excluded
from the latency table.

### Reproduce the report

Capture the complete process while retaining `profile=1` so marker APIs remain
in the trace:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ATV_LOG_LEVEL=off \
  nsys profile --trace=cuda --sample=none --output=/tmp/am1-48-full \
    "$BUILD/bin/bench_nddo_energy" /tmp/am1-48-full-profile.json am1 16 20 1
nsys export --type sqlite --output=/tmp/am1-48-full.sqlite /tmp/am1-48-full.nsys-rep
```

Repeat with `rm1`. Check diagnostics each time; a successful application exit
does not establish trace completeness. These captures used Nsight Systems
2025.3.2.474, SQLite schema 3.20.3. Raw `.nsys-rep` and SQLite files stay outside
the repository; their hashes are retained in the aggregate.

Keep `energy.json`, both `*-48-full-profile.json`, both `*-48-full.sqlite`,
both `*-48-full.nsys-rep`, `executable.sha256`, `CMakeCache.txt`,
`numerical-sources.sha256` and the run-specific `provenance.json` together.
Use the archived provenance record for this measurement; for a new run write
its actual resource, executable and source-archive identities.

```sh
python3 benchmarks/semiempirical/nddo_energy/analyze.py \
  /path/to/sqm-force-closeout-results > /tmp/nddo-energy-summary.json
```

The standard-library script opens SQLite read-only and stores its aggregation
queries, input hashes and selected metadata. No whole-process environment dump
or raw trace is copied into the repository. Allocation ownership needs
callchains; a measured peak needs memory collection from before workspace
construction. The verified interval remains a restricted 48-atom,
fixed-geometry energy/charge workload.

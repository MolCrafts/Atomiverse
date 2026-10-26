# CUDA RHF validation, 2026-10-25

The s/p ERI primitive now retains two adjacent Hermite recurrence layers,
instead of six, and writes only the active domain. No new tensor, kernel,
host transfer or library boundary is introduced. GH200 job **4409180**
completed with exit zero. For 16 waters / 112 AOs, the complete RHF call
improved from 15.710 s to 1.331 s, **11.80×** in the controlled comparison.
The fastest measured eight-core CPU route still takes only 0.909 s.

## Controlled complete-call comparison

Arrhenius GH200, Release FP64, CUDA 13.0.3, GCC 14.3, OpenBLAS 0.3.32,
eight CPU/OpenMP/OpenBLAS threads. Geometries repeat the water pin at
4 Å intervals; Cartesian STO-3G, neutral closed shell, Hcore starts,
tolerance 1e-10, cap 200, DIIS space 8/start 1, no damping or level shift,
energy only. CPU history resets before every sample. CPU and CUDA retain
their existing stopping criteria; iteration counts are reported separately.

One GPU allocation ran **A1 → B1 → B2 → A2** without profiling. A is the
previously accepted eightfold-symmetry implementation; B changes only
`eri_prim_md_device` in `src/cuda/scf/device_integrals.cu`. Each invocation
measures one first solve and three subsequent solves on the same instance.
The table preserves the two three-sample medians per variant. The ratio is
the arithmetic mean of A's two medians divided by that of B's; it is not
a pooled six-sample median.

| Waters / AOs | A1 / A2 warm median, ms | B1 / B2 warm median, ms | A / B | CPU stored / direct warm median, ms |
| --- | ---: | ---: | ---: | ---: |
| 1 / 7 | 12.429 / 12.379 | 8.478 / 8.435 | 1.47 | 0.811 / 1.376 |
| 2 / 14 | 20.138 / 19.986 | 9.920 / 9.706 | 2.04 | 3.040 / 9.314 |
| 4 / 28 | 125.422 / 125.519 | 19.715 / 19.871 | 6.34 | 21.634 / 46.053 |
| 8 / 56 | 1463.606 / 1462.825 | 103.835 / 103.576 | 14.11 | 249.728 / 209.992 |
| 16 / 112 | 15707.428 / 15712.883 | 1338.363 / 1324.188 | 11.80 | 3239.000 / 909.026 |

At 16 waters, A's warm ranges were 15704.038–15717.491 ms and
15706.119–15772.447 ms; B's were 1336.856–1340.485 ms and
1322.901–1327.487 ms. First-solve times were A1/A2
15713.298/15689.224 ms and B1/B2 1343.857/1324.397 ms.
Both variants took 18 SCF iterations. CPU stored/direct took 14.
The best CPU mode varies with size: stored through four waters, direct
at eight and sixteen. At eight waters CUDA is about 2.02× faster than
the best measured CPU mode; at sixteen it takes about 1.46× as long.

The timer encloses `RHF.run(frame)` and final GPU synchronization:
integrals, SCF, allocations, final Frame output and teardown are included.
Construction, context bootstrap, Frame copying and CPU history reset are
excluded. Repeated solves still perform setup. These measurements apply to
this geometry, basis and node; they do not establish a general GPU/CPU ratio.

The revised manual harness rejects nonfinite/unconverged results, checks
Frame energy from a NaN sentinel, checks every repeated energy and external
pins, and exits nonzero on failed comparisons. It reports actual OpenMP
thread count, settings, iterations and conservative memory admission.
The 16 GiB admission budget is neither an allocator cap nor a measured peak.

Build `bench_rhf_cpu_vs_cuda` through `tools/gpu_job.sh` and the public molq
GH200 workflow, then run inside that allocation:

```bash
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
  ./build-hf-cuda/bin/bench_rhf_cpu_vs_cuda 16 3 both 1e-10 16384 stored
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
  ./build-hf-cuda/bin/bench_rhf_cpu_vs_cuda 16 3 cpu 1e-10 16384 direct
```

## Correctness and implementation scope

All five sizes have identical A/B total-energy strings at the logged
14-decimal precision and unchanged iteration counts; this is not a claim
of bitwise identity. The 16-water CUDA total is -1199.38811898183621 Ha.
Its difference from the independent PySCF 2.11.0 Hcore pin
(-1199.388118981794 Ha, `cart=True`, `conv_tol=1e-12`, same water chain)
is about 4.2e-11 Ha. CPU stored differs from CUDA by 7.55e-11 Ha;
CPU direct differs by 1.31e-8 Ha. Both satisfy the existing 1e-6 Ha gate.
One- and eight-water pins reuse the physical regression references.

The job passed all **78** selected checks: 75 CUDA unit/contract checks
plus the RHF, UHF and MP2 physical regression cases. The new primitive
checks cover all 16 s/p shell masks and Hermite orders zero through four,
including repeated centers and Cartesian components. The order-five
force check isolates the two-electron contribution with +/- density and
compares it with the CPU analytic Rys derivative. Their tolerances are
2e-12 Ha and 2e-11 eV/Å. No finite-difference or full SCF loop is added
to a unit test. The two new checks also passed on A before timing.
Memcheck passed all seven integral tests with zero errors; racecheck
passed both new checks with zero hazards, errors or warnings.

The recurrence uses `m & 1`; every active source is written in layer
`m+1` before reuse. The final layer is always zero. Multiplication and
Cartesian recurrence order are preserved. The local array declaration
falls from 10,368 to 3,456 B. `cuobjdump` reports the production ERI
kernel's per-thread stack falling from **11,088 to 4,176 B**, with
128 registers in both variants. These are compiler resource figures,
not a measured global-memory peak or an occupancy measurement.

The change covers the existing s/p energy and analytic-force primitive.
Higher-angular-momentum kernels, SCF dispatch and post-HF algorithms are
unchanged. Dense CUDA HF still requires `nAO^4 <= INT_MAX` (at most 215 AOs).

## Residency and reproducibility

Separate Nsight captures of the frozen A/B HF→MP2 executables enable all
CUDA APIs, memory usage, and both CPU/GPU unified-memory fault collection.
Each capture contains ten scientific windows: first/repeated calls at all
five sizes, from MP2 `initialize_packet` before the RHF iterations through
correlated `publish_energy`. Setup/input and final host output are outside
these windows. This specifically checks the resident HF→MP2 flow; it does
not trace every possible HF basis, force request or parent-driver sequence.

Both variants have zero H2D/D2H copies, allocation/free API calls, dynamic
Device allocation events and UM activity in every window. Capture-setting
checks pass, trace diagnostics are clean and captured API returns are zero.
UM activity tables are absent in the lazy SQLite export; zero activity is
interpreted together with enabled CPU/GPU fault collection and clean diagnostics.
The first window has 10 B of immutable module-static allocations; subsequent
windows have none. Static module initialization is separate from dynamic
algorithm scratch.

The chronological logical Device + Static allocation peak is identical:
**1,814,760,793 B** for each entire HF→MP2 process. Device/Pinned allocation
pairs close completely. This is neither an HF-only peak nor reserved VRAM
or whole-device usage. The device runtime reports 102,005,473,280 B total.
Per-thread compiler stack figures above are a different resource metric.

Profiled 16-water ERI setup kernel time falls from A's
15186.911/15193.431 ms (first/repeated) to B's 808.144/802.137 ms.
The subsequent HF→MP2 scientific windows remain approximately unchanged:
A 520.072/519.950 ms, B 520.678/520.639 ms. These instrumented kernel/window
durations explain the bottleneck; they are separate from the ordinary HF
wall-time samples. Setup remains included in every complete-call timer.

The experiment uses statically linked, separately copied A/B executables
in one build environment. Every paired invocation checks their SHA-256;
linked vendor-library paths agree and no mutable Atomiverse shared library
is loaded. Baseline ERI source SHA-256 is
`905186101dcfba43e8b3994d3b8531446dac491feab2ab5bd7c1c8def9e623a7`;
accepted ERI source is
`834cc723eb47afa2864ac88fe7bdae64c9c489df8c739bf69ab5cd08eeaa7b5f`.
The HF harness SHA-256 is
`38282e0ccfb17a3d08f39ee9375b6cf394c82e8fd157a08639a623c1dd00e502`.

Raw logs, job spec/status, build flags, link inputs, binary hashes,
92-path accepted manifest, allocation analyses and trace SQLite files
are retained in `.claude/campaigns/hf-eri-rolling-20261025/`.
The earlier [HF→MP2 validation](mp2-validation.md) remains a dated record
of the pre-rolling implementation and its broader residency boundary tests.

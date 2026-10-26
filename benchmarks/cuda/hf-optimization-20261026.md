# CUDA RHF optimization, 2026-10-26

The 16-water / 112-AO complete CUDA RHF call now takes approximately
**568 ms**, compared with **918 ms** for the fastest measured eight-core
CPU route, direct HF: about **1.61×** faster in this experiment.
The previous CUDA implementation took approximately 1.33 s.
All scientific tolerances, FP64 precision and SCF controls are unchanged.

## Separate measured changes

Two GH200 allocations ran sequential, independently controlled experiments.
Each comparison used before → after → after → before, with one first solve
and three subsequent solves per size and invocation. Values below are the
arithmetic mean of the two three-sample warm medians, not a pooled median.

| Change | Before / after, ms at 16 waters | Decision |
| --- | ---: | --- |
| Compact the Hermite recurrence box | 1336.395 / 1329.701 | No clear latency gain; retain the smaller local workspace |
| Keep both Jacobi matrices in shared memory when capacity permits | 1329.904 / 1160.977 | Retain; 12.7% less complete-call time |
| Use an odd shared leading dimension | 1150.611 / 1062.155 | Retain; 7.7% less complete-call time |
| Skip a primitive whose existing prefactor is exactly zero in a proven finite domain | 1062.163 / 568.395 | Retain; 46.5% less complete-call time |

The first two rows are job 4435438; the latter two are job 4439121.
Different rows can have different baselines because each has its own
controlled comparison. The changes add no per-solve kernel launches, large
intermediate tensor, scientific host calculation or host transfer.

The compact ERI table uses the existing angular bound
`Rx+Ry+Rz <= 5`: its bounding box needs at most 18 cells, rather than
6×6×6, per parity layer. The two-layer table falls from 3,456 to 288 B;
the compiler-reported ERI stack falls from 4,176 to 1,008 B, with 128
registers in both versions. These resource reductions do not establish
a measured reduction in global-memory traffic or reserved VRAM.

Jacobi rotations, pair ownership, sweep limit, relative stopping criterion,
scaling, CUB norms and sorting are unchanged. Setup queries actual static
shared use and the device limit before selecting shared matrices. The
shared leading dimension is `n|1`; padding never enters arithmetic or norms.
Larger matrices use the same CUDA algorithm with global scratch. cuSOLVER
batched Jacobi remains the selected primitive through order 64; previously
evaluated larger vendor paths download norms during sweeps and do not fit
the CUDA-only requirement on this toolkit.

The ERI early return checks the already-computed FP64 prefactor for exact
zero. It introduces **no near-zero cutoff**. A sufficient finite-arithmetic
domain bounds the recurrence inputs by 1e16; outside it the original CUDA
computation continues. This guard preserves exceptional `0*Inf/NaN`
behavior for extreme coordinates. Normal finite underflow returns zero;
signed-zero bits need not be identical. The guard and the existing angular
bound keep every skipped polynomial intermediate below `DBL_MAX`.

## Complete-call timing and CPU comparison

Arrhenius GH200, Release FP64, CUDA 13.0.3, GCC 14.3, OpenBLAS 0.3.32,
eight OpenMP/OpenBLAS CPU threads. Cartesian STO-3G water chains repeat the
existing physical pin at 4 Å intervals. Neutral closed shell, Hcore starts,
tolerance 1e-10, cap 200, DIIS space 8/start 1, damping and level shift zero,
energy only. CPU history resets before every sample. Each solve performs
setup; no cached integrals or previous electronic guess are used.

| Waters / AOs | Final CUDA warm, ms | CPU stored / direct warm, ms | Fastest CPU / CUDA |
| --- | ---: | ---: | ---: |
| 1 / 7 | 8.553 | 0.812 / 1.436 | 0.095 |
| 2 / 14 | 9.919 | 3.074 / 9.387 | 0.310 |
| 4 / 28 | 21.337 | 22.106 / 46.352 | 1.036 |
| 8 / 56 | 90.474 | 253.811 / 211.187 | 2.334 |
| 16 / 112 | 568.395 | 3253.580 / 917.984 | 1.615 |

CPU entries are medians of three warm solves. CUDA entries average the
two paired round medians. At 16 waters these were 566.803 and 569.987 ms,
with ranges 564.218–569.728 and 569.923–583.023 ms. Their first solves took
566.638 and 569.438 ms. CPU direct had a 917.300 ms first solve and
914.942–937.427 ms warm range. CUDA took 18 iterations; CPU took 14 under
its existing stopping rules. CPU is much faster at one and two waters. The
exact-zero change regresses the four-water CUDA median by about 1.35 ms
relative to its immediate baseline. It is retained for the target workload.

A supplementary CPU-stored-plus-CUDA invocation had CUDA median 570.101 ms,
but included a 923.854 ms warm outlier and a 1287.315 ms first solve.
These observations are retained; the results are latency measurements for
this node and workload, not a guarantee for every call or another basis.

The timer includes `RHF.run(frame)`, final GPU synchronization, integrals,
SCF, allocation/teardown and final Frame output. Object construction,
context bootstrap, input Frame copying and CPU history reset are excluded.
The conservative 16 GiB admission limit is not an allocator cap or a peak
measurement. The standing harness checks finite/converged results, Frame
energy, repeated solves, external pins and CPU agreement, failing nonzero
when a gate is missed.

Run through the public molq GH200 workflow and `tools/gpu_job.sh`, building
`bench_rhf_cpu_vs_cuda`, then execute inside the allocation:

```bash
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
  ./build-hf-cuda/bin/bench_rhf_cpu_vs_cuda 16 3 both 1e-10 16384 stored
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
  ./build-hf-cuda/bin/bench_rhf_cpu_vs_cuda 16 3 cpu 1e-10 16384 direct
```

## Numerical and residency validation

The final CUDA total is -1199.38811898183826 Ha. Its difference from the
independent PySCF 2.11.0 Hcore pin (-1199.388118981794 Ha) is about
4.4e-11 Ha; CPU direct differs from CUDA by about 1.31e-8 Ha. All satisfy
the unchanged 1e-6 Ha physical gate. Padded and exact-zero versions have
identical energy strings at the logged 14-decimal precision and unchanged
iteration counts; no bitwise-equality claim follows.

Both final candidates passed **83/83** selected checks: 80 CUDA
unit/contract checks plus the RHF, UHF and MP2 physical cases, including
analytic forces, frozen-core and spin-scaled correlation. The new
112/116/117-order eigen tests retain 5e-13 eigenvalue, residual and
orthogonality gates; the 112-order tests cover large/tiny finite scaling
and repeated spectra. Existing 64/65/215/216 checks remain; generic order
216 is outside molecular HF's CUDA-only envelope. The new single-method ERI
test verifies exact underflow and preservation of exceptional arithmetic;
existing primitive and raised-force gates stay 2e-12 Ha and 2e-11 eV/Å.

Memcheck passed all 25 selected integral/eigen checks with zero errors.
Racecheck passed both primitive checks and shared-matrix tests at 112/116,
four checks total, with zero hazards, errors or warnings. Both jobs completed
with exit zero.

The final job has nine complete, warning-free Nsight captures: three
HF→MP2 variants, ten solves each, plus final eigen solves at
64/65/112/116/117/215. All enable CUDA APIs, memory usage and CPU/GPU UM
fault collection. The six isolated eigen profiles request CUDA tracing;
the composed profiles additionally trace cuBLAS. All **36** observed
scientific windows have zero H2D/D2H copies, runtime/driver allocation/free
calls, dynamic Device allocation events and UM activity. Recorded API
returns are zero. UM tables are absent in the lazy export; this is
interpreted with enabled collection flags and clean diagnostics.

HF→MP2 boundaries are `initialize_packet` before the HF iterations through
correlated `publish_energy`; eigen boundaries cover the solve before final
result retrieval. Setup/input and final host output are outside those
windows. The first composed window has 10 B of immutable module-static
initialization; later ones have none. These are composed HF→MP2 captures,
separate from the RHF-only ordinary timing executable. They do not establish
residency for every basis, force request or parent-driver sequence.

At 112, the final kernel uses 202,496 B of dynamic shared memory, versus
200,704 B before padding. Actual traces select shared matrices at
65/112/116 and global scratch at 117/215. This is observed dispatch, not
a dimension-only inference. Compiler registers for the shared kernel
fall from 52 to 50. No hardware bank-conflict counter was collected, so
the padding result supports the layout change without proving its sole
hardware cause.

The chronological logical Device + Static peak is **1,814,660,441 B** for
each final comparison process, 100,352 B below the pre-shared version.
All 1,005 Device and 30 Pinned allocation/release pairs close. Immutable
module statics retain 1,429 B; Pinned peak is 16 B. These are captured
process allocations, excluding unobserved driver reservations and per-CTA
shared storage, not whole-device usage or reserved VRAM. The device reports
102,005,473,280 B total memory.

Instrumented 16-water warm ERI setup is 823.130/839.312/331.389 ms for
unpadded shared / padded shared / padded plus exact-zero work avoidance.
Corresponding HF→MP2 windows are 336.697/247.836/247.954 ms; the sums of
18 Jacobi kernels are 255.240/166.540/166.657 ms. These profile durations
explain the two separate improvements and are not ordinary wall samples.

The supported envelope is unchanged: the s/p primitive covers energy and
analytic-force calls; other angular-momentum ERI kernels are unchanged.
Dense CUDA HF still requires `nAO^4 <= INT_MAX`, at most 215 AOs. MP2 remains
energy only and consumes RHF's private device state under the same Backend.

The accepted ERI source SHA-256 is
`910371db21528941ec4f7e4cbbb8b7545cc6ac104ee0f9c44f6e782e3fc88519`;
the eigensolver source is
`386c3b580bae75d51a9715319cc15f561e9fceed796cf3110afc4865202ed402`.
The workspace's later line-wrapping pass changes that file's hash to
`effeca9514b5e4771e032d1bbd06299f361993cdc8b9c82622a69a641f488eca`;
its significant tokens, including comments and literals, match the measured
source. The formatting diff and equivalence check are retained with the data.
The HF harness is unchanged from the
[previous controlled comparison](hf-validation.md).
Frozen static executables, per-run hash checks, common resolved vendor
libraries, build/link flags, HF source/binary manifests, raw timings and
trace analyses are retained in `.claude/campaigns/hf-bounded-workspace-20261026/`
and `.claude/campaigns/hf-jacobi-padding-20261026/`.
Validation uses the frozen HF build recorded there. The workspace's
concurrent molpy/build-system migration is outside that build snapshot;
the optimized scientific code matches the measured snapshot.

# CUDA HF → MP2 validation, 2026-10-25

These timings precede the subsequent s/p ERI rolling-layer optimization.
See the [controlled HF comparison](hf-validation.md) for the newer
implementation, including the faster CPU direct reference.

The supported product is canonical closed-shell CUDA RMP2 energy, including
frozen occupied orbitals and independent OS/SS scales. RHF retains its private
device workspace until correlation finishes. Nested methods borrow their
parent's Backend and leave results in its Frame. MP2 force requests reject.
The dense reference requires `nAO^4 <= INT_MAX`, hence at most 215 AOs.

## Complete-call measurements

Arrhenius GH200, Release, FP64, CUDA toolkit 13.0.3, GCC 14.3 and OpenBLAS
0.3.32. CPU uses eight OpenMP/OpenBLAS threads. Water geometries repeat the
`scf/mp2` water pin at 4 Å intervals; Cartesian STO-3G, neutral closed shell,
Hcore start, no damping/level shift, DIIS space 8/start 1, tolerance 1e-10,
cap 200. The requested extra MP2 budget was 16 GiB.

The timer encloses `MP2.run(frame)` and final GPU synchronization, including
integrals, SCF, correlation, allocation/teardown and result output. Object
construction, process/device-context bootstrap and input Frame copying are
outside the timer. Cold means the first measured solve; warm is the median
of three subsequent solves on the same method instance. Each solve still
performs its own setup. CPU and CUDA retain different stopping criteria;
their iteration counts are reported rather than treated as identical work.

| Waters / AOs | CPU cold / warm, ms | CUDA cold / warm, ms | CPU / CUDA warm | CPU / CUDA iterations |
| --- | ---: | ---: | ---: | ---: |
| 1 / 7 | 2.570 / 1.129 | 75.009 / 13.244 | 0.085 | 10 / 16 |
| 2 / 14 | 4.305 / 3.821 | 27.814 / 20.709 | 0.185 | 12 / 15 |
| 4 / 28 | 35.512 / 34.524 | 127.081 / 127.491 | 0.271 | 13 / 17 |
| 8 / 56 | 1083.909 / 1072.578 | 1472.615 / 1467.298 | 0.731 | 14 / 18 |
| 16 / 112 | 31004.920 / 31024.909 | 15738.517 / 15740.691 | 1.971 | 14 / 18 |

At 16 waters, warm ranges were CPU 30,835–31,096 ms and CUDA
15,733–15,742 ms. Smaller cases remain faster on CPU. These are measurements
of these complete calls on this machine, not a portable speedup guarantee.
The eight-water CUDA warm baseline before the symmetry revision was
2942.192 ms (three repeats, same geometry/tolerance). The final revision
measured 1467.298 ms. No eightfold wall-time claim follows from ERI symmetry.

Reproduce through the repository's GH200 job workflow, building
`bench_mp2_cpu_vs_cuda`, then running:

```bash
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
  ./build-hf-cuda/bin/bench_mp2_cpu_vs_cuda 16 3 both 1e-10 16384
```

## Correctness and ownership

All five sizes pass the 1e-6 Ha comparison on total, correlation and both
spin channels. The largest CPU/CUDA total difference is 7.549e-11 Ha.
Independent offline PySCF 2.11.0, Cartesian STO-3G, Hcore start and
`conv_tol=1e-12` gives the 16-water total -1199.9585752021592 Ha; CUDA gives
-1199.95857520221648 Ha. A separate converged MINAO start differs from CUDA
by 2.972e-9 Ha. The oracle wheel reports no OpenMP support; its timings are
not performance comparisons. External pins and geometries live in
[`regressions/scf/mp2/REFERENCE.md`](../../regressions/scf/mp2/REFERENCE.md).

GH200 job 4407744 passed the RHF, UHF and MP2 external cases, the unchanged
64/65/215 eigenvalue/residual/orthogonality gates, physical convergence checks,
Frame preservation, borrowed execution, failure/retry and resource guards.
Its sole failing test incorrectly requested an unavailable F/STO-3G table.
Only that fixture's supported inputs were corrected; job 4407892 then passed
all five selected checks, including the preserved 216-order generic solver.
Production algorithms were unchanged between those jobs. Memcheck reports
zero errors; racecheck reports zero hazards, errors or warnings. CPU Gaussian
and primitive checks, plus all three external cases, also pass locally.

The scaling test exposed and fixed two CPU contract defects: mutable molecular
single points silently applied an anchor minimum image despite const/force
entries using the supplied geometry, and MP2 returned the correlated total
while leaving only HF energy in the Frame. CPU/GPU now evaluate the same
cluster and publish the same total-energy contract.

## Residency and resources

Nsight captures enable all CUDA APIs, memory usage and both CPU/GPU UM fault
collection. The 19 observed scientific windows comprise ten HF→MP2 solves
(cold/warm at each size), one eigensolve at each 64/65/215 and six MINAO
projections with reused setup and changed geometry. Every window records:

- Zero H2D/D2H copies, runtime/driver allocation/free calls, dynamic Device
  allocation events and UM activity records.
- Clean trace diagnostics and recorded API return codes.

Window boundaries are MP2 `initialize_packet` through `publish_energy`,
eigen `pack_lower`/`cyclic_jacobi` through the start of final result output,
and MINAO cross-overlap through spin-density publication. Setup/input and
final output are outside these windows. Cold immutable module-static events
are separate: 10 B in the first MP2 and 64-order eigen windows, 32 B in the
first MINAO window. They are not dynamic algorithm scratch.

The whole MP2 process's chronological logical Device + Static peak is
1,814,760,793 B. Device/Pinned allocation pairs close completely. This is a
captured process allocation peak, not reserved VRAM or whole-device usage;
the runtime reports 102,005,473,280 B of device memory. The method's default
512 MiB budget bounds extra correlation scratch, excluding HF and library
contexts. No full MO ERI or per-contribution array is materialized.

## Primitive choices and remaining cost

cuBLAS owns the four occupied-tile contractions and DIIS Gram/extrapolation;
CUB owns spin-channel reductions. cuSOLVER LU solves the small Pulay system.
UHF MINAO uses device overlap/normalization, cuSOLVER Cholesky/solve and cuBLAS
density construction, with only immutable element metadata prepared on CPU.

cuSOLVER batched Jacobi is reused through order 64. Larger library candidates
downloaded norms during sweeps, and the installed generic ALGO_1 option was
rejected at setup. Orders 65..215 therefore use bounded, device-controlled
cyclic Jacobi with CUB norms/sorting and setup-owned matrix scratch. The
legacy generic/synthetic branch above 215 retains its original behavior and
is outside the CUDA-only HF envelope.

Real s/p ERIs compute one representative of each eightfold symmetry orbit,
using the existing contraction and the same dense output. The profiled
16-water setup ERI cost remains about 15.19 s, versus 0.523 s in the resident
HF→MP2 science window. Setup is included in the measured complete call.
This is the next measured bottleneck; the result does not justify further
splitting or storing extra tensors merely to use another library.

Raw logs, trace SQL/flags, allocation reconstruction, source/trace hashes and
offline oracle scripts are retained locally in
`.claude/campaigns/hf-cuda-20261025/`.

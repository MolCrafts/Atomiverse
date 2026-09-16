# Reference — plane-wave PBE GGA bulk silicon on CUDA (`cuda::pw::PW`)

## Method

Spin-unpolarized plane-wave Kohn–Sham DFT with **true PBE GGA**
exchange–correlation (Perdew, Burke, Ernzerhof, *Phys. Rev. Lett.* **77**,
3865 (1996)): PBE exchange + PW92-based PBE correlation, with spectral
density gradients on the FFT grid. Pseudopotential is the in-tree analytic
GTH parameterization `GTH-PBE-q4` (Goedecker–Teter–Hutter / Hartwigsen–Goedecker–Hutter).

Public path (both legs): set members on the method object → `build(frame)` →
`run(frame)`.

| Leg | Type | Where the work runs |
| --- | ---- | ------------------- |
| `cpu` | `cpu::pw::PW` | everything on the host |
| `cuda` | `cuda::pw::PW` | Davidson + density side (ρ → V_H → **PBE V_xc** with ∇ρ / −∇·flux → V_eff → Pulay) on the GPU |

## Gate — same-settings CPU parity (the CUDA-slice oracle)

Per the ES optimization order, a **same-settings CPU result is the correctness
oracle for a CUDA method slice**. With `--device=both` the binary runs both
legs on one frame and asserts

\[
|E_\mathrm{cuda} - E_\mathrm{cpu}| < 1.0\times10^{-3}~\mathrm{Ry}.
\]

There is **no absolute literature energy pin** on this slug: the twin uses
analytic GTH at deliberately under-converged ecut/k (tutorial settings), and
no offline QE/CP2K GTH-PBE Si twin was re-run for this CUDA gate. Absolute
digits would be a self-golden if invented; the CPU twin is the only honest
oracle here (`oracle: cpu-twin`).

The 1.0e−3 Ry parity bound is the **design-time** value shared with
`pw-cuda-lda-si`. Re-pin from the first GH200 run; do not tighten at design
time from a single measurement.

## Twin geometry and settings

| Setting | Value |
| ------- | ----- |
| Structure | diamond Si, 2-atom fcc primitive cell |
| Lattice constant \(a\) | 5.43 Å |
| Basis atoms | (0,0,0), \((a/4,a/4,a/4)\) |
| Cell rows | \((0,h,h)/(h,0,h)/(h,h,0)\), \(h = a/2\) |
| Pseudopotential | analytic `GTH-PBE-q4` (both legs) |
| \(E_\mathrm{cut}\) | 15 Ry |
| k-mesh | \(2\times2\times2\) Monkhorst–Pack |
| `xc` | `XCType::PBE` on **both** legs |
| SCF tol | \(10^{-6}\) (density change) |
| `max_iter` | 50 |

## Unit-level evidence (not this binary)

Pointwise device PBE vs host `evaluate_xc(PBE,…)` lives in
`tests/cuda/planewave/test_xc_pbe.cu` (tol **1e-12·max(1,\|ref\|)** including
flux lanes). Full DeviceDensity stage parity (Poisson + PBE XC + V_eff)
lives in `tests/cuda/planewave/test_density.cu`
(`CudaPWDensity.HartreeXcVeffMatchCpuPbe`) at the LDA-stage bounds.

## Performance

**perf_exempt.** Two-atom cell is correctness-only (ES optimization order
step 4). No wall-clock ratio is recorded and none may be inferred.

## Measurement log

| Item | Value |
| ---- | ----- |
| GH200 job id | **1048320** (NVIDIA GH200) |
| Toolchain | `GPU/buildenv-gcccuda/2026.03-cu13.0`, nvcc 13.0 / GNU 14.3, `-DCMAKE_CUDA_ARCHITECTURES=90` |
| Log | `.claude/cuda-pw-pbe-device-1048320.out` |
| Unit suite | **100%** (4/4): `test_cuda_pw_xc_lda`, `test_cuda_pw_xc_pbe`, `test_cuda_pw_guards`, `test_cuda_pw_density` |
| Measured \(E_\mathrm{cpu}\) | **−14.84955549 Ry** (7 iterations) |
| Measured \(E_\mathrm{cuda}\) | **−14.84955549 Ry** (7 iterations) |
| Measured \(\lvert\Delta\rvert\) | **1.421 × 10⁻¹⁴ Ry** |
| Gate (CPU parity) | PASS — 1.421e−14 Ry < 1.0e−03 Ry |

Both legs converge in the same iteration count (7): device Pulay reproduces the
host trajectory, not only the fixed point.

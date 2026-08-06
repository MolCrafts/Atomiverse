# CUDA RHF — device-only HF energy + s/p forces (2026-08-06)

## Goal

After the public **shape** (`cuda::scf::RHF{basis}.run(frame)`), the energy
path for systems the device kernels cover must be **CUDA-only**: integrals +
SCF loop on device, no host ERI materialization. Wave 6 adds **s/p** analytic
forces on device (ends host force fallback for STO-3G H…O / H₂O). Wave 8
wires **level_shift** on the device s/p Fock path.

## Current wiring (`cuda::scf::RHF::launch`)

| Case | Path |
|---|---|
| STO-3G, every atom **s/p** (l ≤ 1; H…O, **mixed Z OK**) | **Device energy**: molecular shell list → `RHFFockDevice` + `launch_*_sp` |
| STO-3G s/p (l ≤ 1; H…O, H₂O) + `compute_force` | **Device forces**: `launch_build_W_rhf` + `launch_rhf_forces_sp` → `frame.fx/y/z` [eV/Å] |
| Higher L / other bases | Host fill S/Hcore/ERI → upload → device SCF; host forces |

`level_shift` (Saunders–Hillier virtual shift) is wired on **both** host-fill
and device s/p Fock paths: orthonormal `F' += λ(I − P'_occ)` from previous
`C_prime`, threshold `|λ| ≤ 1e-4` no-op, post-convergence virtual de-shift on
`eig`. Energy uses unshifted F so converged E is λ-invariant. Device path is
**not** refused when λ ≠ 0.

**DRY:** never retype STO-3G numbers — `basis_data/accessors.h` over `sto3g.h`.

## Process (binding for this loop)

When **writing or optimizing CUDA kernels**, load and follow the matching
Grok CUDA skills (not ad-hoc style):

| Concern | Skill |
|---|---|
| Block/grid size, occupancy, tail | `choose-launch-configuration` |
| Coalescing / layout | `optimize-global-memory-access` |
| Elementwise fuse (e.g. scale·rsqrt) | `fuse-elementwise-ops` |
| Partial tiles / `idx < N` | `handle-boundary-conditions` |
| Numerics (thresh, rsqrt, fp) | `write-numerically-stable-kernel` |
| Warp divergence | `avoid-warp-divergence` |
| Correctness after change | `debug-cuda-kernel-correctness` |

## Waves

| # | Status | Goal |
|---|---|---|
| **0** | **done** | Product path pure-s → `RHFFockDevice` + table-driven contraction |
| **1** | **done** | Löwdin `S^{-1/2}` fully on device |
| **2** | **done** | Geometry pack + V_nuc on device; type-only D2H for path select (superseded by host-Z cache) |
| **3** | **done** | Multi-shell pure-s AO count |
| **4** | **done** | p-shell device integrals (l≤1 MD/OS); H₂O energy cuda-only |
| **5** | **done** | Pure-s device analytic forces; async density energy write; no mid-build fence before integrals; `forces()` via `run()` |
| **6** | **done** | s/p device analytic forces (`launch_rhf_forces_sp`); ends host force fallback for l≤1 |
| **7** | **landed** | Device-only polish: drop mid-build V_nuc D2H/H2D (energy ← `d_V_nuc`); ERI `__launch_bounds__(256,2)` + occupancy-capped grid-stride |
| **8** | **landed** | Device `level_shift` on s/p Fock (`RHFFockDevice` + `SCFSolver`); no host-fill when λ≠0 |
| **9** | **landed** | Path select from host Z before upload (`cache_host_types`); no setup type[] D2H on `run`/`forces` s/p path |
| **10** | **landed** | Device CDIIS (Pulay) Fock mixer + level shift in mix step (CPU MixStage twin); `ws.S` for e=Xᵀ[F,P]_S X; history on device |

## Related CUDA ES (not RHF waves)

| Slice | Status | Notes |
|---|---|---|
| Γ GPW Poisson/FFT foothold | **landed** (`cuda-gpw-foothold`) | `src/cuda/gpw/internal/poisson_fft.{cu,cuh}`; unit vs CPU oracle; **not** full GPW product |
| Padé LDA XC grid foothold | **landed** (`cuda-gpw-xc-lda-foothold`) | `src/cuda/gpw/internal/xc_lda.{cu,cuh}`; `test_cuda_gpw_xc_lda` vs `cpu::xc::pade_lsd`; per-point ~1e-12, E ~1e-10 rel; **not** full GPW product |
| product ρ collocate foothold | **landed** (`cuda-gpw-rho-collocate-foothold`) | `src/cuda/gpw/internal/product_collocate.{cu,cuh}`; `test_cuda_gpw_product_collocate` vs continuous host + `collint_collocate_product`; s/p + multi-s–s; **not** full GPW product |

## Wave 6 layout

- **W** on device: `W = 2 Σ_{i<n_occ} ε_i C_i C_iᵀ` (`launch_build_W_rhf`).
- **Forces** [Ha/Bohr → eV/Å]: raise/lower on l≤1 (raise to d); AO maps live after SCF.
- **API:** `RHF::forces` → `run()` with `compute_force`; `last_force_on_device()`.

## No scientific D2H during calculation (device s/p path)

| Phase | Traffic |
|---|---|
| **Setup** (before SCF) | host Z from `run`/`forces` (`host_Z_`); H2D of α/c/L; device `V_nuc` atomics — **no** type[] D2H on covered path |
| **SCF + s/p forces** | **No** `cudaMemcpy` Device→Host of P, C, ERI, residual, energy, V_nuc |
| Residual | cuBLAS `nrm2` → `d_residual`; converge flag via **mapped** host page |
| Energy | cuBLAS `dot` device pointer mode + kernel reads `d_V_nuc` → `frame.scalars->energy` |
| V_nuc | stays on `d_V_nuc` only (no mid-build host recompute rewrite) |
| **After SCF** | optional residual / V_nuc / `energy_ha_` D2H for host bookkeeping only |

## Host still

- host Z for path select (seeded from host Frame before upload; one-time D2H only if Component-only path never saw host)
- post-calc energy_ha_ / residual mirrors
- host-integral + host-force path for non-s/p bases (higher L)
- (level_shift stays on device s/p energy when covered — not a host-only knob)

## Validation

- GH200 wave 6 (`.claude/merge-cuda-rhf-913053.out`, tip `8337251b`):
  `test_cuda_scf_rhf` + `test_cuda_scf_device_integrals` **100% PASS**.
- GH200 wave 7 (`.claude/merge-cuda-polish-913612.out`, tip `fb9e7ed8` /
  feat `1ae27395`): same pair **100% PASS** (2.29 s + 0.82 s). Configure with
  **`-DCMAKE_CUDA_ARCHITECTURES=90`** on GH200; do not link x86 `libxtb`
  (`-DATV_BUILD_SEMIEMPIRICAL=OFF` or null `ATV_XTB_ROOT`).
- GH200 wave 8 level_shift (worker job **913709**, feat `77eb7f79` / merge
  `ba3a8145`) + post-merge log `.claude/merge-escuda-cuda-915584.out` tip
  `9f2d791c`: `test_cuda_scf_rhf` **Passed 2.04 s** +
  `test_cuda_scf_device_integrals` **Passed 0.85 s**;
  `CUDALevelShiftEnergyInvariant` λ=0.5 matches λ=0 and CPU @ **1e-8 Ha**;
  `last_scf_on_device()` sticky.
- GH200 GPW foothold (worker **915518**, feat `f604577c` / merge `9f2d791c`) +
  same post-merge log: `test_cuda_gpw_device_poisson` **Passed 0.97 s** +
  `test_cuda_pw_poisson` **Passed 1.78 s** (**100%** of 4). Tols: max\|ΔV\|
  **1e-9…1e-10**, Hartree rel **1e-10…1e-12** (cuFFT vs FFTW, not bit-identical).
- GH200 wave 9 path-select-host (worker job **915654**, feat `6f179574` /
  merge `44a5d858`) + post-merge log `.claude/merge-escuda-cuda-915704.out`
  tip `44a5d858`: `test_cuda_scf` + `test_cuda_scf_rhf` +
  `test_cuda_scf_device_integrals` **100% PASS** (0.95 + 2.03 + 0.87 s);
  host `cache_host_types` before upload — no setup type[] D2H on covered
  `run`/`forces`.
- GH200 GPW Padé LDA XC foothold (worker **915678**, feat `90acb1d5` / merge
  `cc450b59`) + same post-merge **915704**: `test_cuda_gpw_xc_lda` **Passed
  0.86 s** (+ poisson/PW LDA still green). Tols: per-point ~**1e-12**, energy
  reduction rel **~1e-10** (device atomic sum vs host).
- GH200 device CDIIS (worker job **915746**, feat `f21a1173` / merge
  `d575e50b`) + post-merge log `.claude/merge-escuda-wave5-915778.out` tip
  `759ca486`: `test_cuda_scf` + `test_cuda_scf_rhf` +
  `test_cuda_scf_device_integrals` **100% PASS** (0.90 + 2.02 + 0.82 s);
  energy/forces/level_shift twins green at existing tols
  (ha_to_ev(1e-6)/1e-8 Ha class). CDIIS after Fock; history on device; tiny
  (m+1) Pulay solve host-only.
- GH200 GPW product ρ collocate foothold (worker **915754**, feat `0d2868e7` /
  merge `759ca486`) + same post-merge **915778**:
  `test_cuda_gpw_product_collocate` **Passed 0.80 s** (+ poisson/XC + full RHF
  suite **6/6 PASS**, 7.37 s total). Tols: device≡continuous host **1e-14**;
  vs collint **1e-10·peak + 1e-14** (p-type **1e-9·peak + 1e-13**); ρ sum rel
  **~1e-10**.
- `tests/cuda/scf/test_rhf.cu`:
  - `CUDAForcesMatchCPU_H2_STO3G` — |ΔF| **1e-5** eV/Å; energy **ha_to_ev(1e-6)**
  - `CUDAForcesMatchFD_H2_STO3G` — FD bar **5e-2** eV/Å
  - `CUDAForcesMatchCPU_H2O_STO3G` — |ΔF| **1e-4** eV/Å; `last_force_on_device()`
  - `CUDAForcesMatchFD_H2O_STO3G` — O-z FD spot **1e-1** eV/Å
  - `CUDALevelShiftEnergyInvariant` — λ=0.5 vs λ=0 and CPU **1e-8 Ha**
- H₂ textbook energy ≈ −1.1167 Ha; pure-s / s/p must not call host RHF for
  forces when l≤1.

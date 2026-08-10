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
| **8** | **landed** | Device `level_shift` on s/p Fock (`RHFFockDevice` + private cycle); no host-fill when λ≠0 |
| **9** | **landed** | Path select from host Z before upload (`cache_host_types`); no setup type[] D2H on `run`/`forces` s/p path |
| **10** | **landed** | Device CDIIS (Pulay) Fock mixer + level shift in mix step (CPU MixStage twin); `ws.S` for e=Xᵀ[F,P]_S X; history on device |

## Related CUDA ES (not RHF waves)

| Slice | Status | Notes |
|---|---|---|
| Γ GPW Poisson/FFT foothold | **landed** (`cuda-gpw-foothold`) | `src/cuda/gpw/internal/poisson_fft.{cu,cuh}`; unit vs CPU oracle; **not** full GPW product |
| Padé LDA XC grid foothold | **landed** (`cuda-gpw-xc-lda-foothold`) | `src/cuda/gpw/internal/xc_lda.{cu,cuh}`; `test_cuda_gpw_xc_lda` vs `cpu::xc::pade_lsd`; per-point ~1e-12, E ~1e-10 rel; **not** full GPW product |
| product ρ collocate foothold | **landed** (`cuda-gpw-rho-collocate-foothold`) | `src/cuda/gpw/internal/product_collocate.{cu,cuh}`; `test_cuda_gpw_product_collocate` vs continuous host + `collint_collocate_product`; s/p + multi-s–s; **not** full GPW product |
| product V·shape integrate foothold | **landed** (`cuda-gpw-product-integrate-foothold`) | `src/cuda/gpw/internal/product_integrate.{cu,cuh}`; reuses `DeviceProduct`/`CollocateGridDesc`; `test_cuda_gpw_product_integrate` vs continuous host + `collint_integrate_product`; s/p + multi-s–s; **not** full GPW product |
| composed KS ρ→V_eff stage unit | **landed** (`cuda-gpw-ks-chain`) | `src/cuda/gpw/internal/ks_chain.{cu,cuh}` reuses collocate/Poisson/XC/integrate; `test_cuda_gpw_ks_chain` + dual-gate reg vs `ks_grid_build_rks`; GH200 **1049858**; LDA single-grid only; **not** full GPW product |
| closed-shell PBE-GGA KS stage | **landed** (`cuda-gpw-ks-gga`) | `SpectralFftDevice` + `KsXcKind::PBE` reuses `xc_pbe_grid`; `test_cuda_gpw_ks_gga` + dual-gate vs `ks_grid_build_rks(PBE)`; GH200 **1050680** (rebased); **not** full GPW product |
| multi-grid product KS stage | **landed** (`cuda-gpw-ks-multigrid`) | `MultigridTransferDevice` + `KsMultiGridChainDevice`; `test_cuda_gpw_ks_multigrid` + dual-gate reg; GH200 **1051042** max\|ΔV_eff\|=8.1e-10; **not** full GPW product |
| **PW method slice** | **landed** (`cuda-pw-method-slice`) | `cuda::pw::PW` method object (`cuda/planewave/pw_solver.{cuh,cu}`) + `internal::DeviceDensity` — density side (ρ(G)→V_H→V_xc→V_eff(r)→Pulay→reductions) device-resident. See below. |
| PW analytic ionic forces | **landed** (`cuda-pw-forces`) | `internal/pw_forces.{cu,cuh}` + `cuda::pw::pw_forces_analytic`; local HF + Ewald + KB; GH200 **1050105** max\|ΔF\|=2.86e-13 eV/Å vs CPU (LSDA reg); **not** spin/smear |
| PW PBE analytic ionic forces | **landed** (`cuda-pw-forces-pbe`) | same public API under `XCType::PBE`; force path shares `den.stream()`; GH200 post-merge **1050801** max\|ΔF\|=2.63e-13 eV/Å (tol 5e-3); E both −13.89672028 Ry; **not** stress/spin/smear |
| PW analytic cell stress | **landed** (`cuda-pw-stress` + `cuda-pw-pbe-stress`) | `internal/pw_stress.{cu,cuh}` + `cuda::pw::pw_stress`; Nielsen–Martin NSCF strain-FD + ρ·Ω₀/Ω remap; GH200 **1050604** max\|Δσ\|=1.36e-12 eV/Å³ (LSDA Si) + **1123959** max\|Δσ\|=5.585e-12 eV/Å³ (PBE Si; `reg_cuda-pw-stress-pbe`); **not** spin/smear/multi-SCF FD product |
| PW collinear LSDA spin | **landed** (`cuda-pw-collinear-spin`) | nspin=2 density + VWN5 spin XC + fixed-occupation UKS; GH200 **1051041** O₂ \|ΔE\|=1.07e-13 Ha (tol 2e-3); forces/stress reject spin; **not** smear |
| PW collinear spin-PBE energy | **landed** (`cuda-pw-spin-pbe`) | true spin-GGA on device (`xc_pbe_spin_grid` + spectral ∇ρ_σ); GH200 **1124645** O₂ \|ΔE\|=1.14e-13 Ha (tol 2e-3); forces/stress/smear still reject |
| PW restricted metallic smear | **landed** (`cuda-pw-metallic-smear`) | FD/Gaussian/MP + extra_bands + Ritz cache; GH200 **1051119** Al \|ΔE\|=1.24e-14 Ry; spin+smear / forces / stress reject |
| RHF non-default stream D2H | **landed** (`cuda-rhf-nondefault-stream`) | residual/V_nuc async on Evaluate stream; GH200 **1051227** 2/2 PASS |

### CUDA PW slice — envelope and evidence

Public shape mirrors `cpu::pw::PW`: public members → `build(frame)` →
`run(frame)` → `PWResult`. The free function `cuda::pw::pw_scf` is **deleted**
(no shim). Device envelope — everything else **throws** `std::invalid_argument`
naming the member and the way out: closed-shell **or collinear fixed-occupation
UKS with `XCType::LSDA` or `XCType::PBE`** (`spin_polarized` + integer mag),
restricted metallic smear (FD/Gaussian/MP; not with spin), `xc=XCType::LSDA` or
`XCType::PBE` (closed-shell or spin), `k_mesh>=1`, `ecut_ry>0`. k-points are
**not** MPI-distributed here. Spin+smear, spin forces, and spin stress still
reject.

Per-iteration PCIe traffic, density side: Pulay Gram ≤ 1 KB (only once history
≥ 2) + ≤ 8 coefficients + 5 energy/residual doubles. **No O(N_G) / O(N_r)
density-side array crosses.**

Band side, after `cuda-pw-kpoint-hoist`: `build_kpoints(prob)` uploads every
per-k table once per run — G→FFT index map, T(G), q = k+G, KB projector
descriptors + radial f_G, **and the initial Davidson subspace**. `init_subspace`
reads only T_G / atom_wfc / gvec / atom_pos_bohr, i.e. it is pure geometry, so
the old loop was recomputing an identical starting guess (including an n_G sort)
and re-uploading it every k every iteration. `bind_kpoint(ik)` is now a pointer
swap plus one β launch: **no allocation, no H2D, no `cudaDeviceSynchronize`**,
and `solve()` seeds the subspace device→device.

β is deliberately **not** cached across k: it is the one O(n_G·n_proj) array
here, so caching it over all k would reach TB scale on a real cell. Its inputs
are cached instead. `build_kpoints` sizes the tables against `cudaMemGetInfo`
and throws with the figure rather than letting a bare `cudaMalloc` fail;
k-batching is the answer if a cell ever exceeds that, and is not implemented.

After `cuda-pw-zheevd`, the subspace eigenproblem runs on device as well:
cuSOLVER `Zheevd` in place on the column-major `d_hs_` cuBLAS just wrote.
`CUBLAS_FILL_MODE_UPPER` reads exactly the `mu <= nu` triangle the CPU
assembly read, and LAPACK ignores the diagonal's imaginary part just as the
CPU forced it to zero — so no transpose/conjugation convention is left to get
wrong (the earlier row-major rebuild existed precisely because that trap is
easy to fall into). Zheevd overwrites `d_hs_` with the eigenvectors, whose
first n_bands columns **are** the Ritz rotation, so the ZGEMMs consume it
directly and the `d_x_` buffer is gone. Removed per Davidson round: a
n_sub² D2H, the host row-major Hermitian rebuild, the host Jacobi sweep
(O(n_sub³)), and a cap×n_bands H2D.

**This wave is deliberately not bit-identical** — the first in the PW slice
that isn't. CPU Jacobi vs device divide-and-conquer: eigenvalues agree to
roundoff, eigenvectors differ by a phase and, inside a degenerate multiplet,
by an arbitrary unitary rotation. The density is invariant under both *only
if a degenerate multiplet is not split across the n_bands cut* — true for
fixed closed-shell occupations on a gapped system, stated in the header rather
than assumed. Individual residuals do differ, so the Davidson path may diverge
from the CPU one; the converged eigenvalues are the contract.

After `cuda-pw-devptr-nrm2`, the residual-expansion norms are device-pointer
too. The CPU solver branches three times per band (converged band, then
rounding-level before and after Gram-Schmidt) and each branch cost a
host-pointer `cublasDznrm2`, i.e. a sync. All three stages now run
**unconditionally** with device-pointer norms into `d_nrm_[3]`, device-computed
guarded reciprocals in `d_inv_[2]` feeding `cublasZdscal` with a device alpha,
and the three predicates folded into one `accept` int — **one sync per band
instead of three**. Predicates and arithmetic are untouched, so results are
bit-identical; the price is that a *rejected* band now pays two extra ZGEMVs it
used to skip. `k_safe_reciprocal` returns 0 rather than inf at zero norm, which
is what keeps the unconditional path finite.

A second cuBLAS handle `blas_dev_` is pinned to `CUBLAS_POINTER_MODE_DEVICE`
rather than toggling `blas_`: the mode also governs GEMM's alpha/beta, and a
throw between set and reset would leave the shared handle wrong. ZGEMV keeps
using `blas_` — its scalars are host constants.

Gram-Schmidt is genuinely sequential across bands (band b projects out the
vectors appended earlier in the same round), so the bands cannot be batched
without changing the algorithm. One flag readback per band is the floor for
this structure.

Still on the host per Davidson round (named, not claimed fixed): the 4-byte
Zheevd `devInfo` check, the n_bands eigenvalues (kernel scalar arguments for
the residual shift and preconditioner), and the accept flag above. **Device
PBE-GGA landed** (`cuda-pw-pbe-device`, GH200 **1048320** — dual-gate
|ΔE|=1.421e-14 Ry on Si GTH-PBE-q4). **Non-default stream landed**
(`cuda-pw-nondefault-stream`, GH200 **1049681** — 9/9 ctest, MAX_EIG_DIFF
5.274e-15 Ha, |ΔE|=1.243e-14 Ry; density owns stream, band solver borrows).
**Large-cell walls (ES order step 4) are filled** — GH200 job **1035278**:
64-atom Si Γ LSDA, CUDA `run(frame)` **9.664 s** vs same-settings CPU **>5400 s**;
216-atom CUDA **191 s**; slice sanity |ΔE|=1.243e-14 Ry. Detail:
[cuda-pw-large-cell.md](cuda-pw-large-cell.md). 8-atom cells remain
correctness-only. **Analytic ionic forces landed** (`cuda-pw-forces`, GH200
**1050105**). **PBE force dual-gate landed** (`cuda-pw-forces-pbe`, GH200
**1050801** max\|ΔF\|=2.63e-13 eV/Å). **Analytic cell stress landed**
(`cuda-pw-stress`, GH200 **1050604** max\|Δσ\|=1.36e-12 eV/Å³, E both
−13.86296636 Ry). **GPW closed-shell PBE KS stage landed**
(`cuda-gpw-ks-gga`, GH200 **1050680** / post-merge **1050825**). **GPW
multi-grid product KS stage landed** (`cuda-gpw-ks-multigrid`, GH200
**1051042**). **Collinear LSDA spin landed** (`cuda-pw-collinear-spin`, GH200
**1051041**). **Restricted metallic smear landed** (`cuda-pw-metallic-smear`,
GH200 **1051119** dual-gate Al \|ΔE\|=1.24e-14 Ry). **RHF non-default stream
D2H landed** (`cuda-rhf-nondefault-stream`, GH200 **1051227** 2/2 PASS).
Residual open depth: spin+smear / spin forces / spin stress / **spin-PBE F/σ**
(+ spin-smear F/σ after spin-smear energy; **spin multi-k dual-gate**;
spin-smear-pbe F/σ); collinear spin-PBE **energy** landed (`cuda-pw-spin-pbe`,
GH200 **1124645**). GPW first product + **device mid-SCF collocate** **landed**
(GH200 **1125709**) + **product multigrid energy** (**1126433**) + **Γ UKS
energy** (**1126430**) — depth residual on es-cpu-cuda queue (DIIS/eigen,
device forces, MG/UKS F/σ + kpoints joint paths, …).

Three latent defects were found and fixed **in the pre-existing** foothold
kernels while wiring them in — all three would have made the SCF silently
wrong, and none was caught by the existing unit tests:

| Where | Defect |
|---|---|
| `pw_scf.cu` (deleted) | called `XCBuilder::step(ws)` with the `XCType::PBE` default — the "CUDA LDA" path was running true GGA |
| `veff_assemble_g` | took `V_loc` as **real**, dropping `Im[Σ_a e^{−iG·R_a}]`; the old test fed a real V_loc so it never surfaced |
| `xc_lda_grid` | computed Slater + **PW92**, which no public `XCType` selects; `XCType::LSDA` is Slater + **VWN5** (now `xc_lsda_grid`, with `IsVwn5NotPw92` guarding regression) |

Evidence, GH200 job **948152** (nvcc 13.0.88, `sm_90`), log
`.claude/cuda-pw-slice-948152.out`; re-run with `sbatch
tools/gpu_job_cuda_pw.sh`. Job **948055** measured the same slice before the
review-fix round and produced **bit-identical** numbers — that equality is the
evidence the fix round changed no physics:

- `test_cuda_pw_{fft,poisson,grid_map,xc_lda,veff,guards,density}` +
  `test_cuda_scf_rhf` — **8/8 PASS**, 8.72 s.
- `diag_band_parity` — **MAX_EIG_DIFF 6.051e-15 Ha**.
- `regressions/pw/pw-cuda-lda-si --device=both` — E_cpu = E_cuda =
  **−15.67177311 Ry**, both 5 iterations, **|Δ| = 2.132e-14 Ry**; inside the
  PseudoDojo/ABINIT window. Parity bound stays at the design-time 1e-3 Ry —
  do not tighten from one cell.
- k-point hoist (job **983622**): all three figures above **bit-identical** to
  the pre-hoist tree — the correctness bar for a wave that only deletes
  redundant recomputation.
- device-pointer `cublasDznrm2` (job **988030**): 8/8 ctest; MAX_EIG_DIFF
  **5.274e-15 Ha** and |E_cuda − E_cpu| **1.243e-14 Ry**, both *exactly* the
  987865 figures — bit-identity is the evidence the unconditional restructure
  preserved every predicate.
- cuSOLVER `Zheevd` (job **987865**): 8/8 ctest; MAX_EIG_DIFF
  **6.051e-15 → 5.274e-15 Ha**; |E_cuda − E_cpu| **2.132e-14 → 1.243e-14 Ry**;
  iterations 5 → 5. The figures **move**, which is the expected signature of a
  different eigensolver — a bit-identical result here would have meant Zheevd
  was not actually on the path. Both land at roundoff, an order of magnitude
  or more below the ~1e-12 / ~1e-10 that was predicted before the run.

**No speed claim for the hoist.** 7 timed runs of the CUDA leg per tree,
same node type and settings: hoisted median **2.31 s** (min 1.79, spread
1.79–2.52), baseline median **2.32 s** (min 1.68, spread 1.68–2.53). The
within-tree spread is ±20 %, far wider than the between-tree difference, so
this 2-atom cell **cannot resolve** the change — process startup and CUDA
context init dominate its ~2 s. The wave is justified structurally (no
allocation and no device sync left in the k-loop; removed work scales as
n_k · max_iter · N_atoms · n_proj · n_G), not by this measurement. Per the ES
optimization order, small cells are correctness-only and never the speed story.

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
- GH200 GPW product V·shape integrate foothold (feat `2b317beb` / merge
  `a9f87206`) + post-merge log `.claude/merge-escuda-wave6-915825.out` tip
  `a9f87206`: `test_cuda_gpw_product_integrate` **Passed 0.95 s** +
  `test_cuda_gpw_product_collocate` **Passed 2.00 s** + `test_cuda_scf_rhf`
  **Passed 2.25 s** (**3/3 PASS**, 5.22 s). Tols: device≡continuous host
  **1e-12·peak + 1e-14**; vs collint **1e-10·peak + 1e-14** (p-type
  **1e-9·peak + 1e-13**); not full GPW product.
- `tests/cuda/scf/test_rhf.cu`:
  - `CUDAForcesMatchCPU_H2_STO3G` — |ΔF| **1e-5** eV/Å; energy **ha_to_ev(1e-6)**
  - `CUDAForcesMatchFD_H2_STO3G` — FD bar **5e-2** eV/Å
  - `CUDAForcesMatchCPU_H2O_STO3G` — |ΔF| **1e-4** eV/Å; `last_force_on_device()`
  - `CUDAForcesMatchFD_H2O_STO3G` — O-z FD spot **1e-1** eV/Å
  - `CUDALevelShiftEnergyInvariant` — λ=0.5 vs λ=0 and CPU **1e-8 Ha**
- H₂ textbook energy ≈ −1.1167 Ha; pure-s / s/p must not call host RHF for
  forces when l≤1.

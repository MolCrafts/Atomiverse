# CUDA RHF — device-only HF energy + pure-s forces (2026-07-30)

## Goal

After the public **shape** (`cuda::scf::RHF{basis}.run(frame)`), the energy
path for systems the device kernels cover must be **CUDA-only**: integrals +
SCF loop on device, no host ERI materialization. Wave 5 adds pure-s analytic
forces on device and removes needless mid-path stream fences.

## Current wiring (`cuda::scf::RHF::launch`)

| Case | Path |
|---|---|
| STO-3G, every atom **s/p** (l ≤ 1; H…O, **mixed Z OK**) | **Device energy**: molecular shell list → `RHFFockDevice` + `launch_*_sp` |
| STO-3G s/p (l ≤ 1; H…O, H₂O) + `compute_force` | **Device forces**: `launch_build_W_rhf` + `launch_rhf_forces_sp` → `frame.fx/y/z` [eV/Å] |
| Higher L / other bases | Host fill S/Hcore/ERI → upload → device SCF; host forces |

`level_shift != 0` is refused on the device s/p energy path (not wired).

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
| **2** | **done** | Geometry pack + V_nuc on device; type-only D2H for path select |
| **3** | **done** | Multi-shell pure-s AO count |
| **4** | **done** | p-shell device integrals (l≤1 MD/OS); H₂O energy cuda-only |
| **5** | **done** | Pure-s device analytic forces; async density energy write; no mid-build fence before integrals; `forces()` via `run()` |
| **6** | **done** | s/p device analytic forces (`launch_rhf_forces_sp`); ends host force fallback for l≤1 |

## Wave 6 layout

- **W** on device: `W = 2 Σ_{i<n_occ} ε_i C_i C_iᵀ` (`launch_build_W_rhf`).
- **Forces** [Ha/Bohr → eV/Å]: raise/lower on l≤1 (raise to d); AO maps live after SCF.
- **API:** `RHF::forces` → `run()` with `compute_force`; `last_force_on_device()`.

## No scientific D2H during calculation (device s/p path)

| Phase | Traffic |
|---|---|
| **Setup** (before SCF) | type[] D2H once (path select + shell table); H2D of α/c/L |
| **SCF + pure-s forces** | **No** `cudaMemcpy` Device→Host of P, C, ERI, residual, energy, V_nuc |
| Residual | cuBLAS `nrm2` → `d_residual`; converge flag via **mapped** host page |
| Energy | cuBLAS `dot` device pointer mode + kernel → `frame.scalars->energy` |
| V_nuc | stays on `d_V_nuc` for the whole SCF |
| **After SCF** | optional residual / V_nuc / `energy_ha_` D2H for host bookkeeping only |

## Host still

- type[] D2H for path select (setup only)  
- post-calc energy_ha_ / residual mirrors  
- p-shell forces; host-integral path for non-s/p bases  

## Validation

- GH200: `tests/cuda/scf/test_rhf.cu` — `CUDAForcesMatchCPU_H2_STO3G`,
  `CUDAForcesMatchFD_H2_STO3G`; pure-s must not call host RHF for forces.
- H₂ textbook energy ≈ −1.1167 Ha; forces match CPU to ~1e-5 eV/Å.

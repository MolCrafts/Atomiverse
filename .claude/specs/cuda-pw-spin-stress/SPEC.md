# cuda-pw-spin-stress

## Lever

| Item | Value |
| --- | --- |
| Track | **cuda** |
| Metric | max‖σ_cuda − σ_cpu‖ [eV/Å³] Voigt under collinear LSDA fixed-occ |
| File / function | `src/cuda/planewave/pw_solver.cu` (`pw_stress` spin reject) + `src/cuda/planewave/internal/pw_stress.cu` (spin SCF + spin density remap NSCF strain-FD); CPU twin `src/cpu/planewave/internal/pw_stress.cpp` (lift spin reject; `run_collinear_scf` + spin ρ_σ remap) |
| Device API | `fetch_rho_spin_r` / `load_rho_spin_r_scaled` on `DeviceDensity` |

## Acceptance

1. `cuda::pw::pw_stress` accepts `spin_polarized=true` with fixed-occ **LSDA** (rejects spin∧smear and spin∧PBE σ — deeper rows).
2. CPU `cpu::pw::pw_stress` same surface (oracle twin).
3. Regression `reg_cuda-pw-spin-stress` dual-gate on O₂ triplet LSDA Γ: finite non-vacuous σ; max|Δσ| < 5e-2 eV/Å³.
4. GH200 log evidence; **perf_exempt** small cell.

## Non-goals

spin-PBE stress, spin∧smear stress, multi-k spin stress, soft tols, self-golden pins.

## Oracle

Same-settings CPU `pw_stress` (not CP2K; not CUDA self-golden).

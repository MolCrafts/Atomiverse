# SPEC — cuda-pw-spin-forces

## Goal

Lift collinear **LSDA** spin reject on `cuda::pw::pw_forces_analytic` and
assemble analytic ionic forces (local HF + Ewald + KB) under fixed-occupation
UKS, dual-gated against same-settings CPU.

## Lever

| File | Function | Metric |
| --- | --- | --- |
| `src/cuda/planewave/pw_solver.cu` | `pw_forces_analytic` | lift spin∧LSDA reject; spin SCF before force assembly |
| `src/cuda/planewave/internal/pw_forces.cu` | `pw_forces_analytic_impl` | total-ρ local HF; per-spin NL re-solve |
| `src/cuda/planewave/internal/device_solver.cu` | `accumulate_nl_forces` | `n_occ` + `occ` (f=1 spin-orbital) |
| `src/cpu/planewave/internal/pw_forces.cpp` | `pw_forces_analytic_impl` | collinear SCF + spin NL (CPU twin / oracle) |

Physics: local HF uses total \(\rho=\rho_\uparrow+\rho_\downarrow\); Ewald is
spin-independent; nonlocal KB sums both channels with \(f=1\) per occupied
spin-orbital (\(N_\uparrow,N_\downarrow\) from multiplicity).

## Acceptance

- GH200 dual-gate O₂ triplet LSDA (UPF, Γ, fixed occ, mult=3):
  \(\max\|F_{\mathrm{cuda}}-F_{\mathrm{cpu}}\| < 5\times 10^{-3}\) eV/Å;
  CPU forces non-vacuous.
- Non-spin closed-shell force path unchanged.
- Fail-closed: spin∧PBE forces still reject (child slug); smear forces reject;
  stress spin reject unchanged.
- Unit: spin∧LSDA no longer throws on empty-geometry gate alone; spin∧smear
  or spin∧PBE still throws with how-to.

## Non-goals

spin-PBE forces, spin stress, spin∧smear, multi-k spin forces, FD re-pin
(energy path unchanged for restricted).

## Oracle

Same-settings `cpu::pw::pw_forces_analytic` after its collinear spin lift
(not CP2K). **perf_exempt** — small-cell correctness foothold.

# cuda-pw-stress

## Lever

| Item | Value |
| --- | --- |
| Method | cuda |
| Files | `src/cuda/planewave/internal/pw_stress.{cu,cuh}`, `device_density.{cu,cuh}` (`fetch_rho_r` / `load_rho_r_scaled` / `nr`), `pw_solver.{cu,cuh}` public wire |
| Function | `cuda::pw::pw_stress` → `internal::pw_stress_analytic_impl` |
| Metric | max\|σ_cuda − σ_cpu\| [eV/Å³] Voigt on small Si (dual-gate); cubic isotropy unit |
| Acceptance | GH200 green: unit + `reg_cuda-pw-stress --device=both`; stated CUDA↔CPU tol; energy path unchanged; device-only NSCF stages |

## Algorithm (CPU twin)

Nielsen–Martin production path (`cpu::pw::pw_stress` / `pw_stress_analytic_impl`):

1. One full SCF at the reference geometry.
2. Hellmann–Feynman cell stress by central FD of the *non-self-consistent* total energy under symmetric strain: density affinely remapped (ρ → ρ·Ω₀/Ω), V_H / V_xc / V_loc / Ewald rebuilt, one Davidson diagonalization per strained cell, `iteration_energy`.
3. Sign: mechanical / CP2K \(\sigma = -(1/\Omega)\,\partial E/\partial\varepsilon\).

CUDA ports the same recipe with `DeviceDensity` + `DeviceBandSolver` for the reference SCF and each NSCF strain evaluation.

## Dual gate

| Gate | Criterion |
| --- | --- |
| Correctness | max\|σ_cuda − σ_cpu\| ≤ 5e-2 eV/Å³ on Si twin; cubic H unit off-diag ~0 |
| Wall | **perf_exempt** — correctness foothold (1 SCF + 12 NSCF; small cell) |

## Non-goals

Spin, smear, multi-SCF strain-FD production path, energy-path changes, variable-cell driver product.

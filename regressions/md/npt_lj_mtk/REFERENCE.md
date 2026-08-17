# npt_lj — MTK NoseHooverNPT on a 64-atom LJ liquid

**One case, multi-device:** `reg_npt_lj_mtk --device=cpu|cuda|both`.
Named `npt_lj_mtk` so the binary does not collide with `mc/npt_lj`.

Public API only: `MD` + `NoseHooverNPT{T, tau_T, P, tau_P, dt}` + `PairLJCut`
+ `NeighborList`. No per-atom Verlet on CUDA.

| Field | Value |
|---|---|
| System | 6³ sc LJ (216 atoms), ρ = 0.8, ε = σ = 1, r_c = 2.5 (L/2 > r_c) |
| Ensemble | MTK NPT, T = 1, τ_T = 0.1, P = 0, τ_P = 1, dt = 0.005 |
| Steps | 80 |
| CPU↔CUDA | \|V_cuda − V_cpu\| / V < 1e-4 |

GH200 **1335945**: CPU and CUDA both `V=269.213535` (`|ΔV|/V = 0`).
LAMMPS 4 Jul 2026 (`develop d3b14f6`) ran the twin deck and printed
`Volume=282.54148` at step 80 — **not** same-settings (LAMMPS
`velocity create 1.0`; ATV uses a deterministic lattice kick at T=1 K,
ε=1 eV). Equal-settings LAMMPS pin is still owed.

# npt_lj — MTK NoseHooverNPT on a 64-atom LJ liquid

**One case, multi-device:** `reg_npt_lj_mtk --device=cpu|cuda|both`.
Named `npt_lj_mtk` so the binary does not collide with `mc/npt_lj`.

Public API only: `MD` + `NoseHooverNPT{T, tau_T, P, tau_P, dt}` + `PairLJCut`
+ `NeighborList`. No per-atom Verlet on CUDA.

| Field | Value |
|---|---|
| System | 6³ sc LJ (216 atoms), ρ = 0.8, ε = σ = 1, r_c = 2.5 (L/2 > r_c) |
| Ensemble | MTK NPT, T* = 1 (T = 1/k_B K with ε = 1 eV), τ_T = 0.1, P = 0, τ_P = 1, dt = 0.005 |
| Steps | 80 |
| CPU↔CUDA | \|V_cuda − V_cpu\| / V < 1e-4 |

GH200 **1336653** (T\*=1, ε=1 eV): CPU `V=269.425432`, CUDA
`V=269.425466`, `|ΔV|/V = 1.27e-7`.
LAMMPS twin deck uses a literal \(a_0\) so ρ=0.8. A digit-identical
LAMMPS volume is not claimed: ATV/LAMMPS RNGs and lj-vs-fs time units
differ. The landed gate is CPU↔CUDA.

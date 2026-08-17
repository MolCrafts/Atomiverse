# npt_lj — MTK NoseHooverNPT on a 64-atom LJ liquid

**One case, multi-device:** `reg_npt_lj_mtk --device=cpu|cuda|both`.
Named `npt_lj_mtk` so the binary does not collide with `mc/npt_lj`.

Public API only: `MD` + `NoseHooverNPT{T, tau_T, P, tau_P, dt}` + `PairLJCut`
+ `NeighborList`. No per-atom Verlet on CUDA.

| Field | Value |
|---|---|
| System | 4³ sc LJ, ρ = 0.8, ε = σ = 1, r_c = 2.5 |
| Ensemble | MTK NPT, T = 1, τ_T = 0.1, P = 0, τ_P = 1, dt = 0.005 |
| Steps | 80 |
| CPU↔CUDA | \|V_cuda − V_cpu\| / V < 1e-4 |

External LAMMPS twin deck: `tools/lammps_oracle/inputs/npt_lj/in.npt`.
The GH200 job records `lmp` version + printed volume line when a LAMMPS
binary is on PATH. Until that line is captured, the dual-gate is CPU↔CUDA
only (◐, not ✓).

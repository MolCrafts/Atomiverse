# cuda-gpw-stress

## Lever
Public `cuda::gpw::gpw_stress` free-fn: CUDA Γ SCF density → fixed-P stress
assembly via CPU analytic twin at state (`gpw_solver.cu` + header). Fail-closed
multi-k. Dual-gate H₂ Γ RKS PADE.

## Acceptance
- GH200 dual-gate max|Δσ| ≤ 5e-5 Ha/Bohr³, |ΔE| ≤ 5e-6 Ha
- unit fail-closed multi-k + unconverged state
- acceptance_met=false until GH200 green

## acceptance_met
false

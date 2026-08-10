# cuda-gpw-uks-gamma acceptance

| Gate | Criterion |
| --- | --- |
| Dual-gate energy | \|E_cuda − E_cpu\| ≤ **5×10⁻⁶ Ha** (high-spin H₂ Γ UKS PADE) |
| Converged | both CPU and CUDA SCF `converged==true` |
| Fail-closed | UKS+PBE, OT, MG, multi-k, MT still throw how-to |
| RKS hold | existing H₂ RKS dual-gate still green |
| Evidence | GH200 sbatch log |
| Perf | perf_exempt (small cell correctness) |

`acceptance_met=true` only when GH200 dual-gate PASS is logged.

## Evidence (2026-08-10)

- GH200 job **1125905** COMPLETED exit 0
- `test_cuda_gpw_full_product` 1/1 PASS (includes H2 high-spin UKS dual-gate)
- `reg_cuda-gpw-uks-gamma`: CPU E=−0.508766760868 Ha; CUDA E=−0.508771520730 Ha;
  \|ΔE\|=**4.760e-06** Ha (tol 5e-6); ⟨S²⟩=2.0 both; **PASS**
- **perf_exempt** small cell

**acceptance_met=true**

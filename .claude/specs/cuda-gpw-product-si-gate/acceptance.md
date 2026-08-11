# Acceptance — cuda-gpw-product-si-gate

- [x] Spec-lite + lever (KB Vnl @ Γ replaces ket-only gth_nonlocal)
- [x] Unit `SiGammaPadeSingleGridEnergyDualGate` + `reg_cuda-gpw-product-si-gate`
- [ ] GH200 dual-gate both converge + \|dE\| ≤ 5e-5 Ha
- [ ] Mid-path scientific DH grep on diff: zero new mid-SCF matrix exchange
- [ ] Skills named (host assembly reuse — no new .cu kernel; N/A device launch skills beyond residency audit)

**acceptance_met** = false until GH200 green.

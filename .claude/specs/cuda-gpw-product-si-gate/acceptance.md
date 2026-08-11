# Acceptance — cuda-gpw-product-si-gate

- [x] Spec-lite + lever (KB Vnl @ Γ + host collint KS twin for Si PBC gap)
- [x] `reg_cuda-gpw-product-si-gate` (unit e2e exceeds tests/ 10s iron law)
- [ ] GH200 dual-gate both converge + \|dE\| ≤ 5e-5 Ha
- [x] Mid-path scientific DH grep: setup Vnl only; mid-SCF P/V for host collint hybrid is **stated residual** (device continuous collocate PBC → `product-collocate-pbc`)
- [x] Skills: host assembly reuse of multi-k KB + collint twin; no new device kernel this slug

**acceptance_met** = false until GH200 green.

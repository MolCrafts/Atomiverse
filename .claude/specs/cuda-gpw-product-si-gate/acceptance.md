# Acceptance — cuda-gpw-product-si-gate

- [x] Spec-lite + lever (KB Vnl @ Γ + host cbg KS twin matching CPU ngrids=1)
- [x] `reg_cuda-gpw-product-si-gate` (unit e2e exceeds tests/ 10s iron law)
- [x] GH200 **1167099** dual-gate both converge + \|dE\|=**2.130e-08** Ha ≤ 5e-5
- [x] Hold gates: full-product \|dE\|=1.4e-11; product-h2o \|dE\|=1.4e-14
- [x] Mid-path: setup Vnl only + **stated** mid-SCF P/V for host cbg hybrid
  (device continuous collocate PBC → `product-collocate-pbc`)
- [x] Skills: host cbg twin of CPU Γ path + multi-k KB reuse; no new device kernel

**acceptance_met=true** (GH200 **1167099** / tip `a30c675d`).

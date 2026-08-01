# Acceptance — `gpw-uks-pbesol`

| # | Criterion | Gate |
|---|-----------|------|
| 1 | `XcKind::PBEsol` UKS build / SCF accepted | product + regression |
| 2 | O₂ triplet SCF converges | regression |
| 3 | \(\lvert E - E_{\mathrm{CP2K}}\rvert \le\) REFERENCE tol | regression |
| 4 | \(\lvert E_{\mathrm{PBEsol}}-E_{\mathrm{PBE}}\rvert > 10^{-4}\) Ha | dual-gate |
| 5 | \(\langle S^2\rangle\) within REFERENCE tol | regression |
| 6 | Frame `kEnergy` [eV] matches internal Ha | regression |
| 7 | \(T_{\mathrm{atv}}/T_{\mathrm{cp2k}} \le 10\) from real twin wall | REFERENCE |

`acceptance_met=true` only when all rows are green on `wf/gpw-uks-pbesol`
after scoped build+test and offline CP2K pin provenance is recorded.

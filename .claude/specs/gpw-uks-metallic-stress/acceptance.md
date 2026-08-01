# Acceptance — gpw-uks-metallic-stress

| Gate | Criterion |
| ---- | --------- |
| Product | Multi-k UKS free-energy analytic stress under FD/MP smear with fractional dual-spin \(P/W\) |
| Dual-gate | Analytic vs multi-SCF FD within sibling metallic-stress tols |
| Oracle | CP2K 2026.1 UKS NUMERICAL stress + SMEAR twin in REFERENCE.md |
| Perf | Fair wall \(\le 10\times\) on matched twin |
| API | No public Config bags |

`acceptance_met=true` only when all gates pass.

# Acceptance — gpw-vdw-stress

| Gate | Criterion |
| ---- | --------- |
| Product | Analytic fixed-\(P\) stress includes nonlocal vdW correlation under strain |
| Dual-gate | Analytic vs multi-SCF FD within stated tol (sibling stress tier) |
| Oracle | CP2K 2026.1 NON_LOCAL + STRESS twin in REFERENCE.md (matched GPW) |
| Perf | Fair wall \(\le 10\times\) on matched twin |
| API | No public Config bags |

`acceptance_met=true` only when all gates pass.

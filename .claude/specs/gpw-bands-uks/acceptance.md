# Acceptance — gpw-bands-uks

| Gate | Criterion |
| ---- | --------- |
| Product | UKS multi-k SCF + NSCF band path with frozen dual \(V_{\mathrm{eff}}^\sigma\) |
| Energy | \(\lvert E_{\mathrm{ATV}}-E_{\mathrm{CP2K}}\rvert \le 10^{-3}\) Ha on twin geometry |
| Bands | Dual-spin path eigenvalues published; insulating gap character gates as in `gpw-bands-si` class |
| Oracle | CP2K 2026.1 GTH-GPW UKS BAND_STRUCTURE twin in REFERENCE.md |
| Perf | Fair wall SCF+bands \(\le 10\times\) CP2K on same twin |
| API | No public Config bags; Frame not required for pure electronic post-process |

`acceptance_met=true` only when all gates pass.

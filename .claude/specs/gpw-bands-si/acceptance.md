# Acceptance — `gpw-bands-si`

| Gate | Status | Evidence |
| --- | --- | --- |
| A1 SCF + sorted evals | met | `reg_gpw-bands-si` ok |
| A2 path Γ/X, n_bands | met | 16 k, n_bands=12 |
| A3 insulating gap | met | gap 0.895 eV > 0 |
| A4 \(E\) vs CP2K | met | \|ΔE\| 2.2e-5 Ha ≤ 1e-3 |
| A5 gap vs CP2K | met | \|Δgap\| 1.2 meV ≤ 50 meV |
| Perf ≤ 10× | met | ratio 3.51 (6.66 / 1.90 s, OMP 8) |

`acceptance_met = true`

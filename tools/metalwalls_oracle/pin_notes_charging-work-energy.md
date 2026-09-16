# Pin notes — charging-work-energy

**Date:** 2026-07-23  
**Slug:** `charging-work-energy`  
**oracle:** literature (MetalWalls JOSS formula identity)  
**oracle_version:** MetalWalls JOSS 5(53) 2373 (2020) eq.6; Scalfi et al. PCCP 22 10480 (2020) / arXiv:1911.09351 eq.12

## Method

No MetalWalls binary was available on the agent host. The offline pin is the
closed-form evaluation of MetalWalls JOSS equation (6) second term on a fixed
parallel-plate charge assignment:

| Region | type | N | Q_i [e] | Ψ [V] |
| ------ | ---: | -: | ------: | ----: |
| Left | 1 | 4 | +0.10 | +0.50 |
| Right | 2 | 4 | −0.10 | −0.50 |

\[
E_{\mathrm{charge}} = -\sum_i \Psi_i Q_i = -0.4~\mathrm{eV}
\]

Cross-check (Scalfi eq. 12): \(Q_L\Delta\psi = 0.4\times 1.0 = 0.4\),
\(E = -q^{\mathrm{T}}\Psi = -0.4\) eV.

Recorded in `regressions/md/charging-work-energy/REFERENCE.md`.

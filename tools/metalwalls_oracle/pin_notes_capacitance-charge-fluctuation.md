# Pin notes — capacitance-charge-fluctuation

**Date:** 2026-07-24  
**Slug:** `capacitance-charge-fluctuation`  
**oracle:** literature (MetalWalls / Scalfi–Salanne / Limmer charge-fluctuation \(C\))  
**oracle_version:** Limmer et al. J. Chem. Phys. 139, 184704 (2013); Scalfi et al. PCCP 22 10480 (2020); MetalWalls JOSS 5(53) 2373 (2020)

## Method

No MetalWalls binary and no LAMMPS ELECTRODE capacitance compute on the agent
host. The offline pin is the closed-form evaluation of

\[
C_{\mathrm{diff}} = \mathrm{Var}(Q)/(k_B T),\qquad
C_{\mathrm{int}} = \langle Q\rangle/\Delta\Psi
\]

on a synthetic primary-electrode charge series:

| \(Q_L\) [e] | 0.4 | 0.6 | 0.4 | 0.6 |
| \(T\) | 300 K |
| \(\Delta\Psi\) | 1.0 V |

\[
\langle Q\rangle=0.5,\quad \mathrm{Var}(Q)=0.01,\quad
C_{\mathrm{diff}}=0.38681727072485284~\mathrm{e/V},\quad
C_{\mathrm{int}}=0.5~\mathrm{e/V}.
\]

Recorded in `regressions/md/capacitance-charge-fluctuation/REFERENCE.md`.

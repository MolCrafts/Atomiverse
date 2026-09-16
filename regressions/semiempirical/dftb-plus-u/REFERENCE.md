# dftb-plus-u

**oracle:** literature (formula)
**oracle_version:** Dudarev PRB 57 1505 (1998); design `.claude/notes/dftb-plus-u.md`

## Method

SCC-DFTB2 (MioSketch) water + oxygen \(U_{\mathrm{eff}}=4\) eV. Energy-only.

## Gates

- \(E_U = \tfrac12 U n(1-n/N)\) on O with \(n=n_{\mathrm{val}}-\Delta q\), \(N=2n_{\mathrm{AO}}\)
- \(E(U)-E(0)=E_U\)
- Nuclear forces: central FD of \(E_{\mathrm{SCC}}+E_U\) matches independent energy FD (tol 5e-4 eV/Å)

## Gaps

- No absolute DFTB+ +U pin (Dudarev shell details differ by engine).
- Stress / multi-k / shell-resolved \(U_\ell\) still open.
- Forces are re-SCC FD, not analytic \(\partial n/\partial R\).

# Reference — FIRE LJ dimer

## Algorithm

Bitzek E., Koskinen P., Gähler F., Moseler M., Gumbsch P.,
*Structural Relaxation Made Simple*, Phys. Rev. Lett. **97**, 170201 (2006).
DOI: 10.1103/PhysRevLett.97.170201

FIRE 1.0, unit mass. ASE-aligned mix/grow knobs; regression uses a larger
pseudo-time scale (`dt0=1.0`, `dt_max=20`) so O(10⁻²) eV/Å LJ forces take
useful geometry steps. Start at \(1.5\,r^*\); force tol \(10^{-3}\) eV/Å.

## Analytic LJ minimum

Unshifted 12-6 LJ: \(r^*=2^{1/6}\sigma\), \(U=-\varepsilon\).

| Quantity | Value |
|---|---|
| ε | 0.0104 eV |
| σ | 3.4 Å |
| r_cut | 2.5 σ |
| start | \(1.2\,r^*\) |
| \(r^*\) | \(2^{1/6}\cdot 3.4 \approx 3.81637\) Å |

`PairLJCut` is energy-shifted so \(V(r_c)=0\); accept energy in
\((-1.1\varepsilon,\,-0.9\varepsilon)\). Force residual \(f_{\max}<0.05\) eV/Å.

Same geometry contract as `regressions/minimizer/lj_dimer` (LBFGS).

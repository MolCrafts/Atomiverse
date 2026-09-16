# Reference — soft-core LJ dimer

## Formula

Beutler / GROMACS one-leg soft-core LJ (appearance):

\[
r_\lambda=(\alpha\sigma^6(1-\lambda)^p+r^6)^{1/6},\quad
V=\lambda\bigl[V_{\mathrm{LJ}}(r_\lambda)-V_{\mathrm{shift}}\bigr]
\]

Hard shift \(V_{\mathrm{shift}}=V_{\mathrm{LJ}}(r_c)\) (PairLJCut convention).

- Beutler et al., *Chem. Phys. Lett.* **222**, 529 (1994),
  DOI 10.1016/0009-2614(94)00397-1
- GROMACS free-energy interactions (one-leg soft-core)

**Not** LAMMPS `lj/cut/soft` algebra.

## Parameters

| Symbol | Value |
|---|---|
| ε | 0.5 eV |
| σ | 3.0 Å |
| r_c | 8.0 Å |
| r | 3.0 Å |
| λ | 0.5 |
| α | 0.5 |
| p | 1 |

## Goldens (offline, 2026-07-23)

| Quantity | Value |
|---|---|
| r_λ | 3.11367244666864 Å |
| E | −0.15722681917773984 eV |
| F_r | 0.7680000000000029 eV/Å |

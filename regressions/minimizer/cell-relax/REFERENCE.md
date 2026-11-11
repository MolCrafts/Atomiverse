# minimizer/cell-relax

One case; every section below is run by `atv_regressions --case minimizer/cell-relax` and keeps its own oracle record.


## Reference — isotropic cell relaxation (LJ dimer)

### Method

0 K isotropic volume relaxation: scale cubic cell and affine atom positions
using virial pressure \(P=\mathrm{tr}(\sigma)/(3V)\) from cut-and-shifted LJ
(`PairLJCut` with `set_compute_stress(true)`).

Target \(P=0\). Positive (compressive) virial ⇒ expand.

### Parameters

| Quantity | Value |
|---|---|
| ε | 0.0104 eV |
| σ | 3.4 Å |
| r_c | 2.5 σ |
| L₀ | 6.0 Å (compressed) |

### Pass

- L₁ > L₀ + 0.05 Å
- |P₁| < 5×10⁻² eV/Å³ (skeleton; not full EOS equilibrium)


## Reference — joint atom+cell LJ dimer at P=0

Analytic LJ 12-6 minimum: \(r_{\min}=2^{1/6}\sigma\).

Gate: |r/r_min − 1| < 0.05 after joint isotropic SD.

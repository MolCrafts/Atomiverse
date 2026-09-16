# Reference — isotropic cell relaxation (LJ dimer)

## Method

0 K isotropic volume relaxation: scale cubic cell and affine atom positions
using virial pressure \(P=\mathrm{tr}(\sigma)/(3V)\) from cut-and-shifted LJ
(`PairLJCut` with `set_compute_stress(true)`).

Target \(P=0\). Positive (compressive) virial ⇒ expand.

## Parameters

| Quantity | Value |
|---|---|
| ε | 0.0104 eV |
| σ | 3.4 Å |
| r_c | 2.5 σ |
| L₀ | 6.0 Å (compressed) |

## Pass

- L₁ > L₀ + 0.05 Å
- |P₁| < 5×10⁻² eV/Å³ (skeleton; not full EOS equilibrium)

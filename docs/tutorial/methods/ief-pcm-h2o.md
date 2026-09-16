# IEF-PCM — H₂O / 6-31G*

*(Technical-report draft; narrative house style TBD — rewrite later.)*

Integral-equation formalism PCM (Cancès–Mennucci–Tomasi): electrostatic free
energy \(\Delta G_{\mathrm{el}} = \tfrac12 \sum_k q_k V_k\).
Regression: `regressions/scf/ief-pcm-h2o`. C-PCM energy: `h2o_rhf_631gs_cpcm`.

## Theory

IEF-PCM solves a more complete boundary-integral problem than conductor-like
PCM (finite dielectric). Observable `e_solv` ≡ \(\Delta G_{\mathrm{el}}\)
(stabilizing ⇒ negative). Surface discretization (Lebedev order, cavity) must
match the pin protocol in REFERENCE.md.

## Practice

```cpp
// RHF + SolventModel::IEFPCM as in the regression
// e_solv = solvent free-energy contribution [Ha]
```

```bash
./build/regressions/reg_ief-pcm-h2o
```

## Results

### Accuracy vs PySCF 2.14.0 (`method='IEF-PCM'`)

| Quantity | Pin (Ha) | Tol |
| -------- | -------: | --- |
| \(\Delta G_{\mathrm{el}}\) | **−0.01265803490853** | **1e-4** (~0.06 kcal/mol; surface model residual) |
| \(E_{\mathrm{gas}}\) | −76.010504988256 | diagnostic |
| \(E_{\mathrm{IEF}}\) total | −76.022025167150 | diagnostic |

### Speed

| Twin | \(T_{\mathrm{PySCF}}\) | \(T_{\mathrm{ATV}}\) | \(r\) | ≤10× |
| ---- | ---------------------: | -------------------: | ----: | ---- |
| vacuum `rhf` | 0.66 s | 6.01 s | **9.08** | ✓ (just under 10) |

Twin omits solvent work → ratio is an **upper bound** on “how expensive IEF is
vs vacuum RHF,” not a pure solver A/B.

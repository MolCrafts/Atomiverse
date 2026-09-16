# GHF analytic gradient — equilateral H₃ / STO-3G

*(Technical-report draft; narrative house style TBD — rewrite later.)*

Real two-component generalized Hartree–Fock **nuclear forces** on the
noncollinear Néel solution of frustrated H₃.
Regression: `regressions/scf/ghf-grad-h3-sto3g`. Energy sibling: `h3_ghf_sto3g`.

## Theory

GHF allows spinors that mix \(\alpha/\beta\). On equilateral H₃ the lowest
solution is a coplanar 120° Néel arrangement. Energy-only GHF shipped earlier;
this page is the **analytic** \(\partial E/\partial R\) of that density (spinor
energy-weighted density + unsymmetrized exchange blocks).

PySCF’s `GHF.nuc_grad_method()` is not implemented; the pin is the analytic
gradient at PySCF AO/dAO integrals on the pinned density, FD-cross-checked.

## Practice

```cpp
cpu::scf::GHF ghf;
ghf.basis = cpu::scf::BasisSet::STO3G;
// equilateral side 1.0 Å; Néel guess as in regression
ghf.run(frame);
auto g = ghf.forces(frame);   // or gradient API used in reg — [eV/Å] on Frame path
```

```bash
./build/regressions/reg_ghf-grad-h3-sto3g
```

## Results

### Accuracy

| Check | Tol |
| ----- | --- |
| Element-wise vs analytic pin | **1e-5 Ha/Bohr** |
| Translational invariance \(\lvert\sum_A F_A\rvert\) | 1e-8 Ha/Bohr |
| Energy sibling \(E_{\mathrm{GHF}}\) | −1.34044034354119 Ha |

### Speed

| Twin | \(T_{\mathrm{PySCF}}\) | \(T_{\mathrm{ATV}}\) | \(r\) | ≤10× |
| ---- | ---------------------: | -------------------: | ----: | ---- |
| proxy `uhf` H₂ | 0.32 s | 0.028 s | **0.088** | ✓ |

Twin is imperfect (different system); treat ratio as rough order-of-magnitude only.

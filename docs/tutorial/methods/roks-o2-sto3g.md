# ROKS — triplet O₂ / LSDA / STO-3G

*(Technical-report draft; narrative house style TBD — rewrite later.)*

Restricted open-shell Kohn–Sham: Roothaan single effective Fock for open shells
with exact \(\langle S^2\rangle = S(S+1)\) for high-spin.
Regression: `regressions/scf/roks-o2-sto3g`. Related: ROHF energy/grad chapters.

## Theory

UHF/UKS break spin symmetry; ROHF/ROKS keep a spin-pure open-shell determinant
via a single effective Fock (Guest–Saunders / PySCF canonical coupling). ROKS
is the KS analogue with an XC functional (here LSDA / VWN-class as in the pin).

## Practice

```cpp
#include <atomiverse/cpu/scf/roks.h>

cpu::scf::ROKS roks(/*n_alpha=*/9, /*n_beta=*/7, cpu::scf::XCType::LSDA);
roks.basis = cpu::scf::BasisSet::STO3G;
roks.run(frame);   // O₂ r = 1.2075 Å
```

```bash
./build/regressions/reg_roks-o2-sto3g
```

## Results

### Accuracy vs PySCF 2.14.0 (`dft.ROKS`, `xc='LDA,VWN'`)

| Quantity | Pin (Ha) | Tol |
| -------- | -------: | --- |
| \(E_{\mathrm{ROKS}}\) | **−147.193715448518** | **1e-6** |
| \(\langle S^2\rangle\) | 2.0 (exact high-spin) | identity |

### Speed

| Twin | \(T_{\mathrm{PySCF}}\) | \(T_{\mathrm{ATV}}\) | \(r\) | ≤10× |
| ---- | ---------------------: | -------------------: | ----: | ---- |
| `rohf` | 0.45 s | 0.59 s | **1.31** | ✓ |

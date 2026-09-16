# Hybrid TDDFT-TDA (B3LYP) — H₂ / STO-3G

*(Technical-report draft; narrative house style TBD — rewrite later.)*

Tamm–Dancoff linear response on a **hybrid** RKS reference: Casida **A** includes
exact-exchange kernel \(\alpha_{\mathrm{hf}}\) plus semi-local \(f^{xc}\).
Regression: `regressions/scf/h2-b3lyp-hybrid-tda`. Related: [Ch. 6 Hybrid DFT](../ch06-hybrid-dft.md).

## Theory

Pure ALDA-TDA ([full-grid ALDA](../index.md) regressions) is not enough for
hybrids. B3LYP mixes HF exchange into the ground-state Fock and into the
response kernel. Under TDA one solves

$$ \mathbf{A}\,\mathbf{X} = \omega\,\mathbf{X} $$

with HF-exchange contributions in \(\mathbf{A}\). Full RPA (A/B) is a separate
method ([full-lr-tddft-h2-pbe](full-lr-tddft-h2-pbe.md)).

## Practice

```cpp
#include <atomiverse/cpu/frame/frame.h>
#include <atomiverse/cpu/scf/rks.h>

cpu::Frame frame(2);
// H–H = 0.74 Å …
cpu::scf::RKS rks(cpu::scf::XCType::B3LYP);
rks.basis = cpu::scf::BasisSet::STO3G;
rks.tol = 1e-10;
rks.run(frame);
auto ex = rks.tda(/*n_states=*/3);   // excitations [eV], oscillator strengths
```

```bash
cmake --build build --target reg_h2-b3lyp-hybrid-tda && ./build/regressions/reg_h2-b3lyp-hybrid-tda
```

## Results

### Accuracy vs PySCF 2.14.0

| Quantity | Pin | Tol |
| -------- | --: | --- |
| \(E_{\mathrm{RKS}}\) (B3LYP) | −1.165418410673 Ha | ~1e-8 |
| S₁ (TDA) | **25.92836482 eV** | 0.05 eV class (reg: micro-eV) |

### Speed

| Twin | \(T_{\mathrm{PySCF}}\) | \(T_{\mathrm{ATV}}\) | \(r\) | ≤10× |
| ---- | ---------------------: | -------------------: | ----: | ---- |
| `rks_tda` B3LYP | 3.68 s | 4.66 s | **1.27** | ✓ |

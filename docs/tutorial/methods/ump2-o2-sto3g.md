# UMP2 — triplet O₂ / STO-3G

*(Technical-report draft; narrative house style TBD — rewrite later.)*

Unrestricted Møller–Plesset second-order correlation on a UHF reference.
Regression: `regressions/scf/ump2-o2-sto3g`. Related narrative:
[Ch. 7 Correlation](../ch07-correlation.md), [Ch. 3 Open shell](../ch03-open-shell.md).

## Theory

Closed-shell RMP2 ([Ch. 7](../ch07-correlation.md)) assumes a restricted HF
reference. Open-shell singlets/triplets need **UMP2**: spin-orbital second-order
perturbation theory on **UHF** orbitals, with same-spin and opposite-spin
channels (unprojected; residual \(\langle S^2\rangle\) contamination is accepted
as in raw PySCF `mp.UMP2`).

$$ E_{\mathrm{corr}}^{\mathrm{UMP2}} = E_{\mathrm{SS}} + E_{\mathrm{OS}}, \qquad
   E_{\mathrm{tot}} = E_{\mathrm{UHF}} + E_{\mathrm{corr}}. $$

## Practice

```cpp
#include <atomiverse/cpu/frame/frame.h>
#include <atomiverse/cpu/scf/uhf.h>
#include <atomiverse/cpu/scf/mp2.h>

// O₂, r = 1.2075 Å, triplet (2S = 2)
cpu::Frame frame(2);
// … set Z=8, positions, tags …
cpu::scf::UHF uhf;
uhf.basis = cpu::scf::BasisSet::STO3G;
uhf.n_alpha = 9; uhf.n_beta = 7;   // or resolve from charge/multiplicity
uhf.tol = 1e-10;
uhf.run(frame);
auto mp = uhf.mp2(frame);          // MP2Result: e_corr, e_tot [Ha]
```

```bash
cmake --build build --target reg_ump2-o2-sto3g
./build/regressions/reg_ump2-o2-sto3g
```

## Results

### Accuracy vs PySCF 2.14.0

Geometry O(0,0,0), O(0,0,1.2075) Å; STO-3G cart; `conv_tol=1e-12` on the pin.

| Quantity | Pin (Ha) | Tol |
| -------- | -------: | --- |
| \(E_{\mathrm{UHF}}\) | −147.633946785450 | 1e-8 (diag.) |
| \(E_{\mathrm{corr}}\) (UMP2) | **−0.096393144130** | **1e-6** |
| \(E_{\mathrm{tot}}\) | **−147.730339929580** | **1e-6** |

### Speed (fair wall)

| Twin | \(T_{\mathrm{PySCF}}\) | \(T_{\mathrm{ATV}}\) | \(r\) | ≤10× |
| ---- | ---------------------: | -------------------: | ----: | ---- |
| `ump2` | 2.51 s | 0.57 s | **0.23** | ✓ |

# Full linear-response TDDFT — H₂ / PBE / STO-3G

*(Technical-report draft; narrative house style TBD — rewrite later.)*

Casida **A/B** (RPA) response on pure GGA RKS (PBE), beyond TDA.
Regression: `regressions/scf/full-lr-tddft-h2-pbe`. Hybrid TDA:
[h2-b3lyp-hybrid-tda](h2-b3lyp-hybrid-tda.md).

## Theory

TDA drops the **B** block. Full TDDFT keeps coupling of excitation and
de-excitation:

$$
\begin{pmatrix} \mathbf{A} & \mathbf{B} \\ \mathbf{B}^* & \mathbf{A}^* \end{pmatrix}
\begin{pmatrix} \mathbf{X} \\ \mathbf{Y} \end{pmatrix}
= \omega
\begin{pmatrix} \mathbf{X} \\ \mathbf{Y} \end{pmatrix}.
$$

For pure functionals, \(f^{xc}\) is semi-local (ALDA-style); hybrids add HF
exchange in both A and B.

## Practice

```cpp
cpu::scf::RKS rks(cpu::scf::XCType::PBE);
rks.basis = cpu::scf::BasisSet::STO3G;
rks.run(frame);
auto ex = rks.tddft(/*n_states=*/3);   // full LR, not TDA
```

```bash
./build/regressions/reg_full-lr-tddft-h2-pbe
```

## Results

### Accuracy vs PySCF 2.14.0 (`TDDFT` / PBE)

| Quantity | Pin | Notes |
| -------- | --: | ----- |
| \(E_{\mathrm{RKS}}\) | −1.152072795159 Ha | H₂ 0.74 Å |
| S₁ (full LR) | see REFERENCE.md eV line | product gate in reg |

### Speed

| Twin | \(T_{\mathrm{PySCF}}\) | \(T_{\mathrm{ATV}}\) | \(r\) | ≤10× |
| ---- | ---------------------: | -------------------: | ----: | ---- |
| `rks_tda` PBE | 5.46 s | 1.45 s | **0.27** | ✓ |

Twin is TDA (conservative for comparing full RPA cost).

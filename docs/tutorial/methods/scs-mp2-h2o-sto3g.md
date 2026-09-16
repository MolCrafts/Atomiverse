# SCS-MP2 — H₂O / STO-3G

*(Technical-report draft; narrative house style TBD — rewrite later.)*

Spin-component-scaled MP2 (Grimme): scale opposite-spin and same-spin
correlation channels separately.

$$ E_{\mathrm{SCS}} = \tfrac{1}{3} E_{\mathrm{SS}} + 1.2\, E_{\mathrm{OS}}. $$

Regression: `regressions/scf/scs-mp2-h2o-sto3g`. Related: [Ch. 7](../ch07-correlation.md).

## Theory

Plain RMP2 overbinds same-spin correlation for many organic systems. SCS-MP2
is a cheap empirical reweighting of channels already available after the MP2
amplitudes — no extra \(O(N^5)\) cost beyond RMP2.

## Practice

```cpp
cpu::scf::RHF rhf;
rhf.basis = cpu::scf::BasisSet::STO3G;
// optional: rhf.mp2_os_scale / mp2_ss_scale members if exposed for SCS
rhf.run(frame);
auto mp = rhf.mp2(frame);   // SCS path as used in reg (see reg source)
```

```bash
./build/regressions/reg_scs-mp2-h2o-sto3g
```

## Results

### Accuracy vs PySCF 2.14.0

| Quantity | Pin (Ha) | Tol |
| -------- | -------: | --- |
| \(E_{\mathrm{OS}}\) (unscaled) | −0.033515820085 | 1e-6 |
| \(E_{\mathrm{SS}}\) (unscaled) | −0.002029831562 | 1e-6 |
| plain \(E_{\mathrm{corr}}\) | −0.035545651647 | 1e-6 |
| **SCS \(E_{\mathrm{corr}}\)** | **−0.040895594623** | **1e-6** |

### Speed

| Twin | \(T_{\mathrm{PySCF}}\) | \(T_{\mathrm{ATV}}\) | \(r\) | ≤10× |
| ---- | ---------------------: | -------------------: | ----: | ---- |
| plain `mp2` | 2.05 s | 0.026 s | **0.012** | ✓ |

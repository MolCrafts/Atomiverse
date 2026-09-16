# BLYP GGA — H₂O / STO-3G

*(Technical-report draft; narrative house style TBD — rewrite later.)*

Pure semi-local GGA: B88 exchange + LYP correlation, **no** HF exchange
(\(\alpha_{\mathrm{hf}}=0\)). Building blocks already lived inside B3LYP; this
exposes `XCType::BLYP` as a first-class functional.
Regression: `regressions/scf/blyp-water-sto3g`. Related: [Ch. 5](../ch05-kohn-sham-dft.md), [Ch. 6](../ch06-hybrid-dft.md).

## Theory

BLYP is the pure-GGA counterpart of B3LYP: same semi-local pieces, no hybrid
mixing. Useful for comparing hybrid vs pure GGA without changing correlation
form.

## Practice

```cpp
cpu::scf::RKS rks(cpu::scf::XCType::BLYP);
rks.basis = cpu::scf::BasisSet::STO3G;
rks.tol = 1e-10;
rks.run(frame);   // water pin geometry
// energy on method / frame.meta (eV on Frame path; Ha on method — see reg)
```

```bash
./build/regressions/reg_blyp-water-sto3g
```

## Results

### Accuracy vs PySCF 2.14.0 (`xc='blyp'`)

| Quantity | Pin (Ha) | Tol |
| -------- | -------: | --- |
| \(E_{\mathrm{BLYP}}\) | **−75.277447482665** | **1e-6** |

### Speed

| Twin | \(T_{\mathrm{PySCF}}\) | \(T_{\mathrm{ATV}}\) | \(r\) | ≤10× |
| ---- | ---------------------: | -------------------: | ----: | ---- |
| `rks` BLYP | 6.57 s | 14.87 s | **2.26** | ✓ |

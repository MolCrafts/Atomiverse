# Frozen-core RMP2 — H₂O / 6-31G*

*(Technical-report draft; narrative house style TBD — rewrite later.)*

Closed-shell MP2 with the lowest \(N_{\mathrm{frozen}}\) occupied MOs excluded
from the correlation sum (PySCF `mp.MP2(mf, frozen=N)` convention).
Regression: `regressions/scf/frozen-core-mp2-h2o`. Related: [Ch. 7](../ch07-correlation.md).

## Theory

Core orbitals contribute little to valence chemistry but cost the same \(O(N^5)\)
terms. Freezing the O 1s on water is the textbook frozen-core demo: \(E_{\mathrm{corr}}\)
rises by ~2.4 mHa relative to full RMP2 on the same geometry/basis.

## Practice

```cpp
cpu::scf::RHF rhf;
rhf.basis = cpu::scf::BasisSet::Pople631Gs;  // 6-31G*
rhf.frozen = 1;   // freeze lowest occupied spatial MO (O 1s)
rhf.run(frame);
auto mp = rhf.mp2(frame);
```

```bash
./build/regressions/reg_frozen-core-mp2-h2o
```

## Results

### Accuracy vs PySCF 2.14.0

Water pin geometry; cart 6-31G\*; frozen=1.

| Quantity | Pin (Ha) | Tol |
| -------- | -------: | --- |
| \(E_{\mathrm{RHF}}\) | −76.010504988257 | 1e-8 |
| \(E_{\mathrm{corr}}\) (frozen) | **−0.186119540118** | **1e-6** |
| \(E_{\mathrm{tot}}\) | **−76.196624528375** | **1e-6** |

Unfrozen RMP2 \(E_{\mathrm{corr}}\approx -0.18852\) Ha on the same setup (diagnostic).

### Speed

| Twin | \(T_{\mathrm{PySCF}}\) | \(T_{\mathrm{ATV}}\) | \(r\) | ≤10× |
| ---- | ---------------------: | -------------------: | ----: | ---- |
| `mp2` | 3.10 s | 0.23 s | **0.073** | ✓ |

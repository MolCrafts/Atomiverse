# CIS(D) — H₂CO / STO-3G

*(Technical-report draft; narrative house style TBD — rewrite later.)*

Head–Gordon CIS(D): doubles correction \(U+V\) on top of closed-shell CIS/TDA
amplitudes. Regression: `regressions/scf/cisd-h2co-sto3g`.

## Theory

CIS (or RHF-TDA) is \(O(N^4)\) but misses dynamical correlation in the excited
state. CIS(D) adds a second-order doubles correction on each CIS root
(Head–Gordon et al., *Chem. Phys. Lett.* **219**, 21 (1994)). PySCF supplies RHF
+ TDA; the product gate is the same spin-orbital \(U+V\) used in-engine.

## Practice

```cpp
cpu::scf::RHF rhf;
rhf.basis = cpu::scf::BasisSet::STO3G;
rhf.run(frame);                    // H2CO geometry as in REFERENCE.md
auto cisd = rhf.cisd(/*n_states=*/3);
// cisd roots: excitation energies [eV] including doubles correction
```

```bash
./build/regressions/reg_cisd-h2co-sto3g
```

## Results

### Accuracy vs offline pin (PySCF TDA + Head–Gordon \(U+V\))

| Quantity | Pin | Tol |
| -------- | --: | --- |
| CIS S₁ | 4.278299 eV | 1e-4 eV |
| **CIS(D) S₁** | **4.544577 eV** | **1e-4 eV** |

### Speed

| Twin | \(T_{\mathrm{PySCF}}\) | \(T_{\mathrm{ATV}}\) | \(r\) | ≤10× |
| ---- | ---------------------: | -------------------: | ----: | ---- |
| `tda` (CIS twin) | 0.89 s | 0.22 s | **0.24** | ✓ |

ATV path includes CIS(D); twin is TDA-only → ratio is optimistic upper-bound on speed.

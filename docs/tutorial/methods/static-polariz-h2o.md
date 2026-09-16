# RHF static polarizability — H₂O / STO-3G

*(Technical-report draft; narrative house style TBD — rewrite later.)*

Dipole polarizability \(\alpha_{ab} = \partial \mu_a / \partial F_b\) from
**CPHF** electric-field response (singlet A+B).
Regression: `regressions/scf/static-polariz-h2o`.

## Theory

Static polarizability is a second energy derivative w.r.t. a uniform field.
CPHF solves the first-order orbital response \(U^{F_b}\) and contracts with
dipole integrals. Equivalent checks: finite-field \(\mathrm{d}\mu/\mathrm{d}F\)
and \(-\partial^2 E/\partial F^2\).

## Practice

```cpp
cpu::scf::RHF rhf;
rhf.basis = cpu::scf::BasisSet::STO3G;
rhf.run(frame);
auto alpha = rhf.polarizability();   // 3×3 [a.u.]
```

```bash
./build/regressions/reg_static-polariz-h2o
```

## Results

### Accuracy vs PySCF 2.14.0 (CPHF / FF cross-check)

Pins and component-wise tols live in `REFERENCE.md` (analytic CPHF on PySCF
MOs/ERIs; FF \(h\sim 2\times 10^{-4}\) a.u.). Product gate: match pin within
reg tolerance (typically \(\sim 10^{-5}\) a.u. class).

### Speed

| Twin | \(T_{\mathrm{PySCF}}\) | \(T_{\mathrm{ATV}}\) | \(r\) | ≤10× |
| ---- | ---------------------: | -------------------: | ----: | ---- |
| vacuum `rhf` | 0.63 s | 0.13 s | **0.20** | ✓ |

ATV includes CPHF polarizability work; twin is SCF-only → ratio optimistic.

# H₂O RHF/STO-3G — external field + point-charge embedding

## Geometry (Å)

Same H₂O as `h2o_rhf_dipole` / multipole oracles:

| Atom | x | y | z |
|------|---|---|---|
| O | 0.0000 | 0.0000 | 0.1173 |
| H | 0.0000 | 0.7572 | −0.4692 |
| H | 0.0000 | −0.7572 | −0.4692 |

## Theory

- RHF / STO-3G, `conv_tol = 1e-12`
- Finite field: F_z = ±1×10⁻⁴ a.u., origin = center of nuclear charge
  (for this geometry R₀ = 0)
- Point charges (Å, e): q₁ = +0.5 at (0, 0, 3.0), q₂ = −0.5 at (0, 0, −3.0)

## PySCF 2.14.0 pins (external oracle, not a runtime dependency)

```text
E0 (vacuum)              = -74.96302313846292  Ha
mu_z                     = -0.6787872654524012 a.u.
E_mm (qmmm.mm_charge)    = -74.98418931920534  Ha
  (QM energy + QM–MM nuclear + electron–MM; NO MM–MM self energy)
MM–MM classical          = q1*q2 / |R1-R2|
                         = (0.5)*(-0.5) / (6.0 * a0⁻¹)
                         = -0.022049050455259  Ha
                         (a0⁻¹ = 1.8897261245650618 Bohr/Å)
```

Atomiverse includes MM–MM in `v_nuc_`, so:

```text
E_atv (embed) = E_mm_pyscf + MM_MM
              = -75.0062383696606 Ha
```

Finite-field central difference:

```text
(E(+F) − E(−F)) / (2F)  ≃  −μ_z   within 1e-5 a.u.
```

Script: one-off external `python3` + PySCF (not run by ctest).

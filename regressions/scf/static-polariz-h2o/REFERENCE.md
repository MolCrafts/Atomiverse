# Reference — H₂O RHF/STO-3G static polarizability (CPHF)

Oracle protocol: docs/tutorial/appendix-pyscf-oracle.md (external run, pinned).
No live PySCF at ctest time.

| Field | Value |
|---|---|
| Geometry (Å) | O(0, 0, 0.1173); H(0, ±0.7572, −0.4692) |
| Basis | STO-3G (Cartesian; pure s/p — identical to spherical for this set) |
| Method | RHF, `conv_tol = 1e-14` |
| E₀ | −74.96302313846286 Ha |
| PySCF version | 2.14.0 |
| Observable | Static dipole polarizability α_ab = ∂μ_a/∂F_b [a.u.] |
| Origin | Center of nuclear charge (origin = 0 for this geometry) |
| Oracle path | Analytic CPHF on PySCF MOs/ERIs: ¹(A+B)·U^{F_b} = −r_b^{vo}, α_ab = −4 Σ_ai U^b_ai ⟨a\|r_a\|i⟩; cross-checked by finite-field dμ/dF and −∂²E/∂F² (h = 2×10⁻⁴ a.u., h1 = F·⟨r⟩ matching Atomiverse `ExternalField`) |
| **α_xx** | **0.04190863** a.u. |
| **α_yy** | **5.20973255** a.u. |
| **α_zz** | **2.12715820** a.u. |
| Off-diagonal | ~0 (C₂v molecular plane yz) |
| Gate | 1×10⁻⁵ a.u. per diagonal component |

## Pin commands (offline; never in ctest)

```bash
bash tools/setup_pyscf_env.sh
# Analytic CPHF (PySCF MOs + ao2mo ERI) and finite-field dμ/dF with the
# repository water geometry; see agent pin log 2026-07-23.
```

## Theory notes

- Closed-shell CPHF electric-field response: S^{(F)} = 0; skeleton Fock
  derivative is the length-gauge dipole matrix (electron charge −1 a.u.
  ⇒ h += F·r). Helgaker, Jørgensen, Olsen, *Molecular Electronic-Structure
  Theory*, response chapter; Pople et al. for the nuclear CPHF twin used
  by `RHF::hessian`.
- STO-3G lacks diffuse / polarization functions — α_xx (out of plane) is
  unphysically small (~0.04 a.u.). The regression is engine-vs-engine at
  identical theory level, not a comparison to experiment (~9–10 a.u.
  isotropic for water).

# scf-review-01-linalg — shared dense linear algebra, permanent gate

Behaviour-preserving refactor gate for spec `scf-review-01-linalg` (hoist the
duplicated dense linear-algebra primitives — `sym_power`, `eig_nonsymmetric`,
`solve_pcg`, and the dRPA solve on `OrbitalHessian` — out of `cpu/scf` into the
`cpu/linalg` infra layer). This case is an **out-of-library** consumer (public
headers + link `atomiverse`, nothing else) that pins the two physical paths
touching the hoisted primitives to previously cited offline oracle values. No
live third-party oracle runs at ctest time; all goldens are hard-coded
literals.

Both numbers below are **reused verbatim** from existing sibling regressions —
this case does not introduce a new oracle number, it re-gates the same
literature values through the refactored code path.

## Part 1 — H₂/STO-3G direct-RPA correlation energy

Path: `RHF::direct_rpa()` (`sym_power` → `OrbitalHessian::direct_rpa`). The
correlation energy is the Furche plasmon formula on the Coulomb-only Casida
spectrum, E_c = ½(Σ Ω_s − Tr A) with Ω_s the positive eigenvalues of the
Hermitian reduction (A−B)^{1/2}(A+B)(A−B)^{1/2}.

Golden source: `regressions/scf/direct-rpa-h2-sto3g/REFERENCE.md`
(PySCF 2.14.0, exact MO-ERI plasmon, offline pin 2026-07-23).

| Atom | x [Å] | y [Å] | z [Å] |
|------|-------|-------|-------|
| H    | 0.0   | 0.0   | 0.0   |
| H    | 0.0   | 0.0   | 0.74  |

- Basis STO-3G (Cartesian), RHF `conv_tol = 1e-12`.

| Quantity | Value | Gate |
|----------|-------|------|
| E_RHF | −1.116759307396 Ha | 1e-6 |
| **E_c^dRPA** | **−0.020633073749 Ha** | **1e-12** (spec honest gate) |
| Ω₁ | 1.570852128278 Ha | 1e-6 |
| Tr A | 1.612118275776 Ha | 1e-6 |
| E_tot^dRPA | −1.137392381145 Ha | 1e-6 |
| plasmon identity \|E_c − ½(Ω₁−Tr A)\| | 0 | 1e-12 |

The spec fixes E_c at **≤1e-12 Ha** as the honest (bitwise-subset) invariant for
the refactor: ATV reproduces the PySCF pin to all 12 printed decimals, so a
1e-12 gate is achievable and correct here (verified 2026-07-24).

## Part 2 — H₂O/STO-3G CPHF static isotropic polarizability

Path: `RHF::polarizability()` (CPHF electric-field linear response, backed by
`solve_pcg`). Static dipole polarizability α_ab = ∂μ_a/∂F_b [a.u.].

Golden source: `regressions/scf/static-polariz-h2o/REFERENCE.md`
(PySCF 2.14.0 analytic CPHF on PySCF MOs/ERIs, cross-checked by finite-field
dμ/dF, offline pin 2026-07-23).

| Atom | x [Å] | y [Å] | z [Å] |
|------|-------|-------|-------|
| O    | 0.0   | 0.0   | 0.1173  |
| H    | 0.0   | 0.7572  | −0.4692 |
| H    | 0.0   | −0.7572 | −0.4692 |

- Basis STO-3G, RHF `conv_tol = 1e-12`. Molecular plane yz (C₂v);
  off-diagonal α ≈ 0.

| Quantity | Value | Gate |
|----------|-------|------|
| α_xx | 0.04190863 a.u. | 1e-6 |
| α_yy | 5.20973255 a.u. | 1e-6 |
| α_zz | 2.12715820 a.u. | 1e-6 |
| **α_iso = Tr(α)/3** | **2.459599793… a.u.** (from the three diagonals) | **1e-6** |
| off-diagonal | ~0 | 1e-4 |

STO-3G lacks diffuse / polarization functions, so α is unphysically small vs
experiment (~9–10 a.u. isotropic); this is an engine-vs-engine pin at identical
theory level, not a comparison to experiment.

## Citations

- Furche, J. Chem. Phys. **114**, 5982 (2001), DOI 10.1063/1.1353585
  (plasmon formula for RPA correlation).
- Ren et al., New J. Phys. **14**, 053020 (2012) (frequency-domain dRPA).
- Zhu & Chan, J. Chem. Theory Comput. **17**, 727 (2021); PySCF `gw.rpa`.
- Helgaker, Jørgensen, Olsen, *Molecular Electronic-Structure Theory*
  (response chapter — CPHF electric-field polarizability).
- Sibling in-tree gates: `regressions/scf/direct-rpa-h2-sto3g`,
  `regressions/scf/static-polariz-h2o`.

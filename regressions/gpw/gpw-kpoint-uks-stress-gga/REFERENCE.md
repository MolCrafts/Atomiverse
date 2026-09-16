# Reference — Multi-k UKS GPW analytic cell stress (RevPBE / RPBE / BLYP)

## Method

Collinear unrestricted Kohn–Sham GPW with **Monkhorst–Pack k-point sampling**
and **Hellmann–Feynman fixed-\(P_k^\sigma\) analytic cell stress** for pure
spin-GGA functionals:

1. One multi-k UKS SCF → occupied Bloch MOs `C_occ_k` / `C_occ_k_beta` and
   band energies → rebuild \(P_k^\sigma\), \(W_k^\sigma\) (occupancy 1 per spin).
2. Central finite difference of the **non-self-consistent** Lagrangian
   \(E_L[\{P_k^\sigma\},\{W_k^\sigma\};\varepsilon]\) under orthorhombic
   diagonal strain (rebuild \(S_k,T_k,V_{\mathrm{nl}}^k\), grid, Poisson,
   **spin GGA XC**; do **not** re-converge SCF).
3. \(\sigma_{aa}=-\Omega^{-1}\partial E_L/\partial\varepsilon_{aa}\)
   (CP2K `STRESS_TENSOR` sign; E Hartree, \(\Omega\) Bohr³).

Oracle twin: multi-SCF strain-FD (`gpw_stress_fd`) on the same method.
Activation gate: \(|E_{\mathrm{RevPBE}}-E_{\mathrm{PBE}}|\) on the same
geometry.

Public path: `cpu::gpw::gpw_stress(atoms, method, h)` with
`n_alpha`/`n_beta` + `k_mesh` + `xc = RevPBE|RPBE|BLYP`. No public `*Config`,
no factories.

## System

| | |
| --- | --- |
| Molecule | H₂, high-spin (M = 3), compressed bond along x |
| Positions [Bohr] | H1 (5.5, 6.0, 6.0), H2 (6.5, 6.0, 6.0) |
| Cell | cubic 12 Bohr, periodic XYZ |
| Basis | DZV-GTH-PADE (H) — in-tree table |
| PP | GTH-PADE-q1 |
| XC | RevPBE / RPBE / BLYP (spin GGA) |
| Cutoff | 100 Ry, local_pp_on_grid |
| k mesh | MONKHORST-PACK 1×1×2 (TR → κ_z = ±¼ folded, w = 1) |
| Spin | Nα=2, Nβ=0 |
| Stress step | h = 2×10⁻³ |

## Offline correctness

### Dual-gate (decisive) — first green 2026-07-26

| Quantity | Value |
| --- | --- |
| E (k-UKS PBE 1×1×2, 100 Ry) | −0.566633887639 Ha |
| **E (k-UKS RevPBE)** | **−0.574211377589 Ha** (6 iters, nk=1) |
| **E (k-UKS RPBE)** | **−0.575044480519 Ha** (6 iters, nk=1) |
| **E (k-UKS BLYP)** | **−0.558884091671 Ha** (6 iters, nk=1) |
| \|E_RevPBE − E_PBE\| | **7.577×10⁻³ Ha** (activation) |
| max\|\σ_an−σ_fd\| RevPBE | **4.63×10⁻¹⁰** Ha/Bohr³ |
| max\|\σ_an−σ_fd\| RPBE | **5.60×10⁻¹⁰** Ha/Bohr³ |
| max\|\σ_an−σ_fd\| BLYP | **1.54×10⁻⁹** Ha/Bohr³ |

RevPBE analytic stress (representative):

| Component | analytic [Ha/Bohr³] |
| --- | --- |
| σ_xx | 4.22391784e-04 |
| σ_yy | −7.94777145e-05 |
| σ_zz | −7.92623369e-05 |

### CP2K NUMERICAL twins (documentation)

| Field | Value |
| ----- | ----- |
| Engine | CP2K 2026.1 (when available) |
| Inputs | `tools/cp2k_oracle/inputs/gpw-kpoint-uks-stress-gga/h2_uks_kpoint_stress_{revpbe,rpbe,blyp}.inp` |
| Note | BASIS_MOLOPT uses SZV-MOLOPT-GTH, not DZV-GTH-PADE. Absolute same-basis |
|       | twin pending. Pin host had **no CP2K module** at first green. |

## Tolerances

| Check | Tol |
| ----- | --- |
| max\|\σ_an − σ_fd\| diag | 1×10⁻⁴ Ha/Bohr³ |
| off-diagonal product | 1×10⁻⁸ |
| \|E_RevPBE − E_PBE\| activation | ≥ 1×10⁻⁴ Ha |
| \|E − pin\| soft | 1×10⁻³ Ha |
| max\|\σ_diag\| floor | > 1×10⁻⁸ Ha/Bohr³ |

## Performance

| Item | Value |
| ---- | ----- |
| `perf_exempt` | **true** |
| reason | No same-basis CP2K fair NUMERICAL stress wall (DZV-GTH-PADE not in BASIS_MOLOPT; no CP2K module on pin host). |
| r = T_atv/T_cp2k | n/a (exempt) |
| `within_10x` | n/a (exempt) |
| T_atv (regression dual-gate, OMP=8, Release) | **~2.7 s** |

## Provenance

- Spec: `.claude/specs/gpw-kpoint-uks-stress-gga/`
- Twin inputs: `tools/cp2k_oracle/inputs/gpw-kpoint-uks-stress-gga/`
- Deps: `gpw-kpoint-uks-stress` ✓, `gpw-uks-revpbe` ✓, `gpw-uks-rpbe` ✓,
  `gpw-uks-blyp` ✓, `gpw-uks-stress` ✓
- Quickstep stress: VandeVondele et al., CPC **167**, 103 (2005); fixed-P cell
  theorem (Nielsen & Martin, PRB **32**, 3780 (1985))

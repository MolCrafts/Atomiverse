# Reference — Multi-k UKS GPW analytic cell stress (high-spin H₂)

## Method

Collinear unrestricted Kohn–Sham GPW (PADE LSD) with **Monkhorst–Pack
k-point sampling** and **Hellmann–Feynman fixed-\(P_k^\sigma\) analytic cell
stress**:

1. One multi-k UKS SCF → occupied Bloch MOs `C_occ_k` / `C_occ_k_beta` and
   band energies `eps_k` / `eps_k_beta` → rebuild \(P_k^\sigma\), \(W_k^\sigma\)
   (occupancy 1 per spin).
2. Central finite difference of the **non-self-consistent** Lagrangian
   \(E_L[\{P_k^\sigma\},\{W_k^\sigma\};\varepsilon]\) under orthorhombic
   diagonal strain at fixed fractional coordinates (rebuild \(S_k,T_k,V_{\mathrm{nl}}^k\),
   grid, Poisson, spin XC; do **not** re-converge SCF).
3. \(\sigma_{aa}=-\Omega^{-1}\partial E_L/\partial\varepsilon_{aa}\)
   (CP2K `STRESS_TENSOR` sign; E Hartree, \(\Omega\) Bohr³).

Oracle twin: multi-SCF strain-FD (`gpw_stress_fd`) on the same UKS+k method.
Closed-shell identity: multi-k UKS (nα=nβ=1) ≡ multi-k RKS stress.

Public path: `cpu::gpw::gpw_stress(atoms, method, h)` with
`n_alpha`/`n_beta` + `k_mesh`. No public `*Config`, no factories.

## System

| | |
| --- | --- |
| Molecule | H₂, high-spin (M = 3), compressed bond along x |
| Positions [Bohr] | H1 (5.5, 6.0, 6.0), H2 (6.5, 6.0, 6.0) |
| Cell | cubic 12 Bohr, periodic XYZ |
| Basis | DZV-GTH-PADE (H) — in-tree table |
| PP | GTH-PADE-q1 |
| XC | PADE LSD |
| Cutoff | 100 Ry, local_pp_on_grid |
| k mesh | MONKHORST-PACK 1×1×2 (TR → κ_z = ±¼ folded, w = 1) |
| Spin | Nα=2, Nβ=0 |
| Stress step | h = 2×10⁻³ |

## Offline correctness

### Dual-gate (decisive)

| Quantity | Value |
| --- | --- |
| **E (k-UKS 1×1×2, 100 Ry)** | **−0.528389371092 Ha** (6 iters, nk=1 TR-folded) |
| σ_xx analytic | 4.26434770e-04 Ha/Bohr³ |
| σ_yy analytic | −8.07875807e-05 Ha/Bohr³ |
| σ_zz analytic | −8.05687961e-05 Ha/Bohr³ |
| max\|\σ_an−σ_fd\| | **3.86×10⁻¹⁰** Ha/Bohr³ |
| Closed-shell k-UKS vs k-RKS max\|\dσ\| | **2.35×10⁻¹⁰** Ha/Bohr³ |

### CP2K NUMERICAL twin

| Field | Value |
| ----- | ----- |
| Engine | CP2K 2026.1 (when available) |
| Input | `tools/cp2k_oracle/inputs/gpw-kpoint-uks-stress/h2_uks_kpoint_stress.inp` |
| Note | BASIS_MOLOPT uses SZV-MOLOPT-GTH, not DZV-GTH-PADE. Absolute same-basis
        twin pending a CP2K data path that exposes DZV-GTH-PADE. Pin host had
        **no CP2K module** at first green (2026-07-26). |

Energy soft pin matches sibling `gpw-kpoint-uks-forces` (same geometry/method).

## Tolerances

| Check | Tol |
| ----- | --- |
| max\|\σ_an − σ_fd\| diag | 1×10⁻⁴ Ha/Bohr³ |
| off-diagonal product | 1×10⁻⁸ |
| closed-shell k-UKS vs k-RKS | 5×10⁻⁴ Ha/Bohr³ |
| \|E − pin\| soft | 1×10⁻³ Ha |
| max\|\σ_diag\| floor | > 1×10⁻⁸ Ha/Bohr³ |

## Performance

| Item | Value |
| ---- | ----- |
| `perf_exempt` | **true** |
| reason | No same-basis CP2K fair NUMERICAL stress wall (DZV-GTH-PADE not in BASIS_MOLOPT; no CP2K module on pin host). |
| r = T_atv/T_cp2k | n/a (exempt) |
| `within_10x` | n/a (exempt) |
| T_atv (regression dual-gate, OMP=8, Release) | **~1.0 s** |

When CP2K is available, run the twin single-rank, record both walls for a
product-only Evaluate pass, and replace this section (target r ≤ 10).

## Provenance

- Spec: `.claude/specs/gpw-kpoint-uks-stress/`
- Twin input: `tools/cp2k_oracle/inputs/gpw-kpoint-uks-stress/h2_uks_kpoint_stress.inp`
- Deps: `gpw-kpoints-uks` ✓, `gpw-kpoint-stress` ✓, `gpw-uks-stress` ✓,
  `gpw-kpoint-uks-forces` ✓
- Quickstep stress: VandeVondele et al., CPC **167**, 103 (2005); fixed-P cell
  theorem (Nielsen & Martin, PRB **32**, 3780 (1985))

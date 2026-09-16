# Reference — GPW analytic forces at multi-k UKS (periodic high-spin H₂)

## Method

Collinear unrestricted Kohn–Sham GPW (PADE LSD) with **Monkhorst–Pack
k-point sampling** and **analytic ionic forces** (one SCF + one gradient
pass over Bloch-phased \(S_k/T_k/V_\mathrm{nl}\), dual \(P_\alpha^k/P_\beta^k\),
and spin-resolved \(V_\mathrm{eff}^{\alpha/\beta}\) on complex \(\chi^k\)).

Public path: `cpu::gpw::GpwDft` with `n_alpha=2`, `n_beta=0`,
`k_mesh = {1,1,2}` + `gpw_forces_analytic(atoms, cfg)`. No public `*Config`,
no factories.

## System

| | |
| --- | --- |
| Molecule | H₂, high-spin (M = 3), compressed bond along x |
| Positions [Bohr] | H1 (5.5, 6.0, 6.0), H2 (6.5, 6.0, 6.0) |
| Cell | cubic 12 Bohr, periodic XYZ |
| Basis | DZV-GTH-PADE (H) — in-tree table |
| PP | GTH-PADE-q1 |
| XC | PADE LSD |
| Cutoff | 100 Ry, NGRIDS 1 |
| k mesh | MONKHORST-PACK 1×1×2 (TR → κ_z = ±¼ folded, w = 1) |
| Spin | Nα=2, Nβ=0 |
| SCF | tol 1e-8 Ha, `local_pp_on_grid` |

## Offline correctness

### Identities (product gate)

- Restoring bond force (F0_x < 0 < F1_x), exact Newton-3 on F_x, |∑F| residual.
- Closed-shell multi-k UKS (nα=nβ=1) matches multi-k RKS forces
  (max\|dF\| ≤ 5×10⁻⁴ Ha/Bohr) — same gate as energy path `gpw-kpoints-uks`.
- Explicit-Γ through k machinery matches Γ-UKS forces (unit test).

### Analytic vs FD

Documented offline on a related off-symmetry geometry when measured; full 6N
FD is outside the ≤ 60 s regression budget (same policy as `gpw-kpoint-forces`).

### CP2K ENERGY_FORCE

| Field | Value |
| ----- | ----- |
| Engine | CP2K 2026.1 (when available) |
| Input | `tools/cp2k_oracle/inputs/gpw-kpoint-uks-forces/h2_uks_kpoint_forces.inp` |
| Note | BASIS_MOLOPT uses SZV-MOLOPT-GTH, not DZV-GTH-PADE. Absolute same-basis
        twin pending a CP2K data path that exposes DZV-GTH-PADE. |

### Atomiverse numbers (first green, 2026-07-25)

| Quantity | Value |
| --- | --- |
| **E (k-UKS 1×1×2, 100 Ry)** | **−0.528389371092 Ha** (6 iters, nk=1 TR-folded) |
| **F₀ [Ha/Bohr]** | **(−0.554544062995, ~0, ~0)** |
| **F₁ [Ha/Bohr]** | **(+0.554544062995, ~0, ~0)** |
| \|∑F\| | 1.6×10⁻¹⁴ Ha/Bohr |
| Closed-shell k-UKS vs k-RKS max\|dF\| | 7.4×10⁻⁷ Ha/Bohr |

## Tolerances

- \|E − pin\| ≤ 1e-5 Ha
- \|F_x − pin\| ≤ 1e-5 Ha/Bohr
- \|∑F\| ≤ 1e-6 Ha/Bohr; Newton-3 on F_x ≤ 1e-6
- Closed-shell k-UKS vs k-RKS ≤ 5e-4 Ha/Bohr

## Performance

| Item | Value |
| ---- | ----- |
| `perf_exempt` | **true** |
| reason | No same-basis CP2K fair wall (DZV-GTH-PADE not in BASIS_MOLOPT; no CP2K module on pin host). |
| r = T_atv/T_cp2k | n/a (exempt) |
| `within_10x` | n/a (exempt) |
| T_atv (regression, OMP=8) | ~1–2 s |

## Provenance

- Spec: `.claude/specs/gpw-kpoint-uks-forces/`
- Deps: `gpw-kpoints-uks` ✓, `gpw-kpoint-forces` ✓, `gpw-uks-forces` ✓
- Quickstep forces: VandeVondele et al., CPC **167**, 103 (2005)

# Reference — GPW analytic forces at k-points (periodic H₂)

## Method

Closed-shell GPW Kohn–Sham DFT (PADE LDA) with **Monkhorst–Pack k-point
sampling** and **analytic ionic forces** (one SCF + one gradient pass over
Bloch-phased S_k / T_k / V_nl and complex χ^k on the density grid).

Public path: `cpu::gpw::GpwDft` with `k_mesh = {1,1,2}` +
`gpw_forces_analytic(atoms, cfg)`. No public `*Config`, no factories.

## System

| | |
| --- | --- |
| Molecule | H₂, compressed bond along x |
| Positions [Bohr] | H1 (5.5, 6.0, 6.0), H2 (6.5, 6.0, 6.0) |
| Cell | cubic 12 Bohr, periodic XYZ |
| Basis | DZV-GTH-PADE (H) — in-tree table |
| PP | GTH-PADE-q1 |
| XC | PADE |
| Cutoff | 100 Ry, NGRIDS 1 |
| k mesh | MONKHORST-PACK 1×1×2 (TR → κ_z = ±¼ folded, w = 1) |
| SCF | tol 1e-8 Ha, `local_pp_on_grid` |

## Offline correctness

### Analytic vs FD (decisive)

On the related off-symmetry geometry
(H1 5.5/6/6, H2 6.7/6.1/5.9, cutoff 120 Ry, same k-mesh), measured
**max\|F_analytic − F_fd\| = 5.98×10⁻⁶ Ha/Bohr** (central h = 10⁻³ Bohr,
Release, OMP=8). That identity proves the k-path force is −∇E of the k-path
energy. Full 6N FD is too slow for the ≤ 60 s regression budget; the
regression freezes the absolute first-green band on the compressed bond
geometry (exact Newton-3, \|∑F\| ~ 10⁻¹⁴).

### Explicit-Γ identity

Forces through the complex k machinery at an explicit κ = (0,0,0) match the
pre-existing real Γ `assemble_gpw_forces` to **1.1×10⁻¹⁰ Ha/Bohr**
(unit-test geometry, 120 Ry).

### CP2K ENERGY_FORCE

| Field | Value |
| ----- | ----- |
| Engine | CP2K 2026.1 (local Arrhenius build) |
| Input | `tools/cp2k_oracle/inputs/gpw-kpoint-forces.inp` |
| Note | BASIS_MOLOPT has DZVP-MOLOPT-GTH, not DZV-GTH-PADE. Exploratory
        SZV-MOLOPT pin (different geometry/basis): E ≈ −1.026 Ha,
        \|F_x\| ≈ 0.18 Ha/Bohr, wall ≈ 120 s. Absolute same-basis twin
        pending a CP2K data path that exposes DZV-GTH-PADE. |

### Atomiverse numbers (first green, 2026-07-25)

| Quantity | Value |
| --- | --- |
| **E (k 1×1×2, 100 Ry)** | **−1.057153829426 Ha** (8 iters) |
| **F₀ [Ha/Bohr]** | **(−0.23541914472, ~0, ~0)** |
| **F₁ [Ha/Bohr]** | **(+0.23541914472, ~0, ~0)** |
| \|∑F\| | 3.2×10⁻¹⁴ Ha/Bohr |
| Offline FD identity (related geom.) | max\|dF\| = 5.98×10⁻⁶ Ha/Bohr |

## Tolerances

- \|E − pin\| ≤ 1e-5 Ha
- \|F_x − pin\| ≤ 1e-5 Ha/Bohr
- \|∑F\| ≤ 1e-6 Ha/Bohr; Newton-3 on F_x ≤ 1e-6

## Performance

| Item | Value |
| ---- | ----- |
| `perf_exempt` | **true** |
| reason | No same-basis CP2K fair wall (DZV-GTH-PADE not in BASIS_MOLOPT). |
| r = T_atv/T_cp2k | n/a (exempt) |
| `within_10x` | n/a (exempt) |
| T_atv (regression, OMP=8) | ~1 s |

## Provenance

- Spec: `.claude/specs/gpw-kpoint-forces/`
- Deps: `gpw-kpoints-si` ✓, `gpw-forces-water` ✓
- Quickstep forces: VandeVondele et al., CPC **167**, 103 (2005)

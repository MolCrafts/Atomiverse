---
slug: gpw-uks-multigrid-kpoints
status: done
created: 2026-07-31
revised: 2026-07-31
domain: gpw / k-point sampling / UKS / multigrid
queue: cp2k-method-loop
deferred: false
depends_on: []
unblocks: []
grilled: false
---

# Spec: `gpw-uks-multigrid-kpoints` — Multi-k collinear UKS with REL_CUTOFF multigrid

**Slug:** `gpw-uks-multigrid-kpoints`
**Domain:** GPW k-point collinear UKS SCF (CPU), CP2K CUTOFF/REL_CUTOFF multigrid
**Queue row:** `gpw-uks-multigrid-kpoints` (GapCensus wave 2 shift; assigned wave 3 shift 2026-07-31)
**Oracle:** CP2K 2026.1 (`cp2k.psmp`, module `CP2K/2026.1` on Arrhenius), offline pin

## Summary

`GpwDft::build` rejects `ngrids > 1` for collinear UKS on the k-point path
(`gpw_solver.cpp:5309`: `reject("multigrid (ngrids > 1) with UKS …")`) — the
multi-k UKS runner (`run_gpw_scf_kpoints_uks`) is still single-grid, so
spin-polarized Bloch SCF always pays the fine CUTOFF grid for every product.
Both parents already shipped:

- `gpw-multigrid-kpoints` — closed-shell multi-k RKS multigrid: image-cell
  product machinery (`collocate_products_cells` / `integrate_products_cells`),
  the Bloch-phased density stack **D(t) = Σ_k w_k Re[P^k e^{−2πi κ·t}]**, and
  the per-cell KS stack **V^k = Σ_t e^{+2πi κ·t} V_t**;
- `gpw-uks-multigrid` — Γ UKS multigrid: per-spin product collocation and
  per-spin restrict+integrate through the same `MultiGrid` levels.

This spec is their product: **per-spin Bloch-phased density / potential
stacks**. No new internal machinery is required — the `internal/multigrid.h`
`_cells` entry points are already spin-agnostic (they take an arbitrary
density-matrix stack); the work is wiring them through the multi-k UKS
runner's four grid chokepoints and opening the build gate.

## Domain basis

Exactly the RKS multi-k multigrid algebra, applied once per spin channel σ:

- **Density:** D^σ(t) = Σ_k w_k Re[P^{σ,k} e^{−2πi κ·t}] weighs the per-level
  product collocation; per-level ρ^σ_l prolong spectrally (`prolong_fft`) to
  the finest grid. ρ = ρ^α + ρ^β feeds Poisson; (ρ^α, ρ^β) feed the spin XC.
- **KS integration:** the two fine effective potentials V^σ_eff restrict back
  per level (`restrict_fft`, exact adjoint) and integrate into the real
  per-cell stacks V^σ_t; the per-k grid KS matrices are the phase sums
  V^{σ,k} = Σ_t e^{+2πi κ·t} V^σ_t (Hermitian, exact transpose of the
  density assembly — the SCF stays variational per spin).
- Γ limit: D^σ(t) = P^σ ∀t — the shipped Γ-UKS product weight. Time-reversal
  folding stays exact (only Re[P e^{−ik·t}] and real V_t enter;
  P^{−k} = conj(P^k) per spin).

## Deliverables

1. `gpw_solver.cpp` `run_gpw_scf_kpoints_uks`: under
   `ngrids > 1 && rel_cutoff > 0` skip the `KpointBasisGrid` build; add the
   same multigrid block as the RKS k runner (products, `tcell_phase`,
   `mg_density_stack` / `mg_density` / `mg_vt_stack` / `mg_phase_V`) and use
   it per spin at every grid chokepoint: `build_uks_k_energy` (OT +
   fixed-density eval), the DIIS loop's inline twin, `rebuild_all_H`, and the
   Hirshfeld dispersion fold. Single-grid path stays bit-identical.
2. `GpwDft::build` k-gate: accept `ngrids > 1 && rel_cutoff > 0` for
   collinear UKS (drop the 5309 reject; `ngrids > 1` without positive
   `rel_cutoff` still throws).
3. Unit gate flip: `tests/cpu/gpw/test_cpu_gpw_solver.cpp`
   `BuildRejectsKpointMultigridUks` → `BuildAcceptsKpointMultigridUks`.
4. Regression `regressions/gpw/gpw-uks-multigrid-kpoints/` + `REFERENCE.md`
   with the offline CP2K 2026.1 pin (twin input
   `tools/cp2k_oracle/inputs/gpw-uks-multigrid-kpoints/h2_uks_kpoints_mg.inp`:
   high-spin H₂ toy of `gpw-kpoints-uks` — 8 Bohr cubic periodic cell,
   DZV-GTH-PADE / GTH-PADE-q1, PADE LSD, multiplicity 3 (Nα=2, Nβ=0),
   CUTOFF 120 Ry, **NGRIDS 4 / REL_CUTOFF 30**, MONKHORST-PACK 2×2×2).

## Acceptance

See `acceptance.md`. Headline: |E_ATV − E_CP2K| in the ~5e-7 Ha class
(`gpw-multigrid-kpoints` precedent 4.7e-7), gated at 1e-3; ATV
multigrid↔single-grid and TR / Γ-limit identities.

## Out of scope

- Nonlocal vdW on the k path (`gpw-vdw-kpoints`, distinct gate at 5326) and
  Γ-UKS nonlocal vdW (`gpw-vdw-uks`, gate 5188) — sibling wave-3 rows in the
  same file; coordinate at merge.
- Multi-k UKS OT density-guess warm start (`guess_P_k_beta` + `use_ot`
  refuses — `second-gen-cpmd-ot-uks-kpoints` owns it).
- Analytic multi-k forces/stress consistency with the multigrid functional
  (force paths keep their own assembly, as Γ and RKS-k multigrid ship).

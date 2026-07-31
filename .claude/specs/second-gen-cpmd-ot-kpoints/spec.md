---
slug: second-gen-cpmd-ot-kpoints
status: approved
created: 2026-07-31
revised: 2026-07-31
domain: dynamics / gpw
queue: cp2k-method-loop
deferred: false
depends_on: [second-gen-cpmd-kpoints, second-gen-cpmd-ot, ot-kpoints]
unblocks: [second-gen-cpmd-ot-kpoints]
grilled: false
---

# Spec: `second-gen-cpmd-ot-kpoints` — multi-k SGCP with OT partial-SCF corrector

**Slug:** `second-gen-cpmd-ot-kpoints`
**Domain:** dynamics (second-generation Car–Parrinello) on multi-k GPW
**Queue row:** `second-gen-cpmd-ot-kpoints` (GapCensus wave 15, worker wave 2)
**Branch:** `wf/second-gen-cpmd-ot-kpoints`

## Summary

Multi-k second-generation Car–Parrinello (Kolafa **ASPC on complex Bloch
densities \(P_k\)**) currently supports only the **diagonalization** SCF
corrector — `GPWEvaluator::build` throws at `gpw_evaluator.cpp:319` for
`use_ot` + a k-mesh under `second_gen_cp`. Both parents exist as product:

- `second-gen-cpmd-kpoints` — multi-k SGCP, diagonalization corrector
- `second-gen-cpmd-ot` / `second-gen-cpmd-uks-ot` — Γ SGCP, OT corrector
- `ot-kpoints` — full-budget multi-k OT SCF (|E_OT − E_DIIS| ≤ 1e-4 Ha pin)

This spec closes the cross product for **closed-shell RKS**: ASPC-predicted
\(P_k\) → **natural-orbital warm start of the per-k OT manifold** → partial
OT corrector (`max_iter = second_gen_cp_max_iter`) → analytic multi-k forces
at the post-corrector state.

## Root cause of the gap

`run_gpw_scf_kpoints` seeds \(P_k\) from `guess_P_k` but always builds the OT
manifold `Cocc_k` from **H0** eigenvectors (`gpw_solver.cpp` init loop:
"still build Cocc_k from H0 for the OT path"). The multi-k OT loop then
immediately overwrites \(P_k\) from `Cocc_k`, so an ASPC prediction would be
**silently ignored** — a budget-capped corrector from a cold H0 manifold gives
garbage forces. The evaluator therefore refused the combination outright.

## Design (mirror of the Γ warm start, `gpw_solver.cpp:3984`)

1. **Solver** (`run_gpw_scf_kpoints` init loop): when `cfg.use_ot` and
   `guess_P_k != nullptr`, purify each predicted \(P_k\) to natural orbitals —
   `internal::diagonalize_k(P_k_guess, orth[ik])` (eigenvalues ascend, largest
   occupations in the last columns), take the last `n_occ` columns, Löwdin
   orthonormalize against \(S_k\) → `Cocc_k[ik]`. Throw when the kept
   orthogonal basis has fewer than `n_occ` columns. Smear path is untouched
   (its manifold expansion overwrites the cold start; the evaluator exposes no
   smearing, so SGCP cannot reach it).
2. **Evaluator** (`GPWEvaluator::build`): narrow the multi-k SGCP `use_ot`
   throw to **multi-k UKS only** (dual-spin k-OT warm start stays refused —
   `run_gpw_scf_kpoints_uks` still cold-starts its manifolds from H0). The
   multi-k RKS SGCP launch branch already passes `guess_p_k` and reads
   `P_k` / `C_occ_k` / `eps_k`, which the k-OT path exports via
   `store_k_payload` on both the converged and budget-exhausted exits — no
   launch-path change.
3. **No new public surface.** Same members as the parents:
   `use_ot`, `k_mesh`, `second_gen_cp*` (via `SecondGenCp{gpw, K, budget}`).

## Out of scope

- Multi-k **UKS** SGCP + OT (dual complex OT manifold warm start) — refused
  with a precise message; follow-on row.
- Metallic smearing / +U under the SGCP corrector (evaluator exposes no smear;
  +U OT k path honors the warm start but is not pinned here).
- CUDA (loop scope excludes CUDA ports).

## Acceptance

See `acceptance.md`. Regression:
`regressions/dynamics/second-gen-cpmd-ot-kpoints/` (family home of all SGCP
dynamics pins), `atv_add_regression(dynamics second-gen-cpmd-ot-kpoints
THREADS 8)`.

## Oracle

**CP2K cannot produce this number even in principle:** CP2K's OT is Γ-only —
`&OT` + `&KPOINTS` aborts ("OT and k-points are not compatible"). The family
policy (per `second-gen-cpmd-kpoints` / `second-gen-cpmd-ot`) applies:
**literature NVE-drift envelope** (Kühne PRL 98, 066401 (2007); Kolafa JCC 25,
335 (2004)) plus **internal corrector A/B identity** — full-budget multi-k OT
vs multi-k diagonalization single point at the initial geometry must agree to
the `ot-kpoints` pin level (≤ 1e-4 Ha). An unexecutable-input sketch is still
recorded at `tools/cp2k_oracle/inputs/second-gen-cpmd-ot-kpoints/` with the
incompatibility documented. `perf_exempt` (no oracle twin can exist).

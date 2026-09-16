# Multi-k Car–Parrinello MD on Bloch \(C_k\) (H₂ NVE, GPW)

*(Technical-report page; CP2K capability loop wave 14.)*

| Field | Value |
| ----- | ----- |
| Slug | `cpmd-kpoints` |
| Regression | `regressions/dynamics/cpmd-kpoints/` |
| Public API | `cpu::electronic::CarParrinello{cpu::gpw::GPWEvaluator{…, k_mesh}, mu_au}` bound with `set_evaluator_ref`; `CpmdCplx` propagator. CP mode is the wrapper type — `cpmd`/`cpmd_mu` are private on `GPWEvaluator`. The nuclear dt is pushed from the MD integrator into the wrapper (`TimestepProvider` → `TimestepConsumer`) — never set on the evaluator |
| Roadmap / queue | `docs/roadmap/dft.md` (AIMD); queue done 2026-07-26 |

## Summary

Car–Parrinello molecular dynamics with **fictitious electronic degrees of
freedom on multi-k closed-shell RKS** GPW: one propagator per sampled Bloch
\(k\), complex coefficients \(C_k\), with the holonomic constraint
\(C_k^\dagger S_k C_k = I\) enforced by electronic RATTLE and the velocity
constraint killing \(\mathrm{Herm}(C_k^\dagger S_k \dot C_k)\).

Production steps run at **fixed density** (`iters == 0`): the nuclei feel
\(-\partial E/\partial R\) at the current electronic state, never a re-converged
SCF. Electronic kinetic energy is
\(T_e = \sum_k w_k \tfrac12 \mu \lVert \dot C_k\rVert_F^2\).

## System

| Quantity | Value |
| -------- | ----- |
| Species | H₂ (periodic), \(R = 1.0\) Bohr compressed bond along \(x\), cell-centred |
| Cell | cubic \(L = 12\) Bohr (≈ 6.350 Å) |
| Spin / XC | closed-shell RKS, \(n_{\mathrm{occ}} = 1\); PADE |
| Basis / PP | DZV-GTH-PADE / GTH-PADE-q1 |
| Cutoff | 100 Ry |
| k-mesh | \(1\times1\times2\), time reversal off → \(n_k = 2\) |
| CPMD | \(\mu = 400\) a.u.; `dt` = 0.1 fs (nuclear = electronic) |
| Trajectory | 1 cold SCF seed + 1 CP baseline + 4 production steps, NVE |

## Gate

Conserved-quantity **envelope**: relative \(E_{\mathrm{cons}}\) drift
\(\delta \le 0.10\) with multi-k Bloch payload diagnostics (`n_kpts >= 2`),
production SCF iterations exactly 0, bounded potential energy, finite
coordinates, and a soft bond window. Unit gates cover the `CpmdCplx` seed and
advance invariants (\(C^\dagger S C = I\), Hermitian velocity constraint), the
multi-k fixed-density path, and the rejected combinations (OT + CPMD,
XL + CPMD). Multi-k **UKS** CPMD ships separately
([cpmd-uks-kpoints](cpmd-uks-kpoints.md)).

Ratio: **perf_exempt** — the gate is an NVE drift envelope, not a wall twin
(same policy as `cpmd-water` and `xl-bomd-kpoints`).

See `regressions/dynamics/cpmd-kpoints/REFERENCE.md`.

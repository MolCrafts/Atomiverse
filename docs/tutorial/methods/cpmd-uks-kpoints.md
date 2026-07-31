# Multi-k UKS Car–Parrinello MD on dual-spin Bloch \(C_k^\sigma\) (H₂ NVE, GPW)

*(Technical-report page; CP2K capability loop worker wave 2.)*

| Field | Value |
| ----- | ----- |
| Slug | `cpmd-uks-kpoints` |
| Regression | `regressions/dynamics/cpmd-uks-kpoints/` |
| Public API | `cpu::electronic::CarParrinello{cpu::gpw::GPWEvaluator{…, n_alpha, n_beta, k_mesh}, mu_au}` bound with `set_evaluator_ref`; one `CpmdCplx` propagator per (k, spin). CP mode is the wrapper type — `cpmd`/`cpmd_mu` are private on `GPWEvaluator`. The nuclear dt is pushed from the MD integrator into the wrapper (`TimestepProvider` → `TimestepConsumer`) |
| Roadmap / queue | `docs/roadmap/dft.md` (AIMD); queue done 2026-07-31 |

## Summary

Car–Parrinello molecular dynamics with **fictitious electronic degrees of
freedom on multi-k collinear UKS** GPW — the `cpmd-kpoints` (multi-k RKS
Bloch CP) × `cpmd-uks` (Γ dual-spin CP) crossing. One complex propagator per
sampled Bloch \(k\) **and spin channel** \(\sigma \in \{\alpha, \beta\}\),
each with the holonomic constraint \(C_k^{\sigma\dagger} S_k C_k^\sigma = I\)
enforced by electronic RATTLE, and the spin-diagonal occupancy-1 EOM

\[
\mu\,\ddot C_k^\sigma
  = -(I - S_k C_k^\sigma C_k^{\sigma\dagger})\,(2 H_k^\sigma C_k^\sigma)
    + \text{RATTLE}
\]

on the spin-resolved KS matrices \(H_k^\alpha / H_k^\beta\) from the
**dual-spin fixed-density multi-k evaluation** (`max_iter == 0` at the
propagated \(\{P_k^\alpha, P_k^\beta\}\), exported through the new
`GpwResult::H_ks_k` / `H_ks_k_beta` payload; `iters == 0` on every
production step). Electronic kinetic energy is
\(T_e = \sum_k w_k \sum_\sigma \tfrac12 \mu \lVert \dot C_k^\sigma
\rVert_F^2\). The multi-k electronic thermostat remains refused (NVE
electrons on the k path; queued `cpmd-thermostat-kpoints`).

## System

| Quantity | Value |
| -------- | ----- |
| Species | H₂ (periodic), \(R = 1.0\) Bohr compressed bond along \(x\), cell-centred |
| Cell | cubic \(L = 12\) Bohr (≈ 6.350 Å) |
| Spin / XC | collinear UKS, \(n_\alpha = n_\beta = 1\) (closed-shell fillings — both channels propagate, molecule stays bound); PADE (LSD) |
| Basis / PP | DZV-GTH-PADE / GTH-PADE-q1 |
| Cutoff | 100 Ry |
| k-mesh | \(1\times1\times2\), time reversal off → \(n_k = 2\) |
| CPMD | \(\mu = 400\) a.u.; `dt` = 0.1 fs (nuclear = electronic) |
| Trajectory | 1 cold SCF seed + 1 CP baseline + 4 production steps, NVE |

## Gate

Conserved-quantity **envelope**: relative \(E_{\mathrm{cons}}\) drift
\(\delta \le 0.10\) with dual-spin multi-k diagnostics (`n_kpts >= 2`,
\(T_e > 0\) once propagated, production SCF iterations exactly 0), plus the
**UKS-vs-RKS closed-shell identity** — dedicated multi-k UKS and multi-k
RKS single points at `tol = 1e-7` must agree within 3×10⁻⁴ eV (anchoring
this path to the CP2K-pinned `gpw-bomd-kpoints` / `gpw-kpoints-uks`
surfaces). Unit gates cover the dual-spin fixed-density
solver export (`H_ks_k_beta`, partial-spin-guess refusal) and the
**Γ-limit identity**: the k-path dual-spin fixed-density eval at one
explicit Γ point reproduces the Γ dual-spin eval (energy and per-spin KS
matrices) — the surface `cpmd-uks` ships on.

Ratio: **perf_exempt** — oracle family lock: CP2K 2026.1 (present on the
pin host) has no fictitious-mass CP propagator, so the CP conserved quantity
with fictitious \(T_e\) is not a number it can print (same policy as
`cpmd-kpoints` / `cpmd-uks`).

See `regressions/dynamics/cpmd-uks-kpoints/REFERENCE.md`.

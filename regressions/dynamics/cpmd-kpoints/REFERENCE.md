# Reference — multi-k Car–Parrinello MD short trajectory (H₂ NVE, GPW)

## Method

**Car–Parrinello molecular dynamics (CPMD)** with fictitious electronic
degrees of freedom on **multi-k closed-shell RKS** Gaussian-and-plane-waves
(GPW / Quickstep) forces (Bloch \(C_k\)):

- Outer loop: `cpu::md::MD` + `VelocityVerlet` (NVE; no thermostat)
- Force/energy: `cpu::gpw::GPWEvaluator` with `compute_forces = true`,
  **`k_mesh = {1,1,2}`**, **`k_time_reversal = false`** (two sampled k),
  **`cpmd = true`**, **`cpmd_mu`**; the electronic dt is **not** set on the
  evaluator — `cpu::md::MD` pushes the nuclear dt in from the `VelocityVerlet`
  integrator through the `TimestepProvider` / `TimestepConsumer` protocol at
  build
- One `cpu::electronic::CpmdCplx` per sampled k propagates occupied Bloch
  coefficients with electronic RATTLE; production steps use fixed-density
  multi-k KS eval (`iters == 0`)
- Conserved diagnostic:
  \(E_{\mathrm{cons}} = T_n + T_e + E_{\mathrm{KS}}\) with
  \(T_e = \sum_k w_k\,\tfrac12\mu\lVert\dot C_k\rVert_F^2\)
- Periodic cubic cell; PADE XC
- Units on the public surface: energy eV, length Å, time fs, force eV/Å

**Literature:**

- Car & Parrinello, *Phys. Rev. Lett.* **55**, 2471 (1985)
- Marx & Hutter, *Ab Initio Molecular Dynamics* (CUP) — CP conserved energy
- Andersen, *J. Comput. Phys.* **52**, 24 (1983) — electronic RATTLE
- Γ product paths: `regressions/dynamics/cpmd-water`, `cpmd-rattle-water`
- Multi-k force pin: `regressions/gpw/gpw-kpoint-forces`
- Multi-k AIMD family twin: `regressions/dynamics/xl-bomd-kpoints`

This is **not** a golden snapshot of Atomiverse total energy. The e2e pin is
the **envelope** on relative \(E_{\mathrm{cons}}\) drift with multi-k Bloch
CP engaged — a wrong \(C_k\) seed, real-only force path, or SCF on production
steps destroys the envelope immediately.

## System

| Quantity | Value |
| -------- | ----- |
| Species | H₂ (2 × H), periodic |
| Geometry | \(R = 1.0\) Bohr compressed bond along \(x\), cell-centred |
| Masses | 1.008 amu each |
| Cell | cubic \(L = 12\) Bohr ≈ 6.350 Å; periodic |
| Spin | closed-shell RKS, \(n_\mathrm{occ}=1\) |
| XC | PADE (LDA) |
| Basis | DZV-GTH-PADE |
| PP | GTH-PADE-q1 |
| CUTOFF | 100 Ry (envelope run) |
| k-mesh | \(1\times1\times2\), `k_time_reversal=false` → \(n_k=2\) |
| SCF seed | `tol = 1e-5`, `max_iter = 60`, DIIS |
| CPMD | \(\mu = 400\) a.u.; production fixed-density |
| Ensemble | NVE |
| `dt` | 0.1 fs (nuclear = electronic) |
| Initial velocity | antisymmetric stretch \(v_x = \pm 0.001\) Å/fs |
| Steps | 1 cold SCF seed + 1 CP baseline + 4 production |

## Offline oracle (CP2K)

**Status:** no `cp2k` binary / CP2K Lmod module on the worker host used for
this pin (Arrhenius login-class, **2026-07-26**) — same inventory as
`cpmd-water` / `xl-bomd-kpoints`. A CP2K twin sketch (short NVE Quickstep
with fictitious electronic DOF + `&KPOINTS`) lives in
`tools/cp2k_oracle/inputs/cpmd-kpoints/`. When a binary is available,
re-run offline and replace this section with version + measured drift +
fair wall.

Until then the regression asserts the **literature CP/NVE conservation
envelope** plus multi-k payload diagnostics (`n_kpts >= 2`), production
`iters == 0`, and soft H–H window.

## Observable and tolerance

Let \(E(t) = E_{\mathrm{KS}}(t) + T_n(t) + T_e(t)\) [eV], with \(E_0\) after
the first CP production step (Te engaged).

\[
\delta = \max_t \frac{|E(t) - E_0|}{\max(1, |E_0|)} \le 0.10
\]

Additional gates:

- Production SCF **iters == 0** (fixed-density multi-k eval)
- `n_kpts >= 2` after seed
- PE finite and bound (\(< -5\) eV after warm-up)
- KE finite and \(\ge 0\), non-trivial after warm-up
- All coordinates finite
- Soft H–H window: \(0.4 < r_{\mathrm{HH}} < 1.5\) Å after production

## Perf gate

| Field | Value |
| ----- | ----- |
| `perf_exempt` | **true** |
| Reason | No offline CP2K binary on host for a fair wall twin of multi-step multi-k CP AIMD (2026-07-26 inventory) |
| `ratio_atv_over_cp2k` | n/a |

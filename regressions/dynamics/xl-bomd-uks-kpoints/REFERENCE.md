# Reference — multi-k UKS XL-BOMD short trajectory (H₂ NVE, GPW forces)

## Method

**Extended-Lagrangian Born–Oppenheimer MD (XL-BOMD)** in its dissipative
(Niklasson) form, on **multi-k collinear UKS** Gaussian-and-plane-waves
(GPW / Quickstep) forces:

- Outer loop: `cpu::md::MD` + `VelocityVerlet` (NVE; no thermostat)
- Force/energy: `cpu::gpw::GPWEvaluator` with `compute_forces = true`,
  **`n_alpha = 1`, `n_beta = 1`** (closed-shell fillings on the UKS path —
  both spin channels non-empty; high-spin H₂ is unbound and cannot pin NVE),
  **`k_mesh = {1,1,2}`**, **`k_time_reversal = false`** (two sampled k),
  **`xl_bomd = true`, `xl_bomd_k = 5`**
- Independent auxiliary complex densities \(P_k^\alpha, P_k^\beta\) are
  propagated per sampled k with the Table-I recursion and used as the
  multi-k UKS SCF initial densities:

  \[
  P^{k\sigma}_{n+1} = 2P^{k\sigma}_n - P^{k\sigma}_{n-1}
            + \kappa\,(D^{k\sigma}_n - P^{k\sigma}_n)
            + \alpha \sum_{m=0}^{K} c_m P^{k\sigma}_{n-m}
  \]

  with \(D^{k\sigma}_n\) the SCF spin density at step \(n\) and
  \((\kappa,\alpha,c_m)\) the \(K=5\) row of Table I in Niklasson et al.
  (2009): \(\kappa = 1.82\), \(\alpha = 0.018\),
  \(c = \{-6, 14, -8, -3, 4, -1\}\).
- SCF is still converged to `tol` each step (SCF-converged XL-BOMD).
- Periodic cubic cell; PADE (LSD) XC
- Units on the public surface: energy eV, length Å, time fs, force eV/Å

**Literature:**

- A. M. N. Niklasson, *Phys. Rev. Lett.* **100**, 123004 (2008)
- A. M. N. Niklasson et al., *J. Chem. Phys.* **130**, 214109 (2009) —
  Table I coefficients (unit-pinned in `test_cpu_gpw_xl_bomd`)
- Prior product paths: `xl-bomd-water` (Γ RKS), `xl-bomd-uks` (Γ UKS),
  `xl-bomd-kpoints` (multi-k RKS)
- Multi-k UKS force pin: `regressions/gpw/gpw-kpoint-uks-forces`

This is **not** a golden snapshot of Atomiverse total energy. The e2e pin is
the **envelope** on relative NVE drift with dual-spin multi-k XL
\(P_k^\sigma\) guesses engaged — a wrong recursion (α-only, RKS total on
the UKS path, shape leak) destroys SCF convergence or the drift envelope
immediately.

## System

| Quantity | Value |
| -------- | ----- |
| Species | H₂ (2 × H), periodic |
| Geometry | \(R = 1.0\) Bohr compressed bond along \(x\), cell-centred |
| Masses | 1.008 amu each |
| Cell | cubic \(L = 12\) Bohr ≈ 6.350 Å; periodic |
| Spin | collinear UKS closed-shell fillings, \(n_\alpha=1\), \(n_\beta=1\) |
| XC | PADE (LSD) |
| Basis | DZV-GTH-PADE |
| PP | GTH-PADE-q1 |
| CUTOFF | 100 Ry (envelope run) |
| k-mesh | \(1\times1\times2\), `k_time_reversal=false` → \(n_k=2\) |
| SCF | `tol = 1e-5`, `max_iter = 60`, DIIS diagonalization |
| XL-BOMD | `K = 5`; dual `XlBomdCplx` per k (α + β) |
| Ensemble | NVE |
| `dt` | 0.5 fs |
| Initial velocity | antisymmetric stretch \(v_x = \pm 0.001\) Å/fs |
| Steps | 1 warm-up + 4 production (chunked) |

## Offline oracle (CP2K)

**Status:** no `cp2k` binary / CP2K Lmod module on the worker host used for
this pin (Arrhenius login-class, **2026-07-26**) — same inventory as
`xl-bomd-kpoints` / `xl-bomd-uks`. A CP2K twin sketch (short NVE Quickstep
UKS MD with ASPC + `&KPOINTS`) lives in
`tools/cp2k_oracle/inputs/xl-bomd-uks-kpoints/`. When a binary is available,
re-run offline and replace this section with version + measured drift +
fair wall.

Until then the regression asserts the **literature XL-BOMD/NVE conservation
envelope** plus multi-k dual-spin payload diagnostics (`n_kpts >= 2`,
`P_k` / `P_k_beta` sizes), closed-shell ⟨S²⟩ ≈ 0 window, per-step SCF
convergence with dual XL \(P_k^\sigma\) guesses, and soft H–H window.
Table-I coefficients remain unit-pinned (`test_cpu_gpw_xl_bomd`).

## Observable and tolerance

Let \(E(t) = E_\mathrm{pot}(t) + E_\mathrm{kin}(t)\) [eV], with \(E_0\) after
one warm-up step (XL histories seeded).

\[
\delta = \max_t \frac{|E(t) - E_0|}{\max(1, |E_0|)} \le 0.05
\]

Additional gates:

- SCF **converged at every sampled step** with dual multi-k XL density guesses
- `n_kpts >= 2` and `P_k.size() == P_k_beta.size() == n_kpts` after warm-up
- PE finite and bound (\(< -5\) eV after warm-up)
- KE finite and \(\ge 0\), non-trivial after warm-up
- ⟨S²⟩ finite and \(\ge 0\) after warm-up and production (multi-k UKS
  stores the integer seed formula \(s_z(s_z+1)+N_\beta\), not full Löwdin)
- All coordinates finite
- Soft H–H window: \(0.4 < r_{\mathrm{HH}} < 1.5\) Å after production

## Perf gate

| Field | Value |
| ----- | ----- |
| `perf_exempt` | **true** |
| Reason | No offline CP2K binary on host for a fair wall twin of multi-step multi-k UKS AIMD (2026-07-26 inventory) |
| `ratio_atv_over_cp2k` | n/a |
| `within_10x` | n/a (exempt) |

## Dual gate summary

| Gate | Result |
| ---- | ------ |
| Correctness | Regression `reg_xl-bomd-uks-kpoints` asserts A6; unit A1–A5 in `test_cpu_gpw_xl_bomd` |
| Speed | `perf_exempt=true` (no CP2K binary for AIMD twin wall) |

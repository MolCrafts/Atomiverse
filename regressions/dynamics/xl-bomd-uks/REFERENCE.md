# Reference — XL-BOMD short trajectory (O₂ triplet NVE, Γ-UKS GPW)

## Method

**Extended-Lagrangian Born–Oppenheimer MD (XL-BOMD)** in its dissipative
(Niklasson) form, on **open-shell Γ-UKS** Gaussian-and-plane-waves (GPW /
Quickstep) forces:

- Outer loop: `cpu::md::MD` + `VelocityVerlet` (NVE; no thermostat)
- Force/energy: `cpu::gpw::GPWEvaluator` with `compute_forces = true`,
  **`n_alpha = 7`, `n_beta = 5`**, **`xl_bomd = true`, `xl_bomd_k = 5`**
- Independent auxiliary AO densities \(P^\alpha, P^\beta\) are propagated
  with the same Table-I recursion and used as SCF initial spin densities:

  \[
  P^\sigma_{n+1} = 2P^\sigma_n - P^\sigma_{n-1} + \kappa\,(D^\sigma_n - P^\sigma_n)
            + \alpha \sum_{k=0}^{K} c_k P^\sigma_{n-k}
  \]

  with \(D^\sigma_n\) the SCF spin density at step \(n\) and
  \((\kappa,\alpha,c_k)\) the \(K=5\) row of Table I in Niklasson et al.
  (2009): \(\kappa = 1.82\), \(\alpha = 0.018\),
  \(c = \{-6, 14, -8, -3, 4, -1\}\).
- SCF is still converged to `tol` each step (SCF-converged XL-BOMD).
- Fixed cubic cell; MT / nonperiodic
- Units on the public surface: energy eV, length Å, time fs, force eV/Å

**Literature:**

- A. M. N. Niklasson, *Phys. Rev. Lett.* **100**, 123004 (2008)
- A. M. N. Niklasson et al., *J. Chem. Phys.* **130**, 214109 (2009) —
  Table I coefficients (unit-pinned in `test_cpu_gpw_xl_bomd`)
- Closed-shell product path: `regressions/dynamics/xl-bomd-water`
- UKS O₂ energy / forces offline CP2K pins:
  `regressions/gpw/gpw-uks-o2-gamma`, `gpw-uks-forces`

This is **not** a golden snapshot of Atomiverse total energy. The e2e pin is
the **envelope** on relative NVE drift with dual-spin XL guesses engaged —
a wrong recursion (α-only, swapped channels, non-density aux) destroys SCF
convergence or the drift envelope immediately.

## System

| Quantity | Value |
| -------- | ----- |
| Species | O₂ (2 × O), isolated / MT |
| Geometry | \(R = 1.2075\) Å, centered in cubic cell |
| Masses | 15.999 amu each |
| Cell | cubic \(L = 12\) Å; `PERIODIC NONE` / `nonperiodic = true` |
| Spin | UKS triplet, \(n_\alpha=7\), \(n_\beta=5\) (GTH-q6, 12 valence e⁻) |
| XC | PADE (LSD) |
| Basis | DZVP-MOLOPT-GTH |
| PP | GTH-PBE-q6 |
| CUTOFF | 200 Ry (envelope run) |
| SCF | `tol = 1e-5`, `max_iter = 80`, DIIS diagonalization |
| XL-BOMD | `K = 5`; dual α/β propagators |
| Ensemble | NVE |
| `dt` | 0.5 fs |
| Initial velocity | antisymmetric stretch \(v_z = \pm 0.0008\) Å/fs |
| Steps | 1 warm-up + 4 production (chunked) |

## Offline oracle (CP2K)

**Status:** no `cp2k` binary / CP2K Lmod module on the worker host used for
this pin (Arrhenius login-class, **2026-07-26**) — same inventory as
`xl-bomd-water`. A CP2K twin sketch (short NVE Quickstep UKS MD; ASPC
extrapolation is CP2K's MD default density-guess propagation) lives in
`tools/cp2k_oracle/inputs/xl-bomd-uks/`. When a binary is available, re-run
offline and replace this section with version + measured drift + fair wall.

Until then the regression asserts the **literature XL-BOMD/NVE conservation
envelope** plus finite/bounded trajectory diagnostics, triplet ⟨S²⟩ window,
and per-step SCF convergence with dual XL spin guesses; Table-I coefficients
remain unit-pinned (`test_cpu_gpw_xl_bomd`).

## Observable and tolerance

Let \(E(t) = E_\mathrm{pot}(t) + E_\mathrm{kin}(t)\) [eV], with \(E_0\) after
one warm-up step (XL histories seeded).

\[
\delta = \max_t \frac{|E(t) - E_0|}{\max(1, |E_0|)} \le 0.05
\]

Additional gates:

- SCF **converged at every sampled step** with dual XL spin density guesses
- PE finite and bound (\(< -100\) eV after warm-up)
- KE finite and \(\ge 0\), non-trivial after warm-up
- ⟨S²⟩ in \([1.5, 2.5]\) (triplet window) after warm-up and production
- All coordinates finite
- Soft O–O window: \(0.9 < r_{\mathrm{OO}} < 1.6\) Å after production

## Perf gate

| Field | Value |
| ----- | ----- |
| `perf_exempt` | **true** |
| Reason | No offline CP2K binary on host for a fair wall twin of multi-step UKS AIMD (2026-07-26 inventory) |
| `ratio_atv_over_cp2k` | n/a |
| `within_10x` | n/a (exempt) |

## Dual gate summary

| Gate | Result |
| ---- | ------ |
| Correctness | Regression `reg_xl-bomd-uks` asserts A7; unit A1–A6 in `test_cpu_gpw_xl_bomd` |
| Speed | `perf_exempt=true` (no CP2K binary for AIMD twin wall) |

# Reference — multi-k second-generation CP (H₂ NVE, GPW)

## Method

**Second-generation Car–Parrinello MD (SGCP / 2nd-gen CP)** on **multi-k
closed-shell RKS** Gaussian-and-plane-waves (GPW / Quickstep) forces:

- Outer loop: `cpu::md::MD` + `VelocityVerlet` (NVE; no thermostat)
- Force/energy: `cpu::gpw::GPWEvaluator` with `compute_forces = true`,
  **`k_mesh = {1,1,2}`**, **`k_time_reversal = false`** (two sampled k),
  **`second_gen_cp = true`**, **`second_gen_cp_k = 1`**,
  **`second_gen_cp_max_iter = 5`**
- Electronics: Kolafa **ASPC** on complex Bloch densities \(P_k\)
  (`cpu::electronic::ASPCPredictorCplx`, one history per k) + **partial SCF
  corrector** (iteration budget ≪ full BOMD); analytic multi-k forces at
  the post-corrector density

**Not BOMD** (production steps cap SCF iters at the corrector budget).  
**Not classical CPMD** (no fictitious electronic mass / \(T_e\)).  
**Not XL-BOMD** (ASPC extrapolation, not Niklasson auxiliary Verlet).

## Literature

- T. D. Kühne, M. Krack, F. R. Mohamed, M. Parrinello, *Phys. Rev. Lett.*
  **98**, 066401 (2007) — second-generation CP
- J. Kolafa, *J. Comput. Chem.* **25**, 335 (2004) — ASPC
- CP2K `EXTRAPOLATION ASPC` + `&KPOINTS`
- Γ product paths: `regressions/dynamics/second-gen-cpmd-water` (+ UKS/OT)
- Multi-k density-guess socket twin: `regressions/dynamics/xl-bomd-kpoints`
- Multi-k force pin: `regressions/gpw/gpw-kpoint-forces`

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
| ASPC \(K\) | 1 (needs \(K+1=2\) seed densities) |
| Corrector SCF | `max_iter = 5` after ASPC warm |
| Ensemble | NVE |
| `dt` | 0.5 fs |
| Initial velocity | antisymmetric stretch \(v_x = \pm 0.001\) Å/fs |
| Steps | 2 seed + 1 baseline + 2 production |

## Offline oracle (CP2K)

**Status:** no `cp2k` binary / CP2K Lmod module on the worker host used for
this pin (Arrhenius login-class, **2026-07-26**) — same inventory as
`xl-bomd-kpoints` / `second-gen-cpmd-water`. A CP2K twin sketch (short NVE
Quickstep MD with `EXTRAPOLATION ASPC` + limited `MAX_SCF` + `&KPOINTS`)
lives in `tools/cp2k_oracle/inputs/second-gen-cpmd-kpoints/`. When a binary
is available, re-run offline and replace this section with version +
measured drift + fair wall.

Until then the regression asserts the **literature SGCP/NVE conservation
envelope** plus multi-k payload diagnostics (`n_kpts >= 2`, `P_k` size),
production corrector-budget gate, and soft H–H window.

## Observable and tolerance

Let \(E(t) = E_\mathrm{pot}(t) + E_\mathrm{kin}(t)\) [eV], with \(E_0\) after
seed + first warm ASPC production step.

\[
\delta = \max_t \frac{|E(t) - E_0|}{\max(1, |E_0|)} \le 0.10
\]

Additional gates:

- Production SCF **iters ≤ `second_gen_cp_max_iter`** once ASPC is warm
- `n_kpts >= 2` and `P_k.size() == n_kpts` after seed
- PE finite and bound (\(< -5\) eV after seed)
- KE finite and \(\ge 0\), non-trivial after seed
- Soft H–H window: \(0.4 < r_{\mathrm{HH}} < 1.5\) Å after production

## Perf gate

| Field | Value |
| ----- | ----- |
| `perf_exempt` | **true** |
| Reason | No offline CP2K binary on host for a fair wall twin of multi-step multi-k SGCP (2026-07-26 inventory) |
| `ratio_atv_over_cp2k` | n/a |

## Honest scope

- Closed-shell multi-k RKS product pin; multi-k UKS dual ASPC is wired in
  `GPWEvaluator` (build guard + force path) but not e2e-pinned here
- Diagonalization corrector only (multi-k + OT refused)
- Soft drift tol vs full BO for partial-SCF heat on short traj

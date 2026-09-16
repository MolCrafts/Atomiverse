# Reference — Second-generation Car–Parrinello MD (O₂ triplet NVE, Γ-UKS GPW)

## Method

**Second-generation Car–Parrinello MD (SGCP / 2nd-gen CP)** on **open-shell
Γ-UKS** Gaussian-and-plane-waves (GPW / Quickstep) forces:

- Outer loop: `cpu::md::MD` + `VelocityVerlet` (NVE; no thermostat)
- Force/energy: `cpu::gpw::GPWEvaluator` with `compute_forces = true`,
  **`n_alpha = 7`, `n_beta = 5`**, **`second_gen_cp = true`**,
  **`second_gen_cp_k = 1`**, **`second_gen_cp_max_iter = 5`**
- Independent spin densities \(P^\alpha, P^\beta\) use dual Kolafa ASPC
  predictors; after ASPC is warm, each MD force step runs a **partial**
  SCF with iteration budget `second_gen_cp_max_iter` (not full BOMD)
- Analytic forces at the post-corrector density (`gpw_forces_from_scf`)
- Fixed cubic cell; MT / nonperiodic
- Units on the public surface: energy eV, length Å, time fs, force eV/Å

**Not BOMD** (production SCF capped at the corrector budget).  
**Not classical CPMD** (no fictitious electronic mass / \(T_e\)).  
**Not XL-BOMD** (ASPC extrapolation, not Niklasson auxiliary Verlet).

**Literature:**

- T. D. Kühne, M. Krack, F. R. Mohamed, M. Parrinello, *Phys. Rev. Lett.*
  **98**, 066401 (2007) — second-generation CP
- J. Kolafa, *J. Comput. Chem.* **25**, 335 (2004) — ASPC
- Closed-shell product path: `regressions/dynamics/second-gen-cpmd-water`
- UKS O₂ energy / forces offline CP2K pins:
  `regressions/gpw/gpw-uks-o2-gamma`, `gpw-uks-forces`
- Related dual-spin AIMD: `regressions/dynamics/xl-bomd-uks`

This is **not** a golden snapshot of Atomiverse total energy. The e2e pin is
the **envelope** on relative NVE drift with dual-spin ASPC + partial SCF
engaged — a broken spin guess (α-only, swapped channels) blows production
iters, ⟨S²⟩, or the drift envelope immediately.

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
| SCF | `tol = 1e-5`, seed `max_iter = 80`, DIIS diagonalization |
| SGCP | ASPC \(K = 1\); corrector budget `max_iter = 5` once warm |
| Ensemble | NVE |
| `dt` | 0.5 fs |
| Initial velocity | antisymmetric stretch \(v_z = \pm 0.0008\) Å/fs |
| Steps | 2 seed (warm ASPC) + 1 baseline production + 2 drift steps |

## Offline oracle (CP2K)

**Status:** no `cp2k` binary / CP2K Lmod module on the worker host used for
this pin (Arrhenius login-class, **2026-07-26**) — same inventory as
`second-gen-cpmd-water` / `xl-bomd-uks`. A CP2K twin sketch (short NVE
Quickstep UKS MD; `EXTRAPOLATION ASPC` + limited `MAX_SCF`) lives in
`tools/cp2k_oracle/inputs/second-gen-cpmd-uks/`. When a binary is available,
re-run offline and replace this section with version + measured drift + fair wall.

Until then the regression asserts the **literature SGCP/NVE conservation
envelope** plus method-identity gates (production SCF iters ≤ corrector
budget, triplet ⟨S²⟩, finite/bounded trajectory).

## Observable and tolerance

Let \(E(t) = E_\mathrm{pot}(t) + E_\mathrm{kin}(t)\) [eV], with \(E_0\) after
the first **warm** production step (ASPC history filled).

\[
\delta = \max_t \frac{|E(t) - E_0|}{\max(1, |E_0|)} \le 0.10
\]

(Soft relative drift vs full BO 0.05 — partial-SCF heat on short traj, same
class as closed-shell `second-gen-cpmd-water`.)

Additional gates:

- Production SCF **iters ≤ `second_gen_cp_max_iter` (5)** once ASPC is warm
- PE finite and bound (\(< -100\) eV after seed)
- KE finite and \(\ge 0\), non-trivial after seed
- ⟨S²⟩ in \([1.5, 2.5]\) (triplet window) after seed and production
- All coordinates finite
- Soft O–O window: \(0.9 < r_{\mathrm{OO}} < 1.6\) Å after production

## Perf gate

| Field | Value |
| ----- | ----- |
| `perf_exempt` | **true** |
| Reason | No offline CP2K binary on host for a fair wall twin of multi-step UKS SGCP AIMD (2026-07-26 inventory) |
| `ratio_atv_over_cp2k` | n/a |
| `within_10x` | n/a (exempt) |

## Dual gate summary

| Gate | Result |
| ---- | ------ |
| Correctness | Regression `reg_second-gen-cpmd-uks` asserts A2/A4; unit A1/A3 in `test_cpu_gpw_second_gen_cpmd` |
| Speed | `perf_exempt=true` (no CP2K binary for AIMD twin wall) |

# Reference — multi-k UKS second-generation CP with OT corrector (H₂ NVE, GPW)

## Method

**Second-generation Car–Parrinello MD (SGCP)** on **multi-k collinear
UKS** GPW with the **Orbital Transformation (OT)** minimizer as the
partial-SCF corrector:

- Outer loop: `cpu::md::MD` + `VelocityVerlet` (NVE; no thermostat)
- Force/energy: `cpu::gpw::GPWEvaluator` with `compute_forces = true`,
  **`n_alpha = 1`, `n_beta = 1`** (collinear UKS path),
  **`k_mesh = {1,1,2}`**, **`k_time_reversal = false`** (two sampled k),
  **`use_ot = true`**, **`second_gen_cp = true`**, **`second_gen_cp_k = 1`**,
  **`second_gen_cp_max_iter = 5`** (via `cpu::electronic::SecondGenCp`)
- Electronics: dual Kolafa **ASPC** on complex Bloch spin densities
  \(P_k^\alpha, P_k^\beta\) (`ASPCPredictorCplx`, one history per
  (k, σ)) → **dual-spin natural-orbital warm start of the per-(k, σ) OT
  manifolds** (eigenvectors of \(X_k^\dagger P_k^\sigma X_k\), largest
  occupations, Löwdin-orthonormalized against \(S_k\); the complex,
  dual-spin mirror of the Γ-UKS `guess_P`/`guess_P_beta` OT warm start
  and of the multi-k RKS `guess_P_k` warm start) → **partial OT**
  corrector (iteration budget ≪ full BOMD) → analytic multi-k UKS forces
  at the post-corrector state

**Not BOMD** (production steps cap OT iters at the corrector budget).  
**Not classical CPMD** (no fictitious electronic mass / \(T_e\)).  
**Not the diagonalization-corrector multi-k UKS SGCP** (that DIIS path
was already product inside `GPWEvaluator::launch`).  
**Not multi-k RKS SGCP-OT** (`second-gen-cpmd-ot-kpoints`).  
**Not XL-BOMD** (ASPC extrapolation, not Niklasson auxiliary Verlet).

## Literature

- T. D. Kühne, M. Krack, F. R. Mohamed, M. Parrinello, *Phys. Rev. Lett.*
  **98**, 066401 (2007) — second-generation CP
- J. Kolafa, *J. Comput. Chem.* **25**, 335 (2004) — ASPC
- J. VandeVondele, J. Hutter, *J. Chem. Phys.* **118**, 4365 (2003) — OT
- Multi-k RKS OT-corrector twin:
  `regressions/dynamics/second-gen-cpmd-ot-kpoints`
- Γ-UKS OT-corrector twin: `regressions/dynamics/second-gen-cpmd-uks-ot`
- Multi-k UKS OT SCF identity pin: `regressions/gpw/ot-uks-kpoints`
  (A2: |E_OT − E_DIIS| ≤ 1e-4 Ha, ⟨S²⟩ identity)

## System

| Quantity | Value |
| -------- | ----- |
| Species | H₂ (2 × H), periodic |
| Geometry | \(R = 1.0\) Bohr compressed bond along \(x\), cell-centred |
| Masses | 1.008 amu each |
| Cell | cubic \(L = 12\) Bohr ≈ 6.350 Å; periodic |
| Spin | collinear UKS, \(n_\alpha = n_\beta = 1\) (dual-spin plumbing on a singlet filling — same family input as `cpmd-uks-kpoints` / `xl-bomd-uks-kpoints`) |
| XC | PADE (LSD) |
| Basis | DZV-GTH-PADE |
| PP | GTH-PADE-q1 |
| CUTOFF | 100 Ry (envelope run) |
| k-mesh | \(1\times1\times2\), `k_time_reversal=false` → \(n_k=2\) |
| ASPC \(K\) | 1 (needs \(K+1=2\) seed densities) |
| Corrector | OT, `max_iter = 5` after ASPC warm |
| Ensemble | NVE |
| `dt` | 0.5 fs |
| Initial velocity | antisymmetric stretch \(v_x = \pm 0.001\) Å/fs |
| Steps | 2 seed + 1 baseline + 2 production |

## Offline oracle (CP2K)

**CP2K cannot produce this number even in principle:** CP2K's OT
implementation is Γ-point-only — an input combining `&OT` with `&KPOINTS`
is refused by CP2K itself ("OT and k-points not compatible", Quickstep SCF
input validation; see the CP2K `&SCF / &OT` reference manual). Adding
`UKS TRUE` does not change the refusal. The unexecutable twin sketch is
still recorded at
`tools/cp2k_oracle/inputs/second-gen-cpmd-ot-uks-kpoints/` with the
incompatibility noted — the same family lock as
`second-gen-cpmd-ot-kpoints` (RKS) and `second-gen-cpmd-uks-ot` (Γ-UKS).
The diagonalization-corrector twin (`second-gen-cpmd-kpoints`) carries the
CP2K-shaped executable input for the k path.

The decisive gates are therefore **internal exactness identities plus the
literature SGCP/NVE envelope** (family policy):

1. **Corrector A/B identity** — full-budget multi-k UKS OT vs multi-k UKS
   diagonalization/DIIS at the initial geometry converge to the same SCF
   fixed point: \(|E_\mathrm{OT} − E_\mathrm{diag}| \le 10^{-4}\) Ha (the
   `ot-uks-kpoints` A2 pin level; same functional, different minimizer),
   with identical ⟨S²⟩ (≤ 1e-6). Multi-k UKS ⟨S²⟩ is the integer seed
   formula \(s_z(s_z+1)+N_\beta\) (= 1 for \(n_\alpha=n_\beta=1\); no
   spatial-orbital overlap correction on the k path — same convention as
   `xl-bomd-uks-kpoints` / `cpmd-uks-kpoints`).
2. **NVE drift envelope** with the dual-spin OT corrector on the k path
   (below).

Measured at this input (2026-07-31, this tree, OMP=8):

| Gate | Value |
| ---- | ----- |
| A/B \(\|E_\mathrm{OT}-E_\mathrm{diag}\|\) | \(3.627\times10^{-11}\) Ha (gate ≤ 1e-4) |
| \(E\) (both) | \(-1.0571538294\) Ha; iters diag/ot 7/11; \(n_k=2\) |
| Seed ⟨S²⟩ OT/diag | 1.0 / 1.0 (integer seed for \(n_\alpha=n_\beta=1\)) |
| Worst NVE \(\|\mathrm{d}E\|/\mathrm{scale}\) | \(5.877\times10^{-4}\) (gate ≤ 0.10) |
| Max prod OT iters | 5 (= budget) |
| \(r_{\mathrm{HH}}\) after production | 0.911 Å |
| Wall | ~1.5 s |

## Observable and tolerance

Let \(E(t) = E_\mathrm{pot}(t) + E_\mathrm{kin}(t)\) [eV], with \(E_0\)
after seed + first warm ASPC production step.

\[
\delta = \max_t \frac{|E(t) - E_0|}{\max(1, |E_0|)} \le 0.10
\]

Additional gates:

- Corrector A/B: \(|E_\mathrm{OT} − E_\mathrm{diag}| \le 10^{-4}\) Ha,
  both converged, \(n_k = 2\), \(|\langle S^2\rangle_\mathrm{OT} −
  \langle S^2\rangle_\mathrm{diag}| \le 10^{-6}\), multi-k UKS seed
  \(\langle S^2\rangle = 1\) for \(n_\alpha=n_\beta=1\)
- Production OT **iters ≤ `second_gen_cp_max_iter`** once ASPC is warm
- `n_kpts >= 2` and **dual-spin** `P_k` / `P_k_beta` payloads sized
  `n_kpts` after seed (a silently-RKS path leaves β empty)
- PE finite and bound (\(< -5\) eV after seed)
- KE finite and \(\ge 0\), non-trivial after seed
- Soft H–H window: \(0.4 < r_{\mathrm{HH}} < 1.5\) Å after production

## Perf gate

| Field | Value |
| ----- | ----- |
| `perf_exempt` | **true** |
| Reason | CP2K's OT is Γ-only (`&OT` + `&KPOINTS` refused by CP2K, UKS included), so no oracle wall twin of multi-k UKS SGCP-OT can exist; family policy per `second-gen-cpmd-ot-kpoints` / `second-gen-cpmd-uks-ot` |
| `ratio_atv_over_cp2k` | n/a |

## Honest scope

- Collinear multi-k UKS product pin on singlet fillings (\(n_\alpha =
  n_\beta = 1\)) — the dual-spin plumbing gates (β payloads, dual ASPC,
  per-(k, σ) OT) are exercised; open-shell k-UKS physics (e.g. O₂ bands)
  rides the same code path via `ot-uks-kpoints`
- No metallic smearing / +U under the SGCP corrector here (the k-UKS OT
  warm start engages on the integer-occupation path; under smearing the
  RKS precedent discards the guess and the evaluator exposes no smearing)
- Soft drift tol vs full BO for partial-SCF heat on short traj

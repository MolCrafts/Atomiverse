# Reference — multi-k second-generation CP with OT corrector (H₂ NVE, GPW)

## Method

**Second-generation Car–Parrinello MD (SGCP)** on **multi-k closed-shell
RKS** GPW with the **Orbital Transformation (OT)** minimizer as the
partial-SCF corrector:

- Outer loop: `cpu::md::MD` + `VelocityVerlet` (NVE; no thermostat)
- Force/energy: `cpu::gpw::GPWEvaluator` with `compute_forces = true`,
  **`k_mesh = {1,1,2}`**, **`k_time_reversal = false`** (two sampled k),
  **`use_ot = true`**, **`second_gen_cp = true`**, **`second_gen_cp_k = 1`**,
  **`second_gen_cp_max_iter = 5`** (via `cpu::electronic::SecondGenCp`)
- Electronics: Kolafa **ASPC** on complex Bloch densities \(P_k\)
  (`ASPCPredictorCplx`, one history per k) → **natural-orbital warm start
  of the per-k OT manifold** (eigenvectors of \(X_k^\dagger P_k X_k\),
  largest occupations, Löwdin-orthonormalized against \(S_k\); the complex
  mirror of the Γ `guess_P` OT warm start) → **partial OT** corrector
  (iteration budget ≪ full BOMD) → analytic multi-k forces at the
  post-corrector state

**Not BOMD** (production steps cap OT iters at the corrector budget).  
**Not classical CPMD** (no fictitious electronic mass / \(T_e\)).  
**Not the diagonalization-corrector SGCP-k** (`second-gen-cpmd-kpoints`).  
**Not XL-BOMD** (ASPC extrapolation, not Niklasson auxiliary Verlet).

## Literature

- T. D. Kühne, M. Krack, F. R. Mohamed, M. Parrinello, *Phys. Rev. Lett.*
  **98**, 066401 (2007) — second-generation CP
- J. Kolafa, *J. Comput. Chem.* **25**, 335 (2004) — ASPC
- J. VandeVondele, J. Hutter, *J. Chem. Phys.* **118**, 4365 (2003) — OT
- Multi-k diagonalization-corrector twin:
  `regressions/dynamics/second-gen-cpmd-kpoints`
- Γ OT-corrector twin: `regressions/dynamics/second-gen-cpmd-ot`
- Multi-k OT SCF identity pin: `regressions/gpw/ot-kpoints`
  (|E_OT − E_DIIS| ≤ 1e-4 Ha)

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
| Corrector | OT, `max_iter = 5` after ASPC warm |
| Ensemble | NVE |
| `dt` | 0.5 fs |
| Initial velocity | antisymmetric stretch \(v_x = \pm 0.001\) Å/fs |
| Steps | 2 seed + 1 baseline + 2 production |

## Offline oracle (CP2K)

**CP2K cannot produce this number even in principle:** CP2K's OT
implementation is Γ-point-only — an input combining `&OT` with `&KPOINTS`
is refused by CP2K itself ("OT and k-points not compatible", Quickstep SCF
input validation; see the CP2K `&SCF / &OT` reference manual). The
unexecutable twin sketch is still recorded at
`tools/cp2k_oracle/inputs/second-gen-cpmd-ot-kpoints/` with the
incompatibility noted. The diagonalization-corrector twin
(`second-gen-cpmd-kpoints`) carries the CP2K-shaped input for the k path.

The decisive gates are therefore **internal exactness identities plus the
literature SGCP/NVE envelope** (family policy per `second-gen-cpmd-kpoints`
and `second-gen-cpmd-ot`):

1. **Corrector A/B identity** — full-budget multi-k OT vs multi-k
   diagonalization/DIIS at the initial geometry converge to the same SCF
   fixed point: \(|E_\mathrm{OT} − E_\mathrm{diag}| \le 10^{-4}\) Ha (the
   `ot-kpoints` pin level; same functional, different minimizer).
2. **NVE drift envelope** with the OT corrector on the k path (below).

## Observable and tolerance

Let \(E(t) = E_\mathrm{pot}(t) + E_\mathrm{kin}(t)\) [eV], with \(E_0\)
after seed + first warm ASPC production step.

\[
\delta = \max_t \frac{|E(t) - E_0|}{\max(1, |E_0|)} \le 0.10
\]

Additional gates:

- Corrector A/B: \(|E_\mathrm{OT} − E_\mathrm{diag}| \le 10^{-4}\) Ha,
  both converged, \(n_k = 2\)
- Production OT **iters ≤ `second_gen_cp_max_iter`** once ASPC is warm
- `n_kpts >= 2` and `P_k.size() == n_kpts` after seed
- PE finite and bound (\(< -5\) eV after seed)
- KE finite and \(\ge 0\), non-trivial after seed
- Soft H–H window: \(0.4 < r_{\mathrm{HH}} < 1.5\) Å after production

## Perf gate

| Field | Value |
| ----- | ----- |
| `perf_exempt` | **true** |
| Reason | CP2K's OT is Γ-only (`&OT` + `&KPOINTS` refused by CP2K), so no oracle wall twin of multi-k SGCP-OT can exist; family policy per `second-gen-cpmd-kpoints` / `second-gen-cpmd-ot` |
| `ratio_atv_over_cp2k` | n/a |

## Honest scope

- Closed-shell multi-k RKS product pin; multi-k **UKS** SGCP + OT is the
  sibling `second-gen-cpmd-ot-uks-kpoints` (dual-spin k-OT warm start)
- No metallic smearing / +U under the SGCP corrector here
- Soft drift tol vs full BO for partial-SCF heat on short traj

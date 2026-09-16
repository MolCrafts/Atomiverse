# Reference — Second-generation CP with OT corrector (H₂O NVE, GPW)

## Method

**Second-generation Car–Parrinello MD (SGCP)** on closed-shell Γ-point GPW
with the **Orbital Transformation (OT)** SCF minimizer as the partial-SCF
corrector:

- Outer loop: `cpu::md::MD` + NVE `VelocityVerlet` (nuclei)
- Force slot: `cpu::gpw::GPWEvaluator` with
  **`second_gen_cp = true`**, **`use_ot = true`**,
  **`second_gen_cp_k = 1`**, **`second_gen_cp_max_iter = 5`**,
  `compute_forces = true`
- Electronics: Kolafa **ASPC** density predictor → natural-orbital warm-start
  of the OT manifold → **partial OT** corrector (iteration budget ≪ full BOMD)
  → analytic forces at the post-corrector density

**Not BOMD** (production steps cap OT iters at the corrector budget).  
**Not classical CPMD** (no fictitious electronic mass / \(T_e\)).  
**Not DIIS SGCP** (`second-gen-cpmd-water` uses diagonalization corrector).  
**Not XL-BOMD** (ASPC extrapolation, not Niklasson auxiliary Verlet).

## Literature

- T. D. Kühne, M. Krack, F. R. Mohamed, M. Parrinello, *Phys. Rev. Lett.*
  **98**, 066401 (2007) — second-generation CP
- J. Kolafa, *J. Comput. Chem.* **25**, 335 (2004) — ASPC
- J. VandeVondele, J. Hutter, *J. Chem. Phys.* **118**, 4365 (2003) — OT
- Twin geometry family: `gpw-bomd-water`, `xl-bomd-water`, `cpmd-water`,
  `second-gen-cpmd-water`, `ot-minimizer-water`

## Setup

| Item | Value |
| ---- | ----- |
| System | Isolated H₂O (CP2K getting-started geometry) |
| Cell | 10 Å cubic, Martyna–Tuckerman (`nonperiodic`) |
| XC / basis / PP | PBE / DZVP-MOLOPT-GTH / GTH-PBE |
| CUTOFF | 200 Ry |
| dt | 0.5 fs |
| ASPC \(K\) | 1 |
| Corrector | OT, `max_iter = 5` |
| Seed | 2 full-OT steps (warm ASPC history, \(K+1\)) |
| Production | 1 baseline + 2 drift steps |

## Offline oracle (CP2K)

**Status:** no `cp2k` binary / Lmod module required on the worker pin host —
same inventory class as `second-gen-cpmd-water` / `cpmd-water`. Offline sketch:

`tools/cp2k_oracle/inputs/second-gen-cpmd-ot/h2o_sgcp_ot_nve.inp`

(`EXTRAPOLATION ASPC`, `EXTRAPOLATION_ORDER 1`, `&OT`, tight `MAX_SCF` on MD).
When a binary is available, re-run via `tools/cp2k_oracle/` and record
version + `.ener` drift + fair wall.

## Pin

| Observable | Gate |
| ---------- | ---- |
| Relative \(\|E-E_0\|/\mathrm{scale}\) | ≤ 0.10 |
| Production OT iters | ≤ `second_gen_cp_max_iter` (5) once ASPC warm |
| O–H distances | < 1.8 Å |
| PE after seed | < −100 eV |

**perf_exempt** — no CP2K twin wall on pin host; literature envelope +
method-identity gates (same class as sibling SGCP dynamics slugs).

## Honest scope

- Closed-shell RKS Γ only; UKS OT + SGCP refused; k-points / smear deferred
- OT natural-orbital warm-start from ASPC \(P\) (not density DIIS mixing)
- Soft drift tol vs full BO for partial-SCF heat on short traj

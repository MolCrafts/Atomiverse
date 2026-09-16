# Reference — Second-generation Car–Parrinello MD (H₂O NVE, GPW)

## Method

**Second-generation Car–Parrinello MD (SGCP / 2nd-gen CP)** on closed-shell
Γ-point GPW:

- Outer loop: `cpu::md::MD` + NVE `VelocityVerlet` (nuclei)
- Force slot: `cpu::gpw::GPWEvaluator` with
  **`second_gen_cp = true`**, **`second_gen_cp_k = 3`**,
  **`second_gen_cp_max_iter = 5`**, `compute_forces = true`
- Electronics: Kolafa **ASPC** density predictor-corrector
  (`cpu::electronic::ASPCPredictor`) + **partial SCF corrector** (iteration budget
  ≪ full BOMD); analytic forces at the post-corrector density

**Not BOMD** (production steps cap SCF iters at the corrector budget).  
**Not classical CPMD** (no fictitious electronic mass / \(T_e\)).  
**Not XL-BOMD** (ASPC extrapolation, not Niklasson auxiliary Verlet).

## Literature

- T. D. Kühne, M. Krack, F. R. Mohamed, M. Parrinello, *Phys. Rev. Lett.*
  **98**, 066401 (2007) — second-generation CP
- J. Kolafa, *J. Comput. Chem.* **25**, 335 (2004) — ASPC
- CP2K exercise `exercises:common:sgcp` — `EXTRAPOLATION ASPC` + limited SCF
- Twin geometry family: `gpw-bomd-water`, `xl-bomd-water`, `cpmd-water`

## Setup

| Item | Value |
| ---- | ----- |
| System | Isolated H₂O (CP2K getting-started geometry) |
| Cell | 10 Å cubic, Martyna–Tuckerman (`nonperiodic`) |
| XC / basis / PP | PBE / DZVP-MOLOPT-GTH / GTH-PBE |
| CUTOFF | 200 Ry |
| dt | 0.5 fs |
| ASPC \(K\) | 1 |
| Corrector SCF | `max_iter = 5` |
| Seed | 2 full-SCF steps (warm ASPC history, \(K+1\)) |
| Production | 1 baseline + 2 drift steps |

## Offline oracle (CP2K)

**Status:** no `cp2k` binary / Lmod module on the worker pin host
(Arrhenius login-class, 2026-07-25) — same inventory as `cpmd-water` /
`xl-bomd-water`. Offline sketch:

`tools/cp2k_oracle/inputs/second_gen_cpmd_water_nve.inp`

(`EXTRAPOLATION ASPC`, `EXTRAPOLATION_ORDER 3`, tight `MAX_SCF` on MD).
When a binary is available, re-run via `tools/cp2k_oracle/` and record
version + `.ener` drift + fair wall.

## Pin

| Observable | Gate |
| ---------- | ---- |
| Relative \(\|E-E_0\|/\mathrm{scale}\) | ≤ 0.10 |
| Production SCF iters | ≤ `second_gen_cp_max_iter` (5) once ASPC warm |
| O–H distances | < 1.8 Å |
| PE after seed | < −100 eV |

**perf_exempt** — no CP2K twin wall on pin host; literature envelope +
method-identity gates.

## Honest scope

- Closed-shell RKS Γ only; UKS / OT / k-points / Langevin refused or deferred
- Diagonalization corrector (not OT single-step)
- Soft drift tol vs full BO for partial-SCF heat on short traj

# dynamics/second-gen-cpmd

One case; every section below is run by `atv_regressions --case dynamics/second-gen-cpmd` and keeps its own oracle record.


## Reference — Second-generation Car–Parrinello MD (H₂O NVE, GPW)

### Method

**Second-generation Car–Parrinello MD (SGCP / 2nd-gen CP)** on closed-shell
Γ-point GPW:

- Outer loop: `cpu::md::MD` + NVE `VelocityVerlet` (nuclei)
- Force slot: `cpu::gpw::GPWEvaluator` with
  **`second_gen_cp = true`**, **`second_gen_cp_k = 1`**,
  **`second_gen_cp_max_iter = 5`**, `compute_forces = true`
- Electronics: Kolafa **ASPC** density predictor-corrector
  (`cpu::electronic::ASPCPredictor`) + **partial SCF corrector** (iteration budget
  ≪ full BOMD); analytic forces at the post-corrector density

**Not BOMD** (production steps cap SCF iters at the corrector budget).  
**Not classical CPMD** (no fictitious electronic mass / \(T_e\)).  
**Not XL-BOMD** (ASPC extrapolation, not Niklasson auxiliary Verlet).

### Literature

- T. D. Kühne, M. Krack, F. R. Mohamed, M. Parrinello, *Phys. Rev. Lett.*
  **98**, 066401 (2007) — second-generation CP
- J. Kolafa, *J. Comput. Chem.* **25**, 335 (2004) — ASPC
- CP2K exercise `exercises:common:sgcp` — `EXTRAPOLATION ASPC` + limited SCF
- Twin geometry family: `gpw-bomd-water`, `xl-bomd-water`, `cpmd-water`

### Setup

| Item | Value |
| ---- | ----- |
| System | Isolated H₂O (CP2K getting-started geometry) |
| Cell | 10 Å cubic, Martyna–Tuckerman (`nonperiodic`) |
| XC / basis / PP | PBE / DZVP-MOLOPT-GTH / GTH-PBE |
| CUTOFF | 200 Ry |
| dt | 0.5 fs |
| ASPC \(K\) | 1 |
| Corrector SCF | `max_iter = 5` |
| Seed | 3 full-SCF steps (warm ASPC history, \(K+2\)) |
| Production | 1 baseline + 2 drift steps |

### Offline oracle (CP2K)

**Status:** no `cp2k` binary / Lmod module on the worker pin host
(Arrhenius login-class, 2026-07-25) — same inventory as `cpmd-water` /
`xl-bomd-water`. Offline sketch:

`tools/cp2k_oracle/inputs/second_gen_cpmd_water_nve.inp`

(`EXTRAPOLATION ASPC`, `EXTRAPOLATION_ORDER 3`, tight `MAX_SCF` on MD).
When a binary is available, re-run via `tools/cp2k_oracle/` and record
version + `.ener` drift + fair wall.

### Pin

| Observable | Gate |
| ---------- | ---- |
| Relative \(\|E-E_0\|/\mathrm{scale}\) | ≤ 0.10 |
| Production SCF iters | ≤ `second_gen_cp_max_iter` (5) once ASPC warm |
| O–H distances | < 1.8 Å |
| PE after seed | < −100 eV |

**perf_exempt** — no CP2K twin wall on pin host; literature envelope +
method-identity gates.

### Honest scope

- Closed-shell RKS Γ only; UKS / OT / k-points / Langevin refused or deferred
- Diagonalization corrector (not OT single-step)
- Soft drift tol vs full BO for partial-SCF heat on short traj


## Reference — Second-generation Car–Parrinello MD (O₂ triplet NVE, Γ-UKS GPW)

### Method

**Second-generation Car–Parrinello MD (SGCP / 2nd-gen CP)** on **open-shell
Γ-UKS** Gaussian-and-plane-waves (GPW / Quickstep) forces:

- Outer loop: `cpu::md::MD` + `VelocityVerlet` (NVE; no thermostat)
- Force/energy: `cpu::gpw::GPWEvaluator` with `compute_forces = true`,
  **`n_alpha = 7`, `n_beta = 5`**, **`second_gen_cp = true`**,
  **`second_gen_cp_k = 1`**, **`second_gen_cp_max_iter = 5`**
- Independent spin densities \(P^\alpha, P^\beta\) use dual Kolafa ASPC
  predictors; after ASPC is warm, each MD force step runs a **partial**
  SCF with iteration budget `second_gen_cp_max_iter` (not full BOMD)
- Analytic forces at the post-corrector density
  (`internal::gpw_forces_from_scf`)
- Fixed cubic cell; MT / nonperiodic
- Units on the public surface: energy eV, length Å, time fs, force eV/Å

**Not BOMD** (production SCF capped at the corrector budget).  
**Not classical CPMD** (no fictitious electronic mass / \(T_e\)).  
**Not XL-BOMD** (ASPC extrapolation, not Niklasson auxiliary Verlet).

**Literature:**

- T. D. Kühne, M. Krack, F. R. Mohamed, M. Parrinello, *Phys. Rev. Lett.*
  **98**, 066401 (2007) — second-generation CP
- J. Kolafa, *J. Comput. Chem.* **25**, 335 (2004) — ASPC
- Closed-shell product path: `regressions/dynamics/second-gen-cpmd`
- UKS O₂ energy / forces offline CP2K pins:
  `regressions/gpw/uks`, `gpw-uks-forces`
- Related dual-spin AIMD: `regressions/dynamics/xl-bomd`

This is **not** a golden snapshot of Atomiverse total energy. The e2e pin is
the **envelope** on relative NVE drift with dual-spin ASPC + partial SCF
engaged — a broken spin guess (α-only, swapped channels) blows production
iters, ⟨S²⟩, or the drift envelope immediately.

### System

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

### Offline oracle (CP2K)

**Status:** no `cp2k` binary / CP2K Lmod module on the worker host used for
this pin (Arrhenius login-class, **2026-07-26**) — same inventory as
`second-gen-cpmd-water` / `xl-bomd-uks`. A CP2K twin sketch (short NVE
Quickstep UKS MD; `EXTRAPOLATION ASPC` + limited `MAX_SCF`) lives in
`tools/cp2k_oracle/inputs/second-gen-cpmd-uks/`. When a binary is available,
re-run offline and replace this section with version + measured drift + fair wall.

Until then the regression asserts the **literature SGCP/NVE conservation
envelope** plus method-identity gates (production SCF iters ≤ corrector
budget, triplet ⟨S²⟩, finite/bounded trajectory).

### Observable and tolerance

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

### Perf gate

| Field | Value |
| ----- | ----- |
| `perf_exempt` | **true** |
| Reason | No offline CP2K binary on host for a fair wall twin of multi-step UKS SGCP AIMD (2026-07-26 inventory) |
| `ratio_atv_over_cp2k` | n/a |
| `within_10x` | n/a (exempt) |

### Dual gate summary

| Gate | Result |
| ---- | ------ |
| Correctness | Regression `reg_second-gen-cpmd-uks` asserts A2/A4; unit A1/A3 in `test_cpu_gpw_second_gen_cpmd` |
| Speed | `perf_exempt=true` (no CP2K binary for AIMD twin wall) |


## Reference — Second-generation CP with OT corrector (H₂O NVE, GPW)

### Method

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

### Literature

- T. D. Kühne, M. Krack, F. R. Mohamed, M. Parrinello, *Phys. Rev. Lett.*
  **98**, 066401 (2007) — second-generation CP
- J. Kolafa, *J. Comput. Chem.* **25**, 335 (2004) — ASPC
- J. VandeVondele, J. Hutter, *J. Chem. Phys.* **118**, 4365 (2003) — OT
- Twin geometry family: `gpw-bomd-water`, `xl-bomd-water`, `cpmd-water`,
  `second-gen-cpmd-water`, `ot-minimizer-water`

### Setup

| Item | Value |
| ---- | ----- |
| System | Isolated H₂O (CP2K getting-started geometry) |
| Cell | 10 Å cubic, Martyna–Tuckerman (`nonperiodic`) |
| XC / basis / PP | PBE / DZVP-MOLOPT-GTH / GTH-PBE |
| CUTOFF | 200 Ry |
| dt | 0.5 fs |
| ASPC \(K\) | 1 |
| Corrector | OT, `max_iter = 5` |
| Seed | 3 full-OT steps (warm ASPC history, \(K+2\)) |
| Production | 1 baseline + 2 drift steps |

### Offline oracle (CP2K)

**Status:** no `cp2k` binary / Lmod module required on the worker pin host —
same inventory class as `second-gen-cpmd-water` / `cpmd-water`. Offline sketch:

`tools/cp2k_oracle/inputs/second-gen-cpmd-ot/h2o_sgcp_ot_nve.inp`

(`EXTRAPOLATION ASPC`, `EXTRAPOLATION_ORDER 1`, `&OT`, tight `MAX_SCF` on MD).
When a binary is available, re-run via `tools/cp2k_oracle/` and record
version + `.ener` drift + fair wall.

### Pin

| Observable | Gate |
| ---------- | ---- |
| Relative \(\|E-E_0\|/\mathrm{scale}\) | ≤ 0.10 |
| Production OT iters | ≤ `second_gen_cp_max_iter` (5) once ASPC warm |
| O–H distances | < 1.8 Å |
| PE after seed | < −100 eV |

**perf_exempt** — no CP2K twin wall on pin host; literature envelope +
method-identity gates (same class as sibling SGCP dynamics slugs).

### Honest scope

- Closed-shell RKS Γ only; UKS OT + SGCP refused; k-points / smear deferred
- OT natural-orbital warm-start from ASPC \(P\) (not density DIIS mixing)
- Soft drift tol vs full BO for partial-SCF heat on short traj


## Reference — Second-generation CP with OT corrector (O₂ triplet NVE, Γ-UKS GPW)

### Method

**Second-generation Car–Parrinello MD (SGCP / 2nd-gen CP)** on **open-shell
Γ-UKS** Gaussian-and-plane-waves (GPW / Quickstep) forces with the **Orbital
Transformation (OT)** SCF as the partial-SCF corrector:

- Outer loop: `cpu::md::MD` + `VelocityVerlet` (NVE; no thermostat)
- Force/energy: `cpu::gpw::GPWEvaluator` with `compute_forces = true`,
  **`n_alpha = 7`, `n_beta = 5`**, **`second_gen_cp = true`**,
  **`second_gen_cp_k = 1`**, **`second_gen_cp_max_iter = 5`**,
  **`use_ot = true`**
- Independent spin densities \(P^\alpha, P^\beta\) use dual Kolafa ASPC
  predictors; each warm MD force step purifies the predicted density to
  **spin-resolved natural orbitals** that seed the OT manifolds, then runs a
  **partial OT** with iteration budget `second_gen_cp_max_iter` (not full BOMD)
- Analytic forces at the post-corrector density
  (`internal::gpw_forces_from_scf`)
- Fixed cubic cell; MT / nonperiodic
- Units on the public surface: energy eV, length Å, time fs, force eV/Å

**Not BOMD** (production OT capped at the corrector budget).  
**Not classical CPMD** (no fictitious electronic mass / \(T_e\)).  
**Not XL-BOMD** (ASPC extrapolation, not Niklasson auxiliary Verlet).  
**Not DIIS SGCP-UKS** — that is `second-gen-cpmd-uks` (`use_ot=false`).  
**Not RKS SGCP-OT** — that is `second-gen-cpmd-ot` (closed-shell).

**Literature:**

- T. D. Kühne, M. Krack, F. R. Mohamed, M. Parrinello, *Phys. Rev. Lett.*
  **98**, 066401 (2007) — second-generation CP
- J. Kolafa, *J. Comput. Chem.* **25**, 335 (2004) — ASPC
- J. VandeVondele, J. Hutter, *J. Chem. Phys.* **118**, 4365 (2003) — OT
- Closed-shell OT product: `regressions/dynamics/second-gen-cpmd`
- UKS DIIS SGCP product: `regressions/dynamics/second-gen-cpmd`
- UKS O₂ energy / forces offline CP2K pins:
  `regressions/gpw/uks`, `gpw-uks-forces`

This is **not** a golden snapshot of Atomiverse total energy. The e2e pin is
the **envelope** on relative NVE drift with dual-spin ASPC + partial OT
engaged — a broken spin guess (α-only, swapped channels, ignored density)
blows production iters, ⟨S²⟩, or the drift envelope immediately.

### System

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
| SCF | `tol = 1e-5`, seed `max_iter = 80`, **OT** |
| SGCP | ASPC \(K = 1\); corrector budget `max_iter = 5` once warm |
| Ensemble | NVE |
| `dt` | 0.5 fs |
| Initial velocity | antisymmetric stretch \(v_z = \pm 0.0008\) Å/fs |
| Steps | 2 seed (warm ASPC) + 1 baseline production + 2 drift steps |

### Offline oracle (CP2K)

**Status:** no `cp2k` binary / CP2K Lmod module on the worker host used for
this pin (Arrhenius login-class, **2026-07-26**) — same inventory as
`second-gen-cpmd-uks` / `second-gen-cpmd-ot` / `xl-bomd-uks`. A CP2K twin
sketch (short NVE Quickstep UKS MD; `EXTRAPOLATION ASPC` + `&OT` + limited
`MAX_SCF`) lives in
`tools/cp2k_oracle/inputs/second-gen-cpmd-uks-ot/`. When a binary is available,
re-run offline and replace this section with version + measured drift + fair wall.

Until then the regression asserts the **literature SGCP/NVE conservation
envelope** plus method-identity gates (production OT iters ≤ corrector
budget, triplet ⟨S²⟩, finite/bounded trajectory).

### Observable and tolerance

Let \(E(t) = E_\mathrm{pot}(t) + E_\mathrm{kin}(t)\) [eV], with \(E_0\) after
the first **warm** production step (ASPC history filled).

\[
\delta = \max_t \frac{|E(t) - E_0|}{\max(1, |E_0|)} \le 0.10
\]

(Soft relative drift vs full BO 0.05 — partial-OT heat on short traj, same
class as `second-gen-cpmd-ot` / `second-gen-cpmd-uks`.)

Additional gates:

- Production OT **iters ≤ `second_gen_cp_max_iter` (5)** once ASPC is warm
- PE finite and bound (\(< -100\) eV after seed)
- KE finite and \(\ge 0\), non-trivial after seed
- ⟨S²⟩ in \([1.5, 2.5]\) (triplet window) after seed and production
- All coordinates finite
- Soft O–O window: \(0.9 < r_{\mathrm{OO}} < 1.6\) Å after production

### Perf gate

| Field | Value |
| ----- | ----- |
| `perf_exempt` | **true** |
| Reason | No offline CP2K binary on host for a fair wall twin of multi-step UKS SGCP-OT AIMD (2026-07-26 inventory) |
| `ratio_atv_over_cp2k` | n/a |
| `within_10x` | n/a (exempt) |

### Dual gate summary

| Gate | Result |
| ---- | ------ |
| Correctness | Regression `reg_second-gen-cpmd-uks-ot`; unit A1/A2 in `test_gpw_second_gen_cpmd` + `test_gpw_xl_bomd` |
| Speed | `perf_exempt=true` (no CP2K binary for AIMD twin wall) |


## Reference — multi-k second-generation CP (H₂ NVE, GPW)

### Method

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

### Literature

- T. D. Kühne, M. Krack, F. R. Mohamed, M. Parrinello, *Phys. Rev. Lett.*
  **98**, 066401 (2007) — second-generation CP
- J. Kolafa, *J. Comput. Chem.* **25**, 335 (2004) — ASPC
- CP2K `EXTRAPOLATION ASPC` + `&KPOINTS`
- Γ product paths: `regressions/dynamics/second-gen-cpmd` (+ UKS/OT)
- Multi-k density-guess socket twin: `regressions/dynamics/xl-bomd`
- Multi-k analytic-force cases: `regressions/gpw/mbd`, `gpw-vdw-kpoint-forces`

### System

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
| ASPC \(K\) | 1 (needs \(K+2=3\) seed densities) |
| Corrector SCF | `max_iter = 5` after ASPC warm |
| Ensemble | NVE |
| `dt` | 0.5 fs |
| Initial velocity | antisymmetric stretch \(v_x = \pm 0.001\) Å/fs |
| Steps | 2 seed + 1 baseline + 2 production |

### Offline oracle (CP2K)

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

### Observable and tolerance

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

### Perf gate

| Field | Value |
| ----- | ----- |
| `perf_exempt` | **true** |
| Reason | No offline CP2K binary on host for a fair wall twin of multi-step multi-k SGCP (2026-07-26 inventory) |
| `ratio_atv_over_cp2k` | n/a |

### Honest scope

- Closed-shell multi-k RKS product pin; multi-k UKS dual ASPC is wired in
  `GPWEvaluator` (build guard + force path) but not e2e-pinned here
- Diagonalization corrector only (multi-k + OT refused)
- Soft drift tol vs full BO for partial-SCF heat on short traj


## Reference — multi-k second-generation CP with OT corrector (H₂ NVE, GPW)

### Method

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

### Literature

- T. D. Kühne, M. Krack, F. R. Mohamed, M. Parrinello, *Phys. Rev. Lett.*
  **98**, 066401 (2007) — second-generation CP
- J. Kolafa, *J. Comput. Chem.* **25**, 335 (2004) — ASPC
- J. VandeVondele, J. Hutter, *J. Chem. Phys.* **118**, 4365 (2003) — OT
- Multi-k diagonalization-corrector twin:
  `regressions/dynamics/second-gen-cpmd`
- Γ OT-corrector twin: `regressions/dynamics/second-gen-cpmd`

### System

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
| ASPC \(K\) | 1 (needs \(K+2=3\) seed densities) |
| Corrector | OT, `max_iter = 5` after ASPC warm |
| Ensemble | NVE |
| `dt` | 0.5 fs |
| Initial velocity | antisymmetric stretch \(v_x = \pm 0.001\) Å/fs |
| Steps | 2 seed + 1 baseline + 2 production |

### Offline oracle (CP2K)

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
   fixed point: \(|E_\mathrm{OT} − E_\mathrm{diag}| \le 10^{-4}\) Ha (same
   functional, different minimizer).
2. **NVE drift envelope** with the OT corrector on the k path (below).

### Observable and tolerance

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

### Perf gate

| Field | Value |
| ----- | ----- |
| `perf_exempt` | **true** |
| Reason | CP2K's OT is Γ-only (`&OT` + `&KPOINTS` refused by CP2K), so no oracle wall twin of multi-k SGCP-OT can exist; family policy per `second-gen-cpmd-kpoints` / `second-gen-cpmd-ot` |
| `ratio_atv_over_cp2k` | n/a |

### Honest scope

- Closed-shell multi-k RKS product pin; multi-k **UKS** SGCP + OT (dual-spin
  k-OT warm start) has no standing regression
- No metallic smearing / +U under the SGCP corrector here
- Soft drift tol vs full BO for partial-SCF heat on short traj

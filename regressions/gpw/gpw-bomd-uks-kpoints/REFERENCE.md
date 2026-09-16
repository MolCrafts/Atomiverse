# Reference — plain multi-k UKS BOMD short trajectory (periodic H₂ NVE, GPW)

## Method

**Plain Born–Oppenheimer MD on multi-k collinear UKS** GPW (Quickstep): full
SCF to `tol` at every step on a Monkhorst–Pack k-mesh, analytic k-summed
forces, and the previous step's converged dual-spin complex densities
\(P_k^\alpha, P_k^\beta\) as the SCF initial guess (this slug's product path —
the k-UKS analogue of the Γ `guess_P`/`guess_P_beta` warm start and of
CP2K's `EXTRAPOLATION USE_PREV_P` on the UKS + k-mesh path).
No auxiliary propagator: not XL-BOMD, not CPMD, not SGCP.

- Outer loop: `cpu::md::MD` + `VelocityVerlet` (NVE)
- Force/energy: `cpu::gpw::GPWEvaluator` with `compute_forces = true`,
  `n_alpha = n_beta = 1` (closed-shell collinear UKS — both spin channels
  non-empty, molecule bound), `k_mesh = {1,1,2}`,
  `k_time_reversal = false` (two sampled k)
- VV bootstrap: one public single point (`cpu::single_point`) seeds forces at
  \(t=0\), matching CP2K's step-0 force evaluation
- Units on the public surface: energy eV, length Å, time fs, force eV/Å

**Literature:** Marx & Hutter, *Ab Initio Molecular Dynamics* (CUP), BOMD
chapter; VandeVondele et al., *Comput. Phys. Commun.* **167**, 103 (2005)
(Quickstep GPW); Monkhorst & Pack, PRB **13**, 5188 (1976).
Multi-k RKS plain BOMD: `regressions/gpw/gpw-bomd-kpoints`. Multi-k UKS
forces: `regressions/gpw/gpw-kpoint-uks-forces`. Multi-k UKS XL:
`regressions/dynamics/xl-bomd-uks-kpoints`.

## System (twin of the CP2K input)

| Quantity | Value |
| -------- | ----- |
| Species | H₂ (2 × H), periodic; bond \(R = 1.0\) Bohr along \(x\), cell-centred |
| Cell | cubic \(L = 12\) Bohr, periodic XYZ |
| Basis | DZV-GTH-PADE — CP2K data file `BASIS_SET` entry `H DZV-GTH-PADE` is numerically identical to ATV's in-tree `kBas_H_DZV_GTH_PADE` |
| PP | GTH-PADE-q1 (`GTH_POTENTIALS`: rloc 0.2, c = −4.18023680, 0.72507482 = ATV table) |
| XC | PADE (LDA), collinear UKS (`LSD` + `MULTIPLICITY 1` ⇔ \(N_\alpha=N_\beta=1\)) |
| CUTOFF | **300 Ry**, NGRIDS 1 (both engines grid-converged; same as RKS multi-k BOMD twin) |
| k mesh | MONKHORST-PACK 1×1×2 → κ_z = ±¼; ATV unreduced (`k_time_reversal=false`) ⇔ CP2K `FULL_GRID .TRUE.` (2 k, w = ½) |
| SCF | ATV `tol = 1e-7`, DIIS; CP2K `EPS_SCF 1e-7`, STANDARD diagonalization, `DIRECT_P_MIXING ALPHA 0.3`, `EXTRAPOLATION USE_PREV_P` |
| Ensemble / dt / steps | NVE, 0.5 fs, 4 steps |
| Initial velocity | antisymmetric stretch \(v_x = \mp 0.001\) Å/fs (CP2K: \(\mp 4.5710290\times10^{-5}\) Bohr/au_t) |

Closed-shell UKS fillings exercise dual \(P_k^\sigma\) without the unbound
high-spin H₂ surface (which cannot pin NVE).

## Offline oracle (CP2K)

| Field | Value |
| ----- | ----- |
| Engine | **CP2K 2026.1** (`cp2k.psmp`, source install `git:5e54ba2`) |
| Binary | `/nobackup/proj/disk/teoroo/personal/jicli594/work/cp2k/install/bin/cp2k.psmp` |
| Env | `source .../cp2k/install/env.sh`; `CP2K_DATA_DIR=.../cp2k/src/data` |
| Host | Arrhenius epyc login node (offline oracle; **never** run at regression time) |
| Input | `tools/cp2k_oracle/inputs/gpw-bomd-uks-kpoints/h2_uks_bomd_kpoints_nve.inp` |
| Launch | `mpirun -np 1 cp2k.psmp -i h2_uks_bomd_kpoints_nve.inp`, `OMP_NUM_THREADS=8` |
| Date | 2026-08-01 |

**Pinned `.ener` trace (CUTOFF 300, UKS):**

| Step | Kin [Ha] | Pot [Ha] | Cons [Ha] |
| ---- | -------- | -------- | --------- |
| 0 | 0.000003839 | **−1.074050184** | −1.074046345 |
| 1 | 0.020127024 | **−1.097682104** | −1.077555080 |
| 2 | 0.046817650 | **−1.125544577** | −1.078726927 |
| 3 | 0.052842084 | **−1.130531455** | −1.077689371 |
| 4 | 0.044574339 | **−1.121507812** | −1.076933473 |

Step-0 `ENERGY| Total FORCE_EVAL ( QS ) energy [hartree]`:
**−1.074050184366365**.
Step-0 `FORCES|`: \(F_x = \mp 0.345535381\) Ha/Bohr (y/z ≈ 1e-11).
SCF iterations: **13** cold (step 0), **8** per production step with
`USE_PREV_P`. On this closed-shell filling the UKS twin prints the **same**
`.ener` Pot rows as the RKS multi-k BOMD twin (`gpw-bomd-kpoints`) to all
printed digits — expected for identical valence occupancy; the dual gate
still exercises the UKS product path (dual \(P_k^\sigma\) payload + warm start).

CP2K NVE conservation over the window:
max \(|\mathrm{Cons}(t) - \mathrm{Cons}(0)| = 4.68\times10^{-3}\) Ha
(relative \(4.4\times10^{-3}\)).

## Same-method band

Identical residual story to `gpw-bomd-kpoints` (Γ-point periodic-GPW model
band on this compressed H₂): at 300 Ry,
\(\Delta E \approx 4.07\times10^{-4}\) Ha,
\(\Delta F_x \approx 5.86\times10^{-3}\) Ha/Bohr. Multi-k UKS vs multi-k
RKS on ATV is degenerate for this closed-shell filling (measured step-0
energy/force match the RKS regression digits).

## Observable, pins, and tolerances

`reg_gpw-bomd-uks-kpoints.cpp` asserts (measured 2026-08-01, OMP=8):

| Gate | Pin / band | Measured |
| ---- | ---------- | -------- |
| Step-0 energy | −1.074050184366365 Ha ± **1e-3** | Δ = 4.07e-4 |
| Step-0 force \(F_x\)(atom 0) | −0.345535381 Ha/Bohr ± **8e-3** | Δ = 5.86e-3 |
| Per-step E_pot (steps 1–4) | `.ener` Pot rows ± **2e-3** Ha | worst Δ = 2.33e-4 |
| NVE drift envelope | \(\max_t \lvert E(t)-E_0\rvert / \max(1,\lvert E_0\rvert) \le 0.05\) | 4.25e-3 |
| Dual-spin multi-k payload | `n_kpts == 2`, full `P_k` **and** `P_k_beta` every step | ✓ |
| Warm start | production `iters ≤ cold + 3` (strict same-geometry win unit-pinned in `test_cpu_gpw_bomd` `PlainMultiKUksRelaunchWarmStartsBothSpins`) | cold 7, worst warm 8 |
| Geometry | finite coords; soft H–H window 0.4–2.5 Å | r = 0.907 Å |
| ⟨S²⟩ | finite ≥ 0 after step-0 and production | 1.0 → 1.0 |

The per-step Pot trace is the decisive MD gate: a broken dual-spin warm start
(partial spin guess → solver throw, shape leak, stale-density corruption), a
wrong k-force, or a VV binding error moves the trace by orders of magnitude
more than the 2e-3 Ha band.

## Perf gate (fair wall, same host / OMP=8 / single rank)

| Metric | Value |
| ------ | ----- |
| \(T_{\mathrm{cp2k}}\) (4-step MD twin, CUTOFF 300; `mpirun -np 1`) | **2.12 s** (best of two; second run 2.24 s) |
| \(T_{\mathrm{atv}}\) (`reg_gpw-bomd-uks-kpoints`: 1 single point + 4 MD steps) | **2.50 s** (second fair run 2.83 s) |
| ratio \(T_{\mathrm{atv}}/T_{\mathrm{cp2k}}\) | **1.18** (≤ **1.33** on second pair) |
| within_10x | **true** |
| perf_exempt | false |

## Provenance

- Spec: `.claude/specs/gpw-bomd-uks-kpoints/` (deleted on close; git history)
- Twin input: `tools/cp2k_oracle/inputs/gpw-bomd-uks-kpoints/h2_uks_bomd_kpoints_nve.inp`
- Offline log: `/tmp/atv-gpw-bomd-uks-kpoints/h2_uks_bomd_kpoints_nve.out` + `.ener`
- Queue: `docs/developer/cp2k-method-queue.md` (`gpw-bomd-uks-kpoints`, wave 1 2026-08-01)
- Unit gate: `tests/cpu/gpw/test_gpw_bomd.cpp` (`PlainMultiKUksRelaunchWarmStartsBothSpins`)

# Reference — plain multi-k BOMD short trajectory (periodic H₂ NVE, GPW)

## Method

**Plain Born–Oppenheimer MD on multi-k closed-shell RKS** GPW (Quickstep):
full SCF to `tol` at every step on a Monkhorst–Pack k-mesh, analytic k-summed
forces, and the previous step's converged complex densities \(P_k\) as the
SCF initial guess (this slug's implementation delta — the k-path analogue of
the Γ `guess_C`/`guess_P` warm start; CP2K's `EXTRAPOLATION USE_PREV_P`).
No auxiliary propagator: not XL-BOMD, not CPMD, not SGCP.

- Outer loop: `cpu::md::MD` + `VelocityVerlet` (NVE)
- Force/energy: `cpu::gpw::GPWEvaluator` with `compute_forces = true`,
  `k_mesh = {1,1,2}`, `k_time_reversal = false` (two sampled k)
- VV bootstrap: one public single point (`cpu::single_point`) seeds forces at
  \(t=0\), matching CP2K's step-0 force evaluation, so step \(n\) here is
  step \(n\) in the CP2K `.ener` trace
- Units on the public surface: energy eV, length Å, time fs, force eV/Å

**Literature:** Marx & Hutter, *Ab Initio Molecular Dynamics* (CUP), BOMD
chapter; VandeVondele et al., *Comput. Phys. Commun.* **167**, 103 (2005)
(Quickstep GPW); Monkhorst & Pack, PRB **13**, 5188 (1976).
Γ product path: `regressions/gpw/gpw-bomd-water`. Multi-k force machinery:
`regressions/gpw/gpw-kpoint-forces`. AIMD family twins:
`regressions/dynamics/xl-bomd-kpoints`, `cpmd-kpoints`,
`second-gen-cpmd-kpoints`.

## System (twin of the CP2K input)

| Quantity | Value |
| -------- | ----- |
| Species | H₂ (2 × H), periodic; bond \(R = 1.0\) Bohr along \(x\), cell-centred |
| Cell | cubic \(L = 12\) Bohr, periodic XYZ |
| Basis | DZV-GTH-PADE — CP2K data file `BASIS_SET` entry `H DZV-GTH-PADE` is numerically identical to ATV's in-tree `kBas_H_DZV_GTH_PADE` |
| PP | GTH-PADE-q1 (`GTH_POTENTIALS`: rloc 0.2, c = −4.18023680, 0.72507482 = ATV table) |
| XC | PADE (LDA) |
| CUTOFF | **300 Ry**, NGRIDS 1 (both engines grid-converged; series below) |
| k mesh | MONKHORST-PACK 1×1×2 → κ_z = ±¼; ATV unreduced (`k_time_reversal=false`) ⇔ CP2K `FULL_GRID .TRUE.` (2 k, w = ½) |
| SCF | ATV `tol = 1e-7`, DIIS; CP2K `EPS_SCF 1e-7`, STANDARD diagonalization, `DIRECT_P_MIXING ALPHA 0.3`, `EXTRAPOLATION USE_PREV_P` |
| Ensemble / dt / steps | NVE, 0.5 fs, 4 steps |
| Initial velocity | antisymmetric stretch \(v_x = \mp 0.001\) Å/fs (CP2K: \(\mp 4.5710290\times10^{-5}\) Bohr/au_t) |

## Offline oracle (CP2K)

| Field | Value |
| ----- | ----- |
| Engine | **CP2K 2026.1** (`cp2k.psmp`, EasyBuild `2026.1-gmpflf-2026.03`) |
| Host | Arrhenius epyc login node (offline oracle; **never** run at regression time) |
| Input | `tools/cp2k_oracle/inputs/gpw-bomd-kpoints/h2_bomd_kpoints_nve.inp` |
| Launch | `mpirun -n 1 cp2k.psmp -i h2_bomd_kpoints_nve.inp`, `OMP_NUM_THREADS=8` |
| Date | 2026-07-31 |

**Pinned `.ener` trace (CUTOFF 300):**

| Step | Kin [Ha] | Pot [Ha] | Cons [Ha] |
| ---- | -------- | -------- | --------- |
| 0 | 0.000003839 | **−1.074050184** | −1.074046345 |
| 1 | 0.020127024 | **−1.097682104** | −1.077555080 |
| 2 | 0.046817650 | **−1.125544577** | −1.078726927 |
| 3 | 0.052842084 | **−1.130531455** | −1.077689371 |
| 4 | 0.044574339 | **−1.121507812** | −1.076933473 |

Step-0 `FORCES|`: \(F_x = \mp 0.345535381\) Ha/Bohr (y/z ≈ 1e-11).
SCF iterations: **14** cold (step 0), **8–9** per production step with
`USE_PREV_P`. CP2K NVE conservation over the window:
max \(|\mathrm{Cons}(t) - \mathrm{Cons}(0)| = 4.68\times10^{-3}\) Ha
(relative \(4.4\times10^{-3}\) — dt = 0.5 fs is aggressive for this
compressed bond; ATV shows the same envelope, below).

## Grid convergence and the same-method band

Single-point `ENERGY_FORCE` series at the initial geometry (both engines,
same mesh):

| CUTOFF [Ry] | CP2K E [Ha] | CP2K F_x | ATV E [Ha] | ATV F_x |
| ----------- | ----------- | -------- | ---------- | ------- |
| 100 | −1.053153626 | −0.226587 | −1.057153829 | −0.235419 |
| 200 | −1.074029855 | −0.345533 | −1.074373792 | −0.340398 |
| 300 | −1.074050184 | −0.345535 | −1.074456793 | −0.339676 |
| 400 | −1.074050699 | −0.345535 | −1.074457666 | −0.339683 |
| 600 | −1.074050706 | −0.345535 | −1.074457654 | −0.339683 |
| 800 | — | — | −1.074457655 | −0.339683 |

Both engines converge cleanly by 300 Ry; the **converged limits differ** by
\(\Delta E = 4.07\times10^{-4}\) Ha and
\(\Delta F = 5.85\times10^{-3}\) Ha/Bohr (1.7 %).

**Γ control (600 Ry):** CP2K Γ −1.074056227 Ha / −0.345543392 Ha/Bohr vs
ATV Γ −1.074463173 Ha / −0.339690838 Ha/Bohr — *identical* deltas
(4.07e−4 / 5.85e−3). The residual is therefore the **pre-existing Γ-point
periodic-GPW model band** on this compressed periodic H₂ (lattice-sum /
periodic-Poisson / PP-tail treatment class — CP2K is insensitive to
`EPS_DEFAULT` 1e-10 → 1e-14), **not** a k-point defect: the multi-k
machinery adds < 1e-5 on top of the Γ story. ATV's k-force is separately
pinned analytic ≡ FD to 6e-6 Ha/Bohr (`gpw-kpoint-forces`), so each engine's
force is the true gradient of its own converged surface. Follow-up
(periodic Γ force parity depth vs CP2K) is a queue matter, named in the
worker summary.

## Observable, pins, and tolerances

`reg_gpw-bomd-kpoints.cpp` asserts (measured values 2026-07-31, OMP=8):

| Gate | Pin / band | Measured |
| ---- | ---------- | -------- |
| Step-0 energy | −1.074050184366365 Ha ± **1e-3** | Δ = 4.07e-4 |
| Step-0 force \(F_x\)(atom 0) | −0.345535381 Ha/Bohr ± **8e-3** | Δ = 5.86e-3 |
| Per-step E_pot (steps 1–4) | `.ener` Pot rows ± **2e-3** Ha | worst Δ = 2.33e-4 |
| NVE drift envelope | \(\max_t \lvert E(t)-E_0\rvert / \max(1,\lvert E_0\rvert) \le 0.05\) | 4.25e-3 (CP2K: 4.4e-3) |
| Multi-k payload | `n_kpts == 2`, full `P_k`, converged every step | ✓ |
| Warm start | production `iters ≤` cold `iters` (strict same-geometry win is unit-pinned in `test_cpu_gpw_bomd`) | 8 ≤ 8 |
| Geometry | finite coords; soft H–H window 0.4–2.5 Å | r = 0.907 Å |

The per-step trace is the decisive MD gate: a broken warm start (shape leak,
stale-density corruption), a wrong k-force, or a VV binding error moves the
trace by orders of magnitude more than the 2e-3 Ha band.

## Perf gate (fair wall, same host / OMP=8 / single rank)

| Metric | Value |
| ------ | ----- |
| \(T_{\mathrm{cp2k}}\) (4-step MD twin, CUTOFF 300) | **33.4 s** |
| \(T_{\mathrm{atv}}\) (`reg_gpw-bomd-kpoints`: 1 single point + 4 MD steps) | **2.14 s** |
| ratio \(T_{\mathrm{atv}}/T_{\mathrm{cp2k}}\) | **0.064** |
| within_10x | **true** |
| perf_exempt | false |

## Provenance

- Spec: `.claude/specs/gpw-bomd-kpoints/` (deleted on close; git history)
- Twin input: `tools/cp2k_oracle/inputs/gpw-bomd-kpoints/h2_bomd_kpoints_nve.inp`
- Queue: `docs/developer/cp2k-method-queue.md` (wave-13 row, staffed wave 1 2026-07-31)

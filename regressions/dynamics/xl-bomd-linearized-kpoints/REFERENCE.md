# Reference — zero-SCF linearized XL-BOMD on the multi-k path (periodic H₂ NVE, GPW)

## Method

**Zero-SCF linearized-functional XL-BOMD** (Niklasson shadow Born–Oppenheimer
potential) on **multi-k RKS and collinear multi-k UKS** Gaussian-and-plane-waves
(GPW / Quickstep) forces:

- Outer loop: `cpu::md::MD` + `VelocityVerlet` (NVE)
- Force/energy: `cpu::gpw::GPWEvaluator` with `compute_forces = true`,
  **`k_mesh = {1,1,2}`**, **`k_time_reversal = false`** (two sampled k),
  wrapped by `cpu::electronic::XlBomdForce{gpw, 5, XlBomdLinearized{}}`
- Cold seed (step 0): one full multi-k SCF → seed one complex `XlBomdCplx`
  history per sampled k (dual per spin for UKS) with \(P_k^{\sigma,\mathrm{SCF}}\)
- Warm steps: auxiliary complex densities \(n_k = P_k^{\mathrm{aux}}\); **one**
  KS build at fixed \(\{n_k\}\) (`max_iter = 0`); **one** generalized
  diagonalization per (k, σ) (`KOrth` + `diagonalize_k`) → \(D_k^\sigma = q[n]\);
  **no iterative SCF** (`last_result().iters == 0`)
- k-weighted shadow energy
  \[
  \mathcal{E} = \sum_k w_k \sum_\sigma \mathrm{Re}\,\mathrm{Tr}\!\left[
  H_k^\sigma[n]\,(D_k^\sigma - n_k^\sigma)\right] + E_{\mathrm{KS}}[n]
  \]
  (recovers \(E_{\mathrm{KS}}\) at \(D_k^\sigma = n_k^\sigma\); RKS is the
  single-spin occ-2 case)
- Forces: k-path assembly (`gpw_forces_from_scf`) at the
  \((D_k, \varepsilon_k, C_k^{\mathrm{occ}})\) payload — since the \(C_k\) are
  exact eigenvectors of \(H_k[n]\), the off-BO energy-weighted density
  \(W = C (C^\dagger H C) C^\dagger\) of the Γ product reduces to
  \(\sum_i f\,\varepsilon_i C C^\dagger\)
- XL `advance(`\(D_k^\sigma\)`)` per (k, σ) with Table-I \(K = 5\) dissipation
- Multi-k UKS solver support added for this slug: `run_gpw_scf_kpoints_uks`
  gained the `max_iter == 0` dual-spin fixed-density branch and
  `GpwResult::H_ks_k_beta`

**Literature:**

- A. M. N. Niklasson, *Phys. Rev. Lett.* **100**, 123004 (2008)
- A. M. N. Niklasson et al., *J. Chem. Phys.* **130**, 214109 (2009) — Table I
  (unit-pinned in `test_cpu_gpw_xl_bomd`)
- A. M. N. Niklasson & M. J. Cawkwell, *Phys. Rev. B* **86**, 174308 (2012);
  Niklasson, *J. Chem. Phys.* **147**, 054103 (2017) — shadow / linearized
  functional
- VandeVondele et al., *Comput. Phys. Commun.* **167**, 103 (2005) — GPW
- Γ product paths: `regressions/dynamics/xl-bomd-linearized-forces` (RKS),
  `xl-bomd-linearized-uks` (UKS); SCF-XL k paths: `xl-bomd-kpoints`,
  `xl-bomd-uks-kpoints`; BO anchor: `regressions/gpw/gpw-bomd-kpoints`

## System (twin of the CP2K input)

Identical twin of `xl-bomd-kpoints` / `gpw-bomd-kpoints` at the
`gpw-bomd-kpoints` pin settings (CUTOFF 300, tol 1e-7):

| Quantity | Value |
| -------- | ----- |
| Species | H₂ (2 × H), periodic; bond \(R = 1.0\) Bohr along \(x\), cell-centred |
| Cell | cubic \(L = 12\) Bohr ≈ 6.350 Å; periodic |
| Basis / PP | DZV-GTH-PADE / GTH-PADE-q1 (identical CP2K data-file rows) |
| XC | PADE (LDA) |
| CUTOFF | 300 Ry (both engines grid-converged; series in `gpw-bomd-kpoints`) |
| k mesh | MONKHORST-PACK 1×1×2, `k_time_reversal=false` ⇔ CP2K `FULL_GRID .TRUE.` (2 k, w = ½) |
| SCF (cold seed) | `tol = 1e-7`, `max_iter = 60`, `mix = 0.3`, DIIS diagonalization |
| XL-BOMD | \(K = 5\); linearized shadow forces on; one `XlBomdCplx` per (k, σ) |
| Ensemble / dt / steps | NVE, 0.5 fs, 1 cold seed + 4 production |
| Initial velocity | antisymmetric stretch \(v_x = \mp 0.001\) Å/fs |

Three legs in one binary:

- **A (anchor):** SCF-XL multi-k (`XlBomdForce{gpw, 5}`) — shipped product
- **B (product under test):** linearized multi-k RKS
- **C (UKS path):** linearized multi-k UKS, `n_alpha = n_beta = 1` —
  spin-symmetric, must reproduce leg B (exact spin-degeneracy identity;
  genuinely exercises the dual-spin \(H_k^\beta\) machinery)

## Offline oracle (CP2K)

CP2K 2026.1 has **no zero-SCF shadow-functional XL product** (its density
extrapolations are ASPC / PS / USE_PREV_P + iterative SCF), so the shadow
dynamics itself cannot be twinned. The **BO surface it must shadow can**:
the converged-SCF k-point NVE twin re-run for this slug.

| Field | Value |
| ----- | ----- |
| Engine | **CP2K 2026.1** (`cp2k.psmp`, EasyBuild `2026.1-gmpflf-2026.03`) |
| Host | Arrhenius epyc login node (offline oracle; **never** run at regression time) |
| Input | `tools/cp2k_oracle/inputs/xl-bomd-linearized-kpoints/h2_xl_linearized_kpoints_nve.inp` |
| Launch | `mpirun -n 1 cp2k.psmp -i h2_xl_linearized_kpoints_nve.inp`, `OMP_NUM_THREADS=8` |
| Date | 2026-07-31 |

**Pinned `.ener` trace (CUTOFF 300)** — byte-identical to the
`gpw-bomd-kpoints` wave-1 pin (same input re-run):

| Step | Kin [Ha] | Pot [Ha] | Cons [Ha] |
| ---- | -------- | -------- | --------- |
| 0 | 0.000003839 | **−1.074050184** | −1.074046345 |
| 1 | 0.020127024 | **−1.097682104** | −1.077555080 |
| 2 | 0.046817650 | **−1.125544577** | −1.078726927 |
| 3 | 0.052842084 | **−1.130531455** | −1.077689371 |
| 4 | 0.044574339 | **−1.121507812** | −1.076933473 |

CP2K NVE conservation over the window: relative 4.4e-3 (dt = 0.5 fs is
aggressive on this compressed bond; the ATV legs show the same class).
The documented engine model band on this system is 4.07e-4 Ha
(constant Γ-point periodic-GPW lattice-sum/PP-tail class, identical at Γ
and on the k path — see `regressions/gpw/gpw-bomd-kpoints/REFERENCE.md`).

## Observable, pins, and tolerances

The VV bootstrap matches CP2K: step 0 is a public `cpu::single_point` (full
SCF + forces at t = 0, which also cold-seeds every per-k XL history), so ATV
step \(n\) aligns with `.ener` row \(n\). The MD component lifecycle is
primed with a 0-step `run` **before** the seeding single point — component
`build()` fires on the first `run()` and `GPWEvaluator::build` deliberately
resets the XL histories, so the reverse order silently wipes the seed and
degrades warm step 1 to a full SCF (caught by the `iters == 0` gate during
development).

`reg_xl-bomd-linearized-kpoints.cpp` asserts (measured 2026-07-31, OMP=8):

| Gate | Pin / band | Measured |
| ---- | ---------- | -------- |
| B/C warm steps | `iters == 0`, `converged`, `n_kpts == 2`, full `P_k` | ✓ (0 / true / 2) |
| B per-step shadow \(E_{\mathrm{pot}}\) vs CP2K Pot rows | ± **2e-3 Ha** | worst **4.066e-4** (≈ the 4.07e-4 model band; production rows 0.6–2.1e-4) |
| A/B trace (linearized vs SCF-XL) | ± **1e-3 Ha** | worst **1.099e-4** |
| C/B spin-degeneracy identity | ± **1e-4 Ha** | worst **2.31e-8** |
| NVE drift envelope (B and C) | \(\le 0.05\) | B **4.056e-3**, C **4.056e-3** (A 4.253e-3; CP2K 4.4e-3) |
| Seed | full SCF (`iters > 0`, converged) | A/B 8 iters, C 7 |
| Geometry | finite coords; soft H–H window 0.4–1.5 Å | r = 0.907–0.908 Å (all legs) |

Measured per-step trace (Pot [Ha], 2026-07-31):

| Step | CP2K | A (SCF-XL) | B (linearized) | C (UKS lin.) |
| ---- | ---- | ---------- | -------------- | ------------ |
| 0 | −1.074050184 | −1.074456793 | −1.074456793 | −1.074456793 |
| 1 | −1.097682104 | −1.097482136 | −1.097521796 | −1.097521784 |
| 2 | −1.125544577 | −1.125333978 | −1.125443925 | −1.125443902 |
| 3 | −1.130531455 | −1.130595564 | −1.130587128 | −1.130587132 |
| 4 | −1.121507812 | −1.121740443 | −1.121722124 | −1.121722125 |

The A/B trace is the decisive shadow gate: a wrong k-weight, a missing
−Tr(H n) term, a spin-channel swap, or forces at the wrong density move the
linearized trace away from the SCF-XL anchor by orders more than the band.
The C/B identity pins the dual-spin path (`H_ks_k_beta`, per-spin
diagonalization, dual advance) at the 1e-8 Ha level with zero free
parameters.

## Perf gate (fair wall, same host / OMP=8 / single rank)

| Metric | Value |
| ------ | ----- |
| \(T_{\mathrm{cp2k}}\) (4-step converged-BOMD k twin, CUTOFF 300) | **1.49 s** (wall, `/usr/bin/time`; CP2K-internal 1.045 s — quiet node; the wave-1 `gpw-bomd-kpoints` measurement of the same input was 33.4 s under load) |
| \(T_{\mathrm{atv}}\) (leg B: 1 seed single point + 4 linearized steps) | **1.96 s** |
| ratio \(T_{\mathrm{atv}}/T_{\mathrm{cp2k}}\) | **1.3** |
| within_10x | **true** |
| perf_exempt | false |
| Regression wall (all three legs) | 7.7 s (≤ 60 s budget) |

## Dual gate summary

| Gate | Result |
| ---- | ------ |
| Correctness | unit `test_cpu_gpw_xl_bomd` (k-shadow identity, UKS-k fixed-density payload, acceptance + stress-guard pins) + e2e `reg_xl-bomd-linearized-kpoints` (CP2K-anchored) |
| Speed | ratio 1.3 ≤ 10 (measured, same host) |

## Provenance

- Spec: `.claude/specs/xl-bomd-linearized-kpoints/` (deleted on close; git)
- Twin input: `tools/cp2k_oracle/inputs/xl-bomd-linearized-kpoints/h2_xl_linearized_kpoints_nve.inp`
- Queue: `docs/developer/cp2k-method-queue.md` row `xl-bomd-linearized-kpoints`

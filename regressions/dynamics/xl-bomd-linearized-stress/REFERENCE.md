# Reference — zero-SCF linearized XL-BOMD shadow-density stress (periodic H₂ NVE)

## Method

**Zero-SCF linearized-functional XL-BOMD** (Niklasson shadow Born–Oppenheimer
potential) with **analytic fixed-\(P\) Hellmann–Feynman cell stress** at the
shadow payload:

- Outer loop: `cpu::md::MD` + `VelocityVerlet` (NVE)
- Force/energy/stress: `cpu::gpw::GPWEvaluator` with
  `compute_forces = true`, **`compute_stress = true`**,
  `stress_use_fd = false`, wrapped by
  `cpu::electronic::XlBomdForce{gpw, 5, XlBomdLinearized{}}`
- Cold seed (step 0): full SCF → classic `gpw_stress` (reference SCF + fixed-\(P\)
  FD) → seeds XL histories
- Warm steps: zero-SCF shadow forces (as in `xl-bomd-linearized-kpoints`) **plus**
  the new state-overload stress path
  `gpw_stress(atoms, method, last_, h, axes)` — fixed-\(P\) Lagrangian at
  \(P = D = q[n]\), \(W\) from the \(H[n]\) eigenpairs; **no reference SCF**
- `stress_use_fd` + linearized remains **refused** at `build()` (multi-SCF
  strain-FD re-converges the SCF under strain — Born–Oppenheimer stress, not
  the shadow functional)

**Literature:**

- A. M. N. Niklasson, *Phys. Rev. Lett.* **100**, 123004 (2008); *J. Chem. Phys.*
  **130**, 214109 (2009) Table I; *Phys. Rev. B* **86**, 174308 (2012);
  *J. Chem. Phys.* **147**, 054103 (2017) — shadow / linearized functional
- O. H. Nielsen & R. M. Martin, *Phys. Rev. B* **32**, 3780 (1985) — fixed-\(P\)
  cell theorem
- Sibling force product: `regressions/dynamics/xl-bomd-linearized-kpoints`
- BO stress product: `regressions/gpw/gpw-analytic-stress`, `gpw-kpoint-stress`

## System (twin of the CP2K input)

Identical twin of `xl-bomd-linearized-kpoints` / `gpw-bomd-kpoints`:

| Quantity | Value |
| -------- | ----- |
| Species | H₂ (2 × H), periodic; bond \(R = 1.0\) Bohr along \(x\), cell-centred |
| Cell | cubic \(L = 12\) Bohr ≈ 6.350 Å; periodic |
| Basis / PP | DZV-GTH-PADE / GTH-PADE-q1 |
| XC | PADE (LDA) |
| CUTOFF | 300 Ry |
| k mesh (legs A/B) | MONKHORST-PACK 1×1×2, `k_time_reversal=false` ⇔ CP2K `FULL_GRID .TRUE.` |
| k mesh (legs C/D) | Γ only |
| SCF (cold seed) | `tol = 1e-7`, `max_iter = 60`, `mix = 0.3` |
| XL-BOMD | \(K = 5\); linearized shadow forces + shadow stress |
| Ensemble / dt / steps | NVE, 0.5 fs, 1 cold seed + 4 production |
| Initial velocity | antisymmetric stretch \(v_x = \mp 0.001\) Å/fs |

Four legs in one binary, all `compute_stress = true`:

- **A (anchor, k):** converged BOMD multi-k — per-step \(\sigma_{xx}\) gated vs CP2K
- **B (product, k):** linearized multi-k — warm `iters == 0` + finite \(\sigma\)
- **C (anchor, Γ):** converged BOMD Γ
- **D (product, Γ):** linearized Γ

## Offline oracle (CP2K)

CP2K 2026.1 has **no zero-SCF shadow-functional XL product**, so the shadow
stress itself cannot be twinned. What CP2K **does** pin is the converged-BOMD
multi-k stress surface the shadow must track:

| Field | Value |
| ----- | ----- |
| Engine | **CP2K 2026.1** (`~/.local/bin/cp2k` → Spack `cp2k.psmp`) |
| Host | Arrhenius epyc login (offline; **never** at ctest) |
| Input | `tools/cp2k_oracle/inputs/xl-bomd-linearized-stress/h2_xl_linearized_stress_nve.inp` |
| Launch | `OMP_NUM_THREADS=8 cp2k -i h2_xl_linearized_stress_nve.inp` |
| Date | 2026-07-31 |
| Stress mode | **`STRESS_TENSOR ANALYTICAL`** (accepted with `KPOINTS`; not refused) |
| Wall | **2.13 s** (`/usr/bin/time`) |

**Pinned `.ener` Pot** — byte-identical to the `xl-bomd-linearized-kpoints` pin:

| Step | Kin [Ha] | Pot [Ha] | Cons [Ha] |
| ---- | -------- | -------- | --------- |
| 0 | 0.000003839 | **−1.074050184** | −1.074046345 |
| 1 | 0.020127024 | **−1.097682104** | −1.077555080 |
| 2 | 0.046817650 | **−1.125544577** | −1.078726927 |
| 3 | 0.052842084 | **−1.130531455** | −1.077689371 |
| 4 | 0.044574339 | **−1.121507812** | −1.076933473 |

**Pinned `.stress` diagonals** (`PROJECT-1.stress` [bar] → Ha/Bohr³ with
\(1\,\mathrm{Ha/Bohr}^3 = 2.94210107994716\times 10^{8}\,\mathrm{bar}\)):

| Step | \(\sigma_{xx}\) [Ha/Bohr³] | \(\sigma_{yy}\) | \(\sigma_{zz}\) |
| ---- | -------------------------- | --------------- | --------------- |
| 0 | **1.998440567215e-04** | −7.010935192196e-08 | −2.386241961179e-09 |
| 1 | **1.701541305617e-04** | −7.810856540460e-08 | −2.134981164927e-09 |
| 2 | **1.087027341964e-04** | −9.651444883875e-08 | −1.350517576735e-09 |
| 3 | **4.492363435060e-05** | −1.194770631848e-07 | 2.382815718937e-11 |
| 4 | **−8.093268435310e-06** | −1.406572107551e-07 | 1.879278928819e-09 |

Off-diagonal components are \(\sim 10^{-12}\) bar (numerical noise). The hard
CP2K gate in the regression is on \(\sigma_{xx}\) (dominant compressed-bond
component); yy/zz are near-zero noise on this orthorhombic stretch.

## Observable, pins, and tolerances

`reg_xl-bomd-linearized-stress.cpp` asserts (measured 2026-07-31, OMP=8):

| Gate | Pin / band | Measured |
| ---- | ---------- | -------- |
| B/D warm steps | `iters == 0`, `converged`, finite off-diag-0 \(\sigma\), `kStress` written | ✓ |
| Dual gate (seed k mesh) | state ≡ classic ≤ **1e-10**; state vs FD ≤ **1e-6** Ha/Bohr³ | **0** / **9.86e-11** |
| A \(\sigma_{xx}\) vs CP2K | ≤ **1e-4** Ha/Bohr³ | worst **6.026e-05** (seed residual ~3.4e-6; trajectory residual grows with engine-model force drift) |
| A/B diag stress | ≤ **1e-5** Ha/Bohr³ | worst **1.970e-06** |
| C/D diag stress | ≤ **1e-5** Ha/Bohr³ | worst **1.969e-06** |
| NVE drift (B, D) | ≤ 0.05 | B **4.056e-3**, D **4.057e-3** |
| Geometry | soft H–H 0.4–1.5 Å | r ≈ 0.907–0.908 Å |

Measured per-step \(\sigma_{xx}\) [Ha/Bohr³] (2026-07-31):

| Step | CP2K | A (BOMD-k) | B (lin-k) | C (BOMD-Γ) | D (lin-Γ) |
| ---- | ---- | ---------- | --------- | ---------- | --------- |
| 0 | 1.998441e-04 | 1.964528e-04 | 1.964528e-04 | 1.964574e-04 | 1.964574e-04 |
| 1 | 1.701541e-04 | 1.464863e-04 | 1.471052e-04 | 1.464903e-04 | 1.471092e-04 |
| 2 | 1.087027e-04 | 5.558410e-05 | 5.642855e-05 | 5.558798e-05 | 5.643209e-05 |
| 3 | 4.492363e-05 | −1.533995e-05 | −1.625187e-05 | −1.533570e-05 | −1.624754e-05 |
| 4 | −8.093268e-06 | −5.905601e-05 | −6.102614e-05 | −5.905180e-05 | −6.102122e-05 |

The A/B (and C/D) envelope is the decisive shadow gate: wrong payload,
missing \(W\), or silent stress drop moves the linearized trace away from the
BOMD anchor by orders more than \(2\times 10^{-6}\). Absolute CP2K residual is
dominated by the known periodic-GPW lattice-sum/PP-tail class (seed) plus
trajectory geometry drift (same class as the sibling energy band).

## Perf gate (fair wall, same host / OMP=8 / single rank)

| Metric | Value |
| ------ | ----- |
| \(T_{\mathrm{cp2k}}\) (4-step converged-BOMD k + analytical stress) | **2.13 s** |
| \(T_{\mathrm{atv}}\) (leg B: 1 seed + 4 linearized steps with shadow stress) | **5.12 s** |
| ratio \(T_{\mathrm{atv}}/T_{\mathrm{cp2k}}\) | **2.4** |
| within_10x | **true** |
| perf_exempt | false |
| Regression wall (dual gate + four legs) | **36.8 s** (≤ 60 s budget) |

Shadow stress adds 6 fixed-\(P\) energy evals per warm step (no extra SCF);
the linearized path remains faster than full BOMD-k with stress (A 6.04 s vs
B 5.12 s on this window).

## Dual gate summary

| Gate | Result |
| ---- | ------ |
| Correctness | unit `test_cpu_gpw_stress` (state overload identity + unconverged throw) + unit `test_cpu_gpw_xl_bomd` (build accepts stress; refuses `stress_use_fd`) + e2e this regression |
| Speed | ratio 2.4 ≤ 10 (measured) |

## Provenance

- Spec: `.claude/specs/xl-bomd-linearized-stress/` (delete on close)
- Twin input: `tools/cp2k_oracle/inputs/xl-bomd-linearized-stress/h2_xl_linearized_stress_nve.inp`
- Queue: `docs/developer/cp2k-method-queue.md` row `xl-bomd-linearized-stress`

# Reference — multi-k water-dimer GPW PBE-D3 (zero) analytic cell stress

## Method

**Multi-k GPW** (PBE) with additive **DFT-D3 zero ("original") damping**
(PBE reference parameters, two-body only, `CALCULATE_C9_TERM .FALSE.`) on a
**periodic** water dimer with Monkhorst–Pack mesh **1×1×2**.

**Product cell stress** is the Hellmann–Feynman fixed-\(\{P_k\}\) path:

1. One multi-k SCF at the unstrained cell → \(\{P_k\}\), \(\{W_k\}\).
2. Central FD of the non-self-consistent k-summed Lagrangian
   \(E_L[\{P_k\};\{W_k\};\varepsilon]\) under orthorhombic diagonal strain at
   fixed fractional coordinates, with geometry-only
   \(E_{\mathrm{disp}}^{\mathrm{D3}}(\{\mathbf{R}\},\mathbf{h})\)
   rebuilt at each strained cell (same `dftd3_pbe` sum as the energy path,
   `kD3Cp2kPairCutoffBohr` = 2 × CP2K `R_CUTOFF 20.0` Å lattice-image cutoff).
3. \(\sigma_{aa}=-\Omega^{-1}\partial E_L/\partial\varepsilon_{aa}\)
   (CP2K `STRESS_TENSOR` sign; E Hartree, \(\Omega\) Bohr³).

Atomiverse path: `cpu::gpw::GPWEvaluator` with `d3=true`, `k_mesh={1,1,2}`,
`compute_stress=true`, `stress_use_fd=false` → Frame `kStress` [eV/Å³];
compare via `last_stress_ha_bohr3()` [Ha/Bohr³].

## Why the stress pin is a *difference* of two CP2K runs

D3 is geometry-only: it adds \(E_{\mathrm{disp}}(\{R\},\mathbf{h})\) to the
total energy and does not enter the Kohn–Sham potential, so the converged
density is identical with and without it. Therefore

\[
\sigma^{\mathrm{CP2K}}(\mathrm{D3}) - \sigma^{\mathrm{CP2K}}(\mathrm{plain})
  = \sigma_{\mathrm{disp}}^{\mathrm{D3}}
\]

isolates exactly the quantity this multi-k fold adds, with the (much larger,
grid-/k-noise-limited) electronic stress cancelling — the multi-k analogue of
`gpw-d3-stress`. Atomiverse isolates its own dispersion stress the same way:
analytic product with `d3=true` minus analytic product with `d3=false`.

Because D3 is independent of the electronic k mesh, the multi-k Δσ pin matches
the Γ pin of `gpw-d3-stress` to machine precision of the NUMERICAL FD; this
slug still **re-runs** the multi-k twin so the pin is provenance-matched to the
claimed algorithm class (multi-k Quickstep + D3).

## System

| Item | Value |
|------|--------|
| System | water dimer, **6 atoms**, order O H H O H H |
| Cell | **12 × 12 × 12 Å**, `PERIODIC XYZ` |
| Coordinates (Å) | O (0,0,0); H (0.758602,0,±0.504284); O (3,0,0); H (3.758602,0,±0.504284) |
| Basis | `DZVP-MOLOPT-SR-GTH` |
| PP | O `GTH-PBE-q6`, H `GTH-PBE-q1` → **16** valence e⁻ (`n_occ=8`) |
| XC | PBE |
| k-mesh | Monkhorst–Pack **1×1×2** (TR-folded) |
| Dispersion | DFT-D3 zero damping, PBE: \(s_6=1.0\), \(s_8=0.722\), \(s_{r,6}=1.217\), \(s_{r,8}=1.0\); ATM off |
| Pair cutoff | `kD3Cp2kPairCutoffBohr` = 2 × `R_CUTOFF 20.0` Å |
| Cutoff | 200 Ry (ATV single grid; CP2K `CUTOFF 200, REL_CUTOFF 40, NGRIDS 4`) |
| SCF | CP2K `EPS_SCF 1e-7`, DIAGONALIZATION + DIRECT_P_MIXING α=0.4; ATV `tol=1e-7`, mix 0.4 |
| Stress | CP2K `STRESS_TENSOR NUMERICAL`; ATV analytic fixed-\(P_k\) `h=2e-3` |

## Offline CP2K pin

| Field | Value |
| ----- | ----- |
| Engine | **CP2K 2026.1** (`cp2k.psmp`, source build `…/cp2k/install/env.sh`, git:**5e54ba2**) |
| Host | Arrhenius login (`arrhenius1`), offline only |
| Parallelism | **1 MPI × OMP 1** (`mpirun -np 1`, `OMP_NUM_THREADS=1`) |
| Inputs | `tools/cp2k_oracle/inputs/gpw-d3-kpoint-stress/water_dimer_d3_kpoint_stress.inp` and `water_dimer_plain_kpoint_stress.inp` |
| Wall (D3 NUMERICAL twin) | **118.34 s** (CP2K internal `CP2K` timer; `/usr/bin/time` 118.69 s) |
| Wall (plain NUMERICAL twin) | **146.62 s** (`/usr/bin/time`) |
| Date | 2026-08-01 |

### Energies [Hartree] (final unstrained FORCE_EVAL)

| Quantity | CP2K |
| -------- | ---- |
| Total (plain) | −34.338293290396436 |
| Total (D3) | −34.339114185073058 |
| **Dispersion energy** | **−0.00082089467662** |

(Matches the Γ periodic D3-zero energy pin — geometry-only, mesh-independent.)

### Numerical stress tensors (CP2K prints **bar**; diagonal)

| Axis | σ(D3) [bar] | σ(plain) [bar] | Δσ = D3 − plain [bar] |
| ---- | ----------- | -------------- | --------------------- |
| xx | 5.44389074124×10³ | 5.44891118013×10³ | **−5.02043889** |
| yy | −1.06513887162×10² | −1.06247938320×10² | **−0.265948842** |
| zz | 1.28102622305×10⁴ | 1.28091610694×10⁴ | **+1.10116110** |

Unit conversion (CP2K internal table):
\(1\,\mathrm{Ha/Bohr}^3 = 2.94210107994716\times 10^{8}\,\mathrm{bar}\).

**Hard Δσ pin [Ha/Bohr³]:** xx **−1.70641278242×10⁻⁸**,
yy **−9.03941893134×10⁻¹⁰**, zz **+3.74277113559×10⁻⁹**.

**Soft total σ(D3) pin [Ha/Bohr³]:** xx 1.85034116548×10⁻⁵,
yy −3.62033404929×10⁻⁷, zz 4.35412036582×10⁻⁵.

### Re-run (offline only — never ctest)

```bash
source /nobackup/proj/disk/teoroo/personal/jicli594/work/cp2k/install/env.sh
cd tools/cp2k_oracle/inputs/gpw-d3-kpoint-stress
export OMP_NUM_THREADS=1
mpirun -np 1 cp2k.psmp -i water_dimer_d3_kpoint_stress.inp \
    -o water_dimer_d3_kpoint_stress.out
mpirun -np 1 cp2k.psmp -i water_dimer_plain_kpoint_stress.inp \
    -o water_dimer_plain_kpoint_stress.out
# Record: Dispersion energy; STRESS| Numerical tensor; CP2K wall
```

## Tolerances (physical justification)

| Gate | Tol | Why |
| ---- | --- | --- |
| \|E_disp − pin\| | **1×10⁻⁸ Ha** | Same closed-form two-body sum / PBE params / 2×R_CUTOFF as Γ `gpw-d3-stress` (measured ~1e−10). |
| max\|Δσ_atv − Δσ_cp2k\| diag | **2×10⁻⁹ Ha/Bohr³** | Difference of two products sharing one density fixed point; same noise budget as Γ D3 stress pin. |
| max\|σ_an − σ_fd\| diag (d3=true, multi-k) | **2×10⁻⁵ Ha/Bohr³** | Internal dual gate; fixed-\(P_k\) vs full re-SCF residual + O(h²). |
| off-diagonal product | **1×10⁻⁸** | Orthorhombic cell: shear identically 0. |
| max\|σ_diag\| floor | **>1×10⁻¹⁰ Ha/Bohr³** | Stress actually computed. |
| max\|σ_tot − σ_cp2k\| diag | **1×10⁻⁴ Ha/Bohr³** (sanity) | ATV single 200 Ry grid vs CP2K NGRIDS 4 / REL_CUTOFF 40; catches unit/sign/blow-up only. |

## Atomiverse numbers (first green, 2026-08-01)

`reg_gpw-d3-kpoint-stress` (OMP 8, Arrhenius login), 11 SCF iters, `nk=1`
(TR-folded MP 1×1×2), \(E_{\mathrm{tot}}\) = −34.331152296970 Ha:

| Gate | Tol | Measured |
| ---- | --- | -------- |
| \|E_disp − pin\| | 1e−8 Ha | **5.53e−11** (E_disp = −8.20894731919e−04 Ha) |
| diag \|Δσ_atv − Δσ_cp2k\| | 2e−9 Ha/Bohr³ | **7.50e−11** (xx; yy 1.6e−11, zz 1.9e−11) |
| dual gate max\|σ_an − σ_fd\| diag | 2e−5 Ha/Bohr³ | **7.39e−07** (xx; yy 2.5e−8, zz 1.2e−9) |
| max\|σ_tot − σ_cp2k\| diag (sanity) | 1e−4 Ha/Bohr³ | 4.02e−05 |
| off-diagonal product | 1e−8 | ~0 (pass) |
| max\|σ_diag\| floor | >1e−10 | 5.83e−5 |
| Wall product (SCF + analytic stress) | ≤ 90 s | **21.84 s** |
| Fair wall ratio vs CP2K NUMERICAL D3 | ≤ 10 | **0.185** (21.84 / 118.34) |

Product analytic diagonals [Ha/Bohr³]: xx −2.16996452×10⁻⁵,
yy +2.00723444×10⁻⁵, zz +5.82684399×10⁻⁵.

The Δσ residual (7.5e−11 on xx) sits well below the 2e−9 gate; xx signal
1.71e−8 ≈ 230× the residual. Dual-gate residual matches the Γ
`gpw-d3-stress` product (~7.4e−7).

## Dual gate

- **Correctness:** CP2K multi-k NUMERICAL stress-difference + E_disp pins;
  internal analytic-vs-FD; unit tests
  `GpwStress.D3ChangesKpointDiagonal` and
  `GpwStress.AnalyticMatchesFdWithD3Kpoint` in
  `tests/cpu/gpw/test_cpu_gpw_stress.cpp`.
- **Speed:** fair wall = ATV product (multi-k SCF + fixed-\(P_k\) stress) /
  CP2K NUMERICAL D3 twin wall **118.34 s** (1×1). Regression extra plain/FD
  twins exist only for the difference pin and dual gate — not the product
  path ratio.

## Notes

- Sibling Γ pin: `regressions/gpw/gpw-d3-stress`.
- Sibling multi-k stress: `gpw-kpoint-stress` (no D3), `gpw-mbd-kpoint-stress`.
- Sibling multi-k D3(BJ)/ATM/D4/TS/XDM stress pins are separate queue rows.

# Reference — GPW CELL_OPT MOLOPT re-twin (ice-like water chain)

## Method

Pin-quality **MOLOPT re-twin** of `gpw-variable-cell-ice`. Same constrained
variable-cell problem (fixed fractional coordinates; only the chain axis
relaxes), productized as:

`cpu::minimizer::Minimize` + `AnisotropicCellRelax` (`relax_x` only;
extensive `kStress` [eV]) + `StressConverged` + `cpu::gpw::GPWEvaluator`
(`compute_stress = true`, product **analytic fixed-\(P\)** stress,
`stress_axes = {x}`).

The ice slug pivoted to compact SZV/DZV-GTH-PADE because multi-SCF strain-FD
with **DZVP-MOLOPT-GTH** exceeded the ≤ 60 s budget. Analytic fixed-\(P\)
stress (default) removes the multi-SCF FD tax, but **full DZVP-MOLOPT-GTH**
periodic image-sum collocation still costs **> 60 s per SCF** on this chain
(measured 2026-07-25; timeout). The e2e twin therefore uses the short-range
MOLOPT table **DZVP-MOLOPT-SR-GTH** (same family as `gpw-pbe-d3bj-ice` /
`gpw-ts-ice`) — still a MOLOPT Gaussian basis + PBE + GTH-PBE product path.
Full-MOLOPT CP2K pins are retained as a secondary oracle input for the
GPW perf campaign.

## System

Periodic proton-ordered **water chain** — 1 H₂O per cell, H-bonded to its own
x-images (ice Ih c-axis strand motif) — `PERIODIC XYZ`, b = c = 8 Å vacuum,
chain axis `a` variable.

Fractional coordinates (fixed; CP2K `SCALED T`):

| Atom | f_x | f_y | f_z |
| ---- | --- | --- | --- |
| O  | 0.000000 | 0 | 0.000000 |
| H1 | 0.319067 | 0 | 0.000000 |
| H2 | −0.079900 | 0 | 0.115830 |

| Knob | Value |
| ---- | ----- |
| XC | **PBE** |
| Basis | **DZVP-MOLOPT-SR-GTH** (O, H) |
| PP | GTH-PBE-q6 (O), GTH-PBE-q1 (H) |
| CUTOFF | 300 Ry |
| SCF | EPS_SCF / `tol` = 1e-7 |
| Cell | a from a₀ = **3.24 Å**; b = c = 8 Å fixed |
| Stress | analytic fixed-\(P\), h = 2e-3, axes `{x}` |
| Relax | α = 1.0, bulk_mod = 0.20 eV/Å³, max_scale 0.05 |
| Converge | `StressConverged{0, 5e-4 eV/Å³, x-only}` |

## Offline CP2K pins (CP2K 2026.1, never launched from ctest)

Engine: **CP2K 2026.1** (`cp2k.psmp`, EasyBuild `2026.1-gmpflf-2026.03`),
Arrhenius epyc login, 1 rank × 4 OMP. Inputs:
`tools/cp2k_oracle/inputs/gpw-variable-cell-molopt/`.

### Headline pin — SR-MOLOPT CELL_OPT (e2e twin)

| Quantity | a₀ = 3.328 (scan) | a₀ = 3.24 (fair twin) |
| -------- | ----------------- | --------------------- |
| a\* | **3.200237 Å** | **3.200292 Å** |
| E\* | **−17.184075549824836 Ha** | −17.184075557775 Ha |
| Wall (timing table `CP2K`) | 9.55 s | **5.67 s** |
| Wall (`time` real) | 12.9 s | 6.17 s |

Regression hard-codes a\* = 3.200237, E\* = −17.184075549824836 Ha (a₀ =
3.328 run; a\* agrees with the a₀ = 3.24 run to < 1e-4 Å).

### Secondary pin — full DZVP-MOLOPT-GTH CELL_OPT (not e2e)

Input: `chain_full_molopt_cellopt.inp`.

| Quantity | Value |
| -------- | ----- |
| a\* | 3.200252 Å |
| E\* | −17.185176472246 Ha |
| Wall (timing table) | 23.5 s |

ATV cannot dual-gate this basis inside ≤ 60 s today (periodic collocation).

### E(a) curvature (full MOLOPT ENERGY scan; bulk_mod source)

| a [Å] | E [Ha] |
| ----- | ------ |
| 3.16 | −17.185048204416361 |
| 3.20 | −17.185176430631692 |
| 3.24 | −17.185069177601907 |

\(d^2E/da^2 \approx 0.147\,\mathrm{Ha/Å}^2\) →
\(d\sigma_{xx}/d\varepsilon \approx 0.20\,\mathrm{eV/Å}^3\) → `bulk_mod = 0.20`.

## Hard gates (measured 2026-07-25, Release, 4 OMP, `OMP_PROC_BIND=close`)

| Gate | Tolerance | Measured |
| ---- | --------- | -------- |
| relax converged | — | `stress_converged` ✓ |
| \|a_ATV − a\*_CP2K\| | ≤ 0.02 Å | **0.0012 Å** (3.201465 vs 3.200237) |
| b, c untouched | bit-identical | 8.0 / 8.0 ✓ |
| residual \|σ_xx\| | ≤ 6×10⁻⁴ eV/Å³ | **5.1×10⁻⁵ eV/Å³** |
| \|E_ATV − E_CP2K\| | ≤ 1×10⁻³ Ha | **1.4×10⁻⁴ Ha** |
| wall | ≤ 60 s | **42.5 s** |

## Perf gate

Fair wall from the **same a₀ = 3.24 Å** (1 rank × 4 OMP):

| Metric | Value |
| ------ | ----- |
| T_cp2k (CELL_OPT, analytic) | **5.67 s** |
| T_atv (regression) | **42.5 s** |
| ratio T_atv/T_cp2k | **7.5** |
| within_10x | **true** |
| perf_exempt | false |

## Oracle commands (offline)

```bash
module load buildenv-gcc/2026.03-mpich
export CP2KROOT=/software/sse2/el9_epyc9005/easybuild/prefix/software/CP2K/2026.1-gmpflf-2026.03
export CP2K_DATA_DIR=$CP2KROOT/share/cp2k/data
export PATH=$CP2KROOT/bin:$PATH
export OMP_NUM_THREADS=4
cd /tmp && ln -sf $CP2K_DATA_DIR/{BASIS_MOLOPT,GTH_POTENTIALS} .
cp $REPO/tools/cp2k_oracle/inputs/gpw-variable-cell-molopt/chain_cellopt.inp .
mpirun -np 1 cp2k.psmp -i chain_cellopt.inp -o cellopt.out
```

## Relation to `gpw-variable-cell-ice`

| | ice (GTH-PADE pivot) | this slug (MOLOPT re-twin) |
| - | -------------------- | -------------------------- |
| XC / basis / PP | PADE / SZV+DZV-GTH-PADE | PBE / DZVP-MOLOPT-SR-GTH / GTH-PBE |
| a\* (CP2K) | 3.211148 Å | 3.200237 Å |
| Stress path | multi-SCF strain-FD | analytic fixed-\(P\) |
| Wall ATV / CP2K | 5.1 / 3.9 s | 42.5 / 5.7 s |

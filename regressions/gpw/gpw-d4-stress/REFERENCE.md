# Reference — periodic water-dimer GPW PBE-D4 analytic cell stress

## Method

Γ-point **GPW** (PBE) with additive **DFT-D4** (Becke–Johnson, two-body,
PBE parameters; ATM off) on a periodic water dimer.

**Product cell stress** is the Hellmann–Feynman fixed-density-matrix path:

1. One SCF at the unstrained cell → \(P\) and energy-weighted \(W\).
2. Central FD of the non-self-consistent Lagrangian
   \(E_L[P;W;\varepsilon]\) under orthorhombic diagonal strain at fixed
   fractional coordinates, with geometry-only
   \(E_{\mathrm{disp}}^{\mathrm{D4}}(\{\mathbf{R}\},\mathbf{h})\)
   rebuilt at each strained cell (same `dftd4_energy` as the energy path).
3. \(\sigma_{aa}=-\Omega^{-1}\partial E_L/\partial\varepsilon_{aa}\)
   (CP2K `STRESS_TENSOR` sign; E Hartree, \(\Omega\) Bohr³).

Oracle twin: multi-SCF strain-FD (`gpw_stress_fd`) on the same `d4=true`
method (full re-SCF includes D4 via `fold_dispersion`).

Atomiverse path: `cpu::gpw::GPWEvaluator` with `d4=true`,
`compute_stress=true`, `stress_use_fd=false` → Frame `kStress` [eV/Å³];
compare via `last_stress_ha_bohr3()` [Ha/Bohr³].

## System

Same geometry as `gpw-d4-water` / `gpw-pbe-d3bj-ice`:

| Item | Value |
|------|--------|
| System | water dimer, **6 atoms** |
| Cell | **12 × 12 × 12 Å**, `PERIODIC XYZ` |
| Coordinates (Å) | O (0,0,0); H (0.758602,0,±0.504284); O (3,0,0); H (3.758602,0,±0.504284) |
| Basis | `DZVP-MOLOPT-SR-GTH` |
| PP | O `GTH-PBE-q6`, H `GTH-PBE-q1` → **16** valence e⁻ (`n_occ=8`) |
| XC | PBE |
| Dispersion | DFT-D4 BJ (PBE: \(s_6=1\), \(s_8=0.95948085\), \(a_1=0.38574991\), \(a_2=4.80688534\)); ATM off |
| Cutoff | 200 Ry |
| Stress step | \(h=2\times 10^{-3}\) |

## Offline pins

### Hard: E_disp (dftd4 3.7.0)

**E_disp = −0.00099458842745846774 Ha** (two-body, periodic, `--mbdscale 0`).

Source: offline **dftd4 3.7.0** on Arrhenius
(`/software/.../dftd4/3.7.0-gfbf-2025b`), same POSCAR as
`tools/cp2k_oracle/inputs/gpw-d4-water/POSCAR`. See sibling
`regressions/gpw/gpw-d4-water/REFERENCE.md`.

### Decisive: dual-gate analytic vs multi-SCF FD

No absolute CP2K `STRESS_TENSOR` pin: Arrhenius **CP2K 2026.1** is built
**without** DFTD4 (`TYPE DFTD4` aborts). Twin input retained under
`tools/cp2k_oracle/inputs/gpw-d4-water/` for hosts with libdftd4.

Decisive gate is **internal dual-gate** \(\max_a|\sigma_a^{\mathrm{an}}
-\sigma_a^{\mathrm{fd}}|\).

### Optional: CP2K / dftd4 re-run (offline only — never ctest)

```bash
# E_disp pin
export PATH=/software/sse2/el9_epyc9005/easybuild/prefix/software/dftd4/3.7.0-gfbf-2025b/bin:$PATH
cd tools/cp2k_oracle/inputs/gpw-d4-water
dftd4 -f pbe --mbdscale 0.0 --json pin.json POSCAR
```

## Tolerances

| Check | Tol |
| ----- | --- |
| \|E_disp − pin\| | **1×10⁻⁸** Ha |
| max\|\σ_an − σ_fd\| diag | **2×10⁻⁵** Ha/Bohr³ |
| off-diagonal product | **1×10⁻⁸** |
| max\|\σ_diag\| floor | > **1×10⁻¹⁰** Ha/Bohr³ |

## Atomiverse numbers (first green, 2026-07-26)

| Quantity | Value |
| --- | --- |
| **E (Γ PBE-D4 200 Ry)** | **−34.3313259463 Ha** (SCF 11 iters) |
| E_disp | **−9.9458849003×10⁻⁴ Ha** (\|d pin\| = 6.3×10⁻¹¹) |
| σ_xx analytic | −2.17585158e-05 Ha/Bohr³ |
| σ_yy analytic | 2.00723219e-05 Ha/Bohr³ |
| σ_zz analytic | 5.82669568e-05 Ha/Bohr³ |
| max\|σ_an−σ_fd\| | **7.39e-07** Ha/Bohr³ |
| T_product (OMP=8, Release) | **21.4 s** |
| T_fd_twin (OMP=8) | **42.3 s** |
| T_dual_gate wall | **~64 s** |

## Performance

| Item | Value |
| ---- | ----- |
| `perf_exempt` | **true** |
| reason | Arrhenius CP2K build lacks DFTD4 (no fair QS stress wall). Dual-gate wall is dominated by multi-SCF FD (6 SCFs), not product analytic. Primary hard gate is E_disp pin parity + analytic↔FD residual (same policy as `gpw-d4-water` / `gpw-dftu-stress`). |

## What is checked

1. Product analytic stress with `d4=true` finishes without throw.
2. `e_disp` within 1e-8 Ha of the dftd4 pin.
3. Dual-gate diagonal residual ≤ 2e-5 Ha/Bohr³.
4. Off-diag product ≈ 0; Frame `kStress` unit conversion.

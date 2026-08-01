# Reference — multi-k GPW analytic cell stress with DFT-D4

## Method

**Monkhorst–Pack multi-k GPW** (PBE, electronic mesh 1×1×2) on a periodic
water dimer with additive **DFT-D4** (Becke–Johnson two-body, PBE parameters;
ATM off). Closes the residual that the Γ fixed-\(P\) stress Lagrangian already
rebuilt \(E_{\mathrm{disp}}^{\mathrm{D4}}\) at every strained cell
(`gpw-d4-stress`) while the multi-k pin for the same fold was missing.

Product cell stress is the Hellmann–Feynman fixed-\(P_k\) path:

1. One multi-k SCF at the unstrained cell → \(\{P_k, W_k\}\).
2. Central FD of the non-self-consistent k-summed Lagrangian
   \(E_L[\{P_k\},\{W_k\};\varepsilon]\) under orthorhombic diagonal strain at
   fixed fractional coordinates, with the geometry-only
   \(E_{\mathrm{disp}}^{\mathrm{D4}}(\{\mathbf R\},\mathbf h)\) **rebuilt at
   every strained cell** (same `dftd4_energy` as the energy path).
3. \(\sigma_{aa} = -\Omega^{-1}\,\partial E_L/\partial\varepsilon_{aa}\)
   (CP2K `STRESS_TENSOR` sign; \(E\) Hartree, \(\Omega\) Bohr³).

Public path: `cpu::gpw::GPWEvaluator` with `k_mesh = {1,1,2}`, `d4 = true`,
`compute_stress = true`, `stress_use_fd = false` → Frame `kStress` [eV/Å³];
compared here via `last_stress_ha_bohr3()`.

## Oracles

| Gate | Oracle |
| ---- | ------ |
| A. \(E_{\mathrm{disp}}\) unstrained | offline **dftd4 3.7.0** — same pin `gpw-d4-water` / `gpw-d4-stress` use; D4 is independent of the electronic k mesh, so the multi-k run must reproduce the Γ pin |
| B. \(E_{\mathrm{disp}}\) **under strain** (±h per axis) | fresh offline dftd4 pins at each strained geometry (below) — the strain derivative being folded is externally validated point by point |
| C. \(\sigma\) | **multi-SCF strain-FD** (`gpw_stress_fd_impl`): full re-SCF at each strained cell on the same 1×1×2 mesh, D4 folded through the energy path |
| D. D4 actually folded | \(\sigma(\mathrm{d4}) - \sigma(\mathrm{plain})\) equals the dftd4 strain-FD term \(-\Delta E_{\mathrm{disp}}/(2h\Omega)\) per axis |

Host CP2K 2026.1 source build (`…/cp2k/install`, git:5e54ba2) has
**no `dftd4` in `cp2kflags`** — `TYPE DFTD4` is not a fair twin on that
binary. Same structure as `gpw-d4-stress` / `gpw-mbd-kpoint-stress`:
strain-FD dual gate + externally pinned dispersion term.

## System

Same geometry as `gpw-d4-water` / `gpw-d4-stress`:

| Item | Value |
|------|-------|
| System | water dimer, **6 atoms**, order **O H H O H H** |
| Cell | 12 × 12 × 12 Å, `PERIODIC XYZ` |
| Coordinates (Å) | O (0,0,0); H (0.758602, 0, ±0.504284); O (3,0,0); H (3.758602, 0, ±0.504284) |
| Basis | `DZVP-MOLOPT-SR-GTH` |
| PP | O `GTH-PBE-q6`, H `GTH-PBE-q1` → **16** valence e⁻ (`n_occ = 8`) |
| XC | PBE |
| Electronic k mesh | MONKHORST-PACK **1×1×2**, time-reversal folded |
| Dispersion | DFT-D4 BJ (PBE: \(s_6=1\), \(s_8=0.95948085\), \(a_1=0.38574991\), \(a_2=4.80688534\)); ATM off |
| Cutoff | 120 Ry (E_disp is geometry-only — cutoff does not enter the pins) |
| Strain step | \(h = 2\times10^{-3}\) |

## Pins

Source: offline **dftd4 3.7.0** (`-f pbe --mbdscale 0.0`), Å→Bohr =
1/0.52917720859. Reproduce:
`tools/cp2k_oracle/inputs/gpw-d4-kpoint-stress/pin_d4_strain.py`
(never launched at ctest time). Output lines:

```
E_disp unstrained            = -9.9458842745846774e-04 Ha
E_disp x +h              = -9.9281871131614977e-04 Ha
E_disp x -h              = -9.9635829706912161e-04 Ha
sigma_D4_xx (FD, dftd4)    = -7.5884329667e-08 Ha/Bohr3
E_disp y +h              = -9.9456763760422703e-04 Ha
E_disp y -h              = -9.9460945924979903e-04 Ha
sigma_D4_yy (FD, dftd4)    = -8.9660422470e-10 Ha/Bohr3
E_disp z +h              = -9.9463925734105045e-04 Ha
E_disp z -h              = -9.9453454831018983e-04 Ha
sigma_D4_zz (FD, dftd4)    = 2.2448317886e-09 Ha/Bohr3
```

The unstrained value is bit-identical to the `gpw-d4-water` pin
(provenance cross-checked at pin time, 2026-08-01).

## Tolerances

| Check | Tol |
| ----- | --- |
| A/B \(\lvert E_{\mathrm{disp}} - \mathrm{dftd4}\rvert\) (unstrained + 6 strained) | 1×10⁻⁸ Ha |
| C \(\sigma_{\mathrm{diag}}\) analytic vs multi-SCF FD | 2×10⁻⁵ Ha/Bohr³ (Γ D4 reg tier) |
| C \(\lvert\sigma\rvert\) nonzero | > 1×10⁻¹⁰ |
| D \(\lvert(\sigma_{\mathrm{d4}}-\sigma_{\mathrm{plain}}) - \sigma^{\mathrm{D4}}_{\mathrm{dftd4}}\rvert\) | 2×10⁻⁹ Ha/Bohr³ |
| E off-diagonals (orthorhombic) | exactly 0 |
| E `kStress` unit conversion | 1×10⁻⁹ |
| Product wall | ≤ 60 s |

## Perf gate

Host CP2K 2026.1 source binary lacks DFTD4 (`cp2kflags` without `dftd4`), so
no fair oracle wall exists for a D4 stress run: **`perf_exempt`** — same
policy as `gpw-d4-stress` / `gpw-d4-water` / `gpw-d4-atm-stress`.

## Measured results (2026-08-01, first green)

| Gate | Tol | Measured |
| ---- | --- | -------- |
| B \(E_{\mathrm{disp}}\) under strain vs dftd4 (6 cells) | 1e−8 Ha | **3.81e−12** |
| A \(E_{\mathrm{disp}}\) multi-k vs Γ pin | 1e−8 Ha | **2.75e−14** |
| C \(\sigma_{\mathrm{diag}}\) analytic vs multi-SCF FD | 2e−5 Ha/Bohr³ | **1.36e−09** |
| D wiring vs dftd4 FD term | 2e−9 Ha/Bohr³ | **4.00e−12** |
| E off-diagonals | exactly 0 | **0** |
| E `kStress` conversion | 1e−9 | exact |
| Product wall | ≤ 60 s | **11.8 s** (FD twin 63.7 s; SCF 11 iters, nk = 1 after TR fold) |

σ_diag (d4, analytic, 120 Ry): xx −3.613e−05, yy −8.570e−05,
zz −3.081e−06 Ha/Bohr³; D4 shifts: xx −7.588e−08, yy −8.94e−10,
zz +2.25e−09 Ha/Bohr³ — matching the dftd4 FD terms per axis.

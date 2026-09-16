# Reference — multi-k GPW analytic cell stress with periodic MBD@rsSCS

## Method

**Monkhorst–Pack multi-k GPW** (PBE, electronic mesh 1×1×2) on a periodic
water dimer with **MBD@rsSCS** many-body dispersion (free-atom
polarizabilities, PBE \(\beta = 0.83\), Fermi steepness \(a = 6\), MBD
lattice sum on its own 2×2×2 mesh). Closes the residual that the Γ fixed-\(P\)
stress Lagrangian already rebuilt \(E_{\mathrm{disp}}^{\mathrm{MBD}}\) at
every strained cell (`gpw-mbd-stress`) while the **k-summed** Lagrangian
(`energy_fixed_P_k`) refused MBD with a throw.

Product cell stress is the Hellmann–Feynman fixed-\(P_k\) path:

1. One multi-k SCF at the unstrained cell → \(\{P_k, W_k\}\).
2. Central FD of the non-self-consistent k-summed Lagrangian
   \(E_L[\{P_k\},\{W_k\};\varepsilon]\) under orthorhombic diagonal strain at
   fixed fractional coordinates, with the geometry-only
   \(E_{\mathrm{disp}}^{\mathrm{MBD}}(\{\mathbf R\},\mathbf h)\) **rebuilt at
   every strained cell** (same `mbd_rsscs_pbe` call as the Γ path and the
   energy path).
3. \(\sigma_{aa} = -\Omega^{-1}\,\partial E_L/\partial\varepsilon_{aa}\)
   (CP2K `STRESS_TENSOR` sign; \(E\) Hartree, \(\Omega\) Bohr³).

Public path: `cpu::gpw::GPWEvaluator` with `k_mesh = {1,1,2}`, `mbd = true`,
`mbd_k_mesh = {2,2,2}`, `compute_stress = true`, `stress_use_fd = false` →
Frame `kStress` [eV/Å³]; compared here via `last_stress_ha_bohr3()`.

## Oracles

| Gate | Oracle |
| ---- | ------ |
| A. \(E_{\mathrm{disp}}\) unstrained | offline **pure-Python pymbd** `mbd_energy_species` — the same pin `gpw-mbd-periodic` / `gpw-mbd-stress` use; MBD is independent of the electronic k mesh, so the multi-k run must reproduce the Γ pin |
| B. \(E_{\mathrm{disp}}\) **under strain** (±h per axis) | fresh offline pymbd pins at each strained geometry (below) — the strain derivative being folded is externally validated point by point |
| C. \(\sigma\) | **multi-SCF strain-FD** (`gpw_stress_fd_impl`): full re-SCF at each strained cell on the same 1×1×2 mesh, MBD folded through the energy path |
| D. MBD actually folded | \(\sigma(\mathrm{mbd}) - \sigma(\mathrm{plain})\) equals the pymbd strain-FD term \(-\Delta E_{\mathrm{disp}}/(2h\Omega)\) per axis |

CP2K 2026.1 stock `PAIR_POTENTIAL TYPE` has no MBD (only DFTD2/D3/D3BJ/D4),
so there is no CP2K stress twin; the strain-FD dual gate plus the externally
pinned dispersion term is the correctness statement for the new fold — same
structure as `gpw-mbd-stress`, strengthened by gate B/D external pins on the
strained cells.

## System

Same geometry as `gpw-mbd-periodic` / `gpw-mbd-stress`:

| Item | Value |
|------|-------|
| System | water dimer, **6 atoms**, order **O H H O H H** |
| Cell | 12 × 12 × 12 Å, `PERIODIC XYZ` |
| Coordinates (Å) | O (0,0,0); H (0.758602, 0, ±0.504284); O (3,0,0); H (3.758602, 0, ±0.504284) |
| Basis | `DZVP-MOLOPT-SR-GTH` |
| PP | O `GTH-PBE-q6`, H `GTH-PBE-q1` → **16** valence e⁻ (`n_occ = 8`) |
| XC | PBE |
| Electronic k mesh | MONKHORST-PACK **1×1×2**, time-reversal folded |
| Dispersion | MBD@rsSCS, free-atom volumes, \(\beta = 0.83\), \(a = 6\), MBD k-mesh 2×2×2 |
| Cutoff | 120 Ry (E_disp is geometry-only — cutoff does not enter the pins) |
| Strain step | \(h = 2\times10^{-3}\) |

## Pins

Source: offline **pure-Python pymbd** `mbd_energy_species` (libmbd git
`88069c6`, `src/pymbd/pymbd.py`), free-atom volume ratios 1, β = 0.83, MBD
k-grid 2×2×2, Å→Bohr = 1/0.52917720859. Reproduce:
`tools/cp2k_oracle/inputs/gpw-mbd-kpoint-stress/pin_mbd_strain.py`
(never launched at ctest time). Output lines:

```
E_disp unstrained            = -1.4965720704989138e-03 Ha
E_disp x +h              = -1.4925684743305467e-03 Ha
E_disp x -h              = -1.5005781658326001e-03 Ha
sigma_MBD_xx (FD, pymbd)    = -1.7171785426e-07 Ha/Bohr3
E_disp y +h              = -1.4965139646472281e-03 Ha
E_disp y -h              = -1.4966306893944292e-03 Ha
sigma_MBD_yy (FD, pymbd)    = -2.5024338482e-09 Ha/Bohr3
E_disp z +h              = -1.4957889518587564e-03 Ha
E_disp z -h              = -1.4973531298823062e-03 Ha
sigma_MBD_zz (FD, pymbd)    = -3.3534037337e-08 Ha/Bohr3
```

The unstrained value is bit-identical to the `gpw-mbd-periodic` pin
(provenance cross-checked at pin time, 2026-07-31).

## Tolerances

| Check | Tol |
| ----- | --- |
| A/B \(\lvert E_{\mathrm{disp}} - \mathrm{pymbd}\rvert\) (unstrained + 6 strained) | 1×10⁻⁸ Ha |
| C \(\sigma_{\mathrm{diag}}\) analytic vs multi-SCF FD | 2×10⁻⁵ Ha/Bohr³ (Γ MBD reg tier) |
| C \(\lvert\sigma\rvert\) nonzero | > 1×10⁻¹⁰ |
| D \(\lvert(\sigma_{\mathrm{mbd}}-\sigma_{\mathrm{plain}}) - \sigma^{\mathrm{MBD}}_{\mathrm{pymbd}}\rvert\) | 2×10⁻⁹ Ha/Bohr³ |
| E off-diagonals (orthorhombic) | exactly 0 |
| E `kStress` unit conversion | 1×10⁻⁹ |
| Product wall | ≤ 60 s |

## Perf gate

CP2K 2026.1 stock has no MBD pair potential, so no fair oracle wall exists
for an MBD stress run: **`perf_exempt`** — same policy as `gpw-mbd-stress` /
`gpw-mbd-kpoint-forces`.

## Measured results (2026-07-31, first green)

| Gate | Tol | Measured |
| ---- | --- | -------- |
| B \(E_{\mathrm{disp}}\) under strain vs pymbd (6 cells) | 1e−8 Ha | **1.78e−15** |
| A \(E_{\mathrm{disp}}\) multi-k vs Γ pin | 1e−8 Ha | **1.07e−11** |
| C \(\sigma_{\mathrm{diag}}\) analytic vs multi-SCF FD | 2e−5 Ha/Bohr³ | **1.36e−09** |
| D wiring vs pymbd FD term | 2e−9 Ha/Bohr³ | **4.00e−12** |
| E off-diagonals | exactly 0 | **0** |
| E `kStress` conversion | 1e−9 | exact |
| Product wall | ≤ 60 s | **13.6 s** (FD twin 22.0 s; SCF 11 iters, nk = 1) |

σ_diag (mbd, analytic, 120 Ry): xx −3.622e−05, yy −8.570e−05,
zz −3.117e−06 Ha/Bohr³; MBD shifts: xx −1.7171e−07, yy −2.500e−09,
zz −3.353e−08 Ha/Bohr³ — matching the pymbd FD terms per axis.

# Reference — multi-k GPW analytic cell stress with free-atom TS dispersion

## Method

**Monkhorst–Pack multi-k GPW** (PBE, electronic mesh 1×1×2) on a periodic
water dimer with **Tkatchenko–Scheffler (TS)** free-atom pairwise dispersion
(\(V/V_{\mathrm{free}} = 1\); Tkatchenko & Scheffler, *Phys. Rev. Lett.*
**102**, 073005 (2009)). Closes the residual that the Γ fixed-\(P\) stress
Lagrangian already rebuilt \(E_{\mathrm{disp}}^{TS}\) at every strained cell
(`gpw-ts-stress`) while the multi-k path needed its dedicated offline pin
(product fold in `energy_fixed_P_k` already ships geometry-only TS via
`ts_pbe`).

Product cell stress is the Hellmann–Feynman fixed-\(P_k\) path:

1. One multi-k SCF at the unstrained cell → \(\{P_k, W_k\}\).
2. Central FD of the non-self-consistent k-summed Lagrangian
   \(E_L[\{P_k\},\{W_k\};\varepsilon]\) under orthorhombic diagonal strain at
   fixed fractional coordinates, with the geometry-only
   \(E_{\mathrm{disp}}^{TS}(\{\mathbf R\},\mathbf h)\) **rebuilt at every
   strained cell** (same `ts_pbe` call as the Γ path and the energy path;
   pure real-space lattice sum, pair cutoff **50 Å**, \(s_R = 0.94\),
   Fermi \(d = 20\)).
3. \(\sigma_{aa} = -\Omega^{-1}\,\partial E_L/\partial\varepsilon_{aa}\)
   (CP2K `STRESS_TENSOR` sign; \(E\) Hartree, \(\Omega\) Bohr³).

Public path: `cpu::gpw::GPWEvaluator` with `k_mesh = {1,1,2}`, `ts = true`,
`compute_stress = true`, `stress_use_fd = false` → Frame `kStress` [eV/Å³];
compared here via `last_stress_ha_bohr3()`.

## Oracles

| Gate | Oracle |
| ---- | ------ |
| A. \(E_{\mathrm{disp}}\) unstrained | offline **libMBD Fortran** `get_ts_energy`, pure-RS branch — the genuine-oracle anchor of `gpw-ts-ice` / `gpw-ts-stress`; TS is independent of the electronic k mesh, so the multi-k run must reproduce the Γ pin |
| B. \(E_{\mathrm{disp}}\) **under strain** (±h per axis) | fresh offline libMBD pins at each strained geometry (below) — the strain derivative being folded is externally validated point by point |
| C. \(\sigma\) | **multi-SCF strain-FD** (`gpw_stress_fd_impl`): full re-SCF at each strained cell on the same 1×1×2 mesh, TS folded through the energy path |
| D. TS actually folded | \(\sigma(\mathrm{ts}) - \sigma(\mathrm{plain})\) equals the libMBD strain-FD term \(-\Delta E_{\mathrm{disp}}/(2h\Omega)\) per axis |

### Why not CP2K QS?

**CP2K 2026.1** stock `PAIR_POTENTIAL TYPE` is only
`DFTD2|DFTD3|DFTD3(BJ)|DFTD4` — there is **no** stock free-atom `TYPE TS`,
hence no CP2K stress twin. libMBD is the canonical offline TS oracle (the
backend FHI-aims / QE / DFTB+ use) — same policy as `gpw-ts-stress` /
`gpw-ts-forces` / `gpw-mbd-kpoint-stress`.

## System

Same geometry as `gpw-ts-ice` / `gpw-ts-stress`:

| Item | Value |
|------|-------|
| System | water dimer, **6 atoms**, order **O H H O H H** |
| Cell | 12 × 12 × 12 Å, `PERIODIC XYZ` |
| Coordinates (Å) | O (0,0,0); H (0.758602, 0, ±0.504284); O (3,0,0); H (3.758602, 0, ±0.504284) |
| Basis | `DZVP-MOLOPT-SR-GTH` |
| PP | O `GTH-PBE-q6`, H `GTH-PBE-q1` → **16** valence e⁻ (`n_occ = 8`) |
| XC | PBE |
| Electronic k mesh | MONKHORST-PACK **1×1×2**, time-reversal folded |
| Dispersion | TS free-atom, pure-RS rcut 50 Å, \(s_R = 0.94\), \(d = 20\) |
| Cutoff | 120 Ry (E_disp is geometry-only — cutoff does not enter the pins) |
| Strain step | \(h = 2\times10^{-3}\) |

## Pins

Source: offline **libMBD Fortran** `get_ts_energy` (libmbd git `88069c6`,
GCC 14.3 / OpenBLAS 0.3.32), pure real-space branch, free-atom
\(\alpha_0/C_6/R_{\mathrm{vdW}}\) from libMBD `ts_vdw_params`, Å→Bohr =
1/0.52917720859. Reproduce:
`tools/cp2k_oracle/inputs/gpw-ts-kpoint-stress/pin_ts_strain.f90`
(never launched at ctest time). Output lines (geometry-only; bit-identical
to the `gpw-ts-stress` pin set):

```
E_disp unstrained           =  -6.0996307733860822E-04 Ha
E_disp x +h            =  -6.1278297256613687E-04 Ha
E_disp x -h            =  -6.0709134569744607E-04 Ha
sigma_TS_xx (FD, libMBD)  =   1.2202142278E-07 Ha/Bohr3
E_disp y +h            =  -6.0992605267944588E-04 Ha
E_disp y -h            =  -6.1000059619617259E-04 Ha
sigma_TS_yy (FD, libMBD)  =  -1.5981205691E-09 Ha/Bohr3
E_disp z +h            =  -6.0996696060335368E-04 Ha
E_disp z -h            =  -6.0995897233070745E-04 Ha
sigma_TS_zz (FD, libMBD)  =   1.7125866055E-10 Ha/Bohr3
```

## Tolerances

| Check | Tol |
| ----- | --- |
| A/B \(\lvert E_{\mathrm{disp}} - \mathrm{libMBD}\rvert\) (unstrained + 6 strained) | 1×10⁻⁸ Ha |
| C \(\sigma_{\mathrm{diag}}\) analytic vs multi-SCF FD | 2×10⁻⁵ Ha/Bohr³ (sibling dispersion-stress tier) |
| C \(\lvert\sigma\rvert\) nonzero | > 1×10⁻¹⁰ |
| D \(\lvert(\sigma_{\mathrm{ts}}-\sigma_{\mathrm{plain}}) - \sigma^{TS}_{\mathrm{libMBD}}\rvert\) | 2×10⁻⁹ Ha/Bohr³ |
| E off-diagonals (orthorhombic) | exactly 0 |
| E `kStress` unit conversion | 1×10⁻⁹ |
| Product wall | ≤ 60 s |

## Perf gate

CP2K 2026.1 stock has no free-atom TS pair potential, so no fair oracle wall
exists for a multi-k TS stress run: **`perf_exempt`** — same policy as
`gpw-ts-stress` / `gpw-ts-forces` / `gpw-mbd-kpoint-stress`.

## Measured results (2026-08-01, wave-7 rebase onto `dev` + dual-gate green)

| Gate | Tol | Measured |
| ---- | --- | -------- |
| B \(E_{\mathrm{disp}}\) under strain vs libMBD (6 cells) | 1e−8 Ha | **9.98e−18** |
| A \(E_{\mathrm{disp}}\) multi-k vs Γ pin | 1e−8 Ha | **6.19e−12** |
| C \(\sigma_{\mathrm{diag}}\) analytic vs multi-SCF FD | 2e−5 Ha/Bohr³ | **1.36e−09** |
| D wiring vs libMBD FD term | 2e−9 Ha/Bohr³ | **4.01e−12** |
| E off-diagonals | exactly 0 | **0** |
| E `kStress` conversion | 1e−9 | exact |
| Product wall | ≤ 60 s | **28.54 s** (FD twin 127.03 s under host load; SCF 11 iters, nk = 1 TR-folded 1×1×2) |

σ_diag (ts, analytic, 120 Ry, MP 1×1×2): xx −3.593e−05, yy −8.570e−05,
zz −3.083e−06 Ha/Bohr³; TS shifts: xx +1.2203e−07, yy −1.596e−09,
zz +1.743e−10 Ha/Bohr³ — matching the libMBD FD terms per axis.

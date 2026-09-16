# Reference — Γ GPW analytic cell stress with free-atom TS dispersion

## Method

**Tkatchenko–Scheffler (TS)** free-atom pairwise dispersion
(\(V/V_{\mathrm{free}} = 1\); Tkatchenko & Scheffler, *Phys. Rev. Lett.*
**102**, 073005 (2009)) folded into the Γ-point **GPW** (PBE) analytic
fixed-\(P\) cell stress. TS is geometry-only, so the stress Lagrangian
rebuilds \(E_{\mathrm{disp}}^{TS}(\{\mathbf R\},\mathbf h)\) at every
strained cell (same `ts_pbe` call as the energy path, pure real-space
lattice sum, pair cutoff **50 Å**, \(s_R = 0.94\), Fermi \(d = 20\)):

1. One SCF at the unstrained cell → \(P, W\).
2. Central FD of the non-self-consistent Lagrangian
   \(E_L[P, W; \varepsilon]\) under orthorhombic diagonal strain at fixed
   fractional coordinates, \(h = 2\times10^{-3}\).
3. \(\sigma_{aa} = -\Omega^{-1}\,\partial E_L/\partial\varepsilon_{aa}\)
   (CP2K `STRESS_TENSOR` sign; Hartree/Bohr³).

Public path: `cpu::gpw::GPWEvaluator` with `ts = true`,
`compute_stress = true`, `stress_use_fd = false` → Frame `kStress` [eV/Å³];
compared here via `last_stress_ha_bohr3()`.

## Oracles

| Gate | Oracle |
| ---- | ------ |
| A. \(E_{\mathrm{disp}}\) unstrained | offline **libMBD Fortran** `get_ts_energy`, pure-RS branch (`ewald_on = .false.`, `dipole_cutoff = 50 Å`, ATV Å→Bohr) — the genuine-oracle anchor of the `gpw-ts-ice` periodic pin |
| B. \(E_{\mathrm{disp}}\) **under strain** (±h per axis) | fresh offline libMBD pins at each strained cell (below) — the strain derivative being folded is externally validated point by point |
| C. \(\sigma\) | **multi-SCF strain-FD** (`gpw_stress_fd_impl`): full re-SCF at each strained cell, TS folded through the energy path |
| D. TS actually folded | \(\sigma(\mathrm{ts}) - \sigma(\mathrm{plain})\) equals the libMBD strain-FD term \(-\Delta E_{\mathrm{disp}}/(2h\Omega)\) per axis |

### Why not CP2K QS?

**CP2K 2026.1** stock `PAIR_POTENTIAL TYPE` is only
`DFTD2|DFTD3|DFTD3(BJ)|DFTD4` — there is **no** stock free-atom `TYPE TS`,
hence no CP2K stress twin. libMBD is the canonical offline TS oracle (the
backend FHI-aims / QE / DFTB+ use) — same policy as `gpw-ts-ice` /
`gpw-ts-forces`, strengthened here by gate B/D external pins on the strained
cells (structure of `gpw-mbd-kpoint-stress`).

## System

Same geometry as `gpw-ts-ice` / `gpw-ts-forces`:

| Item | Value |
|------|-------|
| System | water dimer, **6 atoms**, order **O H H O H H** |
| Cell | 12 × 12 × 12 Å, `PERIODIC XYZ` |
| Coordinates (Å) | O (0,0,0); H (0.758602, 0, ±0.504284); O (3,0,0); H (3.758602, 0, ±0.504284) |
| Basis | `DZVP-MOLOPT-SR-GTH` |
| PP | O `GTH-PBE-q6`, H `GTH-PBE-q1` → 16 valence e⁻ (`n_occ = 8`) |
| XC | PBE, Γ point |
| Dispersion | TS free-atom, pure-RS rcut 50 Å, \(s_R = 0.94\), \(d = 20\) |
| Cutoff | 120 Ry (E_disp is geometry-only — cutoff does not enter the pins) |
| Strain step | \(h = 2\times10^{-3}\) |

## Pins

Source: offline **libMBD Fortran** `get_ts_energy` (libmbd git `88069c6`,
GCC 14.3 / OpenBLAS 0.3.32), pure real-space branch, free-atom
\(\alpha_0/C_6/R_{\mathrm{vdW}}\) from libMBD `ts_vdw_params`, Å→Bohr =
1/0.52917720859. Reproduce:
`tools/cp2k_oracle/inputs/gpw-ts-stress/pin_ts_strain.f90` (never launched
at ctest time). Output lines (2026-07-31):

```
E_disp isolated             =  -5.9764872295816794E-04 Ha
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

Provenance cross-checks at pin time: the isolated value is **bit-identical**
to the shipped `cmbd_ts_energy` C-API pin (`gpw-ts-ice`); the unstrained
periodic value agrees with the shipped periodic pure-RS pin
(−6.099630773386013×10⁻⁴ Ha) to **7×10⁻¹⁸ Ha**, retro-anchoring that pin to a
genuine libMBD run.

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
exists for a TS stress run: **`perf_exempt`** — same policy as `gpw-ts-ice` /
`gpw-ts-forces` / `gpw-mbd-stress`.

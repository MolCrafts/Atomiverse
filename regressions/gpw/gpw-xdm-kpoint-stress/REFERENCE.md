# Reference — multi-k GPW analytic cell stress with free-atom XDM

## Method

**Monkhorst–Pack multi-k GPW** (PBE, electronic mesh 1×1×2) on a periodic
water dimer with **free-atom XDM** (exchange-hole dipole moment) pairwise
dispersion — Becke–Johnson combination rules, QE NC-PBE BJ damping
\(a_1=0.4283\), \(a_2=2.4690\) Å, pure real-space lattice sum with rcut
**50 Å** (same kernel as `gpw-xdm-ice` / `gpw-xdm-stress` / `gpw-xdm-forces`).

Closes the residual that the Γ fixed-\(P\) stress Lagrangian already rebuilt
\(E_{\mathrm{disp}}^{\mathrm{XDM}}\) at every strained cell
(`gpw-xdm-stress`) while the multi-k path needed its dedicated offline pin
(product fold in `energy_fixed_P_k` already ships geometry-only XDM via
`xdm_pbe`).

Product cell stress is the Hellmann–Feynman fixed-\(P_k\) path:

1. One multi-k SCF at the unstrained cell → \(\{P_k, W_k\}\).
2. Central FD of the non-self-consistent k-summed Lagrangian
   \(E_L[\{P_k\},\{W_k\};\varepsilon]\) under orthorhombic diagonal strain at
   fixed fractional coordinates, with the geometry-only
   \(E_{\mathrm{disp}}^{\mathrm{XDM}}(\{\mathbf R\},\mathbf h)\) **rebuilt at
   every strained cell** (same `xdm_pbe` call + rcut as the energy path).
3. \(\sigma_{aa} = -\Omega^{-1}\,\partial E_L/\partial\varepsilon_{aa}\)
   (CP2K `STRESS_TENSOR` sign; \(E\) Hartree, \(\Omega\) Bohr³).

Public path: `cpu::gpw::GPWEvaluator` with `k_mesh = {1,1,2}`, `xdm = true`,
`compute_stress = true`, `stress_use_fd = false` → Frame `kStress` [eV/Å³];
compared here via `last_stress_ha_bohr3()`.

## Oracles

| Gate | Oracle |
| ---- | ------ |
| A. \(E_{\mathrm{disp}}\) unstrained | offline **dual pure-Python** free-atom XDM lattice sum — the same pin `gpw-xdm-ice` / `gpw-xdm-stress` use; XDM is independent of the electronic k mesh, so the multi-k run must reproduce the Γ pin |
| B. \(E_{\mathrm{disp}}\) **under strain** (±h per axis) | fresh dual pure-Python pins at each strained geometry (below) — the strain derivative being folded is externally validated point by point |
| C. \(\sigma\) | **multi-SCF strain-FD** (`gpw_stress_fd_impl`): full re-SCF at each strained cell on the same 1×1×2 mesh, XDM folded through the energy path |
| D. XDM actually folded | \(\sigma(\mathrm{xdm}) - \sigma(\mathrm{plain})\) equals the dual-python strain-FD term \(-\Delta E_{\mathrm{disp}}/(2h\Omega)\) per axis |

### Why not CP2K QS?

**CP2K 2026.1** (`cp2k.psmp` git:5e54ba2) stock `PAIR_POTENTIAL TYPE` is only
`DFTD2|DFTD3|DFTD3(BJ)|DFTD4` — there is **no** stock `TYPE XDM`, so no CP2K
stress twin exists. Hard pins are the dual pure-Python energy sums (same
policy and oracle family as `gpw-xdm-ice` / `gpw-xdm-forces` /
`gpw-xdm-stress`), plus the multi-SCF strain-FD dual gate for the tensor —
same structure as `gpw-mbd-kpoint-stress` / `gpw-ts-kpoint-stress`.

## System

Same geometry as `gpw-xdm-ice` / `gpw-xdm-stress`:

| Item | Value |
|------|-------|
| System | water dimer, **6 atoms**, order **O H H O H H** |
| Cell | 12 × 12 × 12 Å, `PERIODIC XYZ` |
| Coordinates (Å) | O (0,0,0); H (0.758602, 0, ±0.504284); O (3,0,0); H (3.758602, 0, ±0.504284) |
| Basis | `DZVP-MOLOPT-SR-GTH` |
| PP | O `GTH-PBE-q6`, H `GTH-PBE-q1` → **16** valence e⁻ (`n_occ = 8`) |
| XC | PBE |
| Electronic k mesh | MONKHORST-PACK **1×1×2**, time-reversal folded |
| Dispersion | free-atom XDM, QE NC-PBE \(a_1=0.4283\), \(a_2=2.4690\) Å, rcut 50 Å |
| Cutoff | 120 Ry (E_disp is geometry-only — cutoff does not enter the pins) |
| Strain step | \(h = 2\times10^{-3}\) |

## Pins

Source: offline **dual pure-Python** free-atom XDM lattice sum
(`tools/cp2k_oracle/inputs/gpw-xdm-kpoint-stress/pin_xdm_strain.py`, python3
stdlib only; independent implementation sharing only the cited free-atom
data: CRC 88th Ed. α, homoatomic \(C_6/C_8/C_{10}\) from Yan et al. PRA
**54**, 2824 and Chu & Dalgarno JCP **121**, 4083). Never launched at ctest
time. Output lines (2026-08-01 re-run; geometry-only bit-identical to the
`gpw-xdm-stress` pin set):

```
E_disp unstrained            = -2.1148157612879029e-03 Ha
E_disp x +h              = -2.1110849131940963e-03 Ha
E_disp x -h              = -2.1185431863446984e-03 Ha
sigma_XDM_xx (FD, dual-py)  = -1.5989612853e-07 Ha/Bohr3
E_disp y +h              = -2.1147774136190569e-03 Ha
E_disp y -h              = -2.1148546246483736e-03 Ha
sigma_XDM_yy (FD, dual-py)  = -1.6553087314e-09 Ha/Bohr3
E_disp z +h              = -2.1146664855098696e-03 Ha
E_disp z -h              = -2.1149651549478578e-03 Ha
sigma_XDM_zz (FD, dual-py)  = -6.4031024180e-09 Ha/Bohr3
```

Provenance cross-check at pin time: the unstrained value **bit-reproduces**
the shipped `gpw-xdm-ice` periodic pin (−2.1148157612879029×10⁻³ Ha).

## Tolerances

| Check | Tol |
| ----- | --- |
| B \(\lvert E_{\mathrm{disp}} - \mathrm{pin}\rvert\) (6 strained cells, direct `xdm_pbe` + pin Å→Bohr) | 1×10⁻¹² Ha (deterministic geometry-only sum) |
| A \(\lvert E_{\mathrm{disp}} - \mathrm{pin}\rvert\) (product multi-k SCF via Frame) | 1×10⁻⁸ Ha (sibling `gpw-xdm-ice` / `gpw-xdm-stress` tier) |
| C \(\sigma_{\mathrm{diag}}\) analytic vs multi-SCF FD | 2×10⁻⁵ Ha/Bohr³ (sibling dispersion-stress tier) |
| C \(\lvert\sigma\rvert\) nonzero | > 1×10⁻¹⁰ |
| D \(\lvert(\sigma_{\mathrm{xdm}}-\sigma_{\mathrm{plain}}) - \sigma^{\mathrm{XDM}}_{\mathrm{dual\text{-}py}}\rvert\) | 1×10⁻⁸ Ha/Bohr³ |
| E off-diagonals (orthorhombic) | exactly 0 |
| E `kStress` unit conversion | 1×10⁻⁹ |
| Product wall | ≤ 60 s |

## Tolerance justification

- **B 1e−12 Ha:** the in-tree `xdm_pbe` kernel and the dual pure-Python sum
  are two implementations of the same deterministic geometry-only formula on
  shared cited data, evaluated at the pin Å→Bohr constant
  (`1/0.52917720859`); measured residual ~1e−18.
- **A 1e−8 Ha:** product path converts Frame Å geometry with
  `units::kAngstromInBohr` (CODATA 2006), which differs from the pin constant
  at ~4e−9 relative. Same product-path tier as `gpw-xdm-stress`.
- **C 2e−5 Ha/Bohr³:** analytic fixed-\(P_k\) vs multi-SCF FD differ by the
  frozen density-matrix approximation plus SCF residual at `tol = 1e-6`, not
  by the dispersion treatment — same tier as `gpw-mbd-kpoint-stress` /
  `gpw-xdm-stress`.
- **D 1e−8 Ha/Bohr³:** both sides are central FDs at the same \(h\) of the
  same dispersion energy (XDM does not enter the Fock operator, so the
  electronic parts cancel); residual also absorbs the Frame vs pin Å→Bohr
  constant difference on the pin side.

## Perf gate

CP2K 2026.1 stock has no XDM pair potential, so no fair oracle wall exists
for a multi-k XDM stress run: **`perf_exempt`** — same policy as
`gpw-xdm-stress` / `gpw-xdm-forces` / `gpw-mbd-kpoint-stress`.

## Measured results (2026-08-01, first green)

`reg_gpw-xdm-kpoint-stress` (OMP 8, epyc login node), 11 SCF iters,
TR-folded nk = 1 from MP 1×1×2:

| Gate | Tol | Measured |
| ---- | --- | -------- |
| B \(E_{\mathrm{disp}}\) under strain vs dual-py (6 cells) | 1e−12 Ha | **8.67e−19** |
| A \(E_{\mathrm{disp}}\) multi-k vs Γ pin | 1e−8 Ha | **8.62e−12** |
| C \(\sigma_{\mathrm{diag}}\) analytic vs multi-SCF FD | 2e−5 Ha/Bohr³ | **1.37e−09** |
| D shift vs dual-py FD term | 1e−8 Ha/Bohr³ | **1.10e−15** |
| E off-diagonals | exactly 0 | **0** |
| E `kStress` conversion | 1e−9 | exact |
| Product wall | ≤ 60 s | **17.8 s** (FD twin 103.8 s; not product budget) |

Product analytic diagonals [Ha/Bohr³]: xx −3.621e−05, yy −8.570e−05,
zz −3.089e−06. XDM shifts: xx −1.59896e−07, yy −1.655e−09,
zz −6.403e−09 Ha/Bohr³ — matching the dual-python FD terms per axis.

## Notes

- Sibling XDM products: `gpw-xdm-ice` (energy), `gpw-xdm-forces` (ionic
  forces), `gpw-xdm-stress` (Γ stress). Sibling multi-k dispersion stress:
  `gpw-mbd-kpoint-stress`, `gpw-ts-kpoint-stress`.

# Reference — GPW analytic (fixed-\(P\)) cell stress on isolated water

## Method

Closed-shell Γ-point GPW (Quickstep): Kohn–Sham in contracted Gaussians;
Hartree/XC/local-PP on a uniform grid. **Product cell stress** is the
Hellmann–Feynman fixed-density-matrix path:

1. One SCF at the unstrained cell → density matrix \(P\).
2. Central finite difference of the **non-self-consistent** total energy
   \(E[P;\varepsilon]\) under orthorhombic diagonal strain at fixed fractional
   coordinates (rebuild \(T\), \(V_{\mathrm{pp}}\), grid, Poisson, XC; do **not**
   re-converge SCF).
3. \(\sigma_{aa} = -\Omega^{-1}\partial E/\partial\varepsilon_{aa}\)
   (CP2K `STRESS_TENSOR` sign; E Hartree, \(\Omega\) Bohr³).

This is the GPW analogue of plane-wave Nielsen–Martin product stress in this
tree (`pw_stress`). Multi-SCF strain-FD remains `gpw_stress_fd` (oracle).

Shear (xy/xz/yz) requires a triclinic GPW cell and is **0** in v1.

### Included terms (this product)

- Kinetic \(\mathrm{Tr}(P\,T(\varepsilon))\)
- Pseudopotential \(\mathrm{Tr}(P\,V_{\mathrm{pp}}(\varepsilon))\) — nonlocal always;
  local short-range either analytic per-pair or grid \(V_{\mathrm{loc}}^{\mathrm{SR}}\)
  matching `local_pp_on_grid` (same flag as SCF / analytic forces)
- Electrostatics \(E_{\mathrm{es}}[\rho_P - n_c]\) (MT or periodic FFT)
- XC \(E_{\mathrm{xc}}[\rho_P]\) (PADE / PBE / PBEsol grid path)
- Optional Grimme D3 / D3(BJ) geometry-only energy under the same strain

### Deferred

- Closed-form term-by-term Pulay AO stress tables without fixed-\(P\) energy FD
- Off-diagonal / triclinic shear
- UKS, k-point GPW, DFT+U, D4, hybrid XC stress
- Absolute pin to CP2K `STRESS_TENSOR ANALYTICAL` on nonperiodic MT (known to
  disagree grossly with NUMERICAL on this twin — see `gpw-stress-water`)

Atomiverse path: `cpu::gpw::GPWEvaluator` with `compute_stress = true`,
`stress_use_fd = false` (default) → Frame `kStress` [eV/Å³]; compare via
`last_stress_ha_bohr3()` [Ha/Bohr³]. Oracle: free `gpw_stress_fd` or
`stress_use_fd = true`.

## System

Same twin as `gpw-stress-water`: isolated H₂O, CP2K getting-started geometry
(Å), cubic cell 10 Å, Martyna–Tuckerman / `PERIODIC NONE`.

| Atom | x | y | z |
| ---- | - | - | - |
| O | 5.0000 | 5.0000 | 5.0000 |
| H | 5.7586 | 5.0000 | 5.5043 |
| H | 4.2414 | 5.0000 | 5.5043 |

| Knob | Value |
| ---- | ----- |
| XC | PBE |
| Basis | DZVP-MOLOPT-GTH |
| PP | GTH-PBE-q6 (O), GTH-PBE-q1 (H) |
| CUTOFF | 400 Ry |
| SCF | `tol=1e-7` |
| Electrostatics | MT / nonperiodic |
| Stress product | analytic fixed-\(P\), \(h=2\times 10^{-3}\) |
| Stress oracle | multi-SCF strain-FD, same \(h\) |
| `local_pp_on_grid` | true |

## Gates

| Check | Tolerance |
| ----- | --------- |
| \(\max_{\mathrm{diag}}\|\sigma^{\mathrm{an}}-\sigma^{\mathrm{FD}}\|\) | **1×10⁻⁵** Ha/Bohr³ |
| \(\|E - E_{\mathrm{cp2k}}\|\) | **1×10⁻³** Ha |
| Off-diagonal product | exactly 0 (soft 1e-8) |

Offline CP2K energy (FORCE_EVAL QS, unperturbed):  
\(E = -17.212337150586624\) Hartree  
(CP2K 2026.1; same pin as `gpw-stress-water` / `gpw-forces-water`.)

CP2K NUMERICAL stress absolute pin remains on `reg_gpw-stress-water` (product
default may now be analytic; residual vs NUMERICAL stays in that gate).

## Perf note

Analytic product cost class is **1 SCF + 6 fixed-\(P\) energy assemblies**
(vs **6 full SCFs** for multi-SCF FD diagonals). Regression runs both for the
dual gate; wall is dominated by the FD twin. Fair product-only wall is the
analytic Evaluate pass alone.

| Metric | Note |
| ------ | ---- |
| perf_exempt | false for product-only class; dual-gate wall higher |
| Fair twin | analytic vs multi-SCF FD (same engine) — no CP2K ANALYTICAL wall |

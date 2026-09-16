# Reference — water-dimer GPW free-atom XDM ionic forces

## Method

**Exchange-hole dipole moment (XDM)** free-atom pairwise dispersion ionic
forces on top of Γ-point **GPW** (PBE). Same water dimer as `gpw-xdm-ice` /
`gpw-ts-ice`. Free-atom multipole moments and polarizabilities — no Hirshfeld
volume scaling (foothold; density-dependent XDM is out of scope).

Product path:

1. `xdm_pbe` evaluates \(E_{\mathrm{disp}}\) matching the dual pure-Python
   lattice sum (`gpw-xdm-ice` pin).
2. Forces \(F_A = -\partial E_{\mathrm{disp}}/\partial R_A\) [Ha/Bohr] from the
   same geometry-only kernel (analytic; dual-gated by central FD, \(h=10^{-5}\)
   Bohr).
3. `assemble_gpw_forces` with `GpwDft::xdm = true` **adds** these forces to the
   electronic Hellmann–Feynman / Pulay terms (no longer energy-only).

Public API: `cpu::gpw::GpwDft::xdm = true` (and `GPWEvaluator::xdm`) with
`compute_forces=true` / `gpw_forces_*`.

PBE BJ damping: QE NC-PBE \(a_1=0.4283\), \(a_2=2.4690\) Å. Free-atom
\(\alpha,M_1,M_2,M_3\) from in-tree `basis_data/xdm_tables.h`. Periodic
lattice sum: pure real-space, rcut **50 Å**, `pair_image_counts`.

## Reference values (hard gates)

| Quantity | Source | Value |
| -------- | ------ | ----- |
| **E_disp (periodic)** | dual pure-Python / `gpw-xdm-ice` | **−2.1148157612879029×10⁻³ Ha** |
| **E_disp (isolated)** | dual pure-Python | **−2.1020435998516220×10⁻³ Ha** |
| **Force dual-gate** | product \(F\) vs independent central FD \(h=10^{-5}\) Bohr | max\|ΔF\| ≤ **1×10⁻⁶** Ha/Bohr |
| Translation | \(\|\sum_A F_A\|\) | ≤ **1×10⁻⁸** Ha/Bohr |

### Offline force sample (isolated free-atom, product kernel)

| Atom | \(F_x\) | \(F_y\) | \(F_z\) [Ha/Bohr] |
| ---- | ------- | ------- | ----------------- |
| O0 | +1.6875829936003448×10⁻⁴ | 0 | 0 |
| H0 | +6.9736686381575340×10⁻⁵ | 0 | −1.3172272218329521×10⁻⁵ |
| H1 | +6.9736686381575340×10⁻⁵ | 0 | +1.3172272218329521×10⁻⁵ |
| O1 | −1.1378113807647229×10⁻⁴ | 0 | 0 |
| H2 | −9.7225267023356434×10⁻⁵ | 0 | −1.5448899346549854×10⁻⁵ |
| H3 | −9.7225267023356434×10⁻⁵ | 0 | +1.5448899346549854×10⁻⁵ |

Cross-check: dual pure-Python analytic gradient agrees with ATV `xdm_pbe`
forces to \(\sim 10^{-16}\) Ha/Bohr (offline — never in ctest).

### Why not CP2K QS?

**CP2K 2026.1** stock `PAIR_POTENTIAL TYPE` is only
`DFTD2|DFTD3|DFTD3(BJ)|DFTD4` — there is **no** stock `TYPE XDM`. Hard pin is
**dual pure-Python energy + force dual-gate**, same policy as `gpw-xdm-ice` /
`gpw-mbd-forces`. Twin notes under `tools/cp2k_oracle/inputs/gpw-xdm-forces/`.

## Geometry

| Atom | x [Å] | y [Å] | z [Å] |
| ---- | ----- | ----- | ----- |
| O | 0.000000 | 0.000000 | 0.000000 |
| H | 0.758602 | 0.000000 | 0.504284 |
| H | 0.758602 | 0.000000 | −0.504284 |
| O | 3.000000 | 0.000000 | 0.000000 |
| H | 3.758602 | 0.000000 | 0.504284 |
| H | 3.758602 | 0.000000 | −0.504284 |

Cell (SCF only): **ABC 12.0 12.0 12.0 Å**, **PERIODIC XYZ**.
Isolated force pin uses the same nuclear coordinates without lattice images.
Periodic E_disp pin uses rcut 50 Å.

Basis / PP (GPW short SCF; E_disp and F_XDM are basis-independent):
`DZVP-MOLOPT-SR-GTH` + `GTH-PBE-q1` (H) / `GTH-PBE-q6` (O).

## Tolerance

| Check | Tol |
| ----- | --- |
| \|E_disp − pin\| | **1×10⁻⁸** Ha |
| max\|F_product − F_fd\| | **1×10⁻⁶** Ha/Bohr |
| \|Σ F\| | **1×10⁻⁸** Ha/Bohr |
| max\|F − F_dual_python sample\| | **1×10⁻¹⁰** Ha/Bohr |

## Perf gate

XDM evaluation is \(O(N^2\times N_{\mathrm{images}})\) over a 6-atom cell
(microseconds). Full GPW SCF wall dominates if present.
**perf_exempt = true**: primary deliverable is E_disp + force dual-gate parity
(same policy as `gpw-xdm-ice` / `gpw-mbd-forces`). No fair CP2K ENERGY_FORCE
wall exists without stock XDM.

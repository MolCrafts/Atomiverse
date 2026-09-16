# Reference — periodic water-dimer GPW free-atom XDM dispersion energy

## Method

**Exchange-hole dipole moment (XDM)** pairwise dispersion as an additive
correction on top of Γ-point **GPW** (PBE), free-atom multipole moments and
polarizabilities (free-atom limit of XDM — no Hirshfeld volume scaling).
Ice-relevant H-bonded water dimer in a periodic 12 Å cell (same geometry as
`gpw-ts-ice` / `gpw-pbe-d3bj-ice`).

Public API: `cpu::gpw::GpwDft::xdm = true` (and `GPWEvaluator::xdm`).
Dispersion energy is reported on `GpwResult::e_disp` [Hartree] and folded into
`e_total`.

Formula (Becke & Johnson, *J. Chem. Phys.* **127**, 154108 (2007);
postg / QE `xdm_dispersion` combination rules + BJ damping):

\[
\mathrm{fac}=\frac{\alpha_i\alpha_j}{M_{1i}\alpha_j+M_{1j}\alpha_i},\quad
C_6=\mathrm{fac}\,M_{1i}M_{1j},
\]

\[
C_8=\tfrac{3}{2}\,\mathrm{fac}\,(M_{1i}M_{2j}+M_{2i}M_{1j}),
\]

\[
C_{10}=2\,\mathrm{fac}\,(M_{1i}M_{3j}+M_{3i}M_{1j})
+\tfrac{21}{5}\,\mathrm{fac}\,M_{2i}M_{2j},
\]

\[
R_c=\frac{1}{3}\Bigl(\sqrt{C_8/C_6}+(C_{10}/C_6)^{1/4}+\sqrt{C_{10}/C_8}\Bigr),
\quad R_{\mathrm{vdW}}=a_1 R_c+a_2,
\]

\[
E_{\mathrm{disp}}=-\sum_{i\le j}\sum_{\mathbf T}'
\Biggl[
\frac{C_6}{R^6+R_{\mathrm{vdW}}^6}
+\frac{C_8}{R^8+R_{\mathrm{vdW}}^8}
+\frac{C_{10}}{R^{10}+R_{\mathrm{vdW}}^{10}}
\Biggr].
\]

PBE damping: **QE NC-PBE** \(a_1=0.4283\), \(a_2=2.4690\) Å
(Otero-de-la-Roza & Johnson / postg `xdm.param`). Free-atom \(\alpha\) from
CRC Handbook 88th Ed. (same as postg/QE). Free-atom \(M_1,M_2,M_3\) inverted
from literature homo \(C_n\) under the XDM combination rules:

| Z | \(\alpha\) [Bohr³] | \(M_1\) | \(M_2\) | \(M_3\) | Source \(C_n\) |
| - | ------------------ | ------- | ------- | ------- | -------------- |
| H | 4.49979 | 2.88859 | 18.4304 | 241.636 | Yan et al. PRA 54, 2824 |
| O | 5.41216 | 5.74631 | 28.9471 | 262.617 | Chu–Dalgarno JCP 121, 4083 (\(C_6\)); consistent \(C_8,C_{10}\) |

Pair cutoff: **50 Å** pure real-space orthorhombic lattice sum
(`pair_image_counts`). Density-dependent Hirshfeld/BR moments are **out of
scope** (free-atom foothold; same policy as free-atom TS).

## Reference value (hard gate)

**E_disp (free-atom XDM, PBE BJ, periodic water dimer, 12 Å cell, rcut 50 Å) =
−2.114815761287903×10⁻³ Ha**

Cross-checks (offline — never in ctest):

| Source | Geometry | E_disp [Ha] |
| ------ | -------- | ----------- |
| Dual pure-Python lattice sum (this REFERENCE) | isolated | **−2.102043599851622×10⁻³** |
| Dual pure-Python lattice sum | periodic 12 Å, rcut 50 Å | **−2.114815761287903×10⁻³** |
| Atomiverse `xdm_pbe` | same | matches to \(\sim 10^{-16}\) Ha |

### Why not CP2K QS?

**CP2K 2026.1** `PAIR_POTENTIAL TYPE` is only
`DFTD2|DFTD3|DFTD3(BJ)|DFTD4` — there is **no** stock `TYPE XDM`. The
canonical offline XDM oracles are postg / critic2 / QE (density-dependent
full XDM). This slug ships the **free-atom** foothold with a dual pure-Python
pin (same pattern as free-atom TS when stock CP2K is unavailable). Twin note
retained under `tools/cp2k_oracle/inputs/gpw-xdm-ice/`.

## Geometry

| Atom | x [Å] | y [Å] | z [Å] |
| ---- | ----- | ----- | ----- |
| O | 0.000000 | 0.000000 | 0.000000 |
| H | 0.758602 | 0.000000 | 0.504284 |
| H | 0.758602 | 0.000000 | −0.504284 |
| O | 3.000000 | 0.000000 | 0.000000 |
| H | 3.758602 | 0.000000 | 0.504284 |
| H | 3.758602 | 0.000000 | −0.504284 |

Cell: **ABC 12.0 12.0 12.0 Å**, **PERIODIC XYZ**.

Basis / PP (GPW regression; E_disp is basis-independent):
`DZVP-MOLOPT-SR-GTH` + `GTH-PBE-q1` (H) / `GTH-PBE-q6` (O).

## Tolerance

**1 × 10⁻⁸ Ha** on E_disp (regression). Unit tests use tighter checks on the
standalone `xdm_pbe` kernel.

## What is checked

`GpwDft` with `xdm = true` on the pin geometry returns
`last_result().e_disp` within 1e-8 Ha of −2.114815761287903e-03 Ha.
Unit tests also pin isolated dual-Python and analytic forces vs FD.

## Perf gate

XDM evaluation is \(O(N^2\times N_{\mathrm{images}})\) over a 6-atom cell
(microseconds). Full GPW SCF wall is dominated by the electronic structure.
**perf_exempt = true**: primary deliverable is E_disp pin parity; full twin SCF
ratio is unfair for a geometry-only hard gate (same policy as
`gpw-ts-ice` / `gpw-pbe-d3bj-ice`).

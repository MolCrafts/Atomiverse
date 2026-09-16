# Reference — periodic water-dimer GPW PBE-TS dispersion energy

## Method

**Tkatchenko–Scheffler (TS)** pairwise dispersion as an additive correction on
top of Γ-point **GPW** (PBE), free-atom volume ratios \(V/V_{\mathrm{free}}=1\)
(free-atom limit of TS). Ice-relevant H-bonded water dimer in a periodic
12 Å cell (same geometry as `gpw-pbe-d3bj-ice` / `gpw-d4-water`).

Public API: `cpu::gpw::GpwDft::ts = true` (and `GPWEvaluator::ts`).
Dispersion energy is reported on `GpwResult::e_disp` [Hartree] and folded into
`e_total`.

Formula (Tkatchenko & Scheffler, *Phys. Rev. Lett.* **102**, 073005 (2009);
libMBD `get_ts_energy` real-space path):

\[
E_{\mathrm{disp}} = -\sum_{i\le j}\sum_{\mathbf T}'
  C_{6,ij}\,f_{\mathrm{damp}}(R)\,R^{-6}
\]

\[
C_{6,ij}=\frac{2\,C_{6,i}C_{6,j}}
  {(\alpha_j/\alpha_i)\,C_{6,i}+(\alpha_i/\alpha_j)\,C_{6,j}},\quad
R_{0,ij}=R_i+R_j
\]

\[
f_{\mathrm{damp}}=\bigl(1+\exp[-d\,(R/(s_R R_{0,ij})-1)]\bigr)^{-1}
\]

PBE damping: \(s_R=0.94\), \(d=20\). Free-atom \(\alpha_0\), \(C_6\), \(R_{\mathrm{vdW}}\)
from libMBD `ts_vdw_params` (in-tree `basis_data/ts_tables.h`):

| Z | \(\alpha_0\) [Bohr³] | \(C_6\) [Ha·Bohr⁶] | \(R_{\mathrm{vdW}}\) [Bohr] |
| - | -------------------- | ------------------ | -------------------------- |
| H | 4.5 | 6.5 | 3.1 |
| O | 5.4 | 15.6 | 3.19 |

Pair cutoff: **50 Å = 94.561… Bohr** (pure real-space orthorhombic lattice sum;
no Ewald). Hirshfeld density scaling and MBD are **out of scope** (see
`gpw-mbd-ice`).

## Reference value (hard gate)

**E_disp (PBE-TS free-atom, periodic water dimer, 12 Å cell, rcut 50 Å) =
−6.099630773386013×10⁻⁴ Ha**

Cross-checks (offline — never in ctest):

| Source | Geometry | E_disp [Ha] |
| ------ | -------- | ----------- |
| **libMBD** C API `cmbd_ts_energy` (git `88069c6`, built 2026-07-25) | isolated | **−5.9764872295816794×10⁻⁴** |
| Dual pure-Python lattice sum (same formula/tables) | isolated | −5.9764872295816815×10⁻⁴ |
| **libMBD** Fortran `get_ts_energy`, pure-RS branch (`ewald_on = false`, `dipole_cutoff = 50 Å`; driver `tools/cp2k_oracle/inputs/gpw-ts-stress/pin_ts_strain.f90`, 2026-07-31) | periodic 12 Å, rcut 50 Å | **−6.0996307733860822×10⁻⁴** |
| Atomiverse `ts_pbe` (pair_image_counts shell) | periodic 12 Å, rcut 50 Å | −6.099630773386013×10⁻⁴ |
| libMBD with Ewald (`ewald_cutoff_scaling=2.5`) | periodic 12 Å | −6.100334450881942×10⁻⁴ |

Isolated libMBD matches Atomiverse to \({\sim}10^{-19}\) Ha (unit-test hard pin).
The periodic pure-RS pin is anchored by the genuine libMBD no-Ewald run
(agreement 7×10⁻¹⁸ Ha, `gpw-ts-stress` pin driver). Periodic pure-RS and
libMBD Ewald differ by \({\sim}7\times10^{-8}\) Ha (long-range \(1/R^6\) tail +
Ewald vs hard cutoff) — documented; regression hard gate is the pure-RS value
above.

### Why not CP2K QS?

**CP2K 2026.1** `PAIR_POTENTIAL TYPE` is only
`DFTD2|DFTD3|DFTD3(BJ)|DFTD4` — there is **no** stock `TYPE TS`. libMBD is the
canonical offline TS/MBD oracle (also the backend FHI-aims / QE / DFTB+ use).
Twin input retained under `tools/cp2k_oracle/inputs/gpw-ts-ice/` for hosts that
may later expose TS via an external library.

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

## Oracle: libMBD (offline — never launched from ctest)

```bash
# Build libmbd (once); pin isolated free-atom TS on the water dimer
# (see tools/cp2k_oracle/inputs/gpw-ts-ice/pin_libmbd_ts.c for a sketch).
# Isolated: Dispersion energy = -5.9764872295816794e-04 Eh
# Periodic pure-RS pin: dual Python lattice sum (REFERENCE.md table).
```

## Tolerance

**1 × 10⁻⁸ Ha** on E_disp (regression). Unit tests use tighter checks on the
standalone `ts_pbe` kernel.

## What is checked

`GpwDft` with `ts = true` on the pin geometry returns
`last_result().e_disp` within 1e-8 Ha of −6.099630773386013e-04 Ha.
Unit tests also pin isolated libMBD and analytic forces vs FD.

## Perf gate

TS evaluation is \(O(N^2\times N_{\mathrm{images}})\) over a 6-atom cell
(microseconds). Full GPW SCF wall is dominated by the electronic structure.
**perf_exempt = true**: primary deliverable is E_disp pin parity; full twin SCF
ratio is unfair for a geometry-only hard gate (same policy as
`gpw-pbe-d3bj-ice` / `gpw-d4-water`).

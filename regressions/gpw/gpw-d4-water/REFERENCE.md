# Reference — periodic water-dimer GPW PBE-D4 dispersion energy

## Method

Grimme / Caldeweyher **DFT-D4** with Becke–Johnson (rational) damping as an
additive two-body correction on top of Γ-point **GPW** (PBE), for a periodic
water dimer cell (same geometry as `gpw-pbe-d3bj-ice`).

Public API: `cpu::gpw::GpwDft::d4 = true` (and `GPWEvaluator::d4`).
Dispersion energy is reported on `GpwResult::e_disp` [Hartree] and folded into
`e_total`. Three-body ATM is **off** (`s9 = 0`).

PBE-D4 parameters (SAW190103 BJ set; dftd4 `pbe-D4`):

| param | value |
| ----- | ----- |
| \(s_6\) | 1.0 |
| \(s_8\) | 0.95948085 |
| \(a_1\) | 0.38574991 |
| \(a_2\) | 4.80688534 Bohr |

Cutoffs (dftd4 defaults): CN 30 Bohr, disp2 60 Bohr. Periodic EEQ uses an
Ewald-split A-matrix (multicharge-style).

## Reference value (hard gate)

**E_disp (PBE-D4, two-body, periodic water dimer, 12 Å cell) = −0.00099458842745846774 Ha**

Source: offline **dftd4 3.7.0** (`/software/.../dftd4/3.7.0-gfbf-2025b` on
Arrhenius), `dftd4 -f pbe --mbdscale 0.0` on the POSCAR twin below.

Printed:

```text
Dispersion energy:      -9.9458842745847E-04 Eh
```

JSON: `{"version":"3.7.0","energy":-9.9458842745846774E-04}`.

### Why not CP2K QS?

**CP2K 2026.1** (`2026.1-gmpflf-2026.03`) on Arrhenius was built **without**
the DFTD4 library: `TYPE DFTD4` aborts with
`CP2K build without DFTD4` (`qs_dispersion_d4.F`). The twin input is retained
under `tools/cp2k_oracle/inputs/gpw-d4-water/` for hosts that link libdftd4.
The hard pin is therefore **libdftd4 / dftd4 3.7.0**, which is the same
backend CP2K uses when DFTD4 is enabled (CP2K banner would report
`DFT-D4(Version 3.6.0)`-class).

### Context (not the hard gate)

Isolated (non-periodic) two-body on the same nuclear geometry:
**−9.8816586433966×10⁻⁴ Ha** (dftd4 3.7.0, `--mbdscale 0`).

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

Basis / PP for the GPW regression (E_disp is basis-independent):
`DZVP-MOLOPT-SR-GTH` + `GTH-PBE-q1` (H) / `GTH-PBE-q6` (O).

## Oracle: dftd4 3.7.0 (offline — never launched from ctest)

```bash
export PATH=/software/sse2/el9_epyc9005/easybuild/prefix/software/dftd4/3.7.0-gfbf-2025b/bin:$PATH
export LD_LIBRARY_PATH=/software/sse2/el9_epyc9005/easybuild/prefix/software/dftd4/3.7.0-gfbf-2025b/lib:$LD_LIBRARY_PATH
# + FlexiBLAS / GCCcore from buildenv-gcc module
cd tools/cp2k_oracle/inputs/gpw-d4-water
dftd4 -f pbe --mbdscale 0.0 --json pin.json POSCAR
# Dispersion energy: -9.9458842745847E-04 Eh
```

## Tolerance

**1 × 10⁻⁸ Ha** on E_disp. Matching geometry, PBE-D4 parameters, EEQ2019,
and lattice sums are expected to track dftd4 to nano-Hartree on this cell.

## What is checked

`GpwDft` with `d4 = true` on the pin geometry returns
`last_result().e_disp` within 1e-8 Ha of −0.00099458842745846774 Ha.
Unit tests also check the isolated two-body pin on the same nuclei.

## Perf gate

D4 evaluation is \(O(N^2 \times N_{\mathrm{images}})\) over a 6-atom cell
(microseconds–milliseconds). Full GPW SCF wall is dominated by the electronic
structure, not D4. **perf_exempt = true** for this slug: reason — primary
deliverable is E_disp pin parity; CP2K QS wall is unavailable (no DFTD4 build)
and a full twin SCF ratio is unfair for a geometry-only hard gate (same policy
as `gpw-pbe-d3bj-ice`).

# Reference — periodic water-dimer GPW PBE-D4 + ATM (s9)

## Method

Grimme / Caldeweyher **DFT-D4** with Becke–Johnson two-body damping **plus**
the Axilrod–Teller–Muto **three-body** term at \(s_9 = 1\) (dftd4
`pbe-D4-ATM`), as an additive correction on top of Γ-point **GPW** (PBE).

Same geometry and cell as `gpw-d4-water` (two-body-only twin).

Public API:

```cpp
GpwDft gpw;
gpw.d4 = true;
gpw.d4_atm = true;   // s9 = 1
```

`GPWEvaluator::d4_atm` mirrors the solver flag. Dispersion energy is on
`GpwResult::e_disp` [Hartree] and folded into `e_total`.

PBE-D4 parameters (SAW190103 BJ set; dftd4 `pbe-D4-ATM`):

| param | value |
| ----- | ----- |
| \(s_6\) | 1.0 |
| \(s_8\) | 0.95948085 |
| \(s_9\) | **1.0** (ATM on) |
| \(a_1\) | 0.38574991 |
| \(a_2\) | 4.80688534 Bohr |
| alp (ATM) | 16 |

Cutoffs (dftd4 defaults): CN 30 Bohr, disp2 60 Bohr, **disp3 40 Bohr**.
ATM uses **neutral-charge** C6 reweight (`q = 0`), matching simple-dftd4
`disp.f90`.

## Reference value (hard gate)

**E_disp (PBE-D4 + ATM, periodic water dimer, 12 Å cell) = −9.9554475834240288×10⁻⁴ Ha**

Source: offline **dftd4 3.7.0**
(`/software/.../dftd4/3.7.0-gfbf-2025b` on Arrhenius),

```text
dftd4 -f pbe --json pin_periodic_atm.json POSCAR
# Rational (Becke-Johnson) damping: pbe-D4-ATM
#   s9         1.0000
# Dispersion energy:      -9.9554475834240E-04 Eh
```

JSON: `{"version":"3.7.0","energy":-9.9554475834240288E-04}`.

Context (not the hard gate):

| Case | E_disp [Ha] |
| ---- | ----------- |
| Periodic two-body only (`--mbdscale 0`) | −9.9458842745846774×10⁻⁴ |
| Isolated ATM (`s9=1`) | −9.8819741533958208×10⁻⁴ |

ATM contribution on this cell ≈ −9.56×10⁻⁷ Ha.

### Why not CP2K QS?

**CP2K 2026.1** on Arrhenius was built **without** the DFTD4 library
(`CP2K build without DFTD4`). Hard pin is **libdftd4 / dftd4 3.7.0**, the
same backend CP2K uses when DFTD4 is linked. Twin input retained under
`tools/cp2k_oracle/inputs/gpw-d4-water/` for hosts that link libdftd4;
ATM-specific pin files live under `tools/cp2k_oracle/inputs/gpw-d4-atm/`.

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
cd tools/cp2k_oracle/inputs/gpw-d4-atm
dftd4 -f pbe --json pin_periodic_atm.json POSCAR
# Dispersion energy: -9.9554475834240E-04 Eh
```

## Tolerance

**1 × 10⁻⁸ Ha** on E_disp.

## What is checked

`GpwDft` with `d4 = true` and `d4_atm = true` returns
`last_result().e_disp` within 1e-8 Ha of −9.9554475834240288×10⁻⁴ Ha.
Unit tests also pin isolated + periodic free-function ATM energies.

## Perf gate

ATM is \(O(N^3 \times N_{\mathrm{images}}^2)\) on a 6-atom cell (still
milliseconds). Full GPW SCF wall is dominated by the electronic structure.
**perf_exempt = true**: primary deliverable is E_disp pin parity; CP2K QS
wall is unavailable (no DFTD4 build); same policy as `gpw-d4-water`.

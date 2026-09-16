# Reference — GPW DFT-D3 three-body ATM (C9)

## Method

Grimme **DFT-D3** (zero damping, PBE) with the **Axilrod–Teller–Muto
three-body C9** term, as an additive correction on Γ-point **GPW**.

Public API:

- `cpu::gpw::GpwDft::d3 = true`
- `cpu::gpw::GpwDft::d3_atm = true`  (CP2K `CALCULATE_C9_TERM .TRUE.`)

Dispersion energy is on `GpwResult::e_disp` [Hartree] and folded into
`e_total`. Implementation: `dftd3_pbe(..., calculate_c9=true)` matching
CP2K `qs_dispersion_d3` (`s9 = s6`, CN-dependent \(C_6\), damping
`damping_d3(rabc, r0, 4/3, 16)`).

## Reference value (hard gate)

**E_disp (PBE-D3 + ATM, isolated water dimer) = −0.00081414118193 Ha**

Source: offline **CP2K 2026.1** (`2026.1-gmpflf-2026.03` on Arrhenius), twin
`tools/cp2k_oracle/inputs/gpw-d3-atm/water_dimer_d3_atm.inp`.

```text
Dispersion energy:                                           -0.00081414118193
```

Two-body-only baseline on the same geometry (ATM off):
**−0.00081399213205 Ha** (existing unit-test pin). ATM Δ ≈ **−1.49×10⁻⁷ Ha**.

## Geometry

| Atom | x [Å] | y [Å] | z [Å] |
| ---- | ----- | ----- | ----- |
| O | 0.000000 | 0.000000 | 0.000000 |
| H | 0.758602 | 0.000000 | 0.504284 |
| H | 0.758602 | 0.000000 | −0.504284 |
| O | 3.000000 | 0.000000 | 0.000000 |
| H | 3.758602 | 0.000000 | 0.504284 |
| H | 3.758602 | 0.000000 | −0.504284 |

Cell: **ABC 12.0 12.0 12.0 Å**, **PERIODIC NONE** (isolated / MT).

Basis / PP: `DZVP-MOLOPT-SR-GTH` + `GTH-PBE-q1` (H) / `GTH-PBE-q6` (O).

## Oracle: CP2K 2026.1 (offline — never launched from ctest)

```bash
export CP2KROOT=/software/sse2/el9_epyc9005/easybuild/prefix/software/CP2K/2026.1-gmpflf-2026.03
export CP2K_DATA_DIR=$CP2KROOT/share/cp2k/data
export PATH=$CP2KROOT/bin:$PATH
cd tools/cp2k_oracle/inputs/gpw-d3-atm
mpirun -np 1 cp2k.psmp -i water_dimer_d3_atm.inp -o water_dimer_d3_atm.out
# rg 'Dispersion energy' water_dimer_d3_atm.out
```

CP2K wall (this pin, 1 rank × 4 OMP): **T_cp2k ≈ 10.8 s** (TIMING `CP2K`
TOTAL TIME).

## Tolerance

**1 × 10⁻⁹ Ha** on E_disp.

## What is checked

`GpwDft` with `d3 = true` and `d3_atm = true` returns
`last_result().e_disp` within 1e-9 Ha of −0.00081414118193 Ha. Unit tests
check analytic ATM forces vs FD and that default `d3_atm = false` keeps the
two-body pin.

## Perf gate

**perf_exempt = true** — primary deliverable is E_disp pin parity; D3 ATM on
6 atoms is microseconds; CP2K wall is full QS (~11 s) while the hard-gate
regression only needs a short SCF to materialize geometry-only `e_disp`.

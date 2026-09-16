# Reference — electrode total charge observer (parallel-plate foothold)

## oracle

```text
oracle: lammps
oracle_version: LAMMPS (30 Mar 2026 - Development - patch_30Mar2026-1074-g7f680de296)
```

Binary used offline: `/home/jicli594/.local/x86_64/bin/lmp` (no ELECTRODE package
required; pin uses core `compute reduce sum` on group charge).

## Physical system

Parallel-plate **constant-charge control** toy for E1 observe (map §1 / §8):

| Quantity | Value |
| --- | --- |
| Box | orthorhombic \(20 \times 10 \times 10\) Å |
| Left wall | type 1, 4 atoms, \(q_i = +0.125\) e |
| Right wall | type 2, 4 atoms, \(q_i = -0.125\) e |
| Electrolyte placeholders | type 3, 2 atoms, \(q_i = +1\) e (excluded from electrode sums) |
| Ensemble | none (`run 0`) — pure charge accounting |

Analytic totals:

\[
Q_L = 4 \times 0.125 = +0.5\,\mathrm{e},\qquad
Q_R = 4 \times (-0.125) = -0.5\,\mathrm{e}.
\]

## Offline pin procedure

Input twin: `tools/metalwalls_oracle/inputs/lammps/const_pot_charge_observer.in`

```bash
lmp -in tools/metalwalls_oracle/inputs/lammps/const_pot_charge_observer.in
```

Recorded screen lines:

```text
PIN_QL 0.5
PIN_QR -0.5
```

Thermo step 0: `c_ql = 0.5`, `c_qr = -0.5`.

## Assertion

Atomiverse `cpu::ElectrodeChargeObserver({1, 2})` on the same geometry/charges:

| Observable | Pin | Tol |
| --- | ---: | ---: |
| \(Q_L\) [e] | \(+0.5\) | \(10^{-12}\) |
| \(Q_R\) [e] | \(-0.5\) | \(10^{-12}\) |
| \(Q_L + Q_R\) [e] | \(0\) | \(10^{-12}\) |

Tolerance is machine-eps scale: the pin is an exact finite sum of fixed charges
(no force loop, no k-space). Catches wrong type selection, ghost double-counting,
or unit mistakes; not a golden of Atomiverse-only output.

## Why LAMMPS (not MetalWalls)

Summing group charge is standard LAMMPS (`variable atom q` + `compute reduce sum`).
The ELECTRODE package is not installed on this oracle binary; for **this**
observer the reduce-sum path is the correct LAMMPS twin. MetalWalls remains the
oracle for MW-only BO/CG/MaZe charge *solves* (other E1 slugs).

## Perf

```text
perf_exempt: true
perf_exempt_reason: >
  Micro-observer is a single O(N) sum over 10 atoms. Fair wall vs LAMMPS is
  dominated by binary startup (~0.3 s for `run 0` with no pair/kspace), not
  the reduce itself. Not a force-loop twin; ratio would not measure product
  MD throughput.
```

When a longer MD+electrode fair bench exists (const-pot-md-nvt-short), re-record
\(T_{\mathrm{atv}}/T_{\mathrm{oracle}}\) there.

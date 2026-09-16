# Reference — periodic water-dimer GPW PBE-D3(BJ) dispersion energy

## Method

Grimme **DFT-D3 with Becke–Johnson (rational) damping** as an additive
correction on top of Γ-point **GPW** (PBE), for a periodic water dimer cell
(ice-relevant H-bonded pair; not a full ice Ih supercell).

Public API: `cpu::gpw::GpwDft::d3bj = true` (and `GPWEvaluator::d3bj`).
Dispersion energy is reported on `GpwResult::e_disp` [Hartree] and folded into
`e_total`. Coordination numbers use the **CP2K** form (tanh long-range factor)
so E_disp matches offline `TYPE DFTD3(BJ)`. Molecular `cpu::scf::D3BJ` keeps
the Grimme CN (PySCF twin) and is unchanged.

PBE-D3(BJ) parameters (Grimme et al., *J. Comput. Chem.* **32**, 1456 (2011);
CP2K `REFERENCE_FUNCTIONAL PBE`):

| param | value |
| ----- | ----- |
| \(s_6\) | 1.0 |
| \(s_8\) | 0.7875 |
| \(a_1\) | 0.4289 |
| \(a_2\) | 4.4407 Bohr |

Pair cutoff in Atomiverse: **2 × R_CUTOFF 20 Å ≈ 75.59 Bohr**. CP2K prints
`Cutoff Radius [Bohr]: 37.79` for `R_CUTOFF 20` (Å→Bohr); the energy lattice
sum matches the pin when the engine rcut is **twice** that (same 2× convention
as the zero-damping GPW D3 pin). Three-body ATM off.

## Reference value (hard gate)

**E_disp (PBE-D3(BJ), periodic water dimer, 12 Å cell) = −0.00137627769481 Ha**

Source: offline **CP2K 2026.1** (`git:5e54ba2`, module build
`2026.1-gmpflf-2026.03` on Arrhenius), twin input
`tools/cp2k_oracle/inputs/gpw-pbe-d3bj-ice/water_dimer_d3bj.inp`.

Printed line:

```text
Dispersion energy:                                           -0.00137627769481
```

### Context (not the hard gate)

CP2K total QS energy for the same twin (PBE, DZVP-MOLOPT-SR-GTH, GTH-PBE,
CUTOFF 300 Ry, OT): **−34.33962825563683 Ha**. Absolute GPW e_tot parity is
out of scope for this slug (OT/grid/DIIS differences); the regression asserts
**E_disp** and that `e_total` includes it.

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

Basis / PP (CP2K twin and Atomiverse tables): `DZVP-MOLOPT-SR-GTH` +
`GTH-PBE-q1` (H) / `GTH-PBE-q6` (O).

## Oracle: CP2K 2026.1 (offline — never launched from ctest)

```bash
export CP2KROOT=/software/sse2/el9_epyc9005/easybuild/prefix/software/CP2K/2026.1-gmpflf-2026.03
export CP2K_DATA_DIR=$CP2KROOT/share/cp2k/data
export PATH=$CP2KROOT/bin:$MPICH/bin:$PATH   # MPICH 4.3.2 + dep libs on Arrhenius
cd tools/cp2k_oracle/inputs/gpw-pbe-d3bj-ice
mpirun -np 1 cp2k.psmp -i water_dimer_d3bj.inp -o water_dimer_d3bj.out
# rg 'Dispersion energy' water_dimer_d3bj.out
```

CP2K wall (this pin run, 1 rank × 4 OMP): **T_cp2k ≈ 13.4 s** total (TIMING
table `CP2K` TOTAL TIME).

## Tolerance

**1 × 10⁻⁸ Ha** on E_disp. With matching geometry, PBE-BJ parameters, CP2K CN,
and rcut 40 Bohr, Atomiverse is expected to track CP2K dispersion to
nano-Hartree.

## What is checked

`GpwDft` with `d3bj = true` on the pin geometry returns
`last_result().e_disp` within 1e-8 Ha of −0.00137627769481 Ha. Unit tests also
check analytic BJ forces vs FD and that `e_total(d3bj) − e_total(off) = e_disp`.

## Perf gate

D3 evaluation itself is \(O(N^2)\) over a 6-atom cell (microseconds). Full GPW
SCF wall is dominated by the electronic structure, not D3. Fair ratio for a
**dispersion-only** microbenchmark vs CP2K is not meaningful; full twin SCF
ratio depends on cutoff/OT choices. **perf_exempt = true** for this slug:
reason — primary deliverable is E_disp pin parity; CP2K wall includes full QS
(~13 s) while the hard-gate regression only needs a short SCF to materialize
`e_disp` (geometry-only). Document re-measure if a strict full-SCF twin is
required later.

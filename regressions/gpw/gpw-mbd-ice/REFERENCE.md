# Reference — water-dimer GPW PBE-MBD@rsSCS dispersion energy

## Method

**Many-body dispersion MBD@rsSCS** as an additive correction on top of Γ-point
**GPW** (PBE), free-atom volume ratios \(V/V_{\mathrm{free}}=1\). Ice-relevant
H-bonded water dimer (same geometry as `gpw-ts-ice` / `gpw-pbe-d3bj-ice`).

Public API: `cpu::gpw::GpwDft::mbd = true` (and `GPWEvaluator::mbd`).
Dispersion energy is reported on `GpwResult::e_disp` [Hartree] and folded into
`e_total`.

Product path evaluates **isolated** free-atom MBD@rsSCS (geometry-only; basis
independent). Periodic Ewald + k-mesh MBD is a follow-on.

Formula (Ambrosetti et al., *J. Chem. Phys.* **140**, 18A508 (2014); libMBD
Hermann et al., *J. Chem. Phys.* **159**, 174802 (2023)):

1. Range-separated self-consistent screening of dynamical polarizabilities
   (Fermi × Gaussian dipole tensor, \(\beta\) damping).
2. Long-range MBD Hamiltonian diagonalization → zero-point energy difference.

PBE: \(\beta = 0.83\), Fermi steepness \(a = 6\). Free-atom \(\alpha_0\), \(C_6\),
\(R_{\mathrm{vdW}}\) from libMBD `ts_vdw_params` (in-tree `basis_data/ts_tables.h`):

| Z | \(\alpha_0\) [Bohr³] | \(C_6\) [Ha·Bohr⁶] | \(R_{\mathrm{vdW}}\) [Bohr] |
| - | -------------------- | ------------------ | -------------------------- |
| H | 4.5 | 6.5 | 3.1 |
| O | 5.4 | 15.6 | 3.19 |

## Reference value (hard gate)

**E_disp (PBE-MBD@rsSCS free-atom, isolated water dimer) =
−1.4928804171407961×10⁻³ Ha**

Cross-checks (offline — never in ctest):

| Source | Geometry | E_disp [Ha] |
| ------ | -------- | ----------- |
| **libMBD** Fortran `cmbd_mbd_scs_energy` (git `88069c6`) | isolated, ATV Å→Bohr | **−1.4928804171407961×10⁻³** |
| pure-Python `pymbd.mbd_energy` | isolated, ATV Å→Bohr | −1.4928804171425725×10⁻³ |
| pure-Python (libmbd Å→Bohr 0.529177249) | isolated | −1.4928806007885598×10⁻³ |

Difference pure-Python vs Fortran \(\sim 2\times10^{-15}\) Ha (same ATV conversion).

### Why not CP2K QS?

**CP2K 2026.1** stock `PAIR_POTENTIAL TYPE` is only
`DFTD2|DFTD3|DFTD3(BJ)|DFTD4` — MBD is via external libMBD when linked. Hard
pin is **libMBD**, same policy as `gpw-ts-ice`. Twin notes under
`tools/cp2k_oracle/inputs/gpw-mbd-ice/`.

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
MBD pin is isolated free-atom on the same nuclear coordinates.

Basis / PP (GPW regression; E_disp is basis-independent):
`DZVP-MOLOPT-SR-GTH` + `GTH-PBE-q1` (H) / `GTH-PBE-q6` (O).

## Tolerance

**1 × 10⁻⁸ Ha** on E_disp (regression). Unit tests use **1e-12 Ha** on the
standalone `mbd_rsscs_pbe` kernel.

## Perf gate

MBD evaluation is \(O(N^3)\) eigensolve of a \(3N\times 3N\) matrix on a 6-atom
system (microseconds). Full GPW SCF wall is dominated by the electronic
structure. **perf_exempt = true**: primary deliverable is E_disp pin parity
(same policy as `gpw-ts-ice` / `gpw-pbe-d3bj-ice` / `gpw-d4-water`).

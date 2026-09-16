# Reference — water-dimer GPW PBE periodic Ewald + k-mesh MBD@rsSCS

## Method

**Periodic many-body dispersion MBD@rsSCS** (Ewald dipole tensors + Brillouin-zone
k-mesh average) as an additive correction on top of Γ-point **GPW** (PBE), free-atom
volume ratios \(V/V_{\mathrm{free}}=1\). Ice-relevant H-bonded water dimer
geometry (same nuclei as `gpw-mbd-ice` / `gpw-ts-ice`) in a **periodic** cubic cell.

Public API:

```cpp
GpwDft gpw;
gpw.mbd = true;
gpw.mbd_k_mesh[0] = gpw.mbd_k_mesh[1] = gpw.mbd_k_mesh[2] = 2; // periodic
```

Default `mbd_k_mesh={0,0,0}` keeps free-atom **isolated** MBD (`reg_gpw-mbd-ice`).
Dispersion energy on `GpwResult::e_disp` [Hartree], folded into `e_total`.

Formula (Ambrosetti et al., *J. Chem. Phys.* **140**, 18A508 (2014); libMBD
Hermann et al., *J. Chem. Phys.* **159**, 174802 (2023)):

1. Range-separated self-consistent screening of dynamical polarizabilities
   over lattice images (Fermi × Gaussian dipole tensor, no Ewald on this step).
2. Long-range MBD Hamiltonian with **Ewald** dipole tensor, eigenvalues averaged
   over a shifted k-mesh (libMBD `get_kpts`, shift \(1/2\)).

PBE: \(\beta = 0.83\), Fermi steepness \(a = 6\). Free-atom \(\alpha_0\), \(C_6\),
\(R_{\mathrm{vdW}}\) from libMBD `ts_vdw_params` (in-tree `basis_data/ts_tables.h`).

## Reference value (hard gate)

**E_disp (PBE-MBD@rsSCS free-atom volumes, periodic 12 Å cube, k = 2×2×2) =
−1.4965720704989138×10⁻³ Ha**

Cross-checks (offline — never in ctest):

| Source | Geometry | E_disp [Ha] |
| ------ | -------- | ----------- |
| **pure-Python pymbd** `mbd_energy` (libmbd tree, ATV Å→Bohr) | periodic, k=2×2×2 | **−1.4965720704989138×10⁻³** |
| same, k=1×1×1 | periodic | −1.4474832766229184×10⁻³ |
| same, k=3×3×3 | periodic | −1.5008181948612531×10⁻³ |
| isolated free-atom (no lattice) | `reg_gpw-mbd-ice` / libMBD Fortran | −1.4928804171407961×10⁻³ |

Difference periodic (2×2×2) vs isolated \(\sim 3.7\times10^{-6}\) Ha on this box
(weak image coupling at 12 Å).

### Why not CP2K QS?

**CP2K 2026.1** stock `PAIR_POTENTIAL TYPE` is only
`DFTD2|DFTD3|DFTD3(BJ)|DFTD4` — MBD is via external libMBD when linked. Hard
pin is **pure-Python pymbd** (same algorithm as libMBD Fortran), same policy as
`gpw-mbd-ice`. Twin notes under `tools/cp2k_oracle/inputs/gpw-mbd-periodic/`.

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
MBD uses the same cell with **k-mesh 2×2×2** (libMBD shift-½).

Basis / PP (GPW regression; E_disp is basis-independent):
`DZVP-MOLOPT-SR-GTH` + `GTH-PBE-q1` (H) / `GTH-PBE-q6` (O).

## Tolerance

**1 × 10⁻⁸ Ha** on E_disp (regression). Unit tests use **1e-10 Ha** on the
standalone `mbd_rsscs_pbe` periodic kernel.

## Perf gate

MBD evaluation is \(O(N_k \cdot (3N)^3)\) on a 6-atom system (milliseconds).
Full GPW SCF wall is dominated by the electronic structure. **perf_exempt =
true**: primary deliverable is E_disp pin parity (same policy as
`gpw-mbd-ice` / `gpw-ts-ice` / `gpw-pbe-d3bj-ice`).

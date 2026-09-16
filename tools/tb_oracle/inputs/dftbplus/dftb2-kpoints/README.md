# DFTB+ twin — `dftb2-kpoints`

Offline pin input for family oracle **DFTB+**. Not used by ctest.

## Geometry

1-D carbon dimer cell (same as `dftb2-periodic-gamma`):

| Atom | x [Å] | y [Å] | z [Å] |
|------|-------|-------|-------|
| C    | 0.00  | 0.00  | 0.00  |
| C    | 1.40  | 0.00  | 0.00  |

Cell: orthorhombic \(L_x=2.8\), \(L_y=L_z=12.0\) Å.

## Method flags

- SCC-DFTB2 (`SCC = Yes`), no ThirdOrder
- MaxAngularMomentum C = p
- K-mesh: Monkhorst–Pack **4×1×1** (SupercellFolding below), Γ-centered
  CP2K/DFTB+ convention \(\kappa_d=(2r-N-1)/(2N)\)
- Time-reversal symmetry on (DFTB+ default for real H)

## How to run offline

```bash
# requires dftb+ binary + SK tables (e.g. mio-1-1)
cd tools/tb_oracle/inputs/dftbplus/dftb2-kpoints
dftb+ > dftb.out
# total energy in detailed.out — convert H → eV if needed
```

## Pin policy

- Family numerical oracle for periodic multi-k SCC-DFTB2 remains **DFTB+**.
- ATV ships **mio-style sketch** SK (`basis_data/dftb_mio_tables.h`), not a
  full DFTB.org mio-1-1 dump. Absolute energy in the regression is therefore
  the ATV Hamiltonian twin (±0.05 eV) until SK fidelity closes.
- Method: Elstner et al., Phys. Rev. B **58**, 7260 (1998).

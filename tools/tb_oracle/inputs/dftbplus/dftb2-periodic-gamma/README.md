# DFTB+ twin — `dftb2-periodic-gamma`

Offline pin input for family oracle **DFTB+**. Not used by ctest.

## Geometry

1-D carbon dimer cell: two C atoms at (0,0,0) and (1.4,0,0) Å;
orthorhombic cell \(L_x=2.8\), \(L_y=L_z=12.0\) Å. Matches
`regressions/semiempirical/dftb2-periodic-gamma/REFERENCE.md`.

## How to run offline

```bash
# requires dftb+ binary + SK tables (e.g. mio-1-1)
cd tools/tb_oracle/inputs/dftbplus/dftb2-periodic-gamma
dftb+ > dftb.out
# total energy in detailed.out / band.out — convert H → eV if needed
```

## Pin policy

- Family numerical oracle for periodic SCC-DFTB2 remains **DFTB+**.
- ATV ships **mio-style sketch** SK (`basis_data/dftb_mio_tables.h`), not a
  full DFTB.org mio-1-1 dump. Absolute energy in the regression is therefore
  the ATV Hamiltonian twin (±0.05 eV) until SK fidelity
  (`dftb2-dftbplus-energy-pin`) lands bit-identical tables.
- Method citations: Elstner et al., Phys. Rev. B **58**, 7260 (1998).

# DFTB+ twin — `dftb2-periodic-forces`

Offline geometry twin for lattice-image-consistent Γ-PBC nuclear forces under
SCC-DFTB2 (MioSketch). Pin host had no DFTB+ binary; absolute force identity
is **not** claimed. The regression hard-codes analytic vs re-SCC FD of the ATV
lattice-summed energy (see `REFERENCE.md`).

## Geometry

Off-eq 1-D C dimer cell:

| Atom | x [Å] | y | z |
|------|-------|---|---|
| C | 0.00 | 0 | 0 |
| C | 1.20 | 0 | 0 |

Cell: orthorhombic 2.8 × 12 × 12 Å.

## Reproduce with DFTB+ (when binary available)

```bash
cd tools/tb_oracle/inputs/dftbplus/dftb2-periodic-forces
# point SlaterKosterFiles to a mio-style set (ATV tables are sketches)
dftb+ > dftb.out
# parse forces; compare module structure only unless bit-identical SK land
```

Family oracle: **DFTB+** only (never OpenMOPAC for DFTB).

# H2O SCC-DFTB2 (mio-1-1 static SK)

```text
oracle: dftb+
oracle_version: 24.1 + mio-1-1
```

## Method

Self-consistent-charge DFTB2 (Elstner et al., *Phys. Rev. B* **58**, 7260
(1998), DOI [10.1103/PhysRevB.58.7260](https://doi.org/10.1103/PhysRevB.58.7260)).

## Data

In-tree static tables for **all ordered H/C/N/O pairs** transcribed from
DFTB.org / dftbparams mio-1-1 `*.skf` into `basis_data/dftb_mio_sk_tables.h`
(generator: `tools/gen_dftb_mio_tables.py`). Interpolation = DFTB+
`polyInterU1` + `distFudge` (`internal/mio_sk_table.h`). No runtime `.skf`
parser.

## Geometry

Water (Å), cubic box 20 Å (isolated):

| Atom | x    | y    | z   |
|------|------|------|-----|
| O    | 0.00 | 0.00 | 0.0 |
| H    | 0.96 | 0.00 | 0.0 |
| H    |−0.24 | 0.93 | 0.0 |

Same twin as `tools/tb_oracle/inputs/dftbplus/h2o_mio_scc/`.

## Pins / gates

| Quantity | Value | Source |
| -------- | ----- | ------ |
| Total energy (DFTB+) | **−110.9590 eV** (−4.0776668457 Ha) | DFTB+ 24.1, re-run 2026-07-29 |
| ATV absolute tol | **0.05 eV** | covers residual ~0.037 eV |
| Force vs FD | **1×10⁻⁵ eV/Å** | analytic SCC gradient |

## Residual vs DFTB+ (~0.037 eV)

Order-of-magnitude and sign match after real mio tables (sketch era was
~−94.5 eV). Remaining meV-scale gap is under investigation (single-shell
Hubbard U from Us vs DFTB+ shell handling, mixer, onsite SPE). H₂ still
matches DFTB+ to ~1e-8 eV (`dftb2-dftbplus-energy-pin`).

## Re-run oracle

```bash
source tools/tb_oracle/setup_dftbplus_env.sh
tools/tb_oracle/run_dftbplus_pin.sh tools/tb_oracle/inputs/dftbplus/h2o_mio_scc mio
```

# Offline DFTB+ pin — H₂O / mio-1-1 / SCC-DFTB2

```text
oracle: dftb+
oracle_version: 24.1
```

## Geometry

Same as `reg_h2o_dftb2_mio` / `reg_dftb3-water-mio` (Å):

| Atom | x | y | z |
| ---- | - | - | - |
| O | 0 | 0 | 0 |
| H | 0.96 | 0 | 0 |
| H | −0.24 | 0.93 | 0 |

## Pin (re-run 2026-07-29 on this host)

| Quantity | Value |
| -------- | ----- |
| Total energy | **−4.0776668457 Ha = −110.9590 eV** |

```bash
source tools/tb_oracle/setup_dftbplus_env.sh
tools/tb_oracle/run_dftbplus_pin.sh tools/tb_oracle/inputs/dftbplus/h2o_mio_scc mio
```

## ATV status (2026-07-29)

Full H/C/N/O mio-1-1 tables ship in `basis_data/dftb_mio_sk_tables.h`.
ATV SCC-DFTB2 water at this geometry is **~−110.996 eV** (|Δ|≈0.037 eV vs
this pin). `reg_h2o_dftb2_mio` gates |E − DFTB+| < 0.05 eV. Residual is
open (Hubbard-U / mixer detail), not a missing SK table.

See `.claude/notes/dftb-vs-dftbplus-cleanup.md` §1.

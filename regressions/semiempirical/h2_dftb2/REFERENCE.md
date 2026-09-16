# h2_dftb2

```text
oracle: dftb+
oracle_version: 24.1
family_oracle: dftb+
```

## Method

SCC-DFTB2 (`DFTB2`, default MioSketch) on H₂. H–H uses transcribed **mio-1-1**
radial grids (`internal/mio_hh_table.h`); Δq ≡ 0 so γ sketch issues do not
enter. Absolute energy identity with offline DFTB+ 24.1.

## Geometry

H₂, \(r = 0.74\) Å (same as `dftb2-dftbplus-energy-pin` / `tools/tb_oracle/inputs/dftbplus/h2_mio_scc`).

## Pin

| Quantity | Value | Units |
| -------- | ----- | ----- |
| Total energy (DFTB+ 24.1, re-verified 2026-07-29) | **−18.36635052** | eV |

Also gates: \(E(0.74)\) finite and \(\|E(0.74)-E(3.0)\| > 10^{-6}\) eV.

## Twin

`tools/tb_oracle/inputs/dftbplus/h2_mio_scc/` + `source tools/tb_oracle/setup_dftbplus_env.sh`.

# arch-gpw-driver-paths

## Intent

Architecture-only gate for multi-path GPW DIIS Driver stages after
`arch-gpw-driver-node` (Γ RKS DIIS) and `arch-gpw-driver-paths` (Γ UKS +
multi-k RKS/UKS DIIS via path-local stage Ops).

## Oracle

`oracle: harness` — shape assertions only (no energy pin). Energy twins:

- `regressions/gpw/gpw-uks-o2-gamma` (Γ UKS DIIS)
- `regressions/gpw/gpw-kpoints-si` (multi-k RKS DIIS)

## Acceptance

- `GpwDft` is `cpu::Driver<gpw::Phase>` with 5 SCF slots
- Stage default bindings match `ks_build / mix / diagonalize / density / check`
- Path-local Ops construct (no full SCF in this binary)

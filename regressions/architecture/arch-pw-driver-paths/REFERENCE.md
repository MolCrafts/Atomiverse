# Reference — arch-pw-driver-paths

## Intent

Architecture gate: after migrating collinear spin and metallic-smearing
residual free loops onto the same `cpu::Driver<pw::Phase>` stages as
restricted fixed-occupation (`arch-pw-driver-node`), the public multi-path
surface and stage bindings remain correct.

## Lever

Path-local `StepFn` bodies on
`veff → diagonalize → density → mix → check` + shared `drive_stage_loop_`
+ Criterion `PwConverged`. Density-mix order (ρ-space after density rebuild).

## Oracle

`oracle: harness` — shape assertions only (no multi-iter SCF in this binary).
Energy / occupation twins live in product regs:

| Path | Twin |
| --- | --- |
| Restricted metallic smear | `regressions/pw/pw-smearing-al` |
| Collinear fixed-occ / spin-GGA | `regressions/pw/pw-spin-polarized-o2`, `pw-spin-gga` |
| Collinear + metallic | `regressions/pw/pw-uks-fe-smearing`, metallic bands/DOS |

## Acceptance

- `PW` is `cpu::Driver<pw::Phase>` with 5 SCF slots
- Stage default bindings match `veff / diagonalize / density / mix / check`
- Public multi-path members (`spin_polarized`, `smearing`, …) constructible
- No free-loop reimplementation of SCF in the regression

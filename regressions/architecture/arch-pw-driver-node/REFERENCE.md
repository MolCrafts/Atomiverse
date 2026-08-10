# Reference — arch-pw-driver-node

## Intent

Architecture gate: after reshaping `cpu::pw::PW` into
`cpu::Driver<pw::Phase>`, the **restricted fixed-occupation** public surface
(`build(frame)` / `run(frame)` / members) still reproduces the offline
ABINIT-class Si LDA energy band used by `regressions/pw/pw-lda-si`.

## Lever

Self-consistency ownership moves from free `for (iter)` bodies onto Driver
stages (`veff → diagonalize → density → mix → check`) + Criterion
`PwConverged`. Density-mix order (Pulay on ρ), not H-space DIIS.

## Reference value

| Quantity | Value |
| -------- | ----- |
| Oracle | ABINIT-class PseudoDojo deltafactor `Si.djrepo` key `"8.0"` |
| \(E_\mathrm{min}\) | **−16.916316 Ry/cell** |
| Twin band | \([E_\mathrm{min},\, E_\mathrm{min}+1.50]\) Ry (incomplete ecut/k window) |
| Settings | \(a=5.43\) Å, \(E_\mathrm{cut}=15\) Ry, \(k=2^3\), LSDA, fixed occ |

Same provenance as [`regressions/pw/pw-lda-si/REFERENCE.md`](../../pw/pw-lda-si/REFERENCE.md).
Pin is offline literature / ABINIT-class — **not** ATV self-golden.

## Public API only

`cpu::pw::PW{…}.build(frame)` / `.run(frame)` — Frame in; `PWResult` out.
No `*Config`, no free-loop reimplementation of the SCF in the regression.

## Residual (not this gate)

Collinear spin and metallic smearing remain free-loop residual until a
path-migration wave.

# Collinear spin-DFTB2 analytic nuclear forces — O₂

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `dftb2-spin-analytic-forces` |
| Regression | `regressions/semiempirical/dftb2-spin-analytic-forces/` |
| Public API | `cpu::semiempirical::DFTB2` (`multiplicity`, `with_forces`) |
| Track | TB / SE method loop wave 2 () |
| Roadmap | Spin-polarized DFTB analytic gradient (collinear open shell) |
| Family oracle | **DFTB+** (formula / module boundary) |

## Summary

**Closed-form nuclear forces** for **collinear unrestricted** SCC-DFTB2 on
gas-phase triplet O₂ at a stretched bond. Deepens the wave-1 spin energy
foothold (`spin-dftb2-o2`) and the restricted analytic path
(`dftb2-analytic-forces`): open-shell forces no longer re-SCC finite-difference
the full energy. Public surface unchanged — `multiplicity` + `with_forces` on
`DFTB2`. **No live DFTB+ in ctest.**

## Method

Dual \(\alpha/\beta\) densities at the SCC solution; gradient is
Hellmann–Feynman + Pulay + fixed-\(\Delta q\) SCC \(\gamma\) + repulsive, with
spin-dependent onsite shifts on \(\partial S\). See `REFERENCE.md` for the
formula and gates.

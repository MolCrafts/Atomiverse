# se-analytic-hessian

**oracle:** literature (structural)
**oracle_version:** n/a — FD Hessian identity + linear H₂ mode count

## Method

`cpu::vib::Hessian` + `cpu::semiempirical::DFTB2` (MioSketch, forces on) on H₂
at 0.74 Å. Unblocks SE-loop VIB policy for electronic SE force sources.

## Gates

- Hessian 6×6, finite, symmetric to 1e-8 eV/Å²
- Linear rotor → 1 vibrational mode, ω > 100 cm⁻¹, n_imaginary = 0

## Gaps

Absolute cm⁻¹ not compared to DFTB+/experiment (foothold is API + structural).

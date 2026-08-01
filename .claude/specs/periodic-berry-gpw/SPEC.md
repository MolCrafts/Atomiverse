# Spec — periodic-berry-gpw

## Goal

King–Smith–Vanderbilt Berry-phase formal polarization on GPW
(`cpu::gpw::BerryPolarization`), matching the PW product
(`cpu::pw::BerryPolarization`) and offline-pinnable against CP2K Quickstep
`&MOMENTS PERIODIC`.

## Public API

- `cpu::gpw::BerryPolarization` with members `k_perp`, `k_string`, `directions`
- `compute(GpwDft&, atoms)` → `BerryPolarizationResult`
- Free helpers `phase_from_link_dets`, `wrap_centered` (unit-testable)
- No `*Config`, no factories

## Algorithm

1. For each requested lattice direction, build string-major k-mesh
   (\(N_\perp^2\) strings × \(J\) points).
2. Multi-k RKS/UKS SCF on that explicit mesh (`k_time_reversal=false`).
3. Link determinants \(M=C^\dagger S_\mathrm{cross} C\) with phased AO
   overlaps (same \(e^{-ib\cdot r}\) as multi-k MLWF).
4. \(\varphi=\langle\operatorname{Im}\ln\prod\det M\rangle\), closed-shell
   \(p_\mathrm{el}=2\varphi/2\pi\) (quantum 2) or collinear spin quantum 1.
5. Ionic \(p_\mathrm{ion}=\sum Z^\mathrm{val}\tau\); total wrap.

## Acceptance

- Unit: string layout, phase product, wrap, argument gates.
- Regression `regressions/gpw/periodic-berry-gpw`: centrosymmetric Si
  half-quantum lattice + Born ASR \(Z^*=0\) under sublattice displace.
- Offline CP2K twin: `tools/cp2k_oracle/inputs/si_periodic_berry_gpw.inp`
  (fair wall \(T_\mathrm{atv}/T_\mathrm{cp2k}\le 10\)).

## Out of scope

Metallic smearing, non-orthorhombic cells, DFPT Born charges, CUDA.

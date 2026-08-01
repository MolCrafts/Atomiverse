# Spec — periodic-berry-gpw-uks

## Goal

King–Smith–Vanderbilt Berry-phase formal polarization on **collinear UKS**
GPW (`cpu::gpw::BerryPolarization` dual-channel string phases), productizing
the RKS pin `periodic-berry-gpw` and the PW collinear pin `berry-spin-si`.
Offline-pinnable against CP2K Quickstep `&MOMENTS PERIODIC` with `LSD` /
multiplicity 1.

## Public API (product — no surface change)

- `cpu::gpw::BerryPolarization` members `k_perp`, `k_string`, `directions`
- `compute(GpwDft&, atoms)` when `method.is_uks()` (`n_alpha`/`n_beta`)
- Closed-shell path unchanged; no `*Config`, no factories

## Algorithm

Already product-shaped on `dev`:

1. String-major multi-k mesh per lattice direction; TR folding off.
2. Multi-k **UKS** SCF on that explicit mesh (`GpwDft::run`).
3. Per-spin occupied link determinants
   \(M^\sigma=C^{\sigma\dagger}S_\mathrm{cross}C^\sigma\).
4. \(p^\mathrm{el}=(\varphi_\uparrow+\varphi_\downarrow)/2\pi\), wrap quantum 1.
5. Ionic \(p^\mathrm{ion}=\sum Z^\mathrm{val}\tau\) (spin-independent).

## Acceptance

- Unit: existing KSV layout/phase/wrap gates + UKS empty-manifold throw.
- Regression `regressions/gpw/periodic-berry-gpw-uks`: centrosymmetric Si
  half-quantum lattice + Born ASR \(Z^*=0\) on collinear UKS (M=1).
- Offline CP2K twin: `tools/cp2k_oracle/inputs/si_periodic_berry_gpw_uks.inp`
  (fair wall \(T_\mathrm{atv}/T_\mathrm{cp2k}\le 10\) on the matched ENERGY
  job vs one ATV string-mesh SCF+Berry direction).

## Out of scope

Metallic smearing, open-shell multiplicity ≠ 1 physics pins, non-orthorhombic
cells, DFPT Born charges, CUDA, PW-only paths.

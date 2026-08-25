---
title: IPEA-xTB FD Hessian via cpu::vib::Hessian
status: implemented
created: 2026-08-25
chain: tb-method-loop
slug: ipea-xtb-analytic-hessian
oracle: xtb
---

# IPEA-xTB FD Hessian via cpu::vib::Hessian

## Summary

Wire production `cpu::semiempirical::IPEA` (analytic \(\mathbf{F}=-\nabla E\)
already ships) into `cpu::vib::Hessian` as the `vib::Phase::force` evaluator
and pin the resulting finite-difference nuclear Hessian / vibrational
wavenumbers against an offline **xtb 6.7.1** IPEA `--hess` run on **CH4**
(`param_ipea` selected as GFN1). This is the IPEA instance of the
`se-analytic-hessian` / `gfn0-xtb-analytic-hessian` / `gfn-xtb-analytic-hessian`
VIB=ON exception — not a closed-form second-derivative kernel, not `libxtb`,
not GFN1/GFN2 (already shipped). Polar water is **not** this pin (0.5 eV
kernel hole vs xtb, named in `reg_ipea-xtb-small`).

## Domain basis

IPEA-xTB (Ásgeirsson, Bauer, Grimme, *Chem. Sci.* **8**, 4879 (2017), DOI
[10.1039/c7sc00601b](https://doi.org/10.1039/c7sc00601b)) is a GFN1
reparametrization for IPs/EAs. Energy and analytic \(G\) are product
(`reg_ipea-xtb-small`, `reg_ipea-xtb-forces`). `cpu::vib::Hessian` assembles
a 6N central difference of that analytic \(G\) (\(h=10^{-3}\) Å). xtb 6.7.1
`--hess` on IPEA prints an **analytical** Hessian (same xtb path as GFN1);
ATV stays FD of analytic \(G\), the same shape as GFN0/1/2 /
`se-analytic-hessian`. Frequencies follow mass-weighted diagonalization after
projecting translations/rotations (IUPAC-2013 `mass_amu(Z)`).

Oracle: **xtb** `--hess` with `param_ipea-xtb.txt` as `param_gfn1-xtb.txt`
(never DFTB+ / OpenMOPAC). `--vip`/`--vea` is **not** this pin (same-file
`ipea-xtb-vip`). Geometry is the tetrahedral CH4 of `reg_ipea-xtb-small`
(\(r(\mathrm{C{-}H})=1.09\) Å), which already matches xtb to 8.90 meV.

## Design

- Reuse `cpu::vib::Hessian` + `IPEA{with_forces=true}`. No new public type,
  no `*Config`, no `make_*` / `build_*`.
- IPEA sources unchanged except docs: analytic G already accumulates into
  `kAtom_f*` under `ctx.forces_requested`; Hessian already zeros
  accumulators per displacement.
- No closed-form \(\partial^2 E/\partial R_i\partial R_j\) in `ipea.cpp`.
- SE-only cmake **with** `-DATV_BUILD_VIB=ON` (this slug's documented
  exception). HF/MD/CUDA/MINIMIZER stay OFF. CLI=OFF.

## Files

- `tests/cpu/semiempirical/test_xtb_ipea_hessian.cpp` (wiring only; no
  6N `Hessian::run`)
- `regressions/semiempirical/ipea-xtb-analytic-hessian/` (new)
- `tools/tb_oracle/inputs/xtb/ch4_ipea_hess/` (new twin)
- `regressions/CMakeLists.txt` (`ATV_BUILD_VIB` guard, same as family H)
- `tests/CMakeLists.txt` new unit binary under SEMIEMPIRICAL+VIB
- `include/atomiverse/cpu/semiempirical/ipea.h` docs only

## Acceptance

- Unit: IPEA binds as a Semiempirical force kind. `Hessian::run` stays in
  `regressions/`. Existing `test_cpu_xtb_ipea` FD identity stays green.
- Regression: CH4 FD Hessian via `cpu::vib::Hessian` + IPEA vs offline
  xtb 6.7.1 IPEA `--hess` (`REFERENCE.md` pins ω and a Cartesian \(H_{ij}\)).
  Symmetric finite Hessian; nonlinear CH4 → 9 vib modes.
- `oracle: xtb` / `oracle_version: 6.7.1` on REFERENCE.md.
- Perf: \(T_{\mathrm{atv}}/T_{\mathrm{xtb}} \le 10\) or `perf_exempt`.
- Scoped `--target atomiverse test_cpu_xtb_ipea_hessian
  reg_ipea-xtb-analytic-hessian`.

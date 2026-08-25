---
title: GFN0-xTB FD Hessian via cpu::vib::Hessian
status: implemented
created: 2026-08-25
slug: gfn0-xtb-analytic-hessian
oracle: xtb
---

# GFN0-xTB FD Hessian via cpu::vib::Hessian

## Summary

Wire production `cpu::semiempirical::GFN0` (analytic \(\mathbf{F}=-\nabla E\)
already ships) into `cpu::vib::Hessian` as the `vib::Phase::force` evaluator
and pin the resulting finite-difference nuclear Hessian / vibrational
wavenumbers against an offline **xtb 6.7.1** `--gfn 0 --hess` run. This is
the GFN0 instance of the `se-analytic-hessian` VIB=ON exception — not a
closed-form second-derivative kernel, not GFN1/2, not `libxtb`.

## Domain basis

GFN0-xTB (Pracht, Caldeweyher, Ehlert, Grimme, ChemRxiv 2019,
DOI 10.26434/chemrxiv.8326202.v1) is non-SCC. Energy and analytic G are
product (`reg_gfn0-beyond-toy`, `reg_gfn0-xtb-forces`). xtb 6.7.1 `--hess`
assembles a numerical Hessian of that analytic G (same shape as
`cpu::vib::Hessian`: 6N central differences of \(\mathbf{F}\), \(h = 10^{-3}\) Å
on the ATV side). Frequencies follow mass-weighted diagonalization after
projecting translations/rotations.

Oracle: **xtb** `--gfn 0 --hess` (never DFTB+ / OpenMOPAC). Family Hessian
for GFN1/2 stays the later slug `gfn-xtb-analytic-hessian`.

## Design

- Reuse `cpu::vib::Hessian` + `cpu::semiempirical::GFN0{with_forces=true}`.
  No new public type, no `*Config`, no `make_*` / `build_*`.
- GFN0 source is unchanged except docs if needed: analytic G already
  accumulates into `kAtom_f*` under `ctx.forces_requested`; Hessian already
  zeros accumulators per displacement.
- No closed-form \(\partial^2 E/\partial R_i\partial R_j\) in `gfn0.cpp`.
- SE-only cmake **with** `-DATV_BUILD_VIB=ON` (this slug's documented
  exception). HF/MD/CUDA/MINIMIZER stay OFF.

## Files

- `tests/cpu/semiempirical/test_xtb_gfn0.cpp` (Hessian wiring / curvature
  identity; no 6N `Hessian::run` — that is e2e)
- `regressions/semiempirical/gfn0-xtb-analytic-hessian/` (new)
- `tools/tb_oracle/inputs/xtb/<mol>_gfn0_hess/` (new twin)
- `regressions/CMakeLists.txt` (`ATV_BUILD_VIB` guard, same as
  `se-analytic-hessian`)
- `tests/CMakeLists.txt` only if a new unit binary is required

## Acceptance

- Unit: GFN0 binds as a Semiempirical force kind; analytic \(F\) vs energy
  FD on off-eq H₂O stays green (existing `test_cpu_xtb_gfn0`). No
  `Hessian::run` in `tests/`.
- Regression: small-molecule FD Hessian via `cpu::vib::Hessian` + GFN0 vs
  offline xtb 6.7.1 `--gfn 0 --hess` (`REFERENCE.md` pins ω and/or a
  Cartesian \(H_{ij}\)). Symmetric finite Hessian; linear H₂ → 1 real mode
  or nonlinear H₂O → 3N−6 modes as classified.
- `oracle: xtb` / `oracle_version: 6.7.1` on REFERENCE.md.
- Perf: \(T_{\mathrm{atv}}/T_{\mathrm{xtb}} \le 10\) or `perf_exempt`.
- Scoped `--target atomiverse test_cpu_xtb_gfn0 reg_gfn0-xtb-analytic-hessian`.

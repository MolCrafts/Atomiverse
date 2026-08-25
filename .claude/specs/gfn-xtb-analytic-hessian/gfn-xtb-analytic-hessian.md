---
title: GFN1/GFN2-xTB FD Hessian via cpu::vib::Hessian
status: implemented
created: 2026-08-25
slug: gfn-xtb-analytic-hessian
oracle: xtb
---

# GFN1/GFN2-xTB FD Hessian via cpu::vib::Hessian

## Summary

Wire production `cpu::semiempirical::GFN1` and `GFN2` (analytic
\(\mathbf{F}=-\nabla E\) already ships) into `cpu::vib::Hessian` as the
`vib::Phase::force` evaluator and pin the resulting finite-difference
nuclear Hessian / vibrational wavenumbers against offline **xtb 6.7.1**
`--gfn 1 --hess` (primary) and `--gfn 2 --hess` (family close, with the
stated AES-in-Fock hole). This is the GFN1/2 instance of the
`se-analytic-hessian` / `gfn0-xtb-analytic-hessian` VIB=ON exception —
not a closed-form second-derivative kernel, not `libxtb`, not GFN0
(already shipped).

## Domain basis

GFN1-xTB (Grimme, Bannwarth, Shushkov, JCTC **13**, 1989 (2017),
DOI 10.1021/acs.jctc.7b00118) and GFN2-xTB (Bannwarth, Ehlert, Grimme,
JCTC **15**, 1652 (2019), DOI 10.1021/acs.jctc.8b01176) are SCC
tight-binding. Energy and analytic \(G\) are product
(`reg_gfn1-xtb-organic`, `reg_gfn1-xtb-forces`, `reg_gfn2-xtb-water`,
`reg_gfn2-xtb-forces`). `cpu::vib::Hessian` assembles a 6N central
difference of that analytic \(G\) (\(h=10^{-3}\) Å). xtb 6.7.1
`--gfn 1 --hess` / `--gfn 2 --hess` prints an **analytical** Hessian
(xtb log: “analytical hessian”); ATV stays FD of analytic \(G\), the
same shape as GFN0 / `se-analytic-hessian`. Frequencies follow
mass-weighted diagonalization after projecting translations/rotations.

Oracle: **xtb** `--hess` (never DFTB+ / OpenMOPAC). GFN0 FD Hessian
already ships (`reg_gfn0-xtb-analytic-hessian`). GFN2 AES remains
post-SCC (`gfn2-aes-fock`); the GFN2 ω pin carries that Fock hole.

## Design

- Reuse `cpu::vib::Hessian` + `GFN1{with_forces=true}` /
  `GFN2{with_forces=true}`. No new public type, no `*Config`, no
  `make_*` / `build_*`.
- GFN1/GFN2 sources unchanged except docs: analytic G already
  accumulates into `kAtom_f*` under `ctx.forces_requested`; Hessian
  already zeros accumulators per displacement.
- No closed-form \(\partial^2 E/\partial R_i\partial R_j\) in
  `gfn1.cpp` / `gfn2.cpp`.
- SE-only cmake **with** `-DATV_BUILD_VIB=ON` (this slug's documented
  exception). HF/MD/CUDA/MINIMIZER stay OFF. CLI=OFF.

## Files

- `tests/cpu/semiempirical/test_xtb_gfn_hessian.cpp` (wiring only; no
  6N `Hessian::run`)
- `regressions/semiempirical/gfn-xtb-analytic-hessian/` (new)
- `tools/tb_oracle/inputs/xtb/h2o_gfn1_hess/` (new twin)
- `tools/tb_oracle/inputs/xtb/h2o_gfn2_hess/` (new twin)
- `regressions/CMakeLists.txt` (`ATV_BUILD_VIB` guard, same as
  `gfn0-xtb-analytic-hessian`)
- `tests/CMakeLists.txt` new unit binary under SEMIEMPIRICAL+VIB

## Acceptance

- Unit: GFN1 and GFN2 bind as Semiempirical force kinds.
  `Hessian::run` stays in `regressions/`. Existing
  `test_cpu_xtb_gfn1` FD identity stays green.
- Regression: water FD Hessian via `cpu::vib::Hessian` + GFN1 vs
  offline xtb 6.7.1 `--gfn 1 --hess` (`REFERENCE.md` pins ω and a
  Cartesian \(H_{ij}\)). GFN2 same geometry vs `--gfn 2 --hess` with
  the stated AES-in-Fock hole. Symmetric finite Hessian; nonlinear
  H₂O → 3 real modes, \(n_{\mathrm{imag}}=0\).
- `oracle: xtb` / `oracle_version: 6.7.1` on REFERENCE.md.
- Perf: \(T_{\mathrm{atv}}/T_{\mathrm{xtb}} \le 10\) or `perf_exempt`.
- Scoped `--target atomiverse test_cpu_xtb_gfn_hessian
  reg_gfn-xtb-analytic-hessian`.

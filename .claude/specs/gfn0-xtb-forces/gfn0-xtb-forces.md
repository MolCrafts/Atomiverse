---
title: GFN0-xTB analytic nuclear forces
status: implemented
created: 2026-08-25
slug: gfn0-xtb-forces
oracle: xtb
---

# GFN0-xTB analytic nuclear forces

## Summary

Replace the GFN0 `with_forces` path (central finite difference of the total
energy, 6N extra EEQ+EHT solves) with the **closed-form nuclear gradient** of
the same Pracht 2019 Hamiltonian already shipped by `gfn0-beyond-toy`. Pin
Cartesian forces against an offline **xtb 6.7.1** `--gfn 0 --grad` run.
No `libxtb` wrap.

## Domain basis

GFN0-xTB (Pracht, Caldeweyher, Ehlert, Grimme, ChemRxiv 2019,
DOI 10.26434/chemrxiv.8326202.v1) is non-SCC. The energy

\[
E = E_{\mathrm{H0}} + E_{\mathrm{rep}} + E_{\mathrm{IES}} + E_{\mathrm{D4}} + E_{\mathrm{SRB}}
\]

is already product. The gradient is the total derivative of that sum, not a
re-solve:

| Term | Gradient |
| ---- | -------- |
| \(E_{\mathrm{H0}}\) | Hellmann–Feynman + Pulay: \(\mathrm{Tr}(P\,\partial H-\,W\,\partial S)\) plus CN/q chain through the onsite shifts (Pracht `build_dSH0`) |
| \(E_{\mathrm{rep}}\) | pair \(Z_{\mathrm{eff}} e^{-\alpha r^{3/2}}/r\) |
| \(E_{\mathrm{IES}}\) | explicit \(\partial J/\partial R\) at frozen EEQ \(q\), plus \(\partial x/\partial\mathrm{CN}\) |
| \(E_{\mathrm{D4}}\) | in-tree `cpu::scf::dftd4()` two-body analytic \(F=-\nabla E\) (GFN0 BJ scales, \(s_9=0\)) |
| \(E_{\mathrm{SRB}}\) | polar-bond Gaussian plus CN-dependent \(r_0\) chain |

EEQ charge response \(\partial q/\partial R\) is needed only for the H0
onsite \(k_q\) chain (GFN0 is not SCC; there is no coupled-perturbed density).
Integer occupation, molecular, H/C/N/O. Not PBC, not Fermi smear, not GFN1/2.

Oracle: **xtb** `--gfn 0` (never DFTB+ / OpenMOPAC). Formula-level twin:
pprcht/gfn0 (`gfn0_qm.f90` `build_dSH0`, `eeq_module`, `gfn0_srb`,
`gfn0_repulsion`).

## Design

- Same public type `cpu::semiempirical::GFN0`. `with_forces` stays a public
  member; when true, `launch` writes \(F=-\nabla E\) [eV/Å] in **one** pass.
  No `*Config`, no `make_*`.
- Delete the 6N central-FD loop in `src/cpu/semiempirical/gfn0.cpp`.
- Energy numbers of `reg_gfn0-beyond-toy` stay byte-identical (same kernels,
  gradient is additive).
- Native overlap derivatives of the existing STO-nG primitives. D4 reuses
  `dftd4()` with the GFN0 damping already used for energy.

## Files

- `include/atomiverse/cpu/semiempirical/gfn0.h` (doc: analytic G, not FD)
- `src/cpu/semiempirical/gfn0.cpp` (analytic gradient)
- `tests/cpu/semiempirical/test_xtb_gfn0.cpp` (FD identity + net-F=0)
- `regressions/semiempirical/gfn0-xtb-forces/` (new)
- `tools/tb_oracle/inputs/xtb/h2o_gfn0_offeq_grad/` (new twin)
- `tests/CMakeLists.txt` / `regressions/CMakeLists.txt` as needed

## Acceptance

- Unit: analytic \(F\) vs central FD of the **same** total energy on off-eq
  H₂O (\(h=10^{-4}\) Å) to \(O(h^2)\); \(|\sum_i F_i|\) at round-off.
- Regression: off-eq water forces vs offline xtb 6.7.1 `--gfn 0 --grad`
  (`REFERENCE.md`). Energy pin of methanol (`reg_gfn0-beyond-toy`) unmoved.
- SE-only cmake; scoped `--target atomiverse test_cpu_xtb_gfn0 reg_gfn0-xtb-forces`.
- `oracle: xtb` / `oracle_version:` on REFERENCE.md.
- Perf: \(T_{\mathrm{atv}}/T_{\mathrm{xtb}} \le 10\) or `perf_exempt`.

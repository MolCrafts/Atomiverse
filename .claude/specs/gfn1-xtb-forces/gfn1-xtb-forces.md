---
title: GFN1-xTB analytic nuclear forces
status: implemented
created: 2026-08-25
slug: gfn1-xtb-forces
oracle: xtb
---

# GFN1-xTB analytic nuclear forces

## Summary

Replace the GFN1 `with_forces` throw with the **closed-form nuclear gradient**
of the same Grimme 2017 SCC-xTB energy already shipped by `gfn1-xtb-organic`.
Pin Cartesian forces against an offline **xtb 6.7.1** `--gfn 1 --grad` run.
No `libxtb` wrap.

## Domain basis

GFN1-xTB (Grimme, Bannwarth, Shushkov, *J. Chem. Theory Comput.* **13**, 1989
(2017), DOI 10.1021/acs.jctc.7b00118) is SCC. The energy

\[
E = \mathrm{Tr}(P H_0) + \tfrac12 q\,\gamma\,q + \tfrac13 \Gamma q^3
    + E_{\mathrm{rep}} + E_{\mathrm{D3}}
\]

is already product (CHNO, closed shell). The gradient is the total derivative
of that sum at frozen SCC density, not a re-solve:

| Term | Gradient |
| ---- | -------- |
| \(E_{\mathrm{H0}}\) | Hellmann–Feynman + Pulay: \(\mathrm{Tr}(P\,\partial H_0 + [P V_{\mathrm{avg}}-W]\,\partial S)\) plus CN chain through onsite \(h_l - k_{\mathrm{CN}}\mathrm{CN}\) and the shell-poly \(\Pi\) |
| \(E_{\mathrm{ES}}\) | explicit \(\partial\gamma/\partial R\) at frozen Mulliken \(q\) (Klopman–Ohno \(g=2\)); \(\Gamma q^3\) has no explicit \(R\) |
| \(E_{\mathrm{rep}}\) | pair \(Z_{\mathrm{eff}} e^{-\sqrt{\alpha_i\alpha_j}\,r^{3/2}}/r\) |
| \(E_{\mathrm{D3}}\) | in-tree `gpw::dftd3_pbe_bj` analytic \(F=-\nabla E\) (GFN1 BJ scales) |

Integer occupation, molecular, H/C/N/O. Not PBC, not Fermi smear, not
open-shell, not GFN0/GFN2.

Oracle: **xtb** `--gfn 1` (never DFTB+ / OpenMOPAC). Formula-level twin:
tblite `xtb/h0`, `coulomb/charge/gamma`, `repulsion/effective`.

The CH4 energy residual **10.2 meV** vs xtb (`gfn1-xtb-organic`) is a kernel
hole (D3 C6 interpolation, Stewart `intcut`, Anderson vs Broyden) — not a
gradient defect. Analytic \(F\) is judged by FD of the **same** ATV energy;
the xtb force pin uses a tolerance sized from that hole.

## Design

- Same public type `cpu::semiempirical::GFN1`. `with_forces` stays a public
  member; when true and `ctx.forces_requested`, `launch` writes
  \(F=-\nabla E\) [eV/Å] in **one** pass. No `*Config`, no `make_*`.
- `provides_force()` follows `cpu::Semiempirical` (true).
- Energy numbers of `reg_gfn1-xtb-organic` stay the same kernels (gradient is
  additive).
- Native overlap derivatives of the existing Cartesian STO-nG primitives.

## Files

- `include/atomiverse/cpu/semiempirical/gfn1.h`
- `src/cpu/semiempirical/gfn1.cpp`
- `tests/cpu/semiempirical/test_xtb_gfn1.cpp`
- `regressions/semiempirical/gfn1-xtb-forces/` (new)
- `tools/tb_oracle/inputs/xtb/h2o_gfn1_offeq_grad/` (new twin)
- `regressions/CMakeLists.txt`

## Acceptance

- Unit: analytic \(F\) vs central FD of the **same** total energy on off-eq
  H₂O (\(h=10^{-4}\) Å) to \(O(h^2)\); \(|\sum_i F_i|\) at round-off.
  `with_forces` no longer throws.
- Regression: off-eq water forces vs offline xtb 6.7.1 `--gfn 1 --grad`
  (`REFERENCE.md`). Energy pin of CH4 (`reg_gfn1-xtb-organic`) unmoved.
- SE-only cmake; scoped `--target atomiverse test_cpu_xtb_gfn1 reg_gfn1-xtb-forces`.
- `oracle: xtb` / `oracle_version:` on REFERENCE.md.
- Perf: \(T_{\mathrm{atv}}/T_{\mathrm{xtb}} \le 10\) or `perf_exempt`.

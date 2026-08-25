---
title: IPEA-xTB analytic nuclear forces
status: implemented
created: 2026-08-25
chain: tb-method-loop
slug: ipea-xtb-forces
oracle: xtb
---

# IPEA-xTB analytic nuclear forces

## Summary

Replace the IPEA `with_forces` throw with the **closed-form nuclear gradient**
of the same Ásgeirsson 2017 SCC-xTB energy already shipped by `ipea-xtb-small`.
Pin Cartesian forces against an offline **xtb 6.7.1** IPEA `--grad` run
on off-eq CH4 (`param_ipea` selected as GFN1). No `libxtb` wrap.
Off-eq water is the unit FD geometry; its ~0.5 eV polar residual vs xtb
is the energy-path kernel hole already named in `ipea-xtb-small`, not a
gradient defect.

## Domain basis

IPEA-xTB (Ásgeirsson, Bauer, Grimme, *Chem. Sci.* **8**, 4879 (2017), DOI
[10.1039/c7sc00601b](https://doi.org/10.1039/c7sc00601b)) is a GFN1
reparametrization for IPs/EAs. The energy

\[
E = \mathrm{Tr}(P H_0) + \tfrac12 q\,\gamma\,q + \tfrac13 \Gamma q^3
    + E_{\mathrm{rep}} + E_{\mathrm{D3}}
\]

is already product (CHNO, closed shell; extra \((n+1)s\) on C/N/O; H is
1s+2s). The gradient is the total derivative of that sum at frozen SCC
density — the same term list as `gfn1-xtb-forces`, with IPEA numbers:

| Term | Gradient |
| ---- | -------- |
| \(E_{\mathrm{H0}}\) | Hellmann–Feynman + Pulay: \(\mathrm{Tr}(P\,\partial H_0 + [P V_{\mathrm{avg}}-W]\,\partial S)\) plus CN chain through onsite \(h_l - k_{\mathrm{CN}}\mathrm{CN}\) and the shell-poly \(\Pi\) |
| \(E_{\mathrm{ES}}\) | explicit \(\partial\gamma/\partial R\) at frozen Mulliken \(q\) (Klopman–Ohno \(g_{\exp}=1\)); atomic \(\Gamma q^3\) has no explicit \(R\) |
| \(E_{\mathrm{rep}}\) | pair \(Z_{\mathrm{eff}} e^{-\sqrt{\alpha_i\alpha_j}\,r^{3/2}}/r\) |
| \(E_{\mathrm{D3}}\) | in-tree `gpw::dftd3_pbe_bj` analytic \(F=-\nabla E\) (GFN1 BJ scales; IPEA uses the same set) |

Integer occupation, molecular, H/C/N/O. Not PBC, not Fermi smear, not
open-shell, not GFN1/GFN2, not `--vip`/`--vea`.

Oracle: **xtb** with `param_ipea-xtb.txt` as `param_gfn1-xtb.txt` (never
DFTB+ / OpenMOPAC). Formula-level twin: tblite `xtb/ipea1.f90` + GFN1
gradient shape.

The CH4 energy residual **8.90 meV** vs xtb (`ipea-xtb-small`) is a kernel
hole (D3 C6 interpolation, Stewart `intcut`, Anderson vs Broyden) — not a
gradient defect. Analytic \(F\) is judged by FD of the **same** ATV energy;
the xtb force pin uses a tolerance sized from that hole.

## Design

- Same public type `cpu::semiempirical::IPEA`. `with_forces` stays a public
  member; when true and `ctx.forces_requested`, `launch` writes
  \(F=-\nabla E\) [eV/Å] in **one** pass. No `*Config`, no `make_*`.
- `provides_force()` follows `cpu::Semiempirical` (true).
- Energy numbers of `reg_ipea-xtb-small` stay the same kernels (gradient is
  additive).
- Native overlap derivatives of the existing Cartesian STO-nG primitives.

## Files

- `include/atomiverse/cpu/semiempirical/ipea.h`
- `src/cpu/semiempirical/ipea.cpp`
- `tests/cpu/semiempirical/test_xtb_ipea.cpp`
- `regressions/semiempirical/ipea-xtb-forces/` (new)
- `tools/tb_oracle/inputs/xtb/ch4_ipea_offeq_grad/` (new twin)
- `regressions/CMakeLists.txt`
- `tools/tb_oracle/README.md`

## Acceptance

- Unit: analytic \(F\) vs central FD of the **same** total energy on off-eq
  H₂O (\(h=10^{-4}\) Å) to \(O(h^2)\); \(|\sum_i F_i|\) at round-off.
  `with_forces` no longer throws.
- Regression: off-eq CH4 (one C–H stretched to 1.20 Å) vs offline xtb 6.7.1
  IPEA `--grad` (`REFERENCE.md`). Energy pin of eq CH4 (`reg_ipea-xtb-small`)
  unmoved.
- SE-only cmake; scoped `--target atomiverse test_cpu_xtb_ipea
  reg_ipea-xtb-forces`.
- `oracle: xtb` / `oracle_version:` on REFERENCE.md.
- Perf: \(T_{\mathrm{atv}}/T_{\mathrm{xtb}} \le 10\) or `perf_exempt`.

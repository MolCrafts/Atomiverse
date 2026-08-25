---
title: IPEA-xTB energy on a small molecule
status: in-progress
created: 2026-08-25
chain: tb-method-loop
slug: ipea-xtb-small
oracle: xtb
---

# IPEA-xTB energy on a small molecule

## Summary

Land a production **IPEA-xTB** energy Component under `cpu/semiempirical` for
organic H/C/N/O, pinned to an offline **xtb 6.7.1** IPEA-xTB single point on
tetrahedral CH4 (\(r=1.09\) Å). Native static tables — do not wrap `libxtb`.
Own type and tables (not a flag on `GFN1`/`GFN2`). Energy-only foothold
(forces / PBC / spin / ALPB / `--vip` ΔSCC IP are later slugs).

## Domain basis

IPEA-xTB is a special-purpose reparametrization of the GFN1 Hamiltonian for
ionization potentials and electron affinities, with extra \((n+1)s\)
functions (Ásgeirsson, Bauer, Grimme, *Chem. Sci.* **8**, 4879 (2017), DOI
[10.1039/c7sc00601b](https://doi.org/10.1039/c7sc00601b); Koopman, Grimme,
*ACS Omega* **4**, 15120 (2019)). Not a DFTB3 successor; not GFN2.

Energy (same GFN1 skeleton, IPEA numbers):

\[
E = E_{\mathrm{EHT}} + E_{\gamma} + E_{\Gamma} + E_{\mathrm{rep}}
  + E_{\mathrm{D3}} + E_{\mathrm{XB}}.
\]

tblite `ipea1.f90` / xtb `param_ipea-xtb.txt` (vs GFN1):

- Extra 3s on C/N/O (STO-3G, Gram–Schmidt vs valence 2s). H stays 1s+2s.
- \(k_s=2.15\), \(k_p=2.30\), \(k_{sp}=2.25\), \(k_{\mathrm{diff}}=2.25\).
  No H–H / N–H pair \(k\).
- Klopman–Ohno \(g_{\exp}=1\) (GFN1 is 2). Third-order is **atomic**.
- D3(BJ) same damping as GFN1 (\(s_6=1,s_8=2.4,a_1=0.63,a_2=5.0,s_9=0\)).
- XB is identically zero on CHNO.
- `ipeashift` is a `--vip`/`--vea` property shift, **not** this energy path.

Units on the public surface: energy [eV], positions [Å]. Internal work is
Hartree / Bohr. Parameter eV→Ha uses tblite's \(27.21138505\); Frame energy
uses ATV CODATA \(27.211386245988\).

Oracle is **xtb**, never OpenMOPAC or DFTB+.

## Design

**Public type** `ATV::cpu::semiempirical::IPEA : cpu::Semiempirical, cpu::QmEngine`.

Parameters are public members (`scc_tol`, `max_scc`, `mix_beta`, `charge`,
`with_forces`). No `*Config`, no `make_*` / `build_*`. Frame is the only I/O.

- `with_forces` defaults **false**. True throws with a how-to.
  `provides_force()` is false.
- Supported Z: 1, 6, 7, 8. Any other Z throws naming the missing table.
- Closed-shell only (`multiplicity != 1` throws).
- Static in-tree tables in `basis_data/ipea_tables.h` transcribed from
  tblite `xtb/ipea1.f90` (same numbers as xtb `param_ipea-xtb.txt`).

Evaluate usage:

```cpp
cpu::Evaluate eval;
cpu::semiempirical::IPEA ipea;
ipea.with_forces = false;
eval.on(evaluate::Phase::evaluate).add(std::move(ipea));
eval.build();
eval.run(frame);   // frame.meta.get(kEnergy) [eV]
```

## Files to create or modify

- `.claude/specs/ipea-xtb-small/ipea-xtb-small.md` (this file)
- `.claude/specs/ipea-xtb-small/ipea-xtb-small.acceptance.md`
- `include/atomiverse/cpu/semiempirical/ipea.h`
- `include/atomiverse/cpu/semiempirical/basis_data/ipea_tables.h`
- `src/cpu/semiempirical/ipea.cpp`
- `tests/cpu/semiempirical/test_xtb_ipea.cpp`
- `tests/CMakeLists.txt`
- `CMakeLists.txt` (SE sources)
- `regressions/semiempirical/ipea-xtb-small/reg_ipea-xtb-small.cpp`
- `regressions/semiempirical/ipea-xtb-small/REFERENCE.md`
- `regressions/CMakeLists.txt`
- `tools/tb_oracle/inputs/xtb/h2o_ipea/` (mol.xyz + transcript)
- `tools/tb_oracle/README.md` (twin row)

## Tasks

- [x] Offline xtb 6.7.1 IPEA-xTB pin on tetrahedral CH4 \(r=1.09\) Å
- [x] Public `IPEA` Component + static H/C/N/O tables
- [x] Native energy path (overlap, H0, SCC \(g_{\exp}=1\), extra 3s, rep, D3)
- [x] Unit tests (tables, throw, finite energy) — not the pin
- [x] Regression vs the xtb pin
- [x] Dual gate (correctness + 1-thread wall)

## Testing strategy

- Units: `test_cpu_xtb_ipea` — table coverage, unsupported Z, H2O energy
  finite, `with_forces` throws. No literature pin in ctest.
- Regression: `reg_ipea-xtb-small` hard-codes the xtb CH4 number from
  `REFERENCE.md`. Never shells out to xtb.

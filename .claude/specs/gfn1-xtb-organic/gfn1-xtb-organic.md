---
title: GFN1-xTB energy on small organics
status: in-progress
created: 2026-08-25
chain: tb-method-loop
slug: gfn1-xtb-organic
oracle: xtb
---

# GFN1-xTB energy on small organics

## Summary

Land a production **GFN1-xTB** energy Component under `cpu/semiempirical` for
organic H/C/N/O, pinned to an offline **xtb 6.7.1** `--gfn 1` single point on
tetrahedral CH4. Native static tables — do not wrap `libxtb`. Energy-only
foothold (forces / PBC / spin / ALPB are later slugs).

## Domain basis

GFN1-xTB (originally GFN-xTB) is the first-generation Grimme extended
tight-binding Hamiltonian:

- S. Grimme, C. Bannwarth, P. Shushkov, *J. Chem. Theory Comput.* **13**,
  1989–2009 (2017), DOI [10.1021/acs.jctc.7b00118](https://doi.org/10.1021/acs.jctc.7b00118).

Energy (WIREs 2021 recap, same Hamiltonian):

\[
E = E_{\mathrm{EHT}} + E_{\gamma} + E_{\Gamma} + E_{\mathrm{rep}}
  + E_{\mathrm{D3}} + E_{\mathrm{XB}}.
\]

- Minimal valence STOs expanded as contracted Gaussians (Stewart STO-nG;
  H is double-ζ 1s+2s). Generalized eigen \(H C = S C \varepsilon\).
- Diagonal EHT levels \(H_{\mu\mu}=h_l^A - k_{\mathrm{CN},l}^A\,\mathrm{CN}_A\)
  with D3-style geometric CN (tblite `p_kcn` convention: subtract).
- Off-diagonal
  \(H_{\mu\nu}=\tfrac12(H_{\mu\mu}+H_{\nu\nu})\,k_{ll'}\,K_{AB}\,
  (1+k_{\mathrm{EN}}\Delta\mathrm{EN}^2)\,\Pi_{\mu}\Pi_{\nu}\,S_{\mu\nu}\)
  with \(\Pi=1+k_{\mathrm{poly}}\sqrt{R/(R_A+R_B)}\).
- SCC: shell-resolved Klopman–Ohno \(\gamma\) (harmonic-mean hardness,
  \(g_{\exp}=2\)) plus on-site third-order \(E_{\Gamma}=\tfrac13\Gamma_A q_A^3\).
- Pair repulsion \(Z_{\mathrm{eff}}^A Z_{\mathrm{eff}}^B R^{-1}
  \exp(-\sqrt{\alpha_A\alpha_B}\,R^{1.5})\).
- D3(BJ) two-body, GFN1 damping \(s_6=1,\,s_8=2.4,\,a_1=0.63,\,a_2=5.0,\,s_9=0\).
- Halogen-bond correction is identically zero on CHNO.

Units on the public surface: energy [eV], positions [Å]. Internal work is
Hartree / Bohr. Parameter eV→Ha uses tblite's \(27.21138505\) (the number
the tables were published against); Frame energy uses ATV CODATA
\(27.211386245988\).

Oracle is **xtb**, never OpenMOPAC or DFTB+. GFN1 is not a DFTB3 successor.

## Design

**Public type** `ATV::cpu::semiempirical::GFN1 : cpu::Semiempirical, cpu::QmEngine`.

Parameters are public members (`scc_tol`, `max_scc`, `mix_beta`, `charge`,
`with_forces`). No `*Config`, no `make_*` / `build_*`. Frame is the only I/O.

- `with_forces` defaults **false**. True throws with a how-to (forces are a
  later slug). `provides_force()` is false.
- Supported Z: 1, 6, 7, 8. Any other Z throws naming the missing table.
- Closed-shell only (`multiplicity != 1` throws).
- Static in-tree tables in `basis_data/gfn1_tables.h` transcribed from
  tblite `xtb/gfn1.f90` (same numbers as xtb `param_gfn1-xtb.txt`).

Evaluate usage:

```cpp
cpu::Evaluate eval;
cpu::semiempirical::GFN1 gfn1;
gfn1.with_forces = false;
eval.on(evaluate::Phase::evaluate).add(std::move(gfn1));
eval.build();
eval.run(frame);   // frame.meta.get(kEnergy) [eV]
```

## Files to create or modify

- `.claude/specs/gfn1-xtb-organic/gfn1-xtb-organic.md` (this file)
- `.claude/specs/gfn1-xtb-organic/gfn1-xtb-organic.acceptance.md`
- `include/atomiverse/cpu/semiempirical/gfn1.h`
- `include/atomiverse/cpu/semiempirical/basis_data/gfn1_tables.h`
- `src/cpu/semiempirical/gfn1.cpp`
- `tests/cpu/semiempirical/test_xtb_gfn1.cpp`
- `tests/CMakeLists.txt`
- `CMakeLists.txt` (SE sources)
- `regressions/semiempirical/gfn1-xtb-organic/reg_gfn1-xtb-organic.cpp`
- `regressions/semiempirical/gfn1-xtb-organic/REFERENCE.md`
- `regressions/CMakeLists.txt`
- `tools/tb_oracle/inputs/xtb/ch4_gfn1/` (mol.xyz + transcript)
- `tools/tb_oracle/README.md` (twin row)

## Tasks

- [x] Offline xtb 6.7.1 `--gfn 1` pin on tetrahedral CH4 \(r=1.09\) Å
- [x] Public `GFN1` Component + static H/C/N/O tables
- [x] Native energy path (overlap, H0, SCC, rep, D3)
- [x] Unit tests (tables, throw, finite energy) — not the pin
- [x] Regression vs the xtb pin
- [x] Dual gate (correctness + 1-thread wall)

## Testing strategy

- Units: `test_cpu_xtb_gfn1` — table coverage, unsupported Z, H2/CH4 energy
  finite, `with_forces` throws. No literature pin in ctest.
- Regression: `reg_gfn1-xtb-organic` hard-codes the xtb number from
  `REFERENCE.md`. Never shells out to xtb.

---
title: Production GFN0-xTB energy (CHNO)
status: implemented
created: 2026-08-25
slug: gfn0-beyond-toy
oracle: xtb
---

# Production GFN0-xTB energy (beyond toy)

## Summary

Land a **production** GFN0-xTB energy method under `cpu/semiempirical` as
`ATV::cpu::semiempirical::GFN0 : cpu::Semiempirical`. The deleted toy
`XtbGfn0` (`bc14dabd`) was onsite+hop+rep sketch numbers, not the Pracht /
Grimme Hamiltonian. This slug restores the **published** non-SCC GFN0 energy
for H/C/N/O with static in-tree tables, pinned to an offline **xtb 6.7.1**
`--gfn 0` run on methanol. No `libxtb` wrap (`ATV_XTB_ROOT` is leftover from
the never-landed GFN2 CHON campaign).

## Domain basis

GFN0-xTB (Pracht, Caldeweyher, Ehlert, Grimme, ChemRxiv 2019,
DOI 10.26434/chemrxiv.8326202.v1) is a **non-self-consistent** first-order
extended tight-binding method. Electrostatics are classical EEQ charges;
one EHT diagonalization incorporates those charges to first order. Energy:

\[
E = E_{\mathrm{H0}} + E_{\mathrm{rep}} + E_{\mathrm{IES}} + E_{\mathrm{D4}} + E_{\mathrm{SRB}}
\]

| Term | Units | Source |
| ---- | ----- | ------ |
| \(E_{\mathrm{H0}}\) | Hartree (internal), eV on Frame | \(\sum_i n_i\varepsilon_i\); \(H_{\mu\nu}\) in eV |
| \(E_{\mathrm{rep}}\) | Hartree | \(Z_{\mathrm{eff},A}Z_{\mathrm{eff},B}\,e^{-\alpha r^{3/2}}/r\) |
| \(E_{\mathrm{IES}}\) | Hartree | Gaussian-smeared EEQ Coulomb energy |
| \(E_{\mathrm{D4}}\) | Hartree | two-body D4(BJ), GFN0 damping \(s_8=2.85\), \(a_1=0.80\), \(a_2=4.60\), \(s_9=0\) |
| \(E_{\mathrm{SRB}}\) | Hartree | polar-bond short-range correction (B–F only) |

Coordinates internal: Bohr with GFN0's \(a_0=0.52917726\) Å (xtb/pprcht
`gfn0_param`). Public Frame I/O stays Å / eV (ATV CODATA-2018).

**Not this slug:** analytic nuclear forces (FD skeleton only), PBC, spin,
ALPB/GBSA, GFN1/2, wrapping `libxtb`.

## Design

- `cpu::semiempirical::GFN0` inherits `cpu::Semiempirical`. Parameters are
  public members (`charge`) — no `*Config`, no `make_*`.
- Bind with `eval.on(evaluate::Phase::evaluate).add(std::move(gfn0))`.
- Static tables in `basis_data/gfn0_tables.h` transcribed from xtb/pprcht
  `gfn0_param.f90` for **H/C/N/O only**. Unsupported Z throws with a how-to.
- Native kernels (CN erf, EEQ, STO-nG overlap, H0, repulsion, SRB). D4
  two-body reuses in-tree `cpu::scf::dftd4_energy` with GFN0 BJ scales
  (compiled under SE-only when HF is off, same pattern as DFTB2's D3).
- `on(phase).add` ownership as elsewhere. No factories.

## Files

- `include/atomiverse/cpu/semiempirical/gfn0.h` (new)
- `include/atomiverse/cpu/semiempirical/basis_data/gfn0_tables.h` (new)
- `src/cpu/semiempirical/gfn0.cpp` (new)
- `tests/cpu/semiempirical/test_xtb_gfn0.cpp` (new)
- `regressions/semiempirical/gfn0-beyond-toy/` (new)
- `tools/tb_oracle/inputs/xtb/ch3oh_gfn0/` (new)
- `CMakeLists.txt`, `tests/CMakeLists.txt`, `regressions/CMakeLists.txt`

## Acceptance

- Unit: tables cover Z={1,6,7,8}; unsupported Z throws; H₂O energy finite.
- Regression: methanol total energy vs offline xtb 6.7.1 `--gfn 0` pin
  (see `REFERENCE.md`).
- SE-only cmake; scoped `--target atomiverse test_cpu_xtb_gfn0 reg_gfn0-beyond-toy`.
- `oracle: xtb` / `oracle_version:` on REFERENCE.md.
- Perf: \(T_{\mathrm{atv}}/T_{\mathrm{xtb}} \le 10\) or `perf_exempt`.

---
title: GFN2-xTB two-centre CAMM quadrupole (CAMM-vs-R)
status: incomplete
created: 2026-08-25
slug: gfn2-camm-r
oracle: xtb
---

# GFN2-xTB two-centre CAMM quadrupole (CAMM-vs-R)

## Summary

Give every centre — including s-only H — the xtb `mmompop` two-centre
traceless CAMM quadrupole, fold that \(\Theta\) into the AES Fock, and
chain \(\partial\Theta/\partial R\) through \(\partial E_{\mathrm{AES}}/\partial\Theta\)
in the analytic gradient. AES-in-Fock (`gfn2-aes-fock`) does **not** close
angular \(F_{H1,y}\): ATV energy FD stays \(\sim+0.06\) eV/Å vs xtb
\(-0.80\) eV/Å. Frozen-CAMM \(\partial Q/\partial R\) on stretch \(F_{O,x}\)
is the same completeness hole, not a second slug.

Offline **xtb 6.7.1** `--gfn 2 --grad` off-eq water twin already on disk
(`tools/tb_oracle/inputs/xtb/h2o_gfn2_offeq_grad/`). No `libxtb` wrap.

## Domain basis

GFN2-xTB (Bannwarth, Ehlert, Grimme, *J. Chem. Theory Comput.* **15**, 1652
(2019), DOI 10.1021/acs.jctc.8b01176). Production CHNO energy + AES **in
the SCC Fock** + analytic \(G\) + FD Hessian already ship
(`reg_gfn2-xtb-water`, `reg_gfn2-quadrupole-aes`, `reg_gfn2-xtb-forces`,
`reg_gfn2-aes-fock`, `reg_gfn-xtb-analytic-hessian`).

ATV CAMM (until this slug) skipped s-only \(\Theta\): the on-site 1s blob
about the nuclear origin is spherical. xtb `mmompop` (`src/aespot.f90`)
still assigns the **two-centre** origin-shifted quadrupole of every unique
AO pair to **both** endpoints, including H. That \(\Theta_H(R)\) is what
tilts the angular energy surface.

Oracle: **xtb** `--gfn 2 --grad` (never DFTB+ / OpenMOPAC). Same off-eq
geometry as `gfn2-aes-fock` (O–H 0.96 → 1.10 Å):

| Atom | x [Å] | y [Å] | z [Å] |
| ---- | ----- | ----- | ----- |
| O    | 0.00  | 0.00  | 0.00  |
| H    | 1.10  | 0.00  | 0.00  |
| H    | −0.24 | 0.93  | 0.00  |

Integer occupation, molecular, H/C/N/O. Not PBC, not Fermi, not GFN0/1.

A naive dump of the O-centre formula onto H without the two-centre
origin-shift / Fock / \(\partial\Theta/\partial R\) chain is **not**
`mmompop` (eq-H₂O residual 23 meV). This slug is the three together.

**Wave-13:** the three together (tblite `shift_operator` plus-RRS on
every centre, \(v^Q\) in Fock on H, frozen-\(P\) \(\partial\Theta/\partial R\))
still does not close \(F_{H1,y}\) (+0.12 vs xtb −0.80). Minus-RRS
(opab4-signed \(Q\)) still breaks SCC. Not closed.

## Design

- Same public type `cpu::semiempirical::GFN2`. No `*Config`, no `make_*`.
- CAMM quadrupole on **every** atom, same Mulliken partition as the dipole
  foothold: \(\Theta_A=\tfrac12\sum_{\mu\in A}\sum_\nu P_{\mu\nu}\,
  \mathrm{traceless}(q^{\mathrm{shift}}_{\mu\nu}(R_A))\).
- AES Fock: origin-shifted quadrupole operator contracted with \(v^Q\) on
  **both** endpoints of every AO pair (s-only included).
- Analytic \(G\): existing frozen-CAMM AES kernel + dipole CAMM chain, plus
  quadrupole CAMM chain \(\sum_A v^Q_A:\partial\Theta_A/\partial R\) at
  frozen \(P\) (tblite `setdvsdq` / xtb `aniso_grad` + integral derivatives).
- Energy pins of `reg_gfn2-xtb-water` / `reg_gfn2-quadrupole-aes` stay on
  the same kernels; residual may move (H \(\Theta\) is a real energy term).

## Files

- `include/atomiverse/cpu/semiempirical/gfn2.h` (doc: two-centre \(\Theta\) on H)
- `src/cpu/semiempirical/gfn2.cpp` (CAMM, Fock, \(\partial\Theta/\partial R\))
- `tests/cpu/semiempirical/test_gfn2_params.cpp` (FD identity; net force)
- `regressions/semiempirical/gfn2-camm-r/` (new; gates \(F_{H1,y}\))
- `tools/tb_oracle/inputs/xtb/h2o_gfn2_offeq_grad/` (reuse 6.7.1 twin)
- `regressions/CMakeLists.txt` as needed

## Acceptance

- Unit: analytic \(F\) vs central FD of the **same** total energy on off-eq
  H₂O (\(h=10^{-4}\) Å); \(|\sum_i F_i|\) at round-off.
- Regression: off-eq water \(F_{H1,y}\) **and** \(F_{O,x}\) vs offline xtb
  6.7.1 `--gfn 2 --grad` (`REFERENCE.md`).
- SE-only cmake; scoped `--target atomiverse test_cpu_gfn2_params
  reg_gfn2-camm-r`.
- `oracle: xtb` / `oracle_version:` on REFERENCE.md.
- Perf: \(T_{\mathrm{atv}}/T_{\mathrm{xtb}} \le 10\) or `perf_exempt`.

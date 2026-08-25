---
title: GFN2-xTB analytic nuclear forces
status: implemented
created: 2026-08-25
slug: gfn2-xtb-forces
oracle: xtb
---

# GFN2-xTB analytic nuclear forces

## Summary

Replace the GFN2 `with_forces` throw with the **closed-form nuclear gradient**
of the H/C/N/O energy already shipped by `gfn2-xtb-water` +
`gfn2-quadrupole-aes`. Pin Cartesian forces against an offline **xtb 6.7.1**
`--gfn 2 --grad` run on off-eq water. No `libxtb` wrap.

## Domain basis

GFN2-xTB (Bannwarth, Ehlert, Grimme, *J. Chem. Theory Comput.* **15**, 1652
(2019), DOI 10.1021/acs.jctc.8b01176) is SCC. The molecular closed-shell
energy

\[
E = \operatorname{Tr}(P H^{0})
  + E_{\mathrm{IES}}
  + E^{(3)}
  + E_{\mathrm{rep}}
  + E_{\mathrm{D4}}
  + E_{\mathrm{AES}}
\]

is already product (quadrupole AES + qkernel AXC closed the 1.45 meV H₂O
hole). The gradient is the total derivative of that sum at SCC stationarity,
not a re-SCC finite difference:

| Term | Gradient |
| ---- | -------- |
| \(\operatorname{Tr}(P H^{0})\) | Hellmann–Feynman + Pulay: \(\operatorname{Tr}(P\,\partial H^{0}-W\,\partial S)\) plus CN chain through onsite \(\varepsilon\) (tblite `get_hamiltonian_gradient`) |
| \(E_{\mathrm{IES}}\) | \(\tfrac12 q^{\mathrm{T}}(\partial\gamma/\partial R)\,q\) at frozen shell charges; overlap-weighted SCC shift rides with Pulay via Fock \(W\) |
| \(E^{(3)}\) | on-site in \(q\) — no explicit \(R\) at frozen \(q\) |
| \(E_{\mathrm{rep}}\) | pair \(Z_{\mathrm{eff}} e^{-\alpha r^{k_{\mathrm{exp}}}}/r\) |
| \(E_{\mathrm{D4}}\) | in-tree `cpu::scf::dftd4()` two-body + ATM (\(s_9=5\), GFN2 BJ scales) |
| \(E_{\mathrm{AES}}\) | damped multipole kernel at frozen CAMM (tblite `get_multipole_gradient_0d`) + CN-radius chain; AES stays **post-SCC energy** (not a Fock potential — that deepen stays a gap) |

Integer occupation, molecular, H/C/N/O. Not PBC, not Fermi smear, not GFN0/1.

Oracle: **xtb** `--gfn 2` (never DFTB+ / OpenMOPAC). Formula-level twin:
tblite `xtb/h0.f90` `get_hamiltonian_gradient`, `coulomb/multipole.f90`
`get_multipole_gradient_0d`.

Off-eq geometry (O–H stretched 0.96 → 1.10 Å) so \(|F|\neq 0\):

| Atom | x [Å] | y [Å] | z [Å] |
| ---- | ----- | ----- | ----- |
| O    | 0.00  | 0.00  | 0.00  |
| H    | 1.10  | 0.00  | 0.00  |
| H    | −0.24 | 0.93  | 0.00  |

## Design

- Same public type `cpu::semiempirical::GFN2`. `with_forces` stays a public
  member; when true, `launch` writes \(F=-\nabla E\) [eV/Å] in **one** pass.
  No `*Config`, no `make_*`.
- Delete the `with_forces` throw in `src/cpu/semiempirical/gfn2.cpp`.
- Energy numbers of `reg_gfn2-xtb-water` / `reg_gfn2-quadrupole-aes` stay
  on the same kernels (gradient is additive).
- Native overlap derivatives of the existing STO-\(n\)G primitives. D4 reuses
  `dftd4()` with the GFN2 damping already used for energy.

## Files

- `include/atomiverse/cpu/semiempirical/gfn2.h` (doc: analytic G)
- `src/cpu/semiempirical/gfn2.cpp` (analytic gradient)
- `tests/cpu/semiempirical/test_gfn2_params.cpp` (FD identity + net-F=0)
- `regressions/semiempirical/gfn2-xtb-forces/` (new)
- `tools/tb_oracle/inputs/xtb/h2o_gfn2_offeq_grad/` (6.7.1 twin)
- `tests/CMakeLists.txt` / `regressions/CMakeLists.txt` as needed

## Acceptance

- Unit: analytic \(F\) vs central FD of the **same** total energy on off-eq
  H₂O (\(h=10^{-4}\) Å) to \(O(h^2)\); \(|\sum_i F_i|\) at round-off.
- Regression: off-eq water forces vs offline xtb 6.7.1 `--gfn 2 --grad`
  (`REFERENCE.md`). Energy pins of `reg_gfn2-xtb-water` /
  `reg_gfn2-quadrupole-aes` unmoved.
- SE-only cmake; scoped `--target atomiverse test_cpu_gfn2_params
  reg_gfn2-xtb-forces`.
- `oracle: xtb` / `oracle_version:` on REFERENCE.md.
- Perf: \(T_{\mathrm{atv}}/T_{\mathrm{xtb}} \le 10\) or `perf_exempt`.

---
title: GFN2-xTB AES in the SCC Fock
status: implemented
created: 2026-08-25
slug: gfn2-aes-fock
oracle: xtb
---

# GFN2-xTB AES in the SCC Fock

## Summary

Put GFN2 anisotropic electrostatics (charge–dipole, dipole–dipole,
charge–quadrupole + dipole/quadrupole AXC) **inside the SCC Fock**, not as a
post-SCC energy on frozen CAMM. Offline **xtb 6.7.1** `--gfn 2 --grad` off-eq
water twin. No `libxtb` wrap.

Finding: AES-in-Fock does **not** close angular \(F_{H1,y}\) — ATV energy FD
of \(E_{\mathrm{AES}}\) stays \(\sim+0.06\) eV/Å vs xtb \(-0.80\). That slope
is a CAMM-vs-\(R\) hole (xtb `mmompop` two-centre \(\Theta\) on H).

## Domain basis

GFN2-xTB (Bannwarth, Ehlert, Grimme, *J. Chem. Theory Comput.* **15**, 1652
(2019), DOI 10.1021/acs.jctc.8b01176) is SCC. Production CHNO energy +
closed-form \(\mathbf{F}=-\nabla E\) already ship (`reg_gfn2-xtb-water`,
`reg_gfn2-quadrupole-aes`, `reg_gfn2-xtb-forces`). AES is assembled as a
**post-SCC** CAMM energy: the Fock is only \(H^0\) + IES + \(E^{(3)}\). Stretch
\(F_{O,x}\) matches xtb; angular \(F_{H1,y}\) does not (xtb \(\approx-0.80\)
eV/Å vs ATV energy-FD \(\approx+0.06\) eV/Å) because the density is not
stationary with respect to AES.

tblite module boundary (formula twin, not a pin):

| Piece | tblite |
| ----- | ------ |
| AES energy | `coulomb/multipole.f90` `get_energy` (0d `amat_sd/dd/sq` + AXC kernels) |
| AES potential | `get_potential` → `vat`, `vdp`, `vqp` |
| Fock | `scf/potential.f90` `add_pot_to_h1` / `add_vmp_to_h1` |
| CAMM | `wavefunction/mulliken.f90` `get_mulliken_atomic_multipoles` |

ATV already matches xtb AES **energy** (H₂O residual 26 µeV) with the packed
traceless quadrupole /3 that follows xtb `aniso_electro` `gab5` rather than
tblite 0d `amat_sq` packing. The Fock uses the **same** operators: \(\partial
E_{\mathrm{AES}}/\partial q\), \(\partial E_{\mathrm{AES}}/\partial\boldsymbol{\mu}\),
\(\partial E_{\mathrm{AES}}/\partial\Theta\) contracted into \(S\), dipole, and
quadrupole integrals.

Oracle: **xtb** `--gfn 2 --grad` (never DFTB+ / OpenMOPAC). Same off-eq
geometry as `gfn2-xtb-forces` (O–H 0.96 → 1.10 Å):

| Atom | x [Å] | y [Å] | z [Å] |
| ---- | ----- | ----- | ----- |
| O    | 0.00  | 0.00  | 0.00  |
| H    | 1.10  | 0.00  | 0.00  |
| H    | −0.24 | 0.93  | 0.00  |

Integer occupation, molecular, H/C/N/O. Not PBC, not Fermi smear, not GFN0/1.

## Design

- Same public type `cpu::semiempirical::GFN2`. No `*Config`, no `make_*`.
- During SCC, after IES \(v_{\mathrm{sh}}=\gamma q+\Gamma q^2\), build AES
  potentials from the previous CAMM \((q,\boldsymbol{\mu},\Theta)\) and add them to
  \(H\) before the generalized eigenstep:
  - atom `vat` → every shell on that atom, overlap-weighted like IES
  - `vdp` → origin-shifted dipole operator \(\tfrac12(D-R_A S)\)
  - `vqp` → origin-shifted traceless quadrupole (p-shell centres only)
- First SCC iteration has zero moments (AES potential 0). Energy is still the
  explicit \(E_{\mathrm{AES}}(\mathrm{CAMM})\) on the converged density — not
  \(\mathrm{Tr}(P V_{\mathrm{AES}})\).
- Analytic \(\nabla E\) keeps the frozen-CAMM kernel + dipole CAMM chain; Pulay
  \(W\) is now from the AES-containing Fock so the density is variational.
- Energy pins of `reg_gfn2-xtb-water` / `reg_gfn2-quadrupole-aes` stay on the
  same kernels. AES-in-Fock density shift grows the eq-H₂O residual 26→230 µeV
  (CAMM-vs-\(R\)); sibling tols cover it. Naive two-centre \(\Theta\) on H
  is **not** xtb `mmompop` (23 meV).

## Files

- `include/atomiverse/cpu/semiempirical/gfn2.h` (doc: AES in SCC Fock)
- `src/cpu/semiempirical/gfn2.cpp` (AES potential + Fock)
- `tests/cpu/semiempirical/test_gfn2_params.cpp` (FD identity still holds)
- `regressions/semiempirical/gfn2-aes-fock/` (new; gates \(F_{H1,y}\))
- `tools/tb_oracle/inputs/xtb/h2o_gfn2_offeq_grad/` (reuse 6.7.1 twin)
- `regressions/CMakeLists.txt` as needed

## Acceptance

- Unit: analytic \(F\) vs central FD of the **same** total energy on off-eq
  H₂O (\(h=10^{-4}\) Å); \(|\sum_i F_i|\) at round-off.
- Regression: off-eq water \(F_{H1,y}\) vs offline xtb 6.7.1 `--gfn 2 --grad`
  (`REFERENCE.md`). Sibling energy pins unmoved within their tols.
- SE-only cmake; scoped `--target atomiverse test_cpu_gfn2_params
  reg_gfn2-aes-fock`.
- `oracle: xtb` / `oracle_version:` on REFERENCE.md.
- Perf: \(T_{\mathrm{atv}}/T_{\mathrm{xtb}} \le 10\) or `perf_exempt`.

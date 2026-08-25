---
title: IPEA-xTB vertical IP/EA (ΔSCC + empirical shift)
status: implemented
created: 2026-08-25
chain: tb-method-loop
slug: ipea-xtb-vip
oracle: xtb
---

# IPEA-xTB vertical IP/EA (ΔSCC + empirical shift)

## Summary

Land the **IPEA product** — xtb `--vip` / `--vea` vertical ionization
potential and electron affinity — on production `cpu::semiempirical::IPEA`.
Energy, analytic \(G\), and FD Hessian already ship (`reg_ipea-xtb-small`,
`reg_ipea-xtb-forces`, `reg_ipea-xtb-analytic-hessian`). This slug is the
ΔSCC IP/EA with the IPEA empirical shift, **not** the closed-shell energy
pin. Pin to an offline **xtb 6.7.1** IPEA `--vip` / `--vea` run on **CH4**
(same tetrahedral geometry as `reg_ipea-xtb-small`). Polar water is **not**
this pin (0.5 eV kernel hole vs xtb). Native static tables — do not wrap
`libxtb`. Extra-Z tables wait (`ipea-beyond-chno`, same `ipea.cpp`).

## Domain basis

IPEA-xTB (Ásgeirsson, Bauer, Grimme, *Chem. Sci.* **8**, 4879 (2017), DOI
[10.1039/c7sc00601b](https://doi.org/10.1039/c7sc00601b)) is a GFN1
reparametrization for IPs/EAs, with extra \((n+1)s\) functions. xtb computes

\[
\mathrm{IP}_v = E(M^{n+1}) - E(M^{n}) + \Delta_{\mathrm{IPEA}},
\qquad
\mathrm{EA}_v = E(M^{n}) - E(M^{n-1}) + \Delta_{\mathrm{IPEA}},
\]

at **fixed geometry** (vertical). \(n\) is the molecular charge of the
input (0 here). \(E\) is the full IPEA SCC energy (EHT + \(\gamma\) +
\(\Gamma\) + rep + D3); repulsion and D3 cancel in the difference.
\(\Delta_{\mathrm{IPEA}}\) is `ipeashift` from `param_ipea-xtb.txt`
(\(1.78069 \times 0.1\) in tblite eV units → **4.8455 eV** printed by xtb).

xtb `--vip` is two SCCs (N then N−1 electrons); `--vea` is N then N+1;
`--vipea` is all three. Ionized species are odd-electron. xtb fills with
Fermi smearing at the default \(T_{\mathrm{el}}=300\,\mathrm{K}\) on a
**restricted** orbital set (not UHF; not `spin-gfn-xtb`). Neutral CH4 has
HOMO–LUMO 7.751 eV so 300 K is integer; the cation/anion t2 hole/particle
is Fermi-smeared across the degenerate set.

Oracle: **xtb** with `param_ipea-xtb.txt` as `param_gfn1-xtb.txt` (never
DFTB+ / OpenMOPAC). Geometry is the tetrahedral CH4 of
`reg_ipea-xtb-small` (\(r(\mathrm{C{-}H})=1.09\) Å).

## Design

Public members on existing `IPEA` (no new type, no `*Config`, no
`make_*` / `build_*`):

- `bool compute_vip = false` — after the N-electron SCC, run charge+1
  (N−1 electrons) and store \(\mathrm{IP}_v\) [eV].
- `bool compute_vea = false` — likewise charge−1 (N+1 electrons) for
  \(\mathrm{EA}_v\) [eV].
- `double last_vip() const` / `double last_vea() const` — last ΔSCC+shift
  values [eV]. Unset if the corresponding flag was false.

`kEnergy` stays the N-electron total (Evaluate energy slot). VIP/VEA are
property accessors on the Component, same shape as `last_energy()`.

Odd-electron SCC is **restricted** with equal occupation of degenerate
MO shells (window 0.05 Eh) so the CH4 cation t2 hole is 5/3 each, matching
xtb's 300 K Fermi on Td-degenerate integrals. ATV's Stewart/intcut residual
splits that t2 by ≫ kT, so 300 K Fermi would collapse to 2+2+1. Closed-shell
energy / forces / Hessian paths keep integer double occupation
(`n_elec` even, `multiplicity==1`). This is **not** the `gfn-xtb-etemp`
product (GFN2 ground-state Fermi).

`ipeashift` is a static table constant in `ipea_tables.h`
(`kIpeaShiftEv`), transcribed from xtb `ipeashift 1.78069` × 0.1 ×
tblite \(27.21138505\).

Evaluate usage:

```cpp
cpu::Evaluate eval;
cpu::semiempirical::IPEA ipea;
ipea.compute_vip = true;
ipea.compute_vea = true;
eval.on(evaluate::Phase::evaluate).add_ref(ipea);
eval.build();
eval.run(frame);   // kEnergy = E(N) [eV]; ipea.last_vip()/last_vea() [eV]
```

## Files to create or modify

- `.claude/specs/ipea-xtb-vip/ipea-xtb-vip.md` (this file)
- `.claude/specs/ipea-xtb-vip/ipea-xtb-vip.acceptance.md`
- `include/atomiverse/cpu/semiempirical/ipea.h`
- `include/atomiverse/cpu/semiempirical/basis_data/ipea_tables.h` (`kIpeaShiftEv`)
- `src/cpu/semiempirical/ipea.cpp`
- `tests/cpu/semiempirical/test_xtb_ipea.cpp` (VIP/VEA units; no pin)
- `regressions/semiempirical/ipea-xtb-vip/` (new)
- `regressions/CMakeLists.txt`
- `tools/tb_oracle/inputs/xtb/ch4_ipea_vip/` (new twin)
- `tools/tb_oracle/README.md` (twin row)

## Tasks

- [x] Offline xtb 6.7.1 IPEA `--vip` / `--vea` pin on tetrahedral CH4
- [x] `kIpeaShiftEv` + odd-electron degenerate-shell SCC + `compute_vip` / `compute_vea`
- [x] Unit tests (finite VIP/VEA, shift constant, flags default off)
- [x] Regression vs the xtb pin
- [x] Dual gate (correctness + 1-thread wall)

## Testing strategy

- Units: `test_cpu_xtb_ipea` — VIP/VEA finite on CH4/H2O; `last_vip` unset
  unless `compute_vip`; existing energy/FD gates unmoved. No literature
  pin in ctest.
- Regression: `reg_ipea-xtb-vip` hard-codes xtb CH4 IP/EA from
  `REFERENCE.md`. Never shells out to xtb.

## Non-goals

- Extra-Z tables (`ipea-beyond-chno`).
- Polar-water energy hole vs xtb.
- UHF / `spin-gfn-xtb`.
- GFN ground-state `--etemp` (`gfn-xtb-etemp`).
- ALPB, PBC, sTDA, `libxtb`.
- Public `*Config` / `make_*` / `build_*`.
- Live xtb in ctest. OpenMOPAC / DFTB+ as reference (**forbidden**).

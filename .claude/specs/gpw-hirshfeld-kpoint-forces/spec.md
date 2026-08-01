---
slug: gpw-hirshfeld-kpoint-forces
status: approved
created: 2026-08-01
revised: 2026-08-01
domain: gpw / dispersion forces
queue: cp2k-method-loop
deferred: false
depends_on: [gpw-hirshfeld-forces, gpw-hirshfeld-kpoints, gpw-mbd-kpoint-forces]
unblocks: [gpw-hirshfeld-kpoint-forces]
grilled: false
sibling: gpw-ts-kpoint-forces
---

# Spec: `gpw-hirshfeld-kpoint-forces` — multi-k forces with Hirshfeld TS/MBD volumes

**Slug:** `gpw-hirshfeld-kpoint-forces`  
**Domain:** GPW multi-k analytic ionic forces + density-scaled TS/MBD  
**Queue:** CP2K method-loop (wave 5)  
**Status:** approved — pin-first residual (product path already folds volumes)

## Summary

Close the residual that multi-k **energy** Hirshfeld (`gpw-hirshfeld-kpoints`)
and Γ **force** Hirshfeld (`gpw-hirshfeld-forces`) are product-pinned, while
the multi-k force assembly already calls `ts_mbd_vol_ratio` → volume-scaled
`ts_pbe` / `mbd_rsscs_pbe` but has **no dedicated dual-gate pin**. Deliver a
regression that proves:

1. Fixed-\(v\) analytic forces match independent central FD of the same
   volume-scaled energy (TS + MBD@rsSCS).
2. Multi-k product wiring:
   \(F^{k+\mathrm{ts/mbd}+\mathrm{hirsh}}-F^{k,\mathrm{plain}}\) equals
   standalone geometry forces at the SCF volume ratios (not free-atom).
3. Free-atom and Hirshfeld-scaled force differences are nontrivial.

Density-chain \(\partial v_A/\partial R\) remains out of scope (same as Γ).

## Domain basis

Hirshfeld volume ratios from the multi-k SCF density:

\[
v_A = \frac{V_A[\rho]}{V_A^{\mathrm{free}}},
\quad
V_A = \int \frac{\rho_A^{\mathrm{free}}(\mathbf{r})}
                {\sum_B \rho_B^{\mathrm{free}}(\mathbf{r})}
      \rho(\mathbf{r})\,d\mathbf{r}.
\]

TS/MBD free-atom parameters scale as
\(\alpha\propto v\), \(C_6\propto v^2\), \(R\propto v^{1/3}\). Forces hold
\(v\) fixed at the SCF state:

\[
F_A = -\frac{\partial E_{\mathrm{disp}}}{\partial R_A}\Big|_{v}.
\]

Product path: `assemble_gpw_forces_kpoints` → `ts_mbd_vol_ratio(cfg, …,
&scf.volume_ratios)` → `ts_pbe` / `mbd_rsscs_pbe(..., vol_ratio)`.

## Public API

No new symbols. Existing:

```cpp
GpwDft gpw;
gpw.ts = true;          // or gpw.mbd = true
gpw.hirshfeld = true;
gpw.k_mesh[0] = gpw.k_mesh[1] = 1; gpw.k_mesh[2] = 2;
gpw.build(atoms);
const auto& scf = gpw.run(atoms);   // volume_ratios filled post-SCF
auto F = gpw_forces_from_scf(atoms, gpw, scf);
```

## Oracle / twin policy

**CP2K 2026.1** stock `PAIR_POTENTIAL TYPE` is only DFTD2|DFTD3|DFTD3(BJ)|DFTD4
— **no** stock TS/MBD. Hard pin is **libMBD / product dual** (same policy as
`gpw-hirshfeld-forces`). Twin notes under
`tools/cp2k_oracle/inputs/gpw-hirshfeld-kpoint-forces/`.

**oracle:** literature / libMBD (volume-scaled gradient dual)  
**perf_exempt:** true (no fair CP2K ENERGY_FORCE twin)

## Acceptance (summary)

See `acceptance.md`. Green when dual-gates + multi-k product wiring pass and
REFERENCE documents pins; wall ≤ 60 s for the regression.

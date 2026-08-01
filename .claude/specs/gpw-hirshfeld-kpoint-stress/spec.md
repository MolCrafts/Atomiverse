---
slug: gpw-hirshfeld-kpoint-stress
status: approved
created: 2026-08-01
domain: gpw
queue: cp2k-method-loop
depends_on: [gpw-hirshfeld-kpoint-forces, gpw-mbd-kpoint-stress]
---

# Spec: `gpw-hirshfeld-kpoint-stress` — multi-k Hirshfeld-scaled TS/MBD cell stress

## Summary

Deliver a **dedicated dual-gate pin** that the multi-k GPW analytic cell-stress
path folds **Hirshfeld volume ratios** \(v_A = V_A/V_A^{\mathrm{free}}\) into
geometry-only **TS** and **MBD@rsSCS** under orthorhombic strain.

Capability already lives in `gpw_stress` (`energy_fixed_P_k` passes
`scf.volume_ratios` into `ts_pbe` / `mbd_rsscs_pbe` at every strained cell —
fixed-\(v\) definition, density-chain \(\partial v/\partial\varepsilon\) out
of scope). Γ stress fold is gated inside `reg_gpw-hirshfeld-forces`; multi-k
**forces** are product (`reg_gpw-hirshfeld-kpoint-forces`). This slug closes
the missing multi-k **stress** pin.

## Product definition

\[
\sigma_{aa} = -\Omega^{-1}\frac{\partial E_L[\{P_k\},\{W_k\};\varepsilon]}{\partial\varepsilon_{aa}}
\Big|_{v}
\quad\text{with \(v\) fixed at the multi-k SCF density}.
\]

Scaling of free-atom \(\alpha_0\) / \(C_6\) / \(R_{\mathrm{vdW}}\):
\(\alpha\propto v\), \(C_6\propto v^2\), \(R\propto v^{1/3}\).

Public surface:

```cpp
GpwDft gpw;
gpw.ts = true;          // or gpw.mbd = true
gpw.hirshfeld = true;
gpw.k_mesh[0] = gpw.k_mesh[1] = 1;
gpw.k_mesh[2] = 2;
gpw.build(atoms);
const auto& scf = gpw.run(atoms);
auto sig = gpw_stress(atoms, gpw, scf, /*h=*/2e-3);
```

## Oracle family

**CP2K 2026.1** stock `PAIR_POTENTIAL TYPE` is only
`DFTD2|DFTD3|DFTD3(BJ)|DFTD4` — **no** stock TS/MBD. Hard pin is
**libMBD / product dual** (fixed-\(v\) strain-FD of the same energy surface
the stress Lagrangian rebuilds) + multi-k product wiring.

**oracle:** literature / libMBD  
**perf_exempt:** true (no fair CP2K twin algorithm — same policy as
`gpw-hirshfeld-forces` / `gpw-hirshfeld-kpoint-forces` / `gpw-mbd-kpoint-stress`)

## Gates (acceptance)

| Gate | Check |
| ---- | ----- |
| A | Isolated MBD@rsSCS + fixed dual volumes matches hard pin |
| B | Geometry dual under strain: product TS/MBD fixed-\(v\) \(\sigma\) vs central FD of \(E_{\mathrm{disp}}\) |
| C | Multi-k H₂ product wiring: \(\sigma(\mathrm{ts}+\mathrm{hirsh})-\sigma(\mathrm{plain})\approx\sigma_{\mathrm{TS}}(v_{\mathrm{SCF}})\) |
| D | Same wiring for MBD@rsSCS (isolated when `mbd_k_mesh` default zero) |
| E | Residual vs free-atom stress nontrivial; multi-k payload present; wall ≤ 60 s |

## Non-goals

- Density-chain \(\partial v/\partial R\) or \(\partial v/\partial\varepsilon\)
- UKS Hirshfeld stress (see `gpw-hirshfeld-uks*` energy paths)
- Free-atom multi-k TS stress (sibling `gpw-ts-kpoint-stress`)
- Fake `oracle: cp2k` window for TS/MBD

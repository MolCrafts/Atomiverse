# Spec — `cpmd-uks-thermostat-kpoints`

**Slug:** `cpmd-uks-thermostat-kpoints`  
**Loop:** CP2K capability (GPW / periodic DFT)  
**Branch:** `wf/cpmd-uks-thermostat-kpoints`  
**Method:** Multi-k UKS Car–Parrinello MD with Nosé–Hoover electronic
thermostat on every \((k, \sigma)\) channel

## Problem

Product already ships:

- multi-k UKS CPMD NVE — `cpmd-uks-kpoints` (one `CpmdCplx` per
  \((k, \sigma)\), dual-spin fixed-density multi-k eval);
- multi-k RKS CPMD + electronic NH — `cpmd-thermostat-kpoints` (one
  thermostatted `CpmdCplx` per k, k-weighted \(T_e / E_{\mathrm{therm}}\)).

The **combo pin** is missing: multi-k collinear UKS + electronic
Nosé–Hoover engaged on every \((k, \sigma)\) channel, with the Marx–Hutter
extended energy
\(E_{\mathrm{ext}} = T_n + T_e + E_{\mathrm{KS}} + E_{\mathrm{therm}}\)
and a \(T_e\)-control envelope. The plumbing is already product
(`GPWEvaluator` seeds and advances thermostatted `cpmd_k_` / `cpmd_k_beta_`;
unit `BuildAcceptsMultiKUksCpmdThermostat`); this slug is the e2e write-first
capability gate.

## Science

- Car & Parrinello, *Phys. Rev. Lett.* **55**, 2471 (1985) — fictitious-mass
  electronic dynamics.
- Blöchl & Parrinello, *Phys. Rev. B* **45**, 9413 (1992) — electronic
  Nosé–Hoover for adiabaticity control on fictitious \(\dot C\).
- Andersen, *J. Comput. Phys.* **52**, 24 (1983) — electronic RATTLE
  (\(C_k^{\sigma\dagger} S_k C_k^\sigma = I\)).
- Marx & Hutter, *Ab Initio Molecular Dynamics* (CUP) — extended conserved
  energy with electronic NH:
  \(E_{\mathrm{ext}} = T_n + T_e + E_{\mathrm{KS}} + E_{\mathrm{therm}}\),
  with
  \(T_e = \sum_k w_k \sum_\sigma \tfrac12\mu\lVert\dot C_k^\sigma\rVert_F^2\)
  and
  \(E_{\mathrm{therm}} = \sum_k w_k \sum_\sigma E_{\mathrm{therm}}^{(k,\sigma)}\).
- Per-(k, σ) convention (Γ-UKS / multi-k thermostat product): each channel
  targets the full \(K_0\); published totals remain k-weighted.

## Implementation

**Write-first combo pin — no new algorithm path required.** Parents already
productize:

1. Dual-spin multi-k fixed-density eval + per-(k, σ) `CpmdCplx` advance
   (`cpmd-uks-kpoints`).
2. Electronic NH on every multi-k channel with k-weighted \(T_e / E_{\mathrm{therm}}\)
   (`cpmd-thermostat-kpoints`; seed sets `ch.thermostat` on α and β k sets).

Deliverables:

1. **Regression** `regressions/dynamics/cpmd-uks-thermostat-kpoints/` —
   periodic H₂ NVE, mesh \(1\times1\times2\), UKS closed-shell fillings
   \(n_\alpha = n_\beta = 1\), \(\mu = 400\), \(K_0 = 10^{-5}\) Ha,
   \(\tau_e = 50\) fs; gates on \(E_{\mathrm{ext}}\) drift, \(T_e\) ceiling,
   production `iters == 0`, finite Eth, H–H window; plus UKS-vs-RKS
   closed-shell single-point identity (same class as `cpmd-uks-kpoints`).
2. **Offline oracle sketch** (unexecutable — family lock) under
   `tools/cp2k_oracle/inputs/cpmd-uks-thermostat-kpoints/`.
3. **CMake** `atv_add_regression(dynamics cpmd-uks-thermostat-kpoints …)`.
4. **Docs:** REFERENCE.md; roadmap / queue / tutorial rows.

Public surface: `CarParrinello{gpw, mu, ElectronicThermostat{K0, tau_e}}`
with `gpw.n_alpha/n_beta` + `gpw.k_mesh` — no Config bag, Frame in/out.

## Oracle / pin provenance

**Family lock** (same as `cpmd-electronic-thermostat` /
`cpmd-thermostat-kpoints` / `cpmd-uks-kpoints`): CP2K 2026.1 is present on
the host but has **no fictitious-mass Car–Parrinello propagator** — neither
\(T_e[\dot C_k^\sigma]\) nor \(E_{\mathrm{therm}}\) is a number it can
print. Pin = **literature extended-energy / \(T_e\) control envelope**
(Blöchl & Parrinello 1992; Marx & Hutter). The BO surface underneath is
CP2K-anchored by `gpw-kpoints-uks` / `gpw-kpoint-uks-forces` /
`gpw-bomd-uks-kpoints`. `perf_exempt = true` is the stated gate, not a
missing-binary escape.

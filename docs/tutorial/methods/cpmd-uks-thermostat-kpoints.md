# Multi-k UKS Car–Parrinello MD with electronic Nosé–Hoover (H₂ NVE, GPW)

*(Technical-report page; CP2K capability loop worker wave 2.)*

| Field | Value |
| ----- | ----- |
| Slug | `cpmd-uks-thermostat-kpoints` |
| Regression | `regressions/dynamics/cpmd-uks-thermostat-kpoints/` |
| Public API | `cpu::electronic::CarParrinello{gpw, mu_au, ElectronicThermostat{K0, tau_e}}` with multi-k UKS `gpw` (`n_alpha`/`n_beta` + `k_mesh`); one thermostatted `CpmdCplx` per (k, spin). Nuclear dt pushed from the MD integrator (`TimestepProvider` → `TimestepConsumer`) |
| Roadmap / queue | `docs/roadmap/dft.md` (AIMD); queue done 2026-08-01 |

## Summary

Car–Parrinello molecular dynamics with **fictitious electronic degrees of
freedom on multi-k collinear UKS** GPW **and the Nosé–Hoover electronic
thermostat engaged on every \((k, \sigma)\) channel** — the
`cpmd-uks-kpoints` (dual-spin Bloch CP) × `cpmd-thermostat-kpoints`
(multi-k electronic NH) combo pin. One complex propagator per sampled
Bloch \(k\) and spin channel \(\sigma \in \{\alpha, \beta\}\), each with
electronic RATTLE and its own NH variable targeting the full \(K_0\)
(Γ-UKS convention). Published totals are k-weighted dual-spin sums:

\[
T_e = \sum_k w_k \sum_\sigma \tfrac12\mu\lVert\dot C_k^\sigma\rVert_F^2,\qquad
E_{\mathrm{therm}} = \sum_k w_k \sum_\sigma E_{\mathrm{therm}}^{(k,\sigma)}.
\]

The conserved diagnostic is the Marx–Hutter extended energy
\(E_{\mathrm{ext}} = T_n + T_e + E_{\mathrm{KS}} + E_{\mathrm{therm}}\).
Nuclei stay NVE; only the fictitious electronic DOF are thermostatted.
Production steps use dual-spin fixed-density multi-k KS evals
(`iters == 0`).

## System

| Quantity | Value |
| -------- | ----- |
| Species | H₂ (periodic), \(R = 1.0\) Bohr compressed bond along \(x\), cell-centred |
| Cell | cubic \(L = 12\) Bohr (≈ 6.350 Å) |
| Spin / XC | collinear UKS, \(n_\alpha = n_\beta = 1\) (closed-shell fillings); PADE (LSD) |
| Basis / PP | DZV-GTH-PADE / GTH-PADE-q1 |
| Cutoff | 100 Ry |
| k-mesh | \(1\times1\times2\), time reversal off → \(n_k = 2\) |
| CPMD | \(\mu = 400\) a.u.; `dt` = 0.1 fs (nuclear = electronic) |
| Thermostat | \(K_0 = 10^{-5}\) Ha, \(\tau_e = 50\) fs, per (k, σ) |
| Trajectory | 1 cold SCF seed + 1 CP baseline + 4 production steps |

## Implementation

```cpp
cpu::gpw::GPWEvaluator gpw({H}, /*cutoff_ry=*/100.0, /*max_iter=*/80,
                           /*tol=*/1e-5, /*mix=*/0.3,
                           cpu::gpw::XcKind::PADE, /*nonperiodic=*/false,
                           /*compute_forces=*/true);
gpw.n_alpha = 1;  gpw.n_beta = 1;
gpw.k_mesh[2] = 2;  gpw.k_time_reversal = false;

cpu::electronic::CarParrinello cp{
    std::move(gpw), /*mu_au=*/400.0,
    cpu::electronic::ElectronicThermostat{/*K0=*/1e-5, /*tau_e_fs=*/50.0}};
cpu::md::MD md;
md.set_integrator(cpu::md::VelocityVerlet{0.1});
md.set_evaluator_ref(cp);
md.build();
md.run(frame, /*steps=*/6);
```

```bash
cmake --build build --target reg_cpmd-uks-thermostat-kpoints
./build/regressions/reg_cpmd-uks-thermostat-kpoints
```

## Validation

Source: `regressions/dynamics/cpmd-uks-thermostat-kpoints/REFERENCE.md`
(offline pins; never live in ctest). The CP propagator has no CP2K twin
(family lock); the BO surface underneath is CP2K-anchored through
`gpw-kpoints-uks` / `gpw-kpoint-uks-forces` / `gpw-bomd-uks-kpoints`.

| Quantity | Gate | Anchor |
| -------- | ---- | ------ |
| Relative \(E_{\mathrm{ext}}\) drift over 4 production steps | ≤ 0.10 | Literature (Blöchl & Parrinello 1992; Marx & Hutter) |
| Max \(T_e\) over production | ≤ 0.05 eV | Thermostatted control envelope |
| UKS-vs-RKS closed-shell identity, `tol = 1e-7` | ≤ 3×10⁻⁴ eV | Multi-k RKS CP2K-pinned in `gpw-bomd-kpoints` |
| Production SCF iterations | exactly 0 | dual-spin fixed-density multi-k eval |
| \(n_{\mathrm{kpts}}\) after seed | ≥ 2 | 1×1×2, time reversal off |
| \(E_{\mathrm{therm}}\) | finite every production step | NH bookkeeping |

## Performance

| Field | Value |
| ----- | ----- |
| `perf_exempt` | **true** |
| Reason | CP2K 2026.1 has no fictitious-mass CP propagator — no fair wall twin |

## See also

- [cpmd-uks-kpoints](cpmd-uks-kpoints.md) — multi-k UKS CP NVE (no eNH)
- Parent multi-k RKS thermostat: `regressions/dynamics/cpmd-thermostat-kpoints`
- [cpmd-electronic-thermostat](cpmd-electronic-thermostat.md) — Γ electronic NH

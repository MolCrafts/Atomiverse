# Multi-k UKS Car–Parrinello MD on dual-spin Bloch \(C_k^\sigma\) (H₂ NVE, GPW)

*(Technical-report page; CP2K capability loop worker wave 2.)*

| Field | Value |
| ----- | ----- |
| Slug | `cpmd-uks-kpoints` |
| Regression | `regressions/dynamics/cpmd-uks-kpoints/` |
| Public API | `cpu::electronic::CarParrinello{cpu::gpw::GPWEvaluator{…, n_alpha, n_beta, k_mesh}, mu_au}` bound with `set_evaluator_ref`; one `CpmdCplx` propagator per (k, spin). CP mode is the wrapper type — `cpmd`/`cpmd_mu` are private on `GPWEvaluator`. The nuclear dt is pushed from the MD integrator into the wrapper (`TimestepProvider` → `TimestepConsumer`) |
| Roadmap / queue | `docs/roadmap/dft.md` (AIMD); queue done 2026-07-31 |

## Summary

Car–Parrinello molecular dynamics with **fictitious electronic degrees of
freedom on multi-k collinear UKS** GPW — the `cpmd-kpoints` (multi-k RKS
Bloch CP) × `cpmd-uks` (Γ dual-spin CP) crossing. One complex propagator per
sampled Bloch \(k\) **and spin channel** \(\sigma \in \{\alpha, \beta\}\),
each with the holonomic constraint \(C_k^{\sigma\dagger} S_k C_k^\sigma = I\)
enforced by electronic RATTLE, and the spin-diagonal occupancy-1 EOM

\[
\mu\,\ddot C_k^\sigma
  = -(I - S_k C_k^\sigma C_k^{\sigma\dagger})\,(2 H_k^\sigma C_k^\sigma)
    + \text{RATTLE}
\]

on the spin-resolved KS matrices \(H_k^\alpha / H_k^\beta\) from the
**dual-spin fixed-density multi-k evaluation** (`max_iter == 0` at the
propagated \(\{P_k^\alpha, P_k^\beta\}\), exported through the new
`GpwResult::H_ks_k` / `H_ks_k_beta` payload; `iters == 0` on every
production step). Electronic kinetic energy is
\(T_e = \sum_k w_k \sum_\sigma \tfrac12 \mu \lVert \dot C_k^\sigma
\rVert_F^2\). The multi-k **electronic thermostat** (one NH per (k, σ)
channel, k-weighted totals) ships as `cpmd-thermostat-kpoints`
(RKS; `regressions/dynamics/cpmd-thermostat-kpoints/`) **and** the
multi-k UKS eNH combo pin `cpmd-uks-thermostat-kpoints`
(`regressions/dynamics/cpmd-uks-thermostat-kpoints/`); this pin is NVE
electrons.

## System

| Quantity | Value |
| -------- | ----- |
| Species | H₂ (periodic), \(R = 1.0\) Bohr compressed bond along \(x\), cell-centred |
| Cell | cubic \(L = 12\) Bohr (≈ 6.350 Å) |
| Spin / XC | collinear UKS, \(n_\alpha = n_\beta = 1\) (closed-shell fillings — both channels propagate, molecule stays bound); PADE (LSD) |
| Basis / PP | DZV-GTH-PADE / GTH-PADE-q1 |
| Cutoff | 100 Ry |
| k-mesh | \(1\times1\times2\), time reversal off → \(n_k = 2\) |
| CPMD | \(\mu = 400\) a.u.; `dt` = 0.1 fs (nuclear = electronic) |
| Trajectory | 1 cold SCF seed + 1 CP baseline + 4 production steps, NVE |

## Implementation

Public surface only (`cpu::Frame`; parameters are constructor args /
members — no `*Config` bags):

```cpp
// Multi-k UKS GPW forces (DZV-GTH-PADE / GTH-PADE-q1, PADE, 100 Ry)
const cpu::gpw::Kind H{
    molrs::Element::H,
    cpu::gpw::GaussianBasis(molrs::Element::H, "DZV-GTH-PADE"),
    cpu::gpw::GthPotential(molrs::Element::H, "GTH-PADE-q1")};
cpu::gpw::GPWEvaluator gpw({H}, /*cutoff_ry=*/100.0, /*max_iter=*/80,
                           /*tol=*/1e-5, /*mix=*/0.3,
                           cpu::gpw::XcKind::PADE, /*nonperiodic=*/false,
                           /*compute_forces=*/true);
gpw.n_alpha = 1;  gpw.n_beta = 1;          // collinear UKS, closed-shell fillings
gpw.k_mesh[2] = 2;  gpw.k_time_reversal = false;   // 1×1×2, n_k = 2

// CP wrapper: one CpmdCplx per (k, spin); dt pushed from the MD integrator
cpu::electronic::CarParrinello cp{std::move(gpw), /*mu_au=*/400.0};
cpu::md::MD md;
md.set_integrator(cpu::md::VelocityVerlet{0.1});   // fs, NVE
md.set_evaluator_ref(cp);
md.build();
md.run(frame, /*steps=*/6);        // 1 SCF seed + 1 CP baseline + 4 production
```

```bash
cmake --build build --target reg_cpmd-uks-kpoints
./build/regressions/reg_cpmd-uks-kpoints
```

Headers: `cpu/electronic/car_parrinello.h`, `cpu/electronic/cpmd_cplx.h`
(propagator), `cpu/gpw/gpw_evaluator.h` (dual-spin fixed-density multi-k
export `GpwResult::H_ks_k` / `H_ks_k_beta`).

## Validation (vs CP2K-anchored surface + literature envelope)

Source: `regressions/dynamics/cpmd-uks-kpoints/REFERENCE.md` (offline pins;
never live in ctest). The CP propagator itself has no CP2K twin (family
lock, below); the **BO surface underneath is CP2K-anchored** through the
shipped multi-k UKS pins.

| Quantity | Gate (hard-coded) | Anchor / oracle | Notes |
| -------- | ----------------: | --------------- | ----- |
| Relative \(E_{\mathrm{cons}}\) drift \(\delta\) over 4 production steps | \(\le 0.10\) | Literature NVE envelope (Car–Parrinello 1985; Marx & Hutter) | \(E_{\mathrm{cons}} = T_n + T_e + E_{\mathrm{KS}}\), \(E_0\) at first CP step |
| UKS-vs-RKS closed-shell identity, `tol = 1e-7` | \(\le 3\times10^{-4}\) eV (≈ 1.1×10⁻⁵ Ha) | Multi-k RKS path CP2K-pinned in `gpw-bomd-kpoints` (CP2K 2026.1); UKS path in `gpw-kpoints-uks` (A2 holds to 5×10⁻⁶ Ha) | Same geometry / mesh / cutoff; \(n_\alpha=n_\beta=1\) vs \(n_{\mathrm{occ}}=1\) |
| Production SCF iterations | exactly 0 | — | dual-spin fixed-density multi-k eval |
| \(n_{\mathrm{kpts}}\) after seed | \(\ge 2\) | — | 1×1×2, time reversal off |
| \(T_e\) once propagated | \(> 0\) | — | dual-spin sum \(\sum_k w_k \sum_\sigma \tfrac12\mu\lVert\dot C_k^\sigma\rVert_F^2\) |
| PE after warm-up | \(< -5\) eV, finite | — | bound H₂ |
| \(r_{\mathrm{HH}}\) after production | \(0.4\)–\(1.5\) Å | — | soft window |

Unit gates (`tests/`) additionally pin the dual-spin fixed-density solver
export (`H_ks_k_beta`, partial-spin-guess refusal) and the **Γ-limit
identity**: the k-path dual-spin fixed-density eval at one explicit Γ point
reproduces the Γ dual-spin eval (energy and per-spin KS matrices — the
surface `cpmd-uks` ships on).

## Performance

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | ------------------------: | -----------------------: | ----: | ---- | ------ |
| fictitious-mass multi-k UKS CPMD | n/a | n/a | n/a | **perf_exempt** | Oracle family lock (below) |

**perf_exempt** — CP2K 2026.1 (present on the pin host; it pinned
`gpw-bomd-kpoints` on 2026-07-31) has **no fictitious-mass Car–Parrinello
propagator**: its `RUN_TYPE MD` is Born–Oppenheimer and its CP-like method
is second-generation CPMD (ASPC + corrector). The CP conserved quantity with
fictitious \(T_e[\dot C_k^\sigma]\) is not a number CP2K can print for any
input, so no fair same-method wall twin exists (same policy as
`cpmd-kpoints` / `cpmd-uks`).

## Limitations

- Multi-k electronic thermostat remains refused (NVE electrons on the k
  path; queued `cpmd-thermostat-kpoints`).
- Closed-shell fillings only in the pinned case (\(n_\alpha=n_\beta=1\));
  triplet H₂ is unbound and cannot pin NVE (the `xl-bomd-uks-kpoints`
  precedent). Open-shell multi-k CP is exercised only via unit gates.
- CPU backend only; CUDA CP is out of scope for this loop.

## References

- Offline pin: `regressions/dynamics/cpmd-uks-kpoints/REFERENCE.md`
- Car & Parrinello, *Phys. Rev. Lett.* **55**, 2471 (1985)
- Marx & Hutter, *Ab Initio Molecular Dynamics* (CUP) — CP conserved energy
- Andersen, *J. Comput. Phys.* **52**, 24 (1983) — electronic RATTLE
- Parents: [cpmd-kpoints](cpmd-kpoints.md) (multi-k RKS Bloch CP),
  [cpmd-uks](cpmd-uks.md) (Γ dual-spin CP); BO anchors
  `regressions/gpw/gpw-kpoints-uks`, `gpw-kpoint-uks-forces` (CP2K)

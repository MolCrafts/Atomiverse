# Reference — Car–Parrinello MD on Γ-UKS (O₂ triplet NVE, GPW)

## Method

**Car–Parrinello molecular dynamics (CPMD)** with fictitious electronic
degrees of freedom on **collinear Γ-point UKS** GPW:

- Outer loop: `cpu::md::MD` + NVE `VelocityVerlet` (nuclei)
- Force slot: `cpu::gpw::GPWEvaluator` with **`cpmd = true`**,
  **`n_alpha = 7`**, **`n_beta = 5`** (O₂ GTH-q6 triplet),
  **`cpmd_mu = 2000` a.u.**, `compute_forces = true`; the electronic dt is
  **not** set on the evaluator — `cpu::md::MD` pushes the nuclear dt in from
  the `VelocityVerlet` integrator through the `TimestepProvider` /
  `TimestepConsumer` protocol at build
- Propagator: dual `cpu::electronic::Cpmd` instances (α and β) with electronic
  RATTLE; occupancy-1 S-tangent force
  \(\mu\,\ddot C_\sigma = -(I-S C_\sigma C_\sigma^\top)(2 H_\sigma C_\sigma)\)

**Not BOMD** (production steps use fixed-density dual-KS evaluation,
`iters==0`). **Not XL-BOMD** (`xl-bomd-uks`). **Not** second-gen CP.

## Literature

- Car & Parrinello, *Phys. Rev. Lett.* **55**, 2471 (1985)
- H. C. Andersen, *J. Comput. Phys.* **52**, 24 (1983) — RATTLE
- Marx & Hutter, *Ab Initio Molecular Dynamics* (CUP) — CP conserved energy
  \(E_{\mathrm{cons}} = T_{\mathrm{n}} + T_{\mathrm{e}} + E_{\mathrm{KS}}\)
  with \(T_{\mathrm{e}} = T_{\mathrm{e}}^\alpha + T_{\mathrm{e}}^\beta\)
- Twin geometry / UKS family: `xl-bomd-uks`, `gpw-uks-forces`,
  `gpw-uks-o2-gamma`
- Closed-shell CPMD twin: `cpmd-rattle-water`

## Setup

| Item | Value |
| ---- | ----- |
| System | Isolated O₂ triplet (\(R=1.2075\) Å) |
| Cell | 12 Å cubic, Martyna–Tuckerman (`nonperiodic`) |
| XC / basis / PP | PADE LSD / DZVP-MOLOPT-GTH / GTH-PBE-q6 |
| CUTOFF | 200 Ry |
| Spin | \(n_\alpha=7\), \(n_\beta=5\) (multiplicity 3) |
| dt | 0.1 fs (nuclear = electronic) |
| \(\mu\) | 2000 a.u. (AO residual stiffer than PW textbook ~400) |
| Production | 1 SCF seed + 1 CP baseline + 4 CP drift steps |

## Pin

| Observable | Gate |
| ---------- | ---- |
| \(E_{\mathrm{cons}}\) relative drift | ≤ 0.10 |
| Production SCF iters | **0** (fixed-density) |
| ⟨S²⟩ | [1.5, 2.5] (triplet window) |
| O–O distance | [0.9, 1.6] Å |
| PE after warm-up | < −100 eV |

**Offline CP2K twin input:** `tools/cp2k_oracle/inputs/cpmd_uks_o2_nve.inp`
(QS PADE/LSD, DZVP-MOLOPT-GTH, GTH-PADE, MT 12 Å, MULTIPLICITY 3, short NVE).
Pin is a **literature conservation envelope**, not a live CP2K energy twin
in ctest.

**perf_exempt** — no CP2K CPMD-UKS wall twin on the worker host; literature
envelope + structural / spin gates (same inventory class as `cpmd-rattle-water`,
`xl-bomd-uks`).

## Honest scope

- Collinear Γ-UKS only (RKS CPMD remains product on the same flag)
- OT / XL / second-gen-CP / k-point CPMD refused or out of scope
- Electronic RATTLE (pos + vel); geometry transport Löwdin + vel project
- No electronic thermostat

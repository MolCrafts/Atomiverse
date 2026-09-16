# Reference — Car–Parrinello MD with electronic RATTLE (H₂O NVE, GPW)

## Method

**Car–Parrinello molecular dynamics (CPMD)** with fictitious electronic
degrees of freedom on closed-shell Γ-point GPW, using **electronic RATTLE**
to enforce \(C^\top S C = I\) (position) and \(\dot\sigma = 0\) (velocity):

- Outer loop: `cpu::md::MD` + NVE `VelocityVerlet` (nuclei)
- Force slot: `cpu::gpw::GPWEvaluator` with **`cpmd = true`**,
  **`cpmd_mu = 2000` a.u.**, `compute_forces = true`; the electronic dt is
  **not** set on the evaluator — `cpu::md::MD` pushes the nuclear dt in from
  the `VelocityVerlet` integrator through the `TimestepProvider` /
  `TimestepConsumer` protocol at build
- Propagator: `cpu::electronic::Cpmd` — velocity-Verlet + Andersen RATTLE on
  occupied MO coefficients \(C\) (supersedes Löwdin-only `cpmd-water`)

**Not BOMD** (production steps use fixed-density KS evaluation, `iters==0`).
**Not XL-BOMD**. **Not classical nuclear RATTLE** (`FixRattle`).

## Literature

- Car & Parrinello, *Phys. Rev. Lett.* **55**, 2471 (1985)
- H. C. Andersen, *J. Comput. Phys.* **52**, 24 (1983) — RATTLE
- Marx & Hutter, *Ab Initio Molecular Dynamics* (CUP) — CP conserved energy
  \(E_{\mathrm{cons}} = T_{\mathrm{n}} + T_{\mathrm{e}} + E_{\mathrm{KS}}\)
- Twin geometry family: `cpmd-water`, `gpw-bomd-water`, `xl-bomd-water`

## Setup

| Item | Value |
| ---- | ----- |
| System | Isolated H₂O (CP2K getting-started geometry) |
| Cell | 10 Å cubic, Martyna–Tuckerman (`nonperiodic`) |
| XC / basis / PP | PBE / DZVP-MOLOPT-GTH / GTH-PBE |
| CUTOFF | 200 Ry |
| dt | 0.1 fs (nuclear = electronic) |
| \(\mu\) | 2000 a.u. |
| Production | 1 SCF seed + 1 CP baseline + 4 CP drift steps |

## Pin

| Observable | Gate |
| ---------- | ---- |
| \(E_{\mathrm{cons}}\) relative drift | ≤ **0.05** (tighter than Löwdin foothold 0.10) |
| Production SCF iters | **0** (fixed-density) |
| O–H distances | < 1.8 Å |
| PE after warm-up | < −100 eV |

**Offline CP2K twin input:** `tools/cp2k_oracle/inputs/cpmd_rattle_water_nve.inp`
(QS PBE / DZVP-MOLOPT-GTH / GTH-PBE, MT, 10 Å, short NVE). Pin is a
**literature conservation envelope**, not a live CP2K energy twin in ctest.

**perf_exempt** — no CP2K CPMD+RATTLE wall twin on the worker host; literature
envelope + structural gates (same inventory class as `cpmd-water`).

## Honest scope

- Closed-shell RKS Γ only; UKS / OT / k-points refused at `build()`
- Electronic RATTLE (pos + vel); geometry transport still uses Löwdin + vel project
- No electronic thermostat

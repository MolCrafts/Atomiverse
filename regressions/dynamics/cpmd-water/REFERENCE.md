# Reference — Car–Parrinello MD short trajectory (H₂O NVE, GPW)

## Method

**Car–Parrinello molecular dynamics (CPMD)** with fictitious electronic
degrees of freedom on closed-shell Γ-point GPW:

- Outer loop: `cpu::md::MD` + NVE `VelocityVerlet` (nuclei)
- Force slot: `cpu::gpw::GPWEvaluator` with **`cpmd = true`**,
  **`cpmd_mu = 2000` a.u.**, `compute_forces = true`; the electronic dt is
  **not** set on the evaluator — `cpu::md::MD` pushes the nuclear dt in from
  the `VelocityVerlet` integrator through the `TimestepProvider` /
  `TimestepConsumer` protocol at build, so nuclei and electrons share one clock
  by construction
- Propagator: `cpu::electronic::Cpmd` — position Verlet on occupied MO coefficients
  \(C\) with Löwdin re-orthonormalization
- \(E_{\mathrm{cons}}\) is **not** assembled by this case: `cpu::md::CpmdConserved`
  (bound at its default `end_of_step @ -90`, after `KineticEnergy @ -100`) sums
  \(E_{\mathrm{KS}} + T_{\mathrm{n}} + T_e + E_{\mathrm{therm}}^{e}\) onto
  `frame.meta[kConserved]` [eV] each step, and the case reads that key — the
  same channel Thermo's `Conserved` column and a user drift `Criterion` see.
  \(T_e\) likewise comes off `frame.meta[kEnergyElectronKinetic]`, published by
  every CPMD launch, rather than off an evaluator readback

**Not BOMD** (production steps use fixed-density KS evaluation, `iters==0`).
**Not XL-BOMD** (no auxiliary density + SCF; electronic kinetic energy is
physical to the CP Lagrangian).

## Literature

- Car & Parrinello, *Phys. Rev. Lett.* **55**, 2471 (1985)
- Marx & Hutter, *Ab Initio Molecular Dynamics* (CUP) — CP conserved energy
  \(E_{\mathrm{cons}} = T_{\mathrm{n}} + T_{\mathrm{e}} + E_{\mathrm{KS}}\)
- Twin geometry / basis family: `gpw-bomd-water`, `xl-bomd-water`

## Setup

| Item | Value |
| ---- | ----- |
| System | Isolated H₂O (CP2K getting-started geometry) |
| Cell | 10 Å cubic, Martyna–Tuckerman (`nonperiodic`) |
| XC / basis / PP | PBE / DZVP-MOLOPT-GTH / GTH-PBE |
| CUTOFF | 200 Ry |
| dt | 0.1 fs (nuclear = electronic) |
| \(\mu\) | 2000 a.u. (AO residual stiffer than PW textbook ~400) |
| Production | 1 SCF seed + 1 CP baseline + 4 CP drift steps |

## Pin

| Observable | Gate |
| ---------- | ---- |
| \(E_{\mathrm{cons}}\) relative drift | ≤ 0.10 |
| Production SCF iters | **0** (fixed-density) |
| O–H distances | < 1.8 Å |
| PE after warm-up | < −100 eV |

**perf_exempt** — no CP2K CPMD twin wall on the worker host; literature
envelope + structural gates.

## Honest scope

- Closed-shell RKS Γ only; UKS / OT / k-points refused at `build()`
- Löwdin reorth foothold (not full RATTLE)
- No electronic thermostat
- Soft drift tol vs BO (0.10 vs 0.05) for short Löwdin foothold traj

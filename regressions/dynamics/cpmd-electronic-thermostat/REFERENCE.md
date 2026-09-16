# Reference — Car–Parrinello MD with electronic Nosé–Hoover thermostat (H₂O, GPW)

## Method

**Car–Parrinello molecular dynamics (CPMD)** with fictitious electronic
degrees of freedom on closed-shell Γ-point GPW, **electronic RATTLE**, and a
**Nosé–Hoover electronic thermostat** on \(\dot C\):

- Outer loop: `cpu::md::MD` + NVE `VelocityVerlet` (nuclei)
- Force slot: `cpu::gpw::GPWEvaluator` with **`cpmd = true`**,
  **`cpmd_thermostat = true`**, **`cpmd_ekin_target_ha`**, **`cpmd_tau_e`**,
  **`cpmd_mu = 2000` a.u.**, `compute_forces = true`; the electronic dt is
  **not** set on the evaluator — `cpu::md::MD` pushes the nuclear dt in from
  the `VelocityVerlet` integrator through the `TimestepProvider` /
  `TimestepConsumer` protocol at build
- Propagator: `cpu::electronic::Cpmd` — VV + Andersen RATTLE + NH half-steps

**Not BOMD** (production `iters==0`). **Not nuclear NVT** as the pin (nuclei
NVE; only electronic DOF thermostatted). **Not XL-BOMD / SGCP**.

## Literature

- Car & Parrinello, *Phys. Rev. Lett.* **55**, 2471 (1985)
- Andersen, *J. Comput. Phys.* **52**, 24 (1983) — RATTLE
- Blöchl & Parrinello, *Phys. Rev. B* **45**, 9413 (1992) — separate electronic
  thermostats for adiabaticity
- Sprik, *J. Phys. Chem.* **95**, 2283 (1991)
- Marx & Hutter, *Ab Initio Molecular Dynamics* (CUP) — extended conserved
  energy with electronic NH:
  \(E_{\mathrm{ext}} = T_{\mathrm{n}} + T_{\mathrm{e}} + E_{\mathrm{KS}} + E_{\mathrm{therm}}\)
- Twin family: `cpmd-rattle-water`, `cpmd-water`

## Setup

| Item | Value |
| ---- | ----- |
| System | Isolated H₂O (CP2K getting-started geometry) |
| Cell | 10 Å cubic, Martyna–Tuckerman (`nonperiodic`) |
| XC / basis / PP | PBE / DZVP-MOLOPT-GTH / GTH-PBE |
| CUTOFF | 200 Ry |
| dt | 0.1 fs (nuclear = electronic) |
| \(\mu\) | 2000 a.u. |
| \(K_0\) | \(1\times 10^{-5}\) Ha (cool electronic target) |
| \(\tau_e\) | 50 fs |
| Production | 1 SCF seed + 1 CP baseline + 4 CP drift steps |

## Pin

| Observable | Gate |
| ---------- | ---- |
| \(E_{\mathrm{ext}}\) relative drift | ≤ **0.10** |
| Production SCF iters | **0** (fixed-density) |
| \(T_e\) ceiling over production | ≤ **0.05** eV |
| O–H distances | < 1.8 Å |
| PE after warm-up | < −100 eV |

**Offline CP2K twin input:** `tools/cp2k_oracle/inputs/cpmd_electronic_thermostat_h2o.inp`
(QS PBE / DZVP-MOLOPT-GTH / GTH-PBE, MT, 10 Å, short MD). Classical fictitious-mass
CPMD with electronic NH is not a stock CP2K QS ensemble twin — pin is a
**literature extended-energy / \(T_e\) envelope**, not a live CP2K energy twin in ctest.

**perf_exempt** — no CP2K classical-CPMD+e-T wall twin on the worker host;
literature envelope + structural gates (same inventory class as `cpmd-rattle-water`).

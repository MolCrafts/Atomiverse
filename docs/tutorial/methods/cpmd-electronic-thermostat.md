# Car–Parrinello MD + Nosé–Hoover electronic thermostat — H₂O (GPW)

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `cpmd-electronic-thermostat` |
| Regression | `regressions/dynamics/cpmd-electronic-thermostat/` |
| Public API | `cpu::md::MD` + `VelocityVerlet` + `cpu::electronic::CarParrinello{cpu::gpw::GPWEvaluator{…, compute_forces=true}, mu_au, cpu::electronic::ElectronicThermostat{ekin_target_ha, tau_e_fs}}` bound with `set_evaluator_ref`; propagator `cpu::electronic::Cpmd` (RATTLE + NH). The thermostat is the **three-argument wrapper constructor**, not a set of evaluator flags. The nuclear dt is pushed from the integrator into the wrapper by `MD` (`TimestepProvider` → `TimestepConsumer`) — never set on the evaluator |
| Roadmap | MD AIMD electronic thermostat residual closed |
| Dual gate | Correctness from REFERENCE (E_ext drift, Te ceiling, fixed-density); speed **`perf_exempt`** |
| Wave | CP2K capability loop — wave 11 **done** (merged 2026-07-26) |
| Oracle | Offline CP2K geometry/basis twin + literature envelope (never live CP2K in ctest) |

## Summary

**Car–Parrinello molecular dynamics** (Car & Parrinello, *PRL* **55**, 2471
(1985)) with **electronic RATTLE** (Andersen, *J. Comput. Phys.* **52**, 24
(1983)) and a **Nosé–Hoover electronic thermostat** on the fictitious orbital
velocities \(\dot C\) (Blöchl & Parrinello, *PRB* **45**, 9413 (1992); Sprik;
Marx & Hutter). Nuclei integrate under ordinary NVE `VelocityVerlet`. Electrons
share the same \(\Delta t\), advance with velocity-Verlet + RATTLE, and have
their fictitious kinetic energy \(T_e = \tfrac12\mu\lVert\dot C\rVert_F^2\)
regulated toward a cool target \(K_0\) with coupling time \(\tau_e\).

After one cold SCF seed, production force steps are **fixed-density** KS
evaluations (`iters==0`) — not BOMD, not XL-BOMD, not second-generation CP,
and **not** nuclear NVT (only the electronic DOF are thermostatted in this pin).

Extended conserved energy (eV on the public surface; Ha internal OK when cited):

\[
E_{\mathrm{ext}} = T_{\mathrm{n}} + T_{\mathrm{e}} + E_{\mathrm{KS}} + E_{\mathrm{therm}},
\]

with \(E_{\mathrm{therm}} = \tfrac12 Q\xi^2 + 2 K_0\eta\), \(Q = 2 K_0\tau_e^2\),
exposed as `last_electronic_thermostat_ev()`. Pure CP NVE conserves
\(E_{\mathrm{cons}} = T_{\mathrm{n}} + T_{\mathrm{e}} + E_{\mathrm{KS}}\) instead
(`cpmd-rattle-water` / `cpmd-water`).

Scope of this foothold: closed-shell RKS Γ-point GPW, PBE / DZVP-MOLOPT-GTH /
GTH-PBE, Martyna–Tuckerman isolated H₂O. UKS dual-propagator + electronic NH
is allowed by the `Cpmd` API per spin channel; this regression pins RKS only.

## Method

### Extended Lagrangian (electrons + NH)

Occupied KS coefficients \(C\) (\(n_b\times n_{\mathrm{occ}}\)) are classical
fictitious DOF with mass \(\mu\). Closed-shell band-energy gradient with
S-tangent projection:

\[
\mu\,\ddot C = -(I - S C C^\top)\,(4 H C)
\quad(P = 2 C C^\top).
\]

Holonomic orthonormality \(C^\top S C = I\) is enforced by **electronic RATTLE**
(position SHAKE with Lagrange multipliers, then velocity stage killing
\(\mathrm{sym}(C^\top S \dot C)\)).

Optional **Nosé–Hoover** on \(\dot C\): friction \(\xi\) and coordinate \(\eta\)
with target \(K_0\) and thermostat mass \(Q = 2 K_0\tau_e^2\). Symmetric NH
half-steps wrap the mechanical VV+RATTLE step so the extended energy
\(E_{\mathrm{ext}}\) is the literature conserved quantity when the electronic
bath is engaged (Marx & Hutter).

### Nuclear dynamics

Nuclei: NVE Velocity Verlet at the same \(\Delta t\) (fs). No nuclear
thermostat / barostat on this pin — adiabatic separation is maintained by
keeping \(T_e\) cool relative to nuclear KE, not by nuclear NVT.

### Primary observables (envelope, not absolute twin energy)

| Observable | Gate (hard-coded) |
| ---------- | ----------------- |
| Relative \(E_{\mathrm{ext}}\) drift | \(\le 0.10\) |
| Production SCF `iters` | **0** (fixed-density CP) |
| Max \(T_e\) over production | \(\le 0.05\) eV |
| O–H distances | \(< 1.8\) Å |
| PE after warm-up | \(< -100\) eV |

### System (regression settings)

| Quantity | Value |
| -------- | ----- |
| Species | Isolated H₂O (CP2K getting-started geometry) |
| Cell | 10 Å cubic, MT / `nonperiodic = true` |
| XC / basis / PP | PBE / DZVP-MOLOPT-GTH / GTH-PBE-q6,q1 |
| CUTOFF | 200 Ry |
| \(\Delta t\) | 0.1 fs (nuclear = electronic) |
| \(\mu\) | 2000 a.u. |
| \(K_0\) | \(1\times 10^{-5}\) Ha |
| \(\tau_e\) | 50 fs |
| Initial velocity | COM-preserving H stretch \(\pm 0.0015\) Å/fs |
| Steps | 1 warm-up (SCF seed) + 1 CP baseline + 4 production |

| Atom | \(x\) | \(y\) | \(z\) (Å) |
| ---- | ----: | ----: | --------: |
| O | 5.0000 | 5.0000 | 5.0000 |
| H1 | 5.7586 | 5.0000 | 5.5043 |
| H2 | 4.2414 | 5.0000 | 5.5043 |

### Excluded / non-goals

- Nuclear NVT / Nose–Hoover on ions as the dual-gate pin
- BOMD full SCF each step; XL-BOMD; second-generation CP (ASPC + partial SCF)
- Live CP2K classical fictitious-mass CPMD+e-T energy twin (not a stock QS
  ensemble) — geometry/basis twin only at
  `tools/cp2k_oracle/inputs/cpmd_electronic_thermostat_h2o.inp`
- OT seed path, k-points, hybrid / GAPW / PW CP
- Public `*Config` bags / `make_*` factories

## Implementation

Public surface only (`cpu::Frame`, constructor arguments; no `*Config`).

```cpp
constexpr double dt = 0.1; // fs — nuclear and electronic share one step
cpu::gpw::GPWEvaluator gpw(
    {O, H}, /*cutoff_ry*/ 200.0, /*max_iter*/ 60, /*tol*/ 1e-5,
    /*mix*/ 0.3, cpu::gpw::XcKind::PBE, /*nonperiodic*/ true,
    /*compute_forces*/ true);
// Thermostatted CP is the three-argument wrapper constructor: fictitious mass
// mu [a.u. electron mass], then the Nosé–Hoover setpoint. The evaluator has no
// public cpmd_* flags to keep in sync.
cpu::electronic::CarParrinello eval{
    std::move(gpw), /*mu_au=*/2000.0,
    cpu::electronic::ElectronicThermostat{/*ekin_target_ha=*/1e-5, // K0 [Ha]
                                          /*tau_e_fs=*/50.0}};     // [fs]
// No electronic dt: MD pushes the integrator's dt into the wrapper at build.

cpu::md::MD sim;
sim.set_integrator(cpu::md::VelocityVerlet{dt});  // the ONE owner of dt
sim.set_evaluator_ref(eval);        // borrow: readbacks stay live after run()
sim.build();
sim.run(frame, n_steps);

// Extended energy [eV]:
const double e_ext = frame.meta.get(kKinetic)
                   + frame.meta.get(kEnergy)
                   + eval.last_electronic_ke_ev()
                   + eval.last_electronic_thermostat_ev();
```

Propagator primitive (`include/atomiverse/cpu/electronic/cpmd.h`):

```cpp
cpu::electronic::Cpmd cp(2000.0);   // throws if mu <= 0
cp.thermostat = true;
cp.ekin_target_ha = 1e-5;
cp.tau_e_fs = 50.0;
cp.seed(C_occ);              // after cold SCF; NH state reset
cp.align_to_overlap(S);      // after nuclei move
cp.advance(H, S, dt_fs);     // NH½ → VV+RATTLE → NH½
double te = cp.kinetic_ha();
double eth = cp.thermostat_energy_ha();
```

`build()` refuses `cpmd_thermostat` without `cpmd`, non-positive
\(K_0\) / \(\tau_e\), and the usual CP exclusions (OT / XL / second-gen).

```bash
cmake --build build -j$(nproc) --target atomiverse test_cpu_gpw_cpmd \
      reg_cpmd-electronic-thermostat
ctest --test-dir build -R '^(test_cpu_gpw_cpmd)$' --output-on-failure
cmake --build build --target run_reg_cpmd-electronic-thermostat
```

Headers / sources: `include/atomiverse/cpu/electronic/cpmd.h`,
`cpu/gpw/gpw_evaluator.h`; `src/cpu/electronic/cpmd.cpp`,
`src/cpu/gpw/gpw_evaluator.cpp`.
Unit: `tests/cpu/gpw/test_gpw_cpmd.cpp` (NH param guards, Te regulation,
evaluator build accept/reject).

## Validation (vs offline CP2K / literature)

Source: `regressions/dynamics/cpmd-electronic-thermostat/REFERENCE.md`
(offline pin; never live CP2K in ctest).

| Quantity | Atomiverse / pin | Oracle / literature | Tol / gate | Notes |
| -------- | ---------------: | ------------------- | ---------- | ----- |
| \(E_{\mathrm{ext}}\) relative drift | ≤ 0.10 | Marx–Hutter extended energy envelope | 0.10 | NVE nuclei + e-NH |
| Production SCF iters | 0 | CP fixed-density definition | exact | not BOMD |
| Max \(T_e\) | ≤ 0.05 eV | cool electronic target \(K_0=10^{-5}\) Ha | 0.05 eV | runaway gate |
| \(r_{\mathrm{OH}}\) | < 1.8 Å | structural stability | soft | both H |
| PE after warm-up | < −100 eV | PBE water bound | soft | after SCF seed |
| Geometry / basis twin | H₂O MT PBE MOLOPT GTH 200 Ry | `cpmd_electronic_thermostat_h2o.inp` | setup | not energy twin |

Geometry, basis, CUTOFF, \(\mu\), \(K_0\), \(\tau_e\), dt — all from REFERENCE.
Classical fictitious-mass CPMD with electronic NH is **not** a stock CP2K QS
ensemble twin; absolute energy parity is out of scope for this slug.

## Performance

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | ------------------------: | ----------------------: | ----: | ---- | ------ |
| — | n/a | short H₂O NVE (≤60 s class) | — | **perf_exempt** | no CP2K classical-CPMD+e-T wall twin on worker host |

Gate: literature envelope + structural gates (same inventory class as
`cpmd-rattle-water`). wave 11 — ratio **`perf_exempt`**.

## Limitations

- Short foothold trajectory (1 + 1 + 4 steps), not multi-ps production AIMD.
- RKS Γ only in the regression; UKS dual NH not e2e-pinned here
  (`cpmd-uks` is NVE electrons).
- Absolute \(E_{\mathrm{KS}}\) vs CP2K is **not** gated (envelope only).
- Nuclear thermostat / joint ion–electron NVT-CP is out of scope.
- Follow-ons: longer production with reported \(\langle T_e\rangle\), UKS+e-NH
  e2e, optional multi-chain electronic NH (literature; not product yet).

## References

- Offline pin: `regressions/dynamics/cpmd-electronic-thermostat/REFERENCE.md`
- Regression: `regressions/dynamics/cpmd-electronic-thermostat/reg_cpmd-electronic-thermostat.cpp`
- CP2K twin input: `tools/cp2k_oracle/inputs/cpmd_electronic_thermostat_h2o.inp`
- Oracle rules: `tools/cp2k_oracle/README.md`
- Sibling footholds: [cpmd-water](cpmd-water.md),
  [cpmd-rattle-water](cpmd-rattle-water.md)
- Car & Parrinello, *Phys. Rev. Lett.* **55**, 2471 (1985)
- Andersen, *J. Comput. Phys.* **52**, 24 (1983)
- Blöchl & Parrinello, *Phys. Rev. B* **45**, 9413 (1992)
- Marx & Hutter, *Ab Initio Molecular Dynamics* (CUP)

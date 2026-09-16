# AO-basis CPMD on the SCF methods — RHF / RKS / UHF

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `cpmd-rhf-h2` (family page: also `cpmd-rks-water`, `cpmd-uhf-o2`) |
| Regressions | `regressions/dynamics/cpmd-rhf-h2/`, `.../cpmd-rks-water/`, `.../cpmd-uhf-o2/` |
| Public API | `cpu::md::MD` + `VelocityVerlet` + `cpu::electronic::CarParrinelloMethod<cpu::scf::{RHF,RKS,UHF,UKS}>{method, mu_au}` bound with `set_evaluator_ref`; `cpu::md::CpmdConserved` for \(E_{\mathrm{cons}}\); propagator `cpu::electronic::Cpmd` |
| Spec | `esmm-cpmd-04-ao-cpmd` (ac-002 zero production iterations, ac-003 drift envelopes) |
| Dual gate | Correctness = the three regressions; speed **`perf_exempt`** |
| GPW sibling | [cpmd-water](cpmd-water.md) — same physics on the plane-wave/GPW side |

## Summary

**Car–Parrinello molecular dynamics** (Car & Parrinello, *PRL* **55**, 2471
(1985); Marx & Hutter, *Ab Initio Molecular Dynamics*, CUP 2009, §2.4–2.5)
treats the occupied molecular orbitals as classical fictitious degrees of
freedom with mass \(\mu\). On the Gaussian-basis (AO) side the propagated
variables are the occupied **MO coefficient columns** \(C_\sigma\) of the
determinant, and the extended Lagrangian is

\[
\mathcal{L} = \sum_I \tfrac12 M_I \dot R_I^2
            + \sum_\sigma \tfrac12 \mu \lVert \dot C_\sigma\rVert^2
            - E[\{C\},\{R\}]
            - \text{constraints }(C_\sigma^\top S C_\sigma = I),
\]

giving \(\mu\,\ddot C = -(I - S C C^\top)\,\partial E/\partial C\) — the
electronic RATTLE that lives inside `cpu::electronic::Cpmd`.

Two things make this work on an AO method rather than on a plane-wave grid:

1. a **fixed-density evaluation entry** — build the Fock/KS matrix and the
   energy at a *given* \(P\), with no SCF iterations at all; and
2. **forces at a non-eigenstate \(C\)** — the full Lagrange-multiplier form of
   the Pulay weight,
   \(W_\sigma = \mathrm{occ}\cdot C_\sigma (C_\sigma^\top F_\sigma C_\sigma)
   C_\sigma^\top\), which reduces to \(\sum_i \varepsilon_i c_i c_i^\top\)
   exactly when \(C\) are eigenvectors of \(F\).

The first nested step is a **cold seed**: one full converged SCF plus
Born–Oppenheimer forces, whose occupied columns seed the propagator. Every
later step is one fixed-density evaluation plus one electronic advance, so
`iterations()` is **0** forever after. RHF/RKS propagate one channel at
occupancy 2; UHF/UKS propagate α and β independently at occupancy 1 each.

This is **not** Born–Oppenheimer MD (BOMD re-converges the SCF every step) and
**not** XL-BOMD (auxiliary density + SCF). The conserved quantity is

\[
E_{\mathrm{cons}} = T_{\mathrm{n}} + T_{e} + E_{\mathrm{el}}
                    \;(+\; E^{e}_{\mathrm{therm}}).
\]

## API

Car–Parrinello is a **wrapper type**, not a flag on the method: the mode bits
live in `protected` storage on `MeanFieldMethod`, and
`cpu::electronic::CarParrinelloMethod<Method>`
(`cpu/electronic/aimd_method.h`) is the only public way to set them. The
wrapper publicly inherits `Method`, so every ordinary member (`basis`, `tol`,
`max_iter`, …) is still yours to set on the object you move in, and the
wrapper is still an `HF`-kind Component for the force slot.

| Public surface | Meaning |
| --- | --- |
| `CarParrinelloMethod<Method>{Method method, double mu_au}` | Move a configured `RHF` / `RKS` / `UHF` / `UKS` in and enable CP with fictitious electronic mass μ [a.u. electron mass]. Throws `std::invalid_argument` unless μ is finite and > 0. See *Choosing μ* — the AO residual is stiffer than a plane-wave one, so the textbook 400 is often too light. **Nested MD force slot only** — a standalone `run()` / `forces()` on a CP-enabled method throws `std::logic_error` (there is no nuclear trajectory for the electrons to follow at a single point) |
| `CarParrinelloMethod<Method>{Method, double mu_au, ScfElectronicThermostat{ekin_target_ha, tau_e_fs}}` | Same, plus a Nosé–Hoover thermostat on the fictitious DOF (Blöchl & Parrinello, *PRB* **45**, 9413 (1992)): target fictitious kinetic energy \(K_0\) [Hartree] and coupling time τ_e [fs]. The two-argument constructor is NVE electrons |
| `double last_electronic_ke_ev() const` | \(T_e\) [eV] of the last step; also on `frame.meta[kEnergyElectronKinetic]` |
| `double last_electronic_thermostat_ev() const` | \(E^{e}_{\mathrm{therm}}\) [eV]; exactly 0 without the thermostat; also on `frame.meta[kEnergyElectronThermostat]` |

**There is deliberately no `cpmd_dt` member.** The method implements
`cpu::TimestepConsumer` (`timestep_required()` is exactly "CP is on"), and
`cpu::md::MD::build()` pushes the nuclear timestep from the integrator
(`TimestepProvider`) into the method before anything builds. Nuclei and
electrons therefore share **one** clock by construction; wrapping a method in
`CarParrinelloMethod` without an MD integrator throws at `build_on` ("cpmd
requires an MD integrator publishing the timestep"). This is the same protocol
the GPW evaluator uses.

**`determinant()` is not refreshed on production steps.** Propagated orbitals
have no canonical eigenvalues, so the published wavefunction (coefficients,
orbital energies, occupations, \(\langle S^2\rangle\)) stays the **seed SCF's**.
Observers must read it as the launch state, not the current one — a documented
limitation, not an oversight.

**Refusal lattice.** ROHF/ROKS (shared spatial orbitals are not independent
channels), GHF (spinor CP is new physics), `predictor()` (no ASPC-CP hybrid),
`ri != RI::None`, `x2c`, `solvent_eps > 0`, `mom != Off`, `soscf`, non-empty
`point_charges` (the QM/MM × CPMD seam — lifting this gate later *is* the
integration), nonzero `external_field`, `d3bj`/`d4`, `compute_force = false`,
non-positive or non-finite μ / thermostat parameters, and a
near-singular overlap (`n_dropped > 0`) at the seed all throw. In particular
**dispersion is refused because `ScfGradient` carries no d3bj/d4 force terms**
while the total energy includes the dispersion contribution —
\(E_{\mathrm{cons}}\) would drift silently. `xl_bomd` (the `esmm-cpmd-05`
sibling, [xl-bomd-rhf-water](xl-bomd-rhf-water.md)) landed and is mutually
exclusive with `cpmd` in **both** directions — the symmetric row is asserted by
`MeanFieldMethodXlBomd.RefusesCpmd`.

## Usage

```cpp
constexpr double dt = 0.05; // fs — nuclear and electronic share one step

cpu::scf::RKS rks_base{cpu::scf::XCType::LSDA};  // or RHF / UHF{n_a,n_b} / UKS
rks_base.basis = cpu::scf::BasisSet::STO3G;
rks_base.tol   = 1e-9;
// CP is the wrapper type; mu [a.u. electron mass] is its constructor argument.
cpu::electronic::CarParrinelloMethod<cpu::scf::RKS> rks{std::move(rks_base),
                                                        /*mu_au=*/1000.0};
// No electronic dt: MD pushes the integrator's dt into the method at build.

cpu::md::MD sim;
sim.set_integrator(cpu::md::VelocityVerlet{dt});  // the ONE owner of dt
sim.set_evaluator_ref(rks);       // borrow: iterations() stays readable after
sim.set_observer(cpu::md::CpmdConserved{});       // E_cons -> meta[kConserved]
sim.build();

sim.run(frame, 1);          // step 1 = cold SCF seed  (iterations() > 0)
sim.run(frame, n_steps);    // production               (iterations() == 0)

const double e_cons = frame.meta.get(kConserved);              // [eV]
const double t_e    = frame.meta.get(kEnergyElectronKinetic);  // [eV]
```

`cpu::md::CpmdConserved` (`cpu/dynamics/cpmd_conserved.h`) is the single owner
of \(E_{\mathrm{cons}}\): bound at `md::Phase::end_of_step` order −90 (after
`KineticEnergy` at −100), it sums `kEnergy + kKinetic + kEnergyElectronKinetic
+ kEnergyElectronThermostat` onto `frame.meta[kConserved]` [eV]. Do **not**
combine it with another `kConserved` owner (`NoseHooverNPT`, `NphIntegrator`,
`ParrinelloRahmanNPT`, `FixCSVR`) — CPMD is an NVE-nuclei path anyway. A drift
watchdog is ordinary user code: see the `StopWhenDrift` `Criterion` in
[cpmd-water](cpmd-water.md#watching-the-conserved-energy).

### Choosing μ

The AO residual is stiffer than a plane-wave one, so the plane-wave textbook
μ ≈ 400 a.u. is frequently **outside** the adiabatic window. Scanned worst
relative \(E_{\mathrm{cons}}\) drift over 40 production steps:

| Case | μ = 400, dt 0.05 | μ = 1000, dt 0.05 | μ = 2000, dt 0.05 | μ = 400, dt 0.1 |
| ---- | ---- | ---- | ---- | ---- |
| RHF H₂ | 4.2e−07 | **4.2e−07** | 4.2e−07 | 1.6e−06 |
| RKS H₂O | 3.4e−06 | **5.9e−06** | 5.2e−06 | diverges (≈ step 25) |
| UHF O₂ | 1.1e−07 | **2.8e−07** | 2.6e−07 | 5.5e−02 (\(T_e\) → 161 eV) |

μ = 400 looks competitive at 40 steps but is already non-adiabatic: extended to
120 steps the water run dies and the O₂ electrons heat to \(T_e\) ≈ 25 eV,
while μ = 1000 stays flat in both. **μ = 1000 a.u. at dt = 0.05 fs** is the
setting all three regressions ship. Judge a candidate μ by watching
`kEnergyElectronKinetic` stay small and *bounded*, not by the drift over a
short window.

## Pin

- **Unit** (`test_cpu_scf_cpmd`): the full refusal lattice; standalone
  `run()`/`forces()` throws; both dt-gate surfaces; `compute_at_density`
  vs `compute` parity at a converged determinant (restricted and
  unrestricted); seed-then-step returns `iterations() == 0`; the electronic
  meta keys are published.
- **E2E:** three regressions, μ = 1000 a.u., dt = 0.05 fs, 1 SCF seed +
  1 CP baseline + 40 CP drift steps each. Envelopes are **empirical pins**
  (tuned, then frozen at ≈ 9–18× the observed worst drift); the citation
  supplies the *form* of the gate, not the number.

| Regression | Method / system | Observed drift | Envelope | \(T_e\) range [eV] | Wall |
| --- | --- | --- | --- | --- | --- |
| `cpmd-rhf-h2` | RHF/STO-3G, H₂ r = 0.74 Å | 4.195e−07 | **5e−6** | ≡ 0, gated < 1e−20 (see below) | 0.03 s |
| `cpmd-rks-water` | RKS(LSDA)/STO-3G, H₂O (CP2K getting-started) | 5.851e−06 | **5e−5** | 1.1e−07 … 7.4e−02 | 2.0 s |
| `cpmd-uhf-o2` | UHF triplet 9/7, STO-3G, O₂ r = 1.21 Å | 2.810e−07 | **5e−6** | 2.1e−10 … 7.7e−03 | 1.3 s |

Every case additionally pins: production `iterations() == 0` on **every**
post-seed step; `kConserved == kEnergy + kKinetic + T_e + E_therm`
re-assembled from the frame (observed residual bit-exact 0); \(T_e = 0\) and
\(E^{e}_{\mathrm{therm}} = 0\) where they must be; and a bond-length window.
`cpmd-uhf-o2` also pins the seed determinant's \(N_\alpha/N_\beta = 9/7\) and
spin contamination \(\langle S^2\rangle - S(S+1) = 0.0128 < 0.5\).

### Why H₂ shows \(T_e \equiv 0\)

H₂ is homonuclear, so its AO Fock matrix is persymmetric at every bond length,
and in the 2-function STO-3G space the σ_g and σ_u blocks are
**one-dimensional**. The occupied σ_g vector is therefore geometry-independent
and remains an exact S-metric eigenvector along the whole trajectory: the CP
band force \(-(I - S C C^\top)(\partial E/\partial C)\) vanishes identically and
\(T_e\) stays at double-precision zero (observed ≤ 2.3e−32 eV). That is a
symmetry identity, not a broken propagator — and `cpmd-rhf-h2` **gates** it
(\(T_e\) < 1e−20 eV), because a nonzero value there would mean the projector or
the multiplier weight is wrong. The unit suite documents the same symmetry and
uses HeH⁺ (no persymmetry, \(T_e > 0\)) as the counterexample.

`cpmd-rhf-h2` is still the right machinery case — it pins the seed, the
zero-iteration production steps, the dt protocol and the conserved channel. The
*moving* electronic subsystem is what `cpmd-rks-water` and `cpmd-uhf-o2` add.
O₂ is homonuclear too, but its σ_g/σ_u blocks are **three-dimensional**
(1s/2s/2p_z combinations), so the occupied subspace genuinely mixes as the bond
changes and \(T_e > 0\) is a hard gate there.

## Source

| What | Path |
| ---- | ---- |
| CP wrapper (`CarParrinelloMethod`, `ScfElectronicThermostat`) | `include/atomiverse/cpu/electronic/aimd_method.h` |
| Mode storage, `TimestepConsumer`, seed/step | `include/atomiverse/cpu/scf/mean_field_method.h`, `src/cpu/scf/mean_field_method.cpp` |
| Fixed-state gradient (`compute_at_density`) | `include/atomiverse/cpu/scf/internal/gradient.h`, `src/cpu/scf/internal/gradient.cpp` |
| Fixed-density energy | `src/cpu/scf/internal/mean_field_energy.cpp` |
| Propagator | `include/atomiverse/cpu/electronic/cpmd.h` |
| \(E_{\mathrm{cons}}\) assembler | `include/atomiverse/cpu/dynamics/cpmd_conserved.h` |
| Unit tests | `tests/cpu/scf/test_scf_cpmd.cpp` |
| Regressions | `regressions/dynamics/cpmd-rhf-h2/`, `cpmd-rks-water/`, `cpmd-uhf-o2/` |

## Honest non-goals

- PW CPMD, OT-CP, multi-k, CPMD stress — all out of scope here (XL-BOMD on the
  SCF methods shipped separately as `esmm-cpmd-05`:
  [xl-bomd-rhf-water](xl-bomd-rhf-water.md))
- QM/MM × CPMD: the `point_charges` gate stays shut
- The d3bj/d4 gradient gap is **verified and refused**, not fixed
  (`ScfGradient` has no dispersion force terms while `e_total_` includes the
  dispersion energy)
- GGA / hybrid XC gradients at a non-stationary density run the same code path
  but are not pinned by a regression (LSDA only)
- No third-party AO-CPMD twin is run at ctest time (`perf_exempt`): the gates
  are the CP conservation and adiabaticity criteria plus structural sanity

# Car–Parrinello molecular dynamics

> The regression runner has been removed. Regression commands below are historical; case sources and oracle records are retained as archived evidence, not current executable checks.

Car–Parrinello molecular dynamics (CPMD; Car & Parrinello, *PRL* **55**, 2471
(1985); Marx & Hutter, *Ab Initio Molecular Dynamics*, CUP 2009, §2.4–2.5)
treats the occupied orbitals as classical fictitious degrees of freedom with
mass \(\mu\). After one cold SCF seed, production steps do **not** re-converge
the SCF: they evaluate the Kohn–Sham (or Fock) energy and matrix once at the
current density and take analytic nuclear forces at that fixed electronic
state, so the SCF iteration count is 0 on every production step. This is not
[Born–Oppenheimer MD](dynamics-bomd.md#gpw-bomd-water) (SCF every step), not
[XL-BOMD](dynamics-xl-bomd.md#xl-bomd-water) (auxiliary density + SCF), and not
[second-generation CP](dynamics-second-gen-cpmd.md#second-gen-cpmd-water) (ASPC +
partial SCF, no fictitious mass).

CP mode is a **wrapper type**, not a flag on the method. On the GPW side
`cpu::electronic::CarParrinello{cpu::gpw::GPWEvaluator{GpwDft{…},
/*compute_forces=*/true}, mu_au}` (`cpu/electronic/car_parrinello.h`), with an
optional third argument `cpu::electronic::ElectronicThermostat{ekin_target_ha,
tau_e_fs}`; the `cpmd` / `cpmd_mu` mode bits are private on `GPWEvaluator`. On
the Gaussian SCF side `cpu::electronic::CarParrinelloMethod<Method>{method,
mu_au}` for `cpu::scf::{RHF, RKS, UHF, UKS}` (`cpu/electronic/aimd_method.h`),
with optional `ScfElectronicThermostat{ekin_target_ha, tau_e_fs}`. Either
wrapper is bound to `cpu::md::MD` + NVE `cpu::md::VelocityVerlet` with
`set_evaluator_ref` (a borrow, so readbacks stay live after `run()`), and
`cpu::md::CpmdConserved` publishes the conserved energy. The sections cover
closed-shell Γ GPW (H₂O), Γ-UKS GPW (O₂ triplet), multi-k RKS and UKS GPW
(periodic H₂), the electronic Nosé–Hoover thermostat, and the AO methods
(H₂ RHF, H₂O RKS, O₂ UHF); all CPU. The case also runs a
`cpmd-thermostat-kpoints` section (multi-k RKS CP with the electronic
thermostat per k channel; oracle record in the case `REFERENCE.md`).

Case: `ctest -R '^dynamics/car-parrinello$'` — sources and oracle record in [`regressions/dynamics/car-parrinello/`](../../../regressions/dynamics/car-parrinello/). One section of the case per `##` heading below: `atv_regressions --case dynamics/car-parrinello --section <name>`.

## Theory and common API

### Extended Lagrangian

\[
\mathcal{L} = \sum_I \tfrac12 M_I \dot R_I^2
            + \sum_i \tfrac12 \mu \lVert \dot c_i\rVert^2
            - E_{\mathrm{KS}}[C,R]
            - \text{constraints }(C^\top S C = I).
\]

Nuclei integrate with the ordinary MD `VelocityVerlet`. Electrons share the
same \(\Delta t\) and integrate with velocity Verlet + **electronic RATTLE**
(Andersen, *J. Comput. Phys.* **52**, 24 (1983)) on the occupied MO matrix
\(C\): a position stage with Lagrange multipliers restoring \(C^\top S C = I\),
then a velocity stage killing \(\mathrm{sym}(C^\top S \dot C)\). For
closed-shell \(P = 2 C C^\top\) the S-tangent band force is

\[
\mu\,\ddot C = -(I - S C C^\top)\,(4 H C).
\]

The conserved quantity is

\[
E_{\mathrm{cons}} = T_{\mathrm{n}} + T_{\mathrm{e}} + E_{\mathrm{KS}},
\qquad T_{\mathrm{e}} = \tfrac12\mu\sum_i\lVert\dot c_i\rVert^2 ,
\]

with \(T_{\mathrm{e}}\) on `frame.meta[kEnergyElectronKinetic]` [eV] (also
`last_electronic_ke_ev()`) and \(E_{\mathrm{KS}}\) on
`frame.meta.get(kEnergy)`. With the electronic Nosé–Hoover thermostat
(Blöchl & Parrinello, *PRB* **45**, 9413 (1992)) the conserved quantity is the
extended energy \(E_{\mathrm{ext}} = E_{\mathrm{cons}} + E_{\mathrm{therm}}\)
(`kEnergyElectronThermostat`, `last_electronic_thermostat_ev()`). Public
energies are eV on the Frame; internal KS work is Hartree.

### Timestep: one clock

There is deliberately no `cpmd_dt` member. Both wrappers implement
`cpu::TimestepConsumer`, and `cpu::md::MD::build()` pushes the nuclear timestep
from the integrator (`TimestepProvider`) into the wrapper before anything
builds; nuclei and electrons therefore share one step by construction. A
CP-wrapped method without an MD integrator throws at `build_on` ("cpmd
requires an MD integrator publishing the timestep").

```cpp
constexpr double dt = 0.1; // fs — nuclear and electronic share one step
cpu::gpw::GpwDft dft{{O, H}, 200.0, cpu::gpw::XcKind::PBE};
dft.max_iter = 60;  dft.tol = 1e-5;  dft.mix = 0.3;
dft.nonperiodic = true;
cpu::gpw::GPWEvaluator gpw(std::move(dft), /*compute_forces=*/true);
// mu [a.u. electron mass] is a constructor argument of the CP wrapper.
cpu::electronic::CarParrinello eval{std::move(gpw), /*mu_au=*/2000.0};

cpu::md::MD sim;
sim.set_integrator(cpu::md::VelocityVerlet{dt});  // the ONE owner of dt
sim.set_evaluator_ref(eval);
sim.set_observer(cpu::md::CpmdConserved{});       // E_cons -> meta[kConserved]
sim.build();
sim.run(frame, n_steps);

const double e_cons = frame.meta.get(kConserved);  // [eV]
```

### Watching the conserved energy

`cpu::md::CpmdConserved` (`cpu/dynamics/cpmd_conserved.h`) is the single owner
of \(E_{\mathrm{cons}}\): bound at `md::Phase::end_of_step` order −90 (after
`KineticEnergy` at −100), it sums `kEnergy + kKinetic + kEnergyElectronKinetic
+ kEnergyElectronThermostat` onto `frame.meta[kConserved]` [eV], so Thermo's
`Conserved` column reports CP drift instead of a nuclear-only quantity — and it
throws (naming the missing key) rather than substituting 0 when no CPMD
evaluator ran. Do **not** combine it with another `kConserved` owner
(`NoseHooverNPT`, `NphIntegrator`, `ParrinelloRahmanNPT`, `FixCSVR`): two
components would fight over one channel, and CPMD is an NVE-nuclei path.

Because the value lives on the frame, a drift watchdog is ordinary user code —
a `Criterion` written outside the library (the LAMMPS `fix halt` shape):

```cpp
class StopWhenDrift : public cpu::Criterion {
  public:
    explicit StopWhenDrift(double max_rel) : max_rel_(max_rel) {}
    std::string name() const override { return "stop_when_drift"; }
    void launch(...) override {}  // verdict-only — no per-step work
    ATV::Verdict verdict(const cpu::LoopState& s) override {
        const double e = s.frame.meta.get(ATV::kConserved);
        if (!seeded_) { e0_ = e; seeded_ = true; return ATV::Verdict::Continue; }
        return (std::abs(e - e0_) > max_rel_ * std::abs(e0_))
                   ? ATV::Verdict::Stop : ATV::Verdict::Continue;
    }
  private:
    double max_rel_, e0_ = 0.0; bool seeded_ = false;
};

sim.add_criterion(StopWhenDrift{1e-3});  // halt past 0.1 % relative drift
```

Seed on the first poll, not at construction: the baseline must be the step that
engaged \(T_e\) (the cold-SCF seed step has \(T_e = 0\), so anchoring there
would count the electronic turn-on as drift). The GPW sections follow the same
protocol: 1 SCF seed + 1 CP baseline (\(T_e\) on) + production steps, drift
scaled by \(\max(1,\lvert E_0\rvert)\).

### Propagator primitives

`cpu::electronic::Cpmd` (`cpu/electronic/cpmd.h`, real Γ) and
`cpu::electronic::CpmdCplx` (complex Bloch \(C_k\), one per k and spin
channel):

```cpp
cpu::electronic::Cpmd cp(2000.0);  // throws if mu <= 0
cp.thermostat = true;              // optional NH: K0 [Ha], tau_e [fs]
cp.ekin_target_ha = 1e-5;
cp.tau_e_fs = 50.0;
cp.seed(C_occ);             // after cold SCF; NH state reset
cp.align_to_overlap(S);     // after nuclei move
cp.advance(H, S, dt_fs);    // NH½ → VV+RATTLE → NH½
double te_ha = cp.kinetic_ha();
double eth = cp.thermostat_energy_ha();
```

### Build

Sections need the modules in `case.cmake` (GPW sections: `ATV_BUILD_GPW` with
HF + PW, FFTW3; AO sections: `ATV_BUILD_HF`; all: MD and
`ATV_BUILD_REGRESSIONS=ON`), on the CPU toolchain
(`buildenv-gcc/2026.03-mpich`). GPW sections run at 8 threads.

### Oracle and speed policy

Pins are offline; ctest never runs CP2K. CP2K 2026.1 has **no fictitious-mass
Car–Parrinello propagator**: its `RUN_TYPE MD` is Born–Oppenheimer and its
CP-like method is second-generation CPMD (ASPC + corrector). The CP conserved
quantity with fictitious \(T_e\) is not a number CP2K can print for any input,
so every section is gated on the literature conserved-energy envelope plus
fixed-density, structural and (where relevant) spin diagnostics, and speed is
**`perf_exempt`** (no fair same-method wall twin; no third-party AO-CPMD twin
either). Where a CP2K input exists it is a geometry/basis twin, not an energy
twin. Absolute single-point E/F parity is owned by the static GPW pages.

### Common exclusions

- GPW CP refuses an OT seed (`use_ot`) and simultaneous XL-BOMD or
  second-generation CP at `build()`; the AO wrapper refuses `xl_bomd`
  symmetrically.
- Nuclear thermostat / joint ion–electron NVT-CP; nuclear RATTLE (`FixRattle`).
- Hybrid XC, GAPW, PW CP, CPMD stress; multi-ps production trajectories.
- CPU backend only.

### References

- Car & Parrinello, *Phys. Rev. Lett.* **55**, 2471 (1985)
- Andersen, *J. Comput. Phys.* **52**, 24 (1983) — RATTLE
- Blöchl & Parrinello, *Phys. Rev. B* **45**, 9413 (1992) — electronic thermostat
- Marx & Hutter, *Ab Initio Molecular Dynamics* (CUP) — CP conserved energy
- Roadmap: `docs/roadmap/molecular-dynamics.md` §AIMD / Car–Parrinello,
  `docs/roadmap/dft.md` (AIMD)

## cpmd-water — GPW CPMD, H₂O NVE {#cpmd-water}

Closed-shell RKS Γ-point GPW CP on isolated H₂O, reading the conserved energy
from the `kConserved` channel published by `CpmdConserved` (the snippet under
*Timestep: one clock* is this section).

### Setup

| Item | Value |
| ---- | ----- |
| System | Isolated H₂O (CP2K getting-started geometry) |
| Cell | 10 Å cubic, Martyna–Tuckerman (`nonperiodic = true`) |
| XC / basis / PP | PBE / DZVP-MOLOPT-GTH / GTH-PBE |
| CUTOFF | 200 Ry |
| dt | 0.1 fs (nuclear = electronic) |
| \(\mu\) | 2000 a.u. (the AO residual is stiffer than the plane-wave textbook \(\mu\) ~ 400) |
| Protocol | 1 SCF seed + 1 CP baseline + 4 CP drift steps |

### Pin

- **Guards** (no unit test in the tree gates them): mass/seed/advance guards;
  the wrapper refuses OT / `xl_bomd` / non-positive \(\mu,\Delta t\).
- **E2E:** relative \(E_{\mathrm{cons}}\) drift ~8×10⁻⁷ (tol **0.10**) after
  \(T_e\) is engaged, read from `frame.meta[kConserved]`; production
  `iters==0`; O–H distances < 1.8 Å; PE after warm-up < −100 eV; wall ~32 s.

## cpmd-rhf-h2 — AO-basis CPMD on the SCF methods: RHF / RKS / UHF {#cpmd-rhf-h2}

This section's page also covers the `cpmd-rks-water` and `cpmd-uhf-o2`
sections of the case (same wrapper, same protocol).

### Method

On the Gaussian-basis (AO) side the propagated variables are the occupied
**MO coefficient columns** \(C_\sigma\) of the determinant:

\[
\mathcal{L} = \sum_I \tfrac12 M_I \dot R_I^2
            + \sum_\sigma \tfrac12 \mu \lVert \dot C_\sigma\rVert^2
            - E[\{C\},\{R\}]
            - \text{constraints }(C_\sigma^\top S C_\sigma = I),
\]

giving \(\mu\,\ddot C = -(I - S C C^\top)\,\partial E/\partial C\) — the
electronic RATTLE inside `cpu::electronic::Cpmd`. Two things make this work on
an AO method rather than on a plane-wave grid:

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
occupancy 2; UHF/UKS propagate α and β independently at occupancy 1 each. The
conserved quantity is
\(E_{\mathrm{cons}} = T_{\mathrm{n}} + T_{e} + E_{\mathrm{el}}
\;(+\; E^{e}_{\mathrm{therm}})\).

### API

The mode bits live in `protected` storage on `MeanFieldMethod`, and
`cpu::electronic::CarParrinelloMethod<Method>` is the only public way to set
them. The wrapper owns `Method` and implements the nested DriverNode and
timestep protocols. Construct the method with its `basis` and set `tol` /
`max_iter` before moving it in; inspect parameters and results afterwards
through `wrapper.method()`.

| Public surface | Meaning |
| --- | --- |
| `CarParrinelloMethod<Method>{Method method, double mu_au}` | Move a configured `RHF` / `RKS` / `UHF` / `UKS` in and enable CP with fictitious electronic mass μ [a.u. electron mass]. Throws `std::invalid_argument` unless μ is finite and > 0. **Nested MD force slot only** — a standalone `run()` / `forces()` on a CP-enabled method throws `std::logic_error` (there is no nuclear trajectory for the electrons to follow at a single point) |
| `CarParrinelloMethod<Method>{Method, double mu_au, ScfElectronicThermostat{ekin_target_ha, tau_e_fs}}` | Same, plus a Nosé–Hoover thermostat on the fictitious DOF: target fictitious kinetic energy \(K_0\) [Hartree] and coupling time τ_e [fs]. The two-argument constructor is NVE electrons |
| `double last_electronic_ke_ev() const` | \(T_e\) [eV] of the last step; also on `frame.meta[kEnergyElectronKinetic]` |
| `double last_electronic_thermostat_ev() const` | \(E^{e}_{\mathrm{therm}}\) [eV]; exactly 0 without the thermostat; also on `frame.meta[kEnergyElectronThermostat]` |

**`determinant()` is not refreshed on production steps.** Propagated orbitals
have no canonical eigenvalues, so the published wavefunction (coefficients,
orbital energies, occupations, \(\langle S^2\rangle\)) stays the **seed SCF's**.
Observers must read it as the launch state, not the current one.

**Refusal lattice.** ROHF/ROKS (shared spatial orbitals are not independent
channels), GHF (spinor CP is new physics), `predictor()` (no ASPC-CP hybrid),
`ri != RI::None`, `x2c`, `solvent_eps > 0`, `mom != Off`, `soscf`, non-empty
`point_charges` (the QM/MM × CPMD seam — lifting this gate *is* the
integration), nonzero `external_field`, `d3bj`/`d4`, `compute_force = false`,
non-positive or non-finite μ / thermostat parameters, and a near-singular
overlap (`n_dropped > 0`) at the seed all throw. **Dispersion is refused
because `ScfGradient` carries no d3bj/d4 force terms** while the total energy
(`e_total_`) includes the dispersion contribution — \(E_{\mathrm{cons}}\) would
drift silently. `xl_bomd`
([xl-bomd-rhf-water](dynamics-xl-bomd.md#xl-bomd-rhf-water)) is mutually
exclusive with `cpmd` in **both** directions (no unit test in the tree
asserts it).

```cpp
constexpr double dt = 0.05; // fs
cpu::scf::RKS rks_base{cpu::scf::BasisSet::STO3G, cpu::scf::XCType::LSDA};  // or RHF / UHF{basis, n_a, n_b} / UKS
rks_base.tol = 1e-9;
cpu::electronic::CarParrinelloMethod<cpu::scf::RKS> rks{std::move(rks_base),
                                                        /*mu_au=*/1000.0};
cpu::md::MD sim;
sim.set_integrator(cpu::md::VelocityVerlet{dt});
sim.set_evaluator_ref(rks);
sim.set_observer(cpu::md::CpmdConserved{});
sim.build();
sim.run(frame, 1);          // step 1 = cold SCF seed  (iterations() > 0)
sim.run(frame, n_steps);    // production               (iterations() == 0)
const double t_e = frame.meta.get(kEnergyElectronKinetic);  // [eV]
```

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
setting all three sections ship. Judge a candidate μ by watching
`kEnergyElectronKinetic` stay small and *bounded*, not by the drift over a
short window.

### Pin

- **Contract** (no unit test in the tree gates it): the full
  refusal lattice; standalone `run()`/`forces()` throws; both dt-gate surfaces;
  `compute_at_density` vs `compute` parity at a converged determinant
  (restricted and unrestricted); seed-then-step returns `iterations() == 0`;
  the electronic meta keys are published.
- **E2E:** three sections, μ = 1000 a.u., dt = 0.05 fs, 1 SCF seed + 1 CP
  baseline + 40 CP drift steps each. Envelopes are **empirical pins** (tuned,
  then frozen at ≈ 9–18× the observed worst drift); the citation supplies the
  *form* of the gate, not the number. These implement the acceptance
  criteria ac-002 (zero production iterations) and ac-003 (drift envelopes)
  of `esmm-cpmd-04-ao-cpmd`.

| Section | Method / system | Worst observed drift | Envelope | \(T_e\) over production [eV] | Wall |
| --- | --- | --- | --- | --- | --- |
| `cpmd-rhf-h2` | RHF/STO-3G, H₂ r = 0.74 Å | 4.195e−07 | **5e−6** | ≡ 0, gated < 1e−20 (see below) | 0.03 s |
| `cpmd-rks-water` | RKS(LSDA)/STO-3G, H₂O (CP2K getting-started) | 5.851e−06 | **5e−5** | 1.1e−07 … 7.4e−02 | 2.0 s |
| `cpmd-uhf-o2` | UHF triplet 9/7, STO-3G, O₂ r = 1.21 Å | 2.810e−07 | **5e−6** | 2.1e−10 … 7.7e−03 | 1.3 s |

Every section additionally pins: production `iterations() == 0` on **every**
post-seed step; `kConserved == kEnergy + kKinetic + T_e + E_therm`
re-assembled from the frame (observed residual bit-exact 0); \(T_e = 0\) and
\(E^{e}_{\mathrm{therm}} = 0\) where they must be; and a bond-length window.
`cpmd-uhf-o2` also pins the seed determinant's \(N_\alpha/N_\beta = 9/7\) and
spin contamination \(\langle S^2\rangle - S(S+1) = 0.0128 < 0.5\).

#### Why H₂ shows \(T_e \equiv 0\)

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

`cpmd-rhf-h2` is still the right machinery section — it pins the seed, the
zero-iteration production steps, the dt protocol and the conserved channel. The
*moving* electronic subsystem is what `cpmd-rks-water` and `cpmd-uhf-o2` add.
O₂ is homonuclear too, but its σ_g/σ_u blocks are **three-dimensional**
(1s/2s/2p_z combinations), so the occupied subspace genuinely mixes as the bond
changes and \(T_e > 0\) is a hard gate there.

### Source

| What | Path |
| ---- | ---- |
| CP composed node (`CarParrinelloMethod`, `ScfElectronicThermostat`) | `aimd_method.h` |
| Mode storage, `TimestepConsumer`, seed/step | `mean_field_method.h`, `src/cpu/scf/mean_field_method.cpp` |
| Fixed-state gradient (`compute_at_density`) | `gradient.h`, `src/cpu/scf/internal/gradient.cpp` |
| Fixed-density energy | `src/cpu/scf/internal/mean_field_energy.cpp` |
| Propagator / \(E_{\mathrm{cons}}\) assembler | `cpmd.h` / `cpmd_conserved.h` |

### Limitations

- PW CPMD, OT-CP, multi-k and CPMD stress are not on the AO path.
- QM/MM × CPMD: the `point_charges` gate stays shut.
- The d3bj/d4 gradient gap is verified and refused, not fixed.
- GGA / hybrid XC gradients at a non-stationary density run the same code path
  but are not pinned (LSDA only).

## cpmd-uks — GPW CPMD on Γ-UKS, O₂ triplet NVE {#cpmd-uks}

### Method

Collinear Γ-point UKS on the GPW path: occupied MO coefficients for **α and
β** are fictitious degrees of freedom, each advanced by its own `Cpmd`
propagator with electronic RATTLE on \(C_\sigma^\top S C_\sigma = I\):

\[
\mathcal{L}
= \sum_I \tfrac12 M_I \dot R_I^2
+ \sum_{\sigma}\sum_i \tfrac12 \mu \lVert \dot c_{i\sigma}\rVert^2
- E_{\mathrm{KS}}[C_\alpha,C_\beta,R]
- \sum_{\sigma}\text{constraints}\bigl(C_\sigma^\top S C_\sigma = I\bigr),
\]

\[
\mu\,\ddot C_\sigma
= -(I - S C_\sigma C_\sigma^\top)\,(2 H_\sigma C_\sigma),
\qquad
T_{\mathrm{e}} = T_{\mathrm{e}}^\alpha + T_{\mathrm{e}}^\beta
= \tfrac12\mu\sum_{\sigma\in\{\alpha,\beta\}}\sum_i \lVert\dot c_{i\sigma}\rVert^2 .
\]

After one cold UKS SCF seed, production steps evaluate dual-spin KS
energy/forces at the current densities (`iters==0`). Spin densities
\(\rho_\alpha,\rho_\beta\); Hartree from the total density; spin-resolved
**PADE LSD** XC (Goedecker, Teter, Hutter, *Phys. Rev. B* **54**, 1703
(1996)); dual KS matrices \(H_\alpha,H_\beta\). Occupations are explicit
members `n_alpha` / `n_beta` (not a multiplicity bag): O GTH-q6 → 6 valence e⁻
per O → 12 valence e⁻; multiplicity 3 ⇒ \(N_\alpha=7\), \(N_\beta=5\).
Martyna–Tuckerman isolated electrostatics (Martyna & Tuckerman, *J. Chem.
Phys.* **110**, 2810 (1999); `nonperiodic=true` / CP2K `PSOLVER MT`); the
molecule must be **cell-centered**. The geometry is shared with
[gpw-uks-o2-gamma](gpw-uks.md#gpw-uks-o2-gamma), `gpw-uks-forces` and
[xl-bomd-uks](dynamics-xl-bomd.md#xl-bomd-uks), which own absolute E/F parity.

```cpp
cpu::gpw::GpwDft dft{{O}, 200.0, cpu::gpw::XcKind::PADE};
dft.max_iter = 80;  dft.tol = 1e-5;  dft.mix = 0.4;
dft.nonperiodic = true;
dft.n_alpha = 7; // multiplicity 3, 12 valence e⁻
dft.n_beta = 5;
cpu::gpw::GPWEvaluator gpw(std::move(dft), /*compute_forces=*/true);
cpu::electronic::CarParrinello eval{std::move(gpw), /*mu_au=*/2000.0};
// … MD + VelocityVerlet{0.1} + set_evaluator_ref(eval) as above
const double e_cons = frame.meta.get(kKinetic) + frame.meta.get(kEnergy)
                    + eval.last_electronic_ke_ev();
const double s2 = eval.method().last_s2(); // Löwdin ⟨S²⟩ from last full SCF
```

Sources: `gpw_evaluator.h`, `car_parrinello.h`, `cpmd.h`, dual α/β path in
`src/cpu/gpw/gpw_evaluator.cpp`. No unit test in the tree covers the guards
or the electronic RATTLE.

### Setup

| Item | Value |
| ---- | ----- |
| System | Isolated O₂ triplet, \(R=1.2075\) Å |
| Coordinates (Å) | O (6, 6, 5.39625), O (6, 6, 6.60375) — cell-centered |
| Cell | `cpu::Box::cubic(12.0)` (12 × 12 × 12 Å), `PERIODIC NONE` |
| Basis / PP | DZVP-MOLOPT-GTH / GTH-PBE-q6 (ATV); twin lists GTH-PADE-q6 |
| XC | PADE LSD (`XcKind::PADE` / CP2K `&XC_FUNCTIONAL PADE` + `LSD`) |
| Cutoff | **200 Ry** (envelope run; absolute E/F pin uses 400 Ry elsewhere) |
| Masses | O 15.999 amu |
| Initial velocity | antisymmetric stretch \(v_z = -0.0008, +0.0008\) Å/fs (no COM translation) |
| dt / \(\mu\) | **0.1 fs** (nuclear = electronic) / **2000 a.u.** |
| Protocol | 1 SCF seed + 1 CP baseline (\(T_e\) on) + 4 CP drift steps |

### Gates

| Observable | Pin / gate | Tol / window | Notes |
| ---------- | ---------: | ------------ | ----- |
| \(E_{\mathrm{cons}}\) relative drift | literature envelope | **≤ 0.10** | after the \(T_e\)-engaging baseline step |
| Production SCF iters | **0** | exact | fixed-density dual-KS — not BOMD |
| PE after warm-up | bound | **&lt; −100 eV** | cold UKS seed |
| Nuclear KE after warm-up | nontrivial | **&gt; 1e−12 eV** | stretch initial velocities |
| \(\langle S^2\rangle\) (warm-up SCF) | triplet window | **[1.5, 2.5]** | ideal \(S(S+1)=2\); not refreshed on fixed-density steps |
| O–O distance | structural | **[0.9, 1.6] Å** | no explode / collapse |
| Finite trajectory | PE, KE, Te, coords | `isfinite` | all production chunks |

Offline CP2K geometry twin (operator only): `RUN_TYPE MD`, `ENSEMBLE NVE`,
`STEPS 10`, `TIMESTEP 0.1`, `MULTIPLICITY 3`, `LSD`, PADE, DZVP-MOLOPT-GTH, MT
12 Å cell, same O–O geometry as the static UKS family; it does not provide a
numerical \(E_{\mathrm{tot}}\) match.

### Limitations

| In | Out |
| -- | --- |
| Collinear Γ-UKS GPW + dual fictitious e⁻ DOF | Non-collinear |
| Electronic RATTLE (pos + vel), NVE electrons | Electronic thermostat on UKS (not e2e-pinned) |
| Short O₂ NVE drift envelope (≤60 s design) | Multi-ps production |
| PADE LSD + MT isolated | Hybrid XC CPMD; OT-seeded CP |
| RKS and UKS share the `CarParrinello` wrapper | Simultaneous XL / second-gen CP |

## cpmd-kpoints — Multi-k CPMD on Bloch \(C_k\), H₂ NVE {#cpmd-kpoints}

### Method

Fictitious electronic degrees of freedom on **multi-k closed-shell RKS** GPW:
one `CpmdCplx` propagator per sampled Bloch \(k\), complex coefficients
\(C_k\), with the holonomic constraint \(C_k^\dagger S_k C_k = I\) enforced by
electronic RATTLE and the velocity constraint killing
\(\mathrm{Herm}(C_k^\dagger S_k \dot C_k)\). Production steps run at fixed
density (`iters == 0`). Electronic kinetic energy is
\(T_e = \sum_k w_k \tfrac12 \mu \lVert \dot C_k\rVert_F^2\).

### System

| Quantity | Value |
| -------- | ----- |
| Species | H₂ (periodic), \(R = 1.0\) Bohr compressed bond along \(x\), cell-centred |
| Cell | cubic \(L = 12\) Bohr (≈ 6.350 Å) |
| Spin / XC | closed-shell RKS, \(n_{\mathrm{occ}} = 1\); PADE |
| Basis / PP | DZV-GTH-PADE / GTH-PADE-q1 |
| Cutoff | 100 Ry |
| k-mesh | \(1\times1\times2\), time reversal off → \(n_k = 2\) |
| CPMD | \(\mu = 400\) a.u.; `dt` = 0.1 fs (nuclear = electronic) |
| Trajectory | 1 cold SCF seed + 1 CP baseline + 4 production steps, NVE |

### Gate

Relative \(E_{\mathrm{cons}}\) drift \(\delta \le 0.10\) with multi-k Bloch
payload diagnostics (`n_kpts >= 2`), production SCF iterations exactly 0,
bounded potential energy, finite coordinates, and a soft bond window
(0.4–1.5 Å). Unit gates cover the `CpmdCplx` seed and advance invariants
(\(C^\dagger S C = I\), Hermitian velocity constraint), the multi-k
fixed-density path, and the rejected combinations (OT + CPMD, XL + CPMD).
Multi-k UKS CP is [cpmd-uks-kpoints](#cpmd-uks-kpoints).

## cpmd-uks-kpoints — Multi-k UKS CPMD on dual-spin Bloch \(C_k^\sigma\), H₂ NVE {#cpmd-uks-kpoints}

### Method

The [cpmd-kpoints](#cpmd-kpoints) (multi-k RKS Bloch CP) ×
[cpmd-uks](#cpmd-uks) (Γ dual-spin CP) crossing: one complex propagator per
sampled Bloch \(k\) **and spin channel** \(\sigma \in \{\alpha, \beta\}\),
each with \(C_k^{\sigma\dagger} S_k C_k^\sigma = I\) enforced by electronic
RATTLE, and the spin-diagonal occupancy-1 EOM

\[
\mu\,\ddot C_k^\sigma
  = -(I - S_k C_k^\sigma C_k^{\sigma\dagger})\,(2 H_k^\sigma C_k^\sigma)
    + \text{RATTLE}
\]

on the spin-resolved KS matrices \(H_k^\alpha / H_k^\beta\) from the
**dual-spin fixed-density multi-k evaluation** (`max_iter == 0` at the
propagated \(\{P_k^\alpha, P_k^\beta\}\), exported through the
`GpwResult::H_ks_k` / `H_ks_k_beta` payload; `iters == 0` on every production
step). \(T_e = \sum_k w_k \sum_\sigma \tfrac12 \mu \lVert \dot C_k^\sigma
\rVert_F^2\). Electrons are NVE here; the multi-k electronic thermostat (one NH
per (k, σ) channel, k-weighted totals) is the `cpmd-thermostat-kpoints` section
(RKS) and the UKS combination in
[cpmd-uks-thermostat-kpoints](#cpmd-uks-thermostat-kpoints).

### System

Same as [cpmd-kpoints](#cpmd-kpoints) (H₂, 12 Bohr cell, DZV-GTH-PADE /
GTH-PADE-q1, 100 Ry, 1×1×2 unreduced, \(\mu = 400\) a.u., dt 0.1 fs, 1 seed + 1
baseline + 4 production), except spin: collinear UKS, \(n_\alpha = n_\beta = 1\)
(closed-shell fillings — both channels propagate, molecule stays bound); PADE
(LSD).

```cpp
cpu::gpw::GpwDft dft{{H}, 100.0, cpu::gpw::XcKind::PADE};
dft.max_iter = 80;  dft.tol = 1e-5;  dft.mix = 0.3;
dft.nonperiodic = false;
dft.n_alpha = 1;  dft.n_beta = 1;                  // collinear UKS
dft.k_mesh[2] = 2;  dft.k_time_reversal = false;   // 1×1×2, n_k = 2
cpu::gpw::GPWEvaluator gpw(std::move(dft), /*compute_forces=*/true);
cpu::electronic::CarParrinello cp{std::move(gpw), /*mu_au=*/400.0};
cpu::md::MD md;
md.set_integrator(cpu::md::VelocityVerlet{0.1});   // fs, NVE
md.set_evaluator_ref(cp);
md.build();
md.run(frame, /*steps=*/6);        // 1 SCF seed + 1 CP baseline + 4 production
```

Headers: `cpu/electronic/car_parrinello.h`, `cpu/electronic/cpmd_cplx.h`,
`cpu/gpw/gpw_evaluator.h`.

### Validation (CP2K-anchored surface + literature envelope)

The CP propagator has no CP2K twin; the **BO surface underneath is
CP2K-anchored** through the multi-k UKS pins.

| Quantity | Gate (hard-coded) | Anchor / oracle | Notes |
| -------- | ----------------: | --------------- | ----- |
| Relative \(E_{\mathrm{cons}}\) drift \(\delta\) over 4 production steps | \(\le 0.10\) | Literature NVE envelope (Car–Parrinello 1985; Marx & Hutter) | \(E_0\) at first CP step |
| UKS-vs-RKS closed-shell identity, `tol = 1e-7` | \(\le 3\times10^{-4}\) eV (≈ 1.1×10⁻⁵ Ha) | Multi-k RKS path CP2K comparison in [gpw-bomd-kpoints](dynamics-bomd.md#gpw-bomd-kpoints) (CP2K 2026.1); UKS path in [gpw-kpoints-uks](gpw-uks.md#gpw-kpoints-uks) (A2 held to 5×10⁻⁶ Ha) | same geometry / mesh / cutoff; \(n_\alpha=n_\beta=1\) vs \(n_{\mathrm{occ}}=1\) |
| Production SCF iterations | exactly 0 | — | dual-spin fixed-density multi-k eval |
| \(n_{\mathrm{kpts}}\) after seed | \(\ge 2\) | — | 1×1×2, time reversal off |
| \(T_e\) once propagated | \(> 0\) | — | dual-spin sum |
| PE after warm-up | \(< -5\) eV, finite | — | bound H₂ |
| \(r_{\mathrm{HH}}\) after production | \(0.4\)–\(1.5\) Å | — | soft window |

Unit gates additionally pin the dual-spin fixed-density solver export
(`H_ks_k_beta`, partial-spin-guess refusal) and the **Γ-limit identity**: the
k-path dual-spin fixed-density eval at one explicit Γ point reproduces the Γ
dual-spin eval (energy and per-spin KS matrices — the surface `cpmd-uks` ships
on).

### Limitations

- Closed-shell fillings only in the pinned section (\(n_\alpha=n_\beta=1\));
  triplet H₂ is unbound and cannot pin NVE (the
  [xl-bomd-uks-kpoints](dynamics-xl-bomd.md#xl-bomd-uks-kpoints) precedent).
  Open-shell multi-k CP is exercised only via unit gates.
- BO anchors: [gpw-kpoints-uks](gpw-uks.md#gpw-kpoints-uks),
  [gpw-kpoint-uks-forces](gpw-uks.md#gpw-kpoint-uks-forces) (CP2K).

## cpmd-electronic-thermostat — GPW CPMD + Nosé–Hoover electronic thermostat, H₂O {#cpmd-electronic-thermostat}

### Method

The [cpmd-rattle-water](#cpmd-rattle-water) system with a Nosé–Hoover
thermostat on the fictitious orbital velocities \(\dot C\) (Blöchl &
Parrinello 1992; Sprik; Marx & Hutter): friction \(\xi\) and coordinate
\(\eta\) regulate \(T_e = \tfrac12\mu\lVert\dot C\rVert_F^2\) toward a cool
target \(K_0\) with coupling time \(\tau_e\) and thermostat mass
\(Q = 2 K_0\tau_e^2\). Symmetric NH half-steps wrap the mechanical VV+RATTLE
step, so the extended energy

\[
E_{\mathrm{ext}} = T_{\mathrm{n}} + T_{\mathrm{e}} + E_{\mathrm{KS}} + E_{\mathrm{therm}},
\qquad E_{\mathrm{therm}} = \tfrac12 Q\xi^2 + 2 K_0\eta,
\]

is the literature conserved quantity. Nuclei stay NVE: adiabatic separation is
maintained by keeping \(T_e\) cool relative to nuclear KE, not by nuclear NVT.
The `Cpmd` API allows UKS dual-propagator + electronic NH per spin channel;
this section pins closed-shell RKS Γ only.

```cpp
cpu::electronic::CarParrinello eval{
    std::move(gpw), /*mu_au=*/2000.0,
    cpu::electronic::ElectronicThermostat{/*ekin_target_ha=*/1e-5, // K0 [Ha]
                                          /*tau_e_fs=*/50.0}};     // [fs]
// … MD + VelocityVerlet{0.1} + set_evaluator_ref(eval)
const double e_ext = frame.meta.get(kKinetic) + frame.meta.get(kEnergy)
                   + eval.last_electronic_ke_ev()
                   + eval.last_electronic_thermostat_ev();
```

The three-argument constructor throws `std::invalid_argument` on non-finite or
non-positive \(K_0\) / \(\tau_e\); the usual CP exclusions (OT / XL /
second-gen) apply. No unit test in the tree covers the NH parameter guards,
\(T_e\) regulation or evaluator build accept/reject; sources
`src/cpu/electronic/cpmd.cpp`, `src/cpu/gpw/gpw_evaluator.cpp`.

### System

| Quantity | Value |
| -------- | ----- |
| Species | Isolated H₂O (CP2K getting-started geometry) |
| Cell | 10 Å cubic, MT / `nonperiodic = true` |
| XC / basis / PP | PBE / DZVP-MOLOPT-GTH / GTH-PBE-q6,q1 |
| CUTOFF | 200 Ry |
| \(\Delta t\) / \(\mu\) | 0.1 fs (nuclear = electronic) / 2000 a.u. |
| \(K_0\) / \(\tau_e\) | \(1\times 10^{-5}\) Ha / 50 fs |
| Initial velocity | COM-preserving H stretch \(\pm 0.0015\) Å/fs |
| Steps | 1 warm-up (SCF seed) + 1 CP baseline + 4 production |

| Atom | \(x\) | \(y\) | \(z\) (Å) |
| ---- | ----: | ----: | --------: |
| O | 5.0000 | 5.0000 | 5.0000 |
| H1 | 5.7586 | 5.0000 | 5.5043 |
| H2 | 4.2414 | 5.0000 | 5.5043 |

### Gates

| Quantity | Gate | Oracle / literature | Notes |
| -------- | ---: | ------------------- | ----- |
| \(E_{\mathrm{ext}}\) relative drift | ≤ 0.10 | Marx–Hutter extended energy envelope | NVE nuclei + e-NH |
| Production SCF iters | 0 | CP fixed-density definition | not BOMD |
| Max \(T_e\) over production | ≤ 0.05 eV | cool electronic target \(K_0=10^{-5}\) Ha | runaway gate |
| \(r_{\mathrm{OH}}\) | < 1.8 Å | structural stability | both H |
| PE after warm-up | < −100 eV | PBE water bound | after SCF seed |
| Geometry / basis twin | H₂O MT PBE MOLOPT GTH 200 Ry | `cpmd_electronic_thermostat_h2o.inp` | setup, not an energy twin |

Classical fictitious-mass CPMD with electronic NH is not a stock CP2K QS
ensemble; absolute \(E_{\mathrm{KS}}\) vs CP2K is not gated.

### Limitations

- Short trajectory (1 + 1 + 4 steps); no reported \(\langle T_e\rangle\) over a
  long production run.
- UKS + electronic NH is not e2e-pinned on Γ (`cpmd-uks` is NVE electrons);
  multi-chain electronic NH is not implemented.

## cpmd-rattle-water — GPW CPMD + electronic RATTLE, H₂O NVE {#cpmd-rattle-water}

The [cpmd-water](#cpmd-water) system and settings (PBE / DZVP-MOLOPT-GTH /
GTH-PBE, 200 Ry, μ=2000 a.u., dt=0.1 fs, 1 seed + 1 baseline + 4 production)
under the tighter relative drift gate \(\le 0.05\) (versus 0.10 on
`cpmd-water`), with the conserved energy assembled from the evaluator
readbacks:

```cpp
cpu::electronic::CarParrinello eval{std::move(gpw), /*mu_au=*/2000.0};
// … MD + VelocityVerlet{0.1} + set_evaluator_ref(eval)
const double Te = eval.last_electronic_ke_ev(); // [eV]
const double Econs = frame.meta.get(kEnergy) + frame.meta.get(kKinetic) + Te;
```

Gates: \(E_{\mathrm{cons}}\) drift ≤ 0.05; production `iters==0`; O–H < 1.8 Å.
The orthonormality residual, velocity antisymmetry and evaluator guards have
no unit test in the tree.

| In | Out |
| -- | --- |
| Closed-shell RKS Γ GPW | UKS, OT, k-points (other sections) |
| Electronic RATTLE (pos + vel) | Nuclear RATTLE (`FixRattle`) |
| Short H₂O NVE envelope | Multi-ps production |

## cpmd-uks-thermostat-kpoints — Multi-k UKS CPMD with electronic Nosé–Hoover, H₂ {#cpmd-uks-thermostat-kpoints}

*No section of the case runs this variant.*

The [cpmd-uks-kpoints](#cpmd-uks-kpoints) (dual-spin Bloch CP) ×
`cpmd-thermostat-kpoints` (multi-k electronic NH) combination: one
thermostatted `CpmdCplx` per sampled Bloch \(k\) and spin channel
\(\sigma \in \{\alpha, \beta\}\), each with electronic RATTLE and its own NH
variable targeting the full \(K_0\) (Γ-UKS convention). Published totals are
k-weighted dual-spin sums:

\[
T_e = \sum_k w_k \sum_\sigma \tfrac12\mu\lVert\dot C_k^\sigma\rVert_F^2,\qquad
E_{\mathrm{therm}} = \sum_k w_k \sum_\sigma E_{\mathrm{therm}}^{(k,\sigma)}.
\]

The conserved diagnostic is
\(E_{\mathrm{ext}} = T_n + T_e + E_{\mathrm{KS}} + E_{\mathrm{therm}}\); nuclei
stay NVE; production steps use dual-spin fixed-density multi-k KS evals
(`iters == 0`). System as [cpmd-uks-kpoints](#cpmd-uks-kpoints) (H₂, 12 Bohr ≈
6.350 Å, UKS \(n_\alpha = n_\beta = 1\), PADE, 100 Ry, 1×1×2 unreduced,
\(\mu = 400\) a.u., dt 0.1 fs, 1 + 1 + 4 steps) plus the thermostat
\(K_0 = 10^{-5}\) Ha, \(\tau_e = 50\) fs per (k, σ).

```cpp
cpu::electronic::CarParrinello cp{
    std::move(gpw), /*mu_au=*/400.0,     // gpw: multi-k UKS GPWEvaluator
    cpu::electronic::ElectronicThermostat{/*K0=*/1e-5, /*tau_e_fs=*/50.0}};
md.set_evaluator_ref(cp);
md.run(frame, /*steps=*/6);
```

Values recorded when this page was written (no standing regression); the BO
surface underneath is CP2K-anchored through
[gpw-kpoints-uks](gpw-uks.md#gpw-kpoints-uks) /
[gpw-kpoint-uks-forces](gpw-uks.md#gpw-kpoint-uks-forces) / multi-k UKS GPW
BOMD.

| Quantity | Gate | Anchor |
| -------- | ---- | ------ |
| Relative \(E_{\mathrm{ext}}\) drift over 4 production steps | ≤ 0.10 | Literature (Blöchl & Parrinello 1992; Marx & Hutter) |
| Max \(T_e\) over production | ≤ 0.05 eV | thermostatted control envelope |
| UKS-vs-RKS closed-shell identity, `tol = 1e-7` | ≤ 3×10⁻⁴ eV | multi-k RKS CP2K values on [gpw-bomd-kpoints](dynamics-bomd.md#gpw-bomd-kpoints) |
| Production SCF iterations | exactly 0 | dual-spin fixed-density multi-k eval |
| \(n_{\mathrm{kpts}}\) after seed | ≥ 2 | 1×1×2, time reversal off |
| \(E_{\mathrm{therm}}\) | finite every production step | NH bookkeeping |

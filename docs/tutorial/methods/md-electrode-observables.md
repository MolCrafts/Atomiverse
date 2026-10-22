# Electrode observables

Analysis and bookkeeping Components for classical constant-potential electrode
cells. They never solve electrode charges: they read whatever `kAtom_charge`
holds on the Frame (typically after `cpu::md::ConstPotBO` has updated it at
`pre_force`) and publish observables to component members and Frame meta. All
are Components on `cpu::md::MD` (or `cpu::Evaluate`) — no Driver category, and
no charge solve inside `KSpace*`. Parameters are constructor arguments; there is
no `*Config` bag and no factory.

- `cpu::CapacitanceChargeFluctuation` — `Observer`, default `end_of_step` @ +50,
  bound with `set_observer_ref`: differential and integral capacitance from
  electrode-charge statistics.
- `cpu::electrodes::ChargingWork` — `Potential`, bound with `set_evaluator`:
  the charging-work energy \(-\sum_i \Psi_i Q_i\).
- `cpu::PotentialProfile` — `Observer`, default `end_of_step` @ +45, bound with
  `set_observer_ref`: the planar electrostatic potential profile \(\Psi(z)\).

All sections run on CPU only. The case also registers the section
`thermo-integration-capacitance` (`REQUIRES KSPACE MD`, see `case.cmake`), which
has no section on this page.

Case: `ctest -R '^md/electrode-observables$'` — sources and oracle record in [`regressions/md/electrode-observables/`](../../../regressions/md/electrode-observables/). One section of the case per `##` heading below: `atv_regressions --case md/electrode-observables --section <name>`.

## Common setup

- **Build:** each section below requires only the `MD` module (CPU login shell
  with `buildenv-gcc/2026.03-mpich`, as in the top-level build instructions).
- **Oracle: literature** for every section. These are textbook identities for
  constant-potential electrodes; LAMMPS core has no comparable named compute
  (the pin host also lacked the ELECTRODE package) and no MetalWalls binary was
  installed for a live twin, so the LAMMPS-first priority does not apply. Each
  offline pin is the closed-form evaluation of the published formula on a
  synthetic parallel-plate assignment. Never live MetalWalls or LAMMPS in ctest.
- **Units:** \(Q\) [e], \(\Psi\) [V], energy [eV], with
  \(1\,e\cdot 1\,\mathrm{V} = 1\,\mathrm{eV}\).
- **Performance: `perf_exempt`** for every section — each Component is
  \(O(N)\) charge arithmetic on a toy system, and there is no MetalWalls or
  LAMMPS twin binary for these named post-processes on the agent host, so a fair
  wall ratio against an oracle binary is not applicable
  (`ratio_atv_over_oracle` n/a, `within_10x` exempt). A fair twin would need a
  MetalWalls analysis-script harness or a custom LAMMPS post-process with the
  same inputs and constants.
- **MPI:** per-electrode / per-bin charge sums are all-reduced when a domain
  decomposition is active (`owned_count` + `MPI_Allreduce`); energies use the
  existing energy `Reduce`.
- **Out of scope for all sections:** the electrode charge solve itself
  (`ConstPotBO`, E1–E2 of the MetalWalls map), Coulomb / Gaussian-electrode
  terms, and production electrolyte trajectories. The capability map is
  `.claude/notes/metalwalls-capability-map.md` (E1 for charging work, E10 for
  the observers).

A production MD run puts the BO charge update ahead of the observers:

```cpp
cpu::md::ConstPotBO bo(eta);           // [1/Å]
bo.add_electrode(1, 0.0);              // type, Ψ [V]
bo.add_electrode(2, psi_top);
bo.electroneutrality = true;

cpu::md::MD sim;
// … neighbor, Coulomb/PME, integrator …
sim.set_observer_ref(bo);
sim.set_evaluator(cpu::electrodes::ChargingWork({1, 2}, {0.0, psi_top}));
sim.set_observer_ref(cap);             // CapacitanceChargeFluctuation / PotentialProfile
sim.build();
sim.run(frame, nsteps);
```

### References

- A. Marin-Laflèche et al., *J. Open Source Softw.* **5**(53), 2373 (2020)
  (MetalWalls), DOI [10.21105/joss.02373](https://doi.org/10.21105/joss.02373)
- L. Scalfi et al., *Phys. Chem. Chem. Phys.* **22**, 10480 (2020); arXiv:1911.09351

## capacitance-charge-fluctuation — Capacitance from electrode charge fluctuations {#capacitance-charge-fluctuation}

### Summary

Differential and integral capacitance of a constant-potential electrode cell
from electrode total-charge statistics. `cpu::CapacitanceChargeFluctuation` is a
frame-immutable Observer that samples \(Q = \sum_{i\in e_0} q_i\) each launch
(primary electrode = type-list index 0), accumulates moments, and publishes

\[
C_{\mathrm{diff}} = \mathrm{Var}(Q)/(k_{\mathrm{B}}T),\qquad
C_{\mathrm{int}} = \langle Q\rangle/\Delta\Psi
\]

in e/V on members and on Frame meta (`kCapacitanceDiff`,
`kCapacitanceIntegral`). The gate pins the closed-form literature formula on a
four-sample synthetic \(Q\) series (proxy for a short const-pot MD charge
trace).

### Method

At fixed applied electrode potential, the differential capacitance of a
parallel-plate electrochemical cell follows from electrode charge fluctuations
(Limmer et al. 2013; Scalfi et al. 2020; MetalWalls analysis tools):

\[
C_{\mathrm{diff}}
  = \frac{\langle Q^2\rangle - \langle Q\rangle^2}{k_{\mathrm{B}}T}
  = \frac{\mathrm{Var}(Q)}{k_{\mathrm{B}}T}
  \qquad[\mathrm{e}/\mathrm{V}],
\]

with \(Q\) the total charge on the primary electrode [e], \(T\) [K], and

\[
k_{\mathrm{B}} = 8.617333262\times 10^{-5}\,\mathrm{eV\,K^{-1}}
\]

(CODATA 2018, `units::kB_ev_per_k`), so \(k_{\mathrm{B}}T\) is in eV.
Population variance uses the running moments \(\sum Q\), \(\sum Q^2\) over \(N\)
samples:

\[
\mathrm{Var}(Q) = \frac{1}{N}\sum_s Q_s^2 - \Bigl(\frac{1}{N}\sum_s Q_s\Bigr)^2.
\]

When a fixed potential difference \(\Delta\Psi\) [V] is known and nonzero, the
integral capacitance is

\[
C_{\mathrm{int}} = \frac{\langle Q\rangle}{\Delta\Psi}
  \qquad[\mathrm{e}/\mathrm{V}].
\]

If \(\Delta\Psi = 0\), the integral route is disabled (`C_int` = NaN; meta key
`kCapacitanceIntegral` is not written).

| Item | Setting |
| ---- | ------- |
| Geometry | Parallel-plate toy: 4 left wall atoms (type 1), 4 right (type 2); no electrolyte |
| Box | cubic \(L = 20\) Å |
| \(T\) | \(300\) K |
| \(\Delta\Psi\) | \(1.0\) V |
| \(Q_L\) samples [e] | \(0.4,\;0.6,\;0.4,\;0.6\) (equal per-atom share on each wall; \(Q_R = -Q_L\)) |
| \(N\) | \(4\) |
| Primary electrode | type list index 0 → type **1** (left wall) |
| Binding | `cpu::md::MD` with `VelocityVerlet{0.0}` and a zero `FixAddForce` as the energy source; observer at default `end_of_step` @ +50; one `run(frame, 1)` per sample after setting the wall totals |

Closed form:

\[
\langle Q\rangle = 0.5\,\mathrm{e},\qquad
\mathrm{Var}(Q) = 0.01\,\mathrm{e}^2,
\]

\[
k_{\mathrm{B}}T = 8.617333262\times 10^{-5}\times 300 = 0.025851999786\,\mathrm{eV},
\]

\[
C_{\mathrm{diff}}^{\mathrm{pin}} = \frac{0.01}{0.025851999786}
  = 0.38681727072485284\,\mathrm{e/V},\qquad
C_{\mathrm{int}}^{\mathrm{pin}} = 0.5\,\mathrm{e/V}.
\]

**Out of scope:** area-normalized capacitance [µF/cm²], series/parallel
multi-electrode decomposition, free-energy thermodynamic integration of \(C\),
block averaging / bootstrap error bars and finite-sample bias corrections.

### Usage

```cpp
cpu::CapacitanceChargeFluctuation obs(
    /*electrode_types=*/{1, 2},  // index 0 = primary Q
    /*temperature_K=*/300.0,
    /*delta_psi_V=*/1.0);

cpu::md::MD sim;
sim.set_integrator(cpu::md::VelocityVerlet{0.0});
sim.set_evaluator(cpu::md::FixAddForce{0.0, 0.0, 0.0, /*type=*/1});
sim.set_observer_ref(obs);
sim.build();
for (int s = 0; s < 4; ++s) {
    set_wall_total_Q(frame, QL[s], -QL[s]);
    sim.run(frame, 1);
}

// Members [e], [e²], [e/V]
const double mean_Q = obs.mean_charge();
const double var_Q  = obs.charge_variance();
const double C_diff = obs.differential_capacitance();    // Var/(kB T)
const double C_int  = obs.integral_capacitance();        // mean_Q / ΔΨ
// Frame meta: kCapacitanceDiff, kCapacitanceIntegral, kElectrodeCharge
```

Files: `capacitance_charge_fluctuation.h` (public class),
`src/cpu/observe/capacitance_charge_fluctuation.cpp` (sum \(Q\), moments, meta
write-back), `frame_schema.h` (`kCapacitanceDiff`, `kCapacitanceIntegral`,
`kElectrodeCharge`), unit tests in
`tests/cpu/observe/test_capacitance_charge_fluctuation.cpp`.

### Validation (oracle: literature)

Limmer et al., *J. Chem. Phys.* **139**, 184704 (2013) (charge-fluctuation
\(C\)); Scalfi et al. (2020); MetalWalls analysis tools.

| Quantity | Atomiverse gate / pin | Literature closed form | Tol | Notes |
| -------- | --------------------: | ---------------------: | --- | ----- |
| \(N\) samples | **4** | 4 | exact | synthetic series length |
| \(\langle Q\rangle\) [e] | **0.5** | 0.5 | **1e-12** | mean of \(\{0.4,0.6,0.4,0.6\}\) |
| \(\mathrm{Var}(Q)\) [e²] | **0.01** | 0.01 | **1e-12** | population variance |
| \(C_{\mathrm{diff}}\) [e/V] | **0.38681727072485284** | \(0.01/(k_B\cdot 300)\) | **1e-12** | `units::kB_ev_per_k` |
| \(C_{\mathrm{int}}\) [e/V] | **0.5** | \(0.5/1.0\) | **1e-12** | \(\Delta\Psi = 1\) V |
| meta `kCapacitanceDiff` | same as \(C_{\mathrm{diff}}\) | same | **1e-12** | Frame write-back |
| meta `kCapacitanceIntegral` | same as \(C_{\mathrm{int}}\) | same | **1e-12** | Frame write-back |

Tolerance is machine-eps scale: exact rational moments + the shared \(k_B\)
constant. It catches wrong electrode selection, double-counting, unit mistakes
in \(k_B T\), or inverted \(C=\mathrm{Var}/(kT)\) algebra. The section also
checks the computed \(C_{\mathrm{diff}}\) against the printed constant
`PIN_C_DIFF 0.38681727072485284` to 1e-14 (recorded constants: `PIN_MEAN_Q 0.5`,
`PIN_VAR_Q 0.01`, `PIN_C_INT 0.5`).

Offline repro (no MetalWalls/LAMMPS binary required):

```bash
python3 - <<'PY'
kB = 8.617333262e-5
Q = [0.4, 0.6, 0.4, 0.6]
mean = sum(Q)/len(Q)
var = sum(q*q for q in Q)/len(Q) - mean*mean
print(mean, var, var/(kB*300), mean/1.0)
PY
# → 0.5  0.01  0.38681727072485284  0.5
```

### Limitations

- **Primary electrode only for moments.** Capacitance moments use type-list
  index 0; other electrodes are summed for diagnostics (`total_charge(e)`,
  `kElectrodeCharge` pack ≤3) but do not enter \(\mathrm{Var}(Q)\).
- **Synthetic pin, not a production trajectory.** The gate uses four
  hand-assigned \(Q\) samples — it validates algebra and electrode selection,
  not sampling quality, equilibration, or areal \(C\) of a real electrolyte.
- **Population variance, no block error.** No running SEM, bootstrap, or
  decorrelation window; long correlated const-pot series need external analysis.
- **No areal normalization.** Results are in e/V (total device), not µF/cm².

## charging-work-energy — Electrode charging work \(-\sum_i \Psi_i Q_i\) {#charging-work-energy}

### Summary

Under constant applied electrode potentials, the classical potential energy of
an electrochemical cell includes a charging-work term exchanged with the
external circuit: \(E_{\mathrm{charge}} = -\sum_i \Psi_i Q_i\).
`cpu::electrodes::ChargingWork` is a `Potential` that reads fixed per-type
applied potentials \(\Psi\) [V] and current per-atom charges \(Q\) [e] from the
Frame, and accumulates energy [eV] into `kEnergy` and the dedicated channel
`kEnergyChargingWork`. It writes no nuclear forces. The gate pins the
closed-form MetalWalls / Scalfi identity on a fixed parallel-plate assignment
to \(-0.4\) eV within \(1\times 10^{-12}\) eV.

### Method

MetalWalls total energy under constant potential (Marin-Laflèche et al. 2020,
eq. 6):

\[
V(Q) = V_c(Q) - \sum_{i=1}^{N} \Psi_i Q_i .
\]

\(V_c(Q)\) is the Coulomb energy of the full charge distribution and the second
term is the work to charge the electrodes at applied potentials \(\Psi_i\).
Scalfi et al. (2020; arXiv:1911.09351 eq. 12) identify
\(q^{\mathrm{T}}\Psi = Q_L\Delta\psi\) as that work for a symmetric
two-electrode cell and place \(-\,q^{\mathrm{T}}\Psi\) in the
constant-potential Boltzmann weight. This section pins only the second term;
Coulomb \(V_c\) and the Born–Oppenheimer charge solve belong to the
constant-potential regressions (`const-pot-bo-parallel-plate`,
`const-pot-md-nvt-short`, …).

**Pin geometry** (symmetric parallel plate, fixed charges as a proxy for
post-const-pot \(Q\)):

| Region | type | \(N\) | \(Q_i\) [e] | \(\Psi\) [V] |
| ------ | ---: | ----: | ----------: | -----------: |
| Left wall | 1 | 4 | \(+0.10\) | \(+0.50\) |
| Right wall | 2 | 4 | \(-0.10\) | \(-0.50\) |

Positions: left \(z = 0\), right \(z = 10\) Å; cubic box \(L = 20\) Å (unused by
the bookkeeping term).

\[
E_{\mathrm{charge}}
  = -\bigl(4\cdot 0.50\cdot 0.10 + 4\cdot(-0.50)\cdot(-0.10)\bigr)
  = -0.4~\mathrm{eV}.
\]

Scalfi check: \(Q_L = 0.4\,\mathrm{e}\), \(\Delta\psi = 1.0\,\mathrm{V}\),
\(q^{\mathrm{T}}\Psi = Q_L\Delta\psi = 0.4\),
\(E = -q^{\mathrm{T}}\Psi = -0.4\) eV.

| Item | Setting |
| ---- | ------- |
| Operator | \(E = -\sum_i \Psi_i Q_i\) over electrode atoms only |
| Atom selection | by integer `kAtom_type` → fixed \(\Psi\) map |
| Charge source | Frame `kAtom_charge` (caller-owned; no solve) |
| Energy channels | `kEnergy` (add) + `kEnergyChargingWork` (add) |
| Forces | none |
| Domain | owned atoms (`nlocal`); global total via the energy `Reduce` when multi-rank |

### Usage

```cpp
cpu::Frame frame(8);
for (int i = 0; i < 4; ++i) {
    frame.span(kAtom_x)[i] = static_cast<double>(i);
    frame.span(kAtom_y)[i] = 0.0;
    frame.span(kAtom_z)[i] = 0.0;
    frame.span(kAtom_type)[i] = 1;
    frame.span(kAtom_charge)[i] = 0.10;   // e
    frame.span(kAtom_mass)[i] = 12.011;
    frame.span(kAtom_tag)[i] = i + 1;
}
for (int i = 4; i < 8; ++i) {
    frame.span(kAtom_x)[i] = static_cast<double>(i - 4);
    frame.span(kAtom_y)[i] = 0.0;
    frame.span(kAtom_z)[i] = 10.0;
    frame.span(kAtom_type)[i] = 2;
    frame.span(kAtom_charge)[i] = -0.10;  // e
    frame.span(kAtom_mass)[i] = 12.011;
    frame.span(kAtom_tag)[i] = i + 1;
}
frame.box = cpu::Box::cubic(20.0);

cpu::Evaluate eval;
eval.set_evaluator(
    cpu::electrodes::ChargingWork(/*types=*/{1, 2},
                                 /*psi_V=*/{0.5, -0.5}));  // V
eval.build();
eval.run(frame);

const double e_tot = frame.meta.get(kEnergy);                 // −0.4 eV
const double e_cw  = frame.meta.get(kEnergyChargingWork);     // −0.4 eV
```

On classical MD, bind the same Component after the const-pot charge update so
\(Q\) on the Frame is current (see **Common setup**). Files: `charging_work.h`,
`src/cpu/electrodes/charging_work.cpp` (launch: \(-\sum\Psi_i Q_i\), meta
write), unit tests in `tests/cpu/electrodes/test_charging_work.cpp` (parallel
plate, Scalfi form, ZeroEnergy clear).

### Validation (oracle: literature)

`oracle_version`: MetalWalls JOSS 5(53) 2373 (2020) eq.6; Scalfi et al. PCCP 22
10480 (2020) / arXiv:1911.09351 eq.12. Public path: `cpu::Evaluate` +
`ChargingWork` only; tolerance absolute \(1\times 10^{-12}\) eV (exact rational
arithmetic).

| Quantity | Atomiverse / pin | Literature identity | Tol | Notes |
| -------- | ---------------: | ------------------: | --- | ----- |
| \(E_{\mathrm{charge}}\) (`kEnergy`) | **−0.4 eV** | **−0.4 eV** | **\(1\times 10^{-12}\) eV** | gate |
| \(E_{\mathrm{charge}}\) (`kEnergyChargingWork`) | **−0.4 eV** | **−0.4 eV** | **\(1\times 10^{-12}\) eV** | dedicated channel |
| Scalfi \(E = -Q_L\Delta\psi\) | −0.4 eV | \(Q_L=0.4\), \(\Delta\psi=1.0\) | \(1\times 10^{-12}\) eV | unit-test cross-check |

### Limitations

- **No Coulomb \(V_c\).** MetalWalls total \(V(Q) = V_c - \sum\Psi Q\) is not
  assembled here; pair/PME/Gaussian-electrode terms live elsewhere.
- **Type-keyed \(\Psi\).** Applied potentials are fixed per atom type at
  construction; re-tagging electrode membership mid-run is not supported.
- **No forces.** Charging work does not contribute nuclear forces; electrode
  freeze / piston / stress are other map items.
- **Literature oracle**, not a LAMMPS ELECTRODE binary pin — by design for this
  formula identity.

## potential-profile-observer — Slab electrostatic potential profile \(\Psi(z)\) {#potential-profile-observer}

### Summary

Planar slab electrostatic potential profile \(\Psi(z)\) for classical electrode
/ electric-double-layer (EDL) slabs. `cpu::PotentialProfile` is a
frame-immutable Observer that bins `kAtom_charge` along \(z\) into a planar
density \(\rho(z)\) and integrates the planar Poisson equation in MD metal units
to \(E_z(z)\) [V/Å] and \(\Psi(z)\) [V]. Gauge: \(\Psi(z_{\min})=0\),
\(E_z(z_{\min})=0\) at the left edge of the bin window. The gate pins the
discrete piecewise-constant Poisson scheme on a synthetic two-sheet capacitor
against the closed-form continuum identity \(E_{\mathrm{gap}}=4\pi
k_{\mathrm{e}}\sigma\).

### Method

For planar symmetry the field and potential satisfy

\[
\frac{\mathrm{d} E_z}{\mathrm{d} z}
  = 4\pi\,k_{\mathrm{e}}\,\rho(z)
  = \frac{\rho(z)}{\varepsilon_0},
\qquad
E_z = -\frac{\mathrm{d}\Psi}{\mathrm{d} z},
\]

with

\[
k_{\mathrm{e}} = 14.399645\,\mathrm{eV\cdot Å\cdot e^{-2}}
\]

(`units::coulomb`, LAMMPS metal `qqr2e`), \(\rho\) [e/Å³], \(E_z\) [V/Å],
\(\Psi\) [V]. Ideal infinite parallel-plate capacitor with surface charge
\(\pm\sigma\):

\[
E_{\mathrm{gap}} = 4\pi k_{\mathrm{e}}\,\sigma
  \qquad\text{(between plates; \(0\) outside)}.
\]

**Discrete scheme.** Charges on owned atoms are histogrammed into
\(n_{\mathrm{bins}}\) bins over \([z_{\min}, z_{\max})\). Per bin,

\[
\rho_i = \frac{Q_i}{A\,\Delta z},\qquad
A = L_x L_y,\qquad
\Delta z = \frac{z_{\max}-z_{\min}}{n_{\mathrm{bins}}}.
\]

Assuming piecewise-constant \(\rho\) in each bin, the left-edge field and
potential are advanced analytically; samples at bin centres are

\[
E_i = E_{\mathrm{left}} + 4\pi k_{\mathrm{e}}\,\rho_i\,\tfrac{\Delta z}{2},
\]

\[
\Psi_i = \Psi_{\mathrm{left}}
  - \Bigl(
      E_{\mathrm{left}}\,\tfrac{\Delta z}{2}
      + \tfrac12\,4\pi k_{\mathrm{e}}\,\rho_i\,
        \bigl(\tfrac{\Delta z}{2}\bigr)^2
    \Bigr),
\]

then the right edge is advanced by a full bin step. The reported potential drop
is \(\Delta\Psi = \Psi(z_{\max})-\Psi(z_{\min})\) with left-edge gauge zero.

| Item | Setting |
| ---- | ------- |
| Geometry | Two point sheets (proxy for planar walls): \(Q=+1\,e\) at \(z=5.5\) Å, \(Q=-1\,e\) at \(z=14.5\) Å |
| Box | orthorhombic \(L_x=L_y=10\) Å, \(L_z=20\) Å |
| Area \(A\) | \(100\) Å² |
| Bin window | \(z\in[0,20)\) Å |
| \(n_{\mathrm{bins}}\) | \(20\) |
| \(\Delta z\) | \(1\) Å |
| \(\sigma = Q/A\) | \(0.01\) e/Å² |
| Binding | `cpu::md::MD` with `VelocityVerlet{0.0}` and a zero `FixAddForce`; observer at default `end_of_step` @ +45; one `run(frame, 1)` |

Closed form / discrete pin:

\[
E_{\mathrm{gap}}^{\mathrm{pin}}
  = 4\pi\times 14.399645\times 0.01
  = 1.80951275785204\,\mathrm{V/Å},
\]

\[
\Psi(z_{\max})^{\mathrm{pin}}
  = -E_{\mathrm{gap}}\times 9
  = -16.28561482066836\,\mathrm{V}
\]

(centre-to-centre plate separation \(9\) Å under this binning). Mid-gap field
is bin 9 (\(z=9.5\)); vacuum bins 0 and 19 have \(E_z=0\); \(\rho\) at the
charged bins is \(\pm 0.01\) e/Å³.

**Out of scope:** non-\(z\) axes, cylindrical / spherical profiles, full 3D
Poisson; time-averaged running \(\langle\rho\rangle\), \(\langle\Psi\rangle\)
over MD trajectories (single-shot / per-launch profiles only).

### Usage

```cpp
cpu::Frame frame(2);
// atom 0: z=5.5 Å, q=+1 e; atom 1: z=14.5 Å, q=-1 e
frame.box = cpu::Box::orthorhombic(10.0, 10.0, 20.0);

cpu::PotentialProfile obs(/*n_bins=*/20, /*z_min=*/0.0, /*z_max=*/20.0);

cpu::md::MD sim;
sim.set_integrator(cpu::md::VelocityVerlet{0.0});
sim.set_evaluator(cpu::md::FixAddForce{0.0, 0.0, 0.0, /*type=*/1});
sim.set_observer_ref(obs);
sim.build();
sim.run(frame, 1);

// Members [e/Å³], [V/Å], [V]
const auto& rho = obs.rho();              // bin 5 → +0.01; bin 14 → −0.01
const auto& Ez  = obs.field();            // mid-gap bin 9 → E_gap
const auto& Psi = obs.psi();              // bin 0 → 0; bin 19 → ΔΨ
const double drop = obs.potential_drop(); // Ψ(z_max) − Ψ(z_min)
const double drop_meta = frame.meta.get(kPotentialDrop);
```

The scalar drop lands on `kPotentialDrop`; full profiles live on the members
`rho()`, `field()`, `psi()`. Files: `potential_profile.h`,
`src/cpu/observe/potential_profile.cpp` (bin charges, planar Poisson, meta
write-back), `frame_schema.h` (`kPotentialDrop`), unit tests in
`tests/cpu/observe/test_potential_profile.cpp`.

### Validation (oracle: literature)

Planar Poisson / parallel-plate continuum (Jackson, *Classical
Electrodynamics*, ch. 1); MetalWalls potential-profile analysis tools; Scalfi
et al. (2020) EDL profiles. The pin is the closed-form \(E_{\mathrm{gap}} = 4\pi
k_{\mathrm{e}}\sigma\) plus the discrete piecewise-constant \(\rho\) integration
on the two-sheet geometry above.

| Quantity | Atomiverse gate / pin | Literature closed form | Tol | Notes |
| -------- | --------------------: | ---------------------: | --- | ----- |
| Continuum \(E_{\mathrm{gap}}\) [V/Å] | **1.80951275785204** | \(4\pi k_{\mathrm{e}}\sigma\) | **1e-14** | `units::coulomb` identity |
| \(E_z\) mid-gap bin 9 [V/Å] | **1.80951275785204** | same \(E_{\mathrm{gap}}\) | **1e-12** | \(z=9.5\) Å |
| \(E_z\) vacuum left bin 0 | **0** | 0 outside plates | **1e-12** | left of \(+Q\) sheet |
| \(E_z\) vacuum right bin 19 | **0** | 0 outside plates | **1e-12** | right of \(-Q\) sheet |
| \(\Psi\) bin 0 [V] | **0** | gauge \(\Psi(z_{\min})=0\) | **1e-12** | left-edge zero |
| \(\Psi\) bin 19 [V] | **−16.28561482066836** | \(-E_{\mathrm{gap}}\times 9\) | **1e-12** | potential drop |
| `potential_drop()` [V] | same as \(\Psi\) bin 19 | same | **1e-12** | member |
| meta `kPotentialDrop` [V] | same | same | **1e-12** | Frame write-back |
| \(\rho\) bin 5 [e/Å³] | **0.01** | \(Q/(A\Delta z)\) | **1e-12** | left sheet |
| \(\rho\) bin 14 [e/Å³] | **−0.01** | \(-Q/(A\Delta z)\) | **1e-12** | right sheet |
| \(\Psi\) drop vs \(-E\times 9\) | identity | algebraic | **1e-12** | centre-to-centre 9 Å |

Tolerance is machine-eps scale: exact double arithmetic of the discrete scheme
plus the shared `units::coulomb` constant. It catches wrong area, inverted
Poisson sign, gauge mistakes, or bin-edge / bin-centre off-by-one errors.
Recorded constants: `PIN_E_GAP 1.80951275785204`, `PIN_PSI_DROP
-16.28561482066836`, `PIN_RHO_LEFT 0.01`, `PIN_RHO_RIGHT -0.01`.

Offline repro (same \(k_e\) as `units::coulomb`):

```bash
python3 - <<'PY'
import math
ke = 14.399645
sigma = 0.01
E_gap = 4.0 * math.pi * ke * sigma
print(E_gap, -E_gap * 9.0)
PY
# → 1.80951275785204  -16.28561482066836
```

### Limitations

- **\(z\)-axis only.** No \(x\)/\(y\) profiles, no cylindrical / spherical
  Poisson.
- **Synthetic pin, not a production EDL trajectory.** The gate uses two static
  point sheets — it validates planar Poisson algebra and units, not sampling
  quality, electrolyte layering, or long-run \(\langle\Psi(z)\rangle\).
- **Per-launch snapshot.** No built-in running or block average of \(\rho\) or
  \(\Psi\); time averages need external accumulation.
- **Left-edge gauge.** \(\Psi(z_{\min})=0\) and \(E_z(z_{\min})=0\) assume a
  vacuum / neutralized reference at the left window edge; asymmetric or charged
  systems need a gauge choice consistent with that assumption.
